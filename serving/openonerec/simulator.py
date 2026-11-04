"""Event-driven policy experiment for fixed-beam three-SID recommendation.

This is an opt-in frontend, not a calibrated OpenOneRec performance model.
"""
from dataclasses import dataclass
import heapq

from .state import Directory, UserState, nonnegative_int
from .timing import build_stage_graph


@dataclass
class Work:
    request: object
    destination: str
    mode: str
    source: str = None
    stage: int = 0
    ready: bool = False
    allocated: bool = False
    prefix_copy: object = None
    private_prefix_bytes: int = 0
    branch_bytes: int = 0
    first_start_ns: int = None


class Simulator:
    def __init__(self, gpu_capacities, timing, *, cpu_capacity_bytes=0,
                 beam_width=8, kv_bytes_per_token=1024, scratch_bytes=0,
                 max_batch_size=8, half_life_ns=1_000_000_000, ttl_ns=0,
                 writeback=True, aging_threshold_ns=1_000_000_000):
        if not gpu_capacities or 'cpu' in gpu_capacities:
            raise ValueError('provide nonempty named GPU capacities; cpu is reserved')
        for name, value in gpu_capacities.items():
            if not isinstance(name, str) or not name:
                raise ValueError('GPU names must be nonempty strings')
            nonnegative_int(value, 'GPU capacity')
        for name, value in [('beam_width', beam_width), ('kv_bytes_per_token', kv_bytes_per_token),
                            ('max_batch_size', max_batch_size), ('half_life_ns', half_life_ns),
                            ('aging_threshold_ns', aging_threshold_ns)]:
            if nonnegative_int(value, name) == 0:
                raise ValueError(f'{name} must be positive')
        for name, value in [('cpu_capacity_bytes', cpu_capacity_bytes),
                            ('scratch_bytes', scratch_bytes), ('ttl_ns', ttl_ns)]:
            nonnegative_int(value, name)
        if not isinstance(writeback, bool):
            raise ValueError('writeback must be a boolean')
        self.gpus = dict(gpu_capacities)
        self.timing = timing
        self.beam_width = beam_width
        self.kv_bytes_per_token = kv_bytes_per_token
        self.scratch_bytes = scratch_bytes
        self.max_batch_size = max_batch_size
        self.writeback = writeback
        self.aging_threshold_ns = aging_threshold_ns
        self.users = {}
        self.directory = Directory(dict(gpu_capacities, cpu=cpu_capacity_bytes),
                                   self.users, half_life_ns, ttl_ns)
        self.queues = {name: [] for name in self.gpus}
        self.busy_until = {name: 0 for name in self.gpus}
        self.events = []
        self.sequence = 0
        self.now = 0
        self.records = []
        self.responses = []
        self.started = False

    def _event(self, at, kind, payload):
        nonnegative_int(at, 'event time')
        self.sequence += 1
        heapq.heappush(self.events, (at, self.sequence, kind, payload))

    def _duration(self, destination, graph, estimate=False):
        method = self.timing.estimate_duration_ns if estimate else self.timing.duration_ns
        value = method(destination, graph)
        return nonnegative_int(value, 'timing duration_ns')

    def _transfer(self, source, destination, size, estimate=False):
        method = self.timing.estimate_transfer_ns if estimate else self.timing.transfer_ns
        value = method(source, destination, size)
        return nonnegative_int(value, 'timing transfer_ns')

    def _graph(self, stage, works):
        return build_stage_graph(stage, [len(w.request.prefix_token_ids) for w in works],
                                 self.beam_width, [w.mode for w in works])

    def _choices(self, request, destination):
        yield 'recompute', None, 0
        for location in self.directory.capacities:
            if self.directory.valid(request.prefix_key, location, self.now):
                if location == destination:
                    yield 'local', location, 0
                else:
                    mode = 'cpu_reload' if location == 'cpu' else 'remote'
                    yield mode, location, self._transfer(location, destination,
                                                        len(request.prefix_token_ids) * self.kv_bytes_per_token, estimate=True)

    def _route(self, request):
        options = []
        required = (len(request.prefix_token_ids) + 3 * self.beam_width) * self.kv_bytes_per_token + self.scratch_bytes
        for destination, capacity in self.gpus.items():
            if required > capacity:
                continue
            for mode, source, acquisition in self._choices(request, destination):
                work = Work(request, destination, mode, source)
                service = sum(self._duration(destination, self._graph(stage, [work]), estimate=True) for stage in range(3))
                # Queue estimate is explicitly a policy heuristic, not a backend clock.
                queue = len(self.queues[destination]) * service
                score = max(self.now, self.busy_until[destination]) + queue + acquisition + service
                options.append((score, destination, mode, source or ''))
        if not options:
            raise ValueError(f'request {request.request_id} peak workspace exceeds every GPU capacity')
        _, destination, mode, source = min(options)
        self.queues[destination].append(Work(request, destination, mode, source or None))
        self.records.append(dict(event='route', time_ns=self.now, request_id=request.request_id,
                                 destination=destination, mode=mode))

    def _allocate(self, work):
        request, destination = work.request, work.destination
        key = request.prefix_key
        # Re-evaluate the joint acquisition mode at admission: queued copies may
        # have been evicted, or a prior same-user request may have published one.
        modes = []
        for mode, source, delay in self._choices(request, destination):
            candidate = Work(request, destination, mode, source)
            cost = delay + self._duration(destination, self._graph(0, [candidate]), estimate=True)
            modes.append((cost, mode, source or '', delay))
        _, work.mode, source, delay = min(modes)
        work.source = source or None
        copy = self.directory.valid(key, source, self.now) if source else None
        if copy:
            copy.pins += 1  # Pin source before eviction for target capacity.
        prefix_size = len(request.prefix_token_ids) * self.kv_bytes_per_token
        private = 0 if work.mode == 'local' else prefix_size
        branch = 3 * self.beam_width * self.kv_bytes_per_token + self.scratch_bytes
        if not self.directory.make_room(destination, private + branch, self.now):
            if copy:
                copy.pins -= 1
            return False
        self.directory.workspace[destination] += private + branch
        self.directory.record_peak(destination)
        work.private_prefix_bytes, work.branch_bytes = private, branch
        work.allocated = True
        if work.mode == 'local':
            work.prefix_copy = copy
            copy.last_used_ns = self.now
            work.ready = True
        elif copy:
            copy.last_used_ns = self.now
            delay = self._transfer(source, destination, prefix_size)
            self._event(self.now + delay, 'acquired', (work, copy))
        else:
            work.ready = True
        self.records.append(dict(event='acquire', time_ns=self.now, request_id=request.request_id,
                                 destination=destination, mode=work.mode, ready=work.ready))
        return True

    def _publish_prefix(self, work):
        if work.prefix_copy is not None:
            return
        directory, key, destination = self.directory, work.request.prefix_key, work.destination
        existing = directory.valid(key, destination, self.now)
        if existing:
            existing.pins += 1
            directory.workspace[destination] -= work.private_prefix_bytes
            work.private_prefix_bytes = 0
            work.prefix_copy = existing
        elif (key, destination) not in directory.copies:
            # Atomic ownership transfer: workspace bytes become retained bytes.
            size = work.private_prefix_bytes
            directory.workspace[destination] -= size
            copy = directory.reserve(key, destination, size, self.now)
            if copy is None:
                directory.workspace[destination] += size
                return
            copy.ready, copy.pins = True, 1
            work.prefix_copy = copy
            work.private_prefix_bytes = 0
            self.records.append(dict(event='publish', time_ns=self.now, location=destination,
                                     request_id=work.request.request_id))

    def _start_writeback(self, work):
        source = work.prefix_copy
        if not self.writeback or source is None or not self.directory.capacities['cpu']:
            return
        if self.directory.valid(source.key, 'cpu', self.now):
            return
        copy = self.directory.reserve(source.key, 'cpu', source.size_bytes, self.now)
        if copy is None:
            return
        source.pins += 1
        delay = self._transfer(work.destination, 'cpu', source.size_bytes)
        self._event(self.now + delay, 'writeback', (source, copy, work.request.request_id))
        self.records.append(dict(event='writeback_start', time_ns=self.now,
                                 request_id=work.request.request_id))

    def _finish_batch(self, destination, batch):
        for work in batch:
            if work.stage == 0:
                self._publish_prefix(work)
            if work.stage < 2:
                work.stage += 1
                work.ready = True
            else:
                self._start_writeback(work)
                if work.prefix_copy:
                    work.prefix_copy.pins -= 1
                    work.prefix_copy.last_used_ns = self.now
                self.directory.workspace[destination] -= work.branch_bytes + work.private_prefix_bytes
                self.queues[destination].remove(work)
                request = work.request
                self.responses.append(dict(request_id=request.request_id, user_id=request.user_id,
                                           arrival_time_ns=request.arrival_time_ns,
                                           response_time_ns=self.now, latency_ns=self.now-request.arrival_time_ns,
                                           waiting_ns=work.first_start_ns-request.arrival_time_ns,
                                           destination=destination, mode=work.mode,
                                           deadline_missed=bool(request.budget_ns and self.now-request.arrival_time_ns > request.budget_ns)))

    def _schedule(self):
        for destination, queue in self.queues.items():
            if self.busy_until[destination] > self.now:
                continue
            # FCFS admission, independent of observed hotness. If the oldest
            # unallocated request cannot fit, later requests cannot jump it.
            for work in queue:
                if not work.allocated and not self._allocate(work):
                    break
            ready = [w for w in queue if w.allocated and w.ready]
            if not ready:
                continue
            def priority(work):
                request = work.request
                age = self.now - request.arrival_time_ns
                if age >= self.aging_threshold_ns:
                    return (0, request.arrival_time_ns, request.request_id)
                remaining = sum(self._duration(destination, self._graph(stage, [work]), estimate=True)
                                for stage in range(work.stage, 3))
                slack = (request.budget_ns - age - remaining) if request.budget_ns else float('inf')
                return (1, slack, request.arrival_time_ns, request.request_id)

            ready.sort(key=priority)
            oldest = ready[0]
            batch = [w for w in ready if w.stage == oldest.stage][:self.max_batch_size]
            duration = self._duration(destination, self._graph(oldest.stage, batch))
            for work in batch:
                work.ready = False
                if work.first_start_ns is None:
                    work.first_start_ns = self.now
            self.busy_until[destination] = self.now + duration
            self._event(self.now + duration, 'batch', (destination, batch))
            self.records.append(dict(event='batch_start', time_ns=self.now, destination=destination,
                                     stage=oldest.stage, request_ids=[w.request.request_id for w in batch],
                                     remaining_budget_ns=[None if not w.request.budget_ns else
                                                          w.request.budget_ns-(self.now-w.request.arrival_time_ns)
                                                          for w in batch], duration_ns=duration))

    def run(self, requests):
        if self.started:
            raise ValueError('Simulator instances are single-use')
        self.started = True
        requests = list(requests)
        if len({r.request_id for r in requests}) != len(requests):
            raise ValueError('request_id values must be unique')
        for request in sorted(requests, key=lambda r: (r.arrival_time_ns, r.request_id)):
            self._event(request.arrival_time_ns, 'arrival', request)
        while self.events:
            self.now = self.events[0][0]
            # Drain all simultaneous arrivals/completions before scheduling.
            while self.events and self.events[0][0] == self.now:
                _, _, kind, payload = heapq.heappop(self.events)
                if kind == 'arrival':
                    self.users.setdefault(payload.user_id, UserState()).observe(self.now)
                    self._route(payload)
                elif kind == 'acquired':
                    work, source = payload
                    source.pins -= 1
                    source.last_used_ns = self.now
                    work.ready = True
                    self._publish_prefix(work)
                elif kind == 'batch':
                    self._finish_batch(*payload)
                elif kind == 'writeback':
                    source, copy, request_id = payload
                    source.pins -= 1
                    copy.ready = True
                    copy.last_used_ns = self.now
                    self.records.append(dict(event='writeback_complete', time_ns=self.now,
                                             request_id=request_id))
            self._schedule()
        if any(self.queues.values()):
            raise RuntimeError('scheduler cannot make progress with current memory capacities')
        return dict(timing_label=self.timing.label, responses=self.responses, events=self.records,
                    drained_time_ns=self.now, peak_bytes=self.directory.peak,
                    evictions=self.directory.evictions)
