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
    cached_prefix_tokens: int = 0
    source_copy: object = None


class Simulator:
    def __init__(self, gpu_capacities, timing, *, cpu_capacity_bytes=0,
                 beam_width=8, kv_bytes_per_token=1024, scratch_bytes=0,
                 max_batch_size=8, half_life_ns=1_000_000_000, ttl_ns=0,
                 writeback=True, aging_threshold_ns=1_000_000_000,
                 execution_mode='stage', max_batch_rows=None, max_batch_tokens=None,
                 max_active_requests=None):
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
        if execution_mode not in ('stage', 'wave', 'fixed_step'):
            raise ValueError('execution_mode must be stage, wave or fixed_step')
        for name, value in [('max_batch_rows', max_batch_rows),
                            ('max_batch_tokens', max_batch_tokens),
                            ('max_active_requests', max_active_requests)]:
            if value is not None and nonnegative_int(value, name) == 0:
                raise ValueError(f'{name} must be positive')
        self.execution_mode = execution_mode
        self.max_batch_rows = max_batch_rows
        self.max_batch_tokens = max_batch_tokens
        self.max_active_requests = max_active_requests
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
                                 self.rows(works[0].request), [w.mode for w in works],
                                 cached_prefix_tokens=[w.cached_prefix_tokens for w in works])

    def rows(self, request):
        return request.beam_rows if request.beam_rows is not None else self.beam_width

    def peak_workspace(self, request):
        return ((len(request.prefix_token_ids) + 3 * self.rows(request))
                * self.kv_bytes_per_token + self.scratch_bytes)

    def batch_error(self, batch):
        """Pure resource/shape validation shared by automatic and policy scheduling.

        Row capacity reserves each request's full beam, including at stage zero.
        Token capacity bounds peak attention positions times reserved beam rows;
        it is a conservative work bound, not resident shared-prefix KV bytes.
        """
        if not batch or len(batch) > self.max_batch_size:
            return 'batch exceeds request capacity'
        if self.execution_mode == 'stage' and len({w.stage for w in batch}) != 1:
            return 'batch members must have the same SID stage'
        if self.execution_mode == 'fixed_step' and any(w.stage != 0 for w in batch):
            return 'fixed-step batches must begin at SID stage zero'
        if self.max_batch_rows is not None and sum(self.rows(w.request) for w in batch) > self.max_batch_rows:
            return 'batch exceeds beam-row capacity'
        tokens = sum(self.rows(w.request) * (len(w.request.prefix_token_ids) + 3) for w in batch)
        if self.max_batch_tokens is not None and tokens > self.max_batch_tokens:
            return 'batch exceeds token capacity'
        return None

    def request_error(self, request):
        work = Work(request, '', 'recompute')
        return self.batch_error([work])

    def _batch_duration(self, destination, batch, estimate=False):
        # Heterogeneous waves are decomposed into serial homogeneous launches.
        # No fused-kernel or overlap speedup is inferred from arithmetic timing.
        stages = range(3) if self.execution_mode == 'fixed_step' else sorted({w.stage for w in batch})
        duration = 0
        for stage in stages:
            members = batch if self.execution_mode == 'fixed_step' else [w for w in batch if w.stage == stage]
            for width in sorted({self.rows(w.request) for w in members}):
                group = [w for w in members if self.rows(w.request) == width]
                duration += self._duration(destination, self._graph(stage, group), estimate=estimate)
        return duration

    def _launch_batch(self, destination, batch):
        error = self.batch_error(batch)
        if error:
            raise ValueError(error)
        duration = self._batch_duration(destination, batch)
        for work in batch:
            work.ready = False
            if work.first_start_ns is None:
                work.first_start_ns = self.now
        self.busy_until[destination] = self.now + duration
        self._event(self.now + duration, 'batch', (destination, batch))
        self.records.append(dict(event='batch_start', time_ns=self.now, destination=destination,
                                 stage=batch[0].stage, stages=[w.stage for w in batch],
                                 execution_mode=self.execution_mode,
                                 beam_rows=[self.rows(w.request) for w in batch],
                                 request_ids=[w.request.request_id for w in batch],
                                 remaining_budget_ns=[None if not w.request.budget_ns else
                                     w.request.budget_ns-(self.now-w.request.arrival_time_ns) for w in batch],
                                 duration_ns=duration))

    def _choices(self, request, destination):
        yield 'recompute', None, 0
        for location in self.directory.capacities:
            self.directory.expire(self.now, request.prefix_key, location)
            copy, reused = self.directory.best_prefix(request, location, self.now)
            if copy is not None:
                if location == destination:
                    yield 'local', location, 0
                else:
                    mode = 'cpu_reload' if location == 'cpu' else 'remote'
                    yield mode, location, self._transfer(location, destination,
                                                        reused * self.kv_bytes_per_token, estimate=True)

    def _route(self, request):
        options = []
        required = self.peak_workspace(request)
        if self.request_error(request):
            raise ValueError(self.request_error(request))
        for destination, capacity in self.gpus.items():
            if required > capacity:
                continue
            for mode, source, acquisition in self._choices(request, destination):
                work = Work(request, destination, mode, source)
                if source:
                    _, work.cached_prefix_tokens = self.directory.best_prefix(request, source, self.now)
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

    def _allocate(self, work, acquisition=None):
        request, destination = work.request, work.destination
        key = request.prefix_key
        if self.max_active_requests is not None and sum(w.allocated for w in self.queues[destination]) >= self.max_active_requests:
            return False
        # Re-evaluate the joint acquisition mode at admission: queued copies may
        # have been evicted, or a prior same-user request may have published one.
        if acquisition is None:
            modes = []
            for mode, source, delay in self._choices(request, destination):
                candidate = Work(request, destination, mode, source)
                copy = None
                if source:
                    copy, candidate.cached_prefix_tokens = self.directory.best_prefix(request, source, self.now)
                private = (0 if mode == 'local' and copy.key == key else
                           len(request.prefix_token_ids) * self.kv_bytes_per_token)
                required = private + 3 * self.rows(request) * self.kv_bytes_per_token + self.scratch_bytes
                if not self.directory.can_fit(destination, required, protected=copy):
                    continue
                cost = delay + self._duration(destination, self._graph(0, [candidate]), estimate=True)
                modes.append((cost, mode, source or '', delay))
            if not modes:
                return False
            _, work.mode, source, delay = min(modes)
            work.source = source or None
        else:
            work.mode, work.source = acquisition
            source = work.source
        copy, reused = (self.directory.best_prefix(request, source, self.now)
                        if source else (None, 0))
        work.cached_prefix_tokens = reused
        if copy:
            copy.pins += 1  # Pin source before eviction for target capacity.
        prefix_size = len(request.prefix_token_ids) * self.kv_bytes_per_token
        exact_local = work.mode == 'local' and copy is not None and copy.key == key
        private = 0 if exact_local else prefix_size
        branch = 3 * self.rows(request) * self.kv_bytes_per_token + self.scratch_bytes
        if not self.directory.make_room(destination, private + branch, self.now):
            if copy:
                copy.pins -= 1
            return False
        self.directory.workspace[destination] += private + branch
        self.directory.record_peak(destination)
        work.private_prefix_bytes, work.branch_bytes = private, branch
        work.allocated = True
        if exact_local:
            work.prefix_copy = copy
            copy.last_used_ns = self.now
            work.ready = True
        elif work.mode == 'local' and copy:
            # Partial local reuse keeps a conservative full private destination
            # allocation, rather than an unsafe alias to an evictable parent.
            work.source_copy = copy
            copy.last_used_ns = self.now
            work.ready = True
        elif copy:
            copy.last_used_ns = self.now
            delay = self._transfer(source, destination, reused * self.kv_bytes_per_token)
            self._event(self.now + delay, 'acquired', (work, copy))
        else:
            work.ready = True
        self.records.append(dict(event='acquire', time_ns=self.now, request_id=request.request_id,
                                 destination=destination, mode=work.mode, ready=work.ready,
                                 reused_prefix_tokens=reused,
                                 missing_prefix_tokens=len(request.prefix_token_ids)-reused))
        return True

    def _publish_prefix(self, work, completed=False):
        # A transferred partial prefix is not a completed full prefix. Publication
        # waits for incremental stage-0 prefill to finish, never transfer start.
        if not completed and work.cached_prefix_tokens < len(work.request.prefix_token_ids):
            return
        if work.source_copy is not None:
            work.source_copy.pins -= 1
            work.source_copy.last_used_ns = self.now
            work.source_copy = None
        if work.prefix_copy is not None:
            return
        directory, key, destination = self.directory, work.request.prefix_key, work.destination
        directory.expire(self.now, key, destination)
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
            copy = directory.reserve(key, destination, size, self.now,
                                     work.request.prefix_token_ids)
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
        self.directory.expire(self.now, source.key, 'cpu')
        if self.directory.valid(source.key, 'cpu', self.now):
            return
        copy = self.directory.reserve(source.key, 'cpu', source.size_bytes, self.now,
                                      source.prefix_token_ids)
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
                self._publish_prefix(work, completed=True)
            if self.execution_mode == 'fixed_step':
                work.stage = 2
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
            batch = []
            for work in ready:
                if self.batch_error(batch + [work]) is None:
                    batch.append(work)
            self._launch_batch(destination, batch)

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
