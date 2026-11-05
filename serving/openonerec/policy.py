"""Causal, immutable policy-search boundary for the synthetic lifecycle model.

Policies receive snapshots, never the simulator or its future event queue. Actions
are validated before admission. Only WaitAction advances the event clock; it
stops at the next event, bounded timer or aging boundary, and drains all
simultaneous arrivals/completions. Timing
providers must implement pure estimates and reset their own external backend
state between episodes when applicable.
"""
from copy import deepcopy
from dataclasses import asdict, dataclass
import heapq
import random

from .simulator import Simulator, Work
from .state import RecRequest, UserState, nonnegative_int


@dataclass(frozen=True)
class RouteAction:
    request_id: str
    destination: str


@dataclass(frozen=True)
class AcquireAction:
    request_id: str
    mode: str
    source: str = None


@dataclass(frozen=True)
class BatchAction:
    destination: str
    request_ids: tuple

    def __post_init__(self):
        object.__setattr__(self, 'request_ids', tuple(self.request_ids))


@dataclass(frozen=True)
class WaitAction:
    """Wait to next event, aging boundary or timer (default: now + max_wait_ns)."""
    until_ns: int = None


@dataclass(frozen=True)
class RequestObservation:
    request: object
    destination: str = None
    mode: str = None
    source: str = None
    stage: int = 0
    allocated: bool = False
    ready: bool = False
    age_ns: int = 0
    remaining_budget_ns: int = None


@dataclass(frozen=True)
class CopyObservation:
    key: tuple
    location: str
    size_bytes: int
    ready: bool
    valid: bool
    pins: int
    last_used_ns: int


@dataclass(frozen=True)
class DeviceObservation:
    location: str
    capacity_bytes: int
    used_bytes: int
    workspace_bytes: int
    busy: bool


@dataclass(frozen=True)
class UserObservation:
    user_id: str
    observations: int
    last_arrival_ns: int
    arrival_rate_hz: float


@dataclass(frozen=True)
class Observation:
    time_ns: int
    requests: tuple
    copies: tuple
    devices: tuple
    users: tuple
    legal_actions: tuple
    completed_request_ids: tuple
    done: bool


@dataclass(frozen=True)
class StepResult:
    observation: Observation
    done: bool


class IllegalAction(ValueError):
    """Rejected actions leave the episode unchanged and execute no backend work."""


class PolicyEnvironment:
    """Resettable event-driven environment, independent of Simulator.run policy.

    Route chooses a GPU, acquire chooses an exact current prefix source, and
    batch chooses an explicit same-stage set. Admission is FCFS per device;
    aged ready work must be included before younger work on that device. Waiting
    cannot skip an aged action that is currently feasible. legal_actions lists
    legal singleton batches; larger batches can be checked with validate().
    """
    def __init__(self, gpu_capacities, timing, *, max_wait_ns=1_000_000_000, **simulator_options):
        if nonnegative_int(max_wait_ns, 'max_wait_ns') == 0:
            raise ValueError('max_wait_ns must be positive')
        self.max_wait_ns = max_wait_ns
        self._capacities = dict(gpu_capacities)
        self._timing = timing
        self._options = deepcopy(simulator_options)
        self._sim = None
        self._pending = {}
        self._decisions = []
        self.seed = None

    def reset(self, requests, *, seed=0):
        nonnegative_int(seed, 'seed')
        requests = tuple(requests)
        if any(not isinstance(r, RecRequest) for r in requests):
            raise TypeError('requests must contain immutable RecRequest values')
        if len({r.request_id for r in requests}) != len(requests):
            raise ValueError('request_id values must be unique')
        self._sim = Simulator(self._capacities, self._timing, **self._options)
        for request in requests:
            peak = ((len(request.prefix_token_ids) + 3 * self._sim.beam_width)
                    * self._sim.kv_bytes_per_token + self._sim.scratch_bytes)
            if peak > max(self._sim.gpus.values()):
                raise ValueError(f'request {request.request_id} peak workspace exceeds every GPU capacity')
        self.seed = seed
        self.rng = random.Random(seed)
        self._pending = {}
        self._decisions = []
        for request in sorted(requests, key=lambda r: (r.arrival_time_ns, r.request_id)):
            self._sim._event(request.arrival_time_ns, 'arrival', request)
        self._drain_current()
        return self.observe()

    def _require_reset(self):
        if self._sim is None:
            raise RuntimeError('reset must be called before using the environment')

    @property
    def done(self):
        self._require_reset()
        return not (self._pending or self._sim.events or any(self._sim.queues.values()))

    def _works(self):
        return {w.request.request_id: w for q in self._sim.queues.values() for w in q}

    def _aged(self, request):
        return self._sim.now - request.arrival_time_ns >= self._sim.aging_threshold_ns

    @staticmethod
    def _order(request):
        return request.arrival_time_ns, request.request_id

    def _acquisition(self, work, action):
        s = self._sim
        if work.allocated:
            raise IllegalAction('request is already allocated')
        queue = s.queues[work.destination]
        if next(w for w in queue if not w.allocated) is not work:
            raise IllegalAction('admission must respect device FCFS order')
        source = action.source
        if action.mode == 'recompute':
            if source is not None:
                raise IllegalAction('recompute has no source')
            copy = None
        else:
            expected = ('local' if source == work.destination else
                        'cpu_reload' if source == 'cpu' else 'remote')
            if source not in s.directory.capacities or action.mode != expected:
                raise IllegalAction('acquisition mode/source mismatch')
            copy = s.directory.valid(work.request.prefix_key, source, s.now)
            if copy is None:
                raise IllegalAction('prefix source is not valid and ready')
        private = 0 if action.mode == 'local' else len(work.request.prefix_token_ids) * s.kv_bytes_per_token
        required = private + 3 * s.beam_width * s.kv_bytes_per_token + s.scratch_bytes
        if not s.directory.can_fit(work.destination, required, protected=copy):
            raise IllegalAction('insufficient unpinned memory for acquisition')

    def validate(self, action):
        """Pure legality check; no estimates, execution, eviction or pin updates."""
        self._require_reset()
        s = self._sim
        if self.done:
            raise IllegalAction('episode is complete')
        works = self._works()
        if isinstance(action, (RouteAction, AcquireAction)):
            if not isinstance(action.request_id, str) or not action.request_id:
                raise IllegalAction('request_id must be a nonempty string')
        if isinstance(action, (RouteAction, BatchAction)):
            if not isinstance(action.destination, str) or not action.destination:
                raise IllegalAction('destination must be a nonempty string')
        if isinstance(action, AcquireAction):
            if not isinstance(action.mode, str) or (action.source is not None and not isinstance(action.source, str)):
                raise IllegalAction('mode and source must be strings')
        if isinstance(action, BatchAction):
            if any(not isinstance(i, str) or not i for i in action.request_ids):
                raise IllegalAction('batch request IDs must be nonempty strings')
        if isinstance(action, RouteAction):
            request = self._pending.get(action.request_id)
            if request is None or action.destination not in s.gpus:
                raise IllegalAction('route requires an arrived, unrouted request and known GPU')
            aged = sorted((r for r in self._pending.values() if self._aged(r)), key=self._order)
            if aged and request is not aged[0]:
                raise IllegalAction('oldest aged request must be routed first')
            peak = (len(request.prefix_token_ids) + 3 * s.beam_width) * s.kv_bytes_per_token + s.scratch_bytes
            if peak > s.gpus[action.destination]:
                raise IllegalAction('request peak workspace exceeds destination capacity')
        elif isinstance(action, AcquireAction):
            work = works.get(action.request_id)
            if work is None:
                raise IllegalAction('acquisition requires a routed request')
            self._acquisition(work, action)
        elif isinstance(action, BatchAction):
            if action.destination not in s.gpus or s.busy_until[action.destination] > s.now:
                raise IllegalAction('batch destination must be an idle GPU')
            ids = action.request_ids
            if not ids or len(ids) > s.max_batch_size or len(set(ids)) != len(ids):
                raise IllegalAction('batch must be nonempty, unique and within max_batch_size')
            batch = [works.get(i) for i in ids]
            if any(w is None or w.destination != action.destination or not w.allocated or not w.ready for w in batch):
                raise IllegalAction('batch members must be ready on the destination')
            if len({w.stage for w in batch}) != 1:
                raise IllegalAction('batch members must have the same SID stage')
            aged = sorted((w for w in s.queues[action.destination] if w.ready and self._aged(w.request)),
                          key=lambda w: self._order(w.request))
            if aged and aged[0] not in batch:
                raise IllegalAction('batch must include the oldest aged ready request')
        elif isinstance(action, WaitAction):
            if action.until_ns is not None:
                try:
                    nonnegative_int(action.until_ns, 'until_ns')
                except ValueError as error:
                    raise IllegalAction(str(error)) from error
                if not s.now < action.until_ns <= s.now + self.max_wait_ns:
                    raise IllegalAction('wait timer must be in the future and within max_wait_ns')
            if any(self._action_is_aged(a) for a in self._candidate_actions() if self._legal(a)):
                raise IllegalAction('cannot wait while aged work can make progress')
        else:
            raise IllegalAction('unknown action type')
        return True

    def estimate(self, action):
        """Pure hypothetical nanoseconds for a legal action, never backend execution.

        Routes estimate all three stages with recomputation, excluding queueing.
        Acquisition estimates transport only (local/recompute cost zero); batch
        estimates the exact selected stage. Wait has no forecast: revealing its
        next external event time would leak future workload information.
        """
        self.validate(action)
        s = self._sim
        if isinstance(action, RouteAction):
            work = Work(self._pending[action.request_id], action.destination, 'recompute')
            return sum(s._duration(action.destination, s._graph(stage, [work]), estimate=True)
                       for stage in range(3))
        if isinstance(action, AcquireAction):
            if action.mode in ('recompute', 'local'):
                return 0
            work = self._works()[action.request_id]
            return s._transfer(action.source, work.destination,
                               len(work.request.prefix_token_ids) * s.kv_bytes_per_token, estimate=True)
        if isinstance(action, BatchAction):
            works = self._works()
            batch = [works[i] for i in action.request_ids]
            return s._duration(action.destination, s._graph(batch[0].stage, batch), estimate=True)
        raise IllegalAction('wait has no causal duration estimate')

    def _action_is_aged(self, action):
        if isinstance(action, RouteAction):
            return self._aged(self._pending[action.request_id])
        if isinstance(action, AcquireAction):
            return self._aged(self._works()[action.request_id].request)
        if isinstance(action, BatchAction):
            return any(self._aged(self._works()[i].request) for i in action.request_ids)
        return False

    def _legal(self, action):
        try:
            self.validate(action)
            return True
        except IllegalAction:
            return False

    def _candidate_actions(self):
        s = self._sim
        for request in sorted(self._pending.values(), key=self._order):
            for destination in s.gpus:
                yield RouteAction(request.request_id, destination)
        for queue in s.queues.values():
            for work in queue:
                rid = work.request.request_id
                if not work.allocated:
                    yield AcquireAction(rid, 'recompute')
                    for source in s.directory.capacities:
                        mode = 'local' if source == work.destination else 'cpu_reload' if source == 'cpu' else 'remote'
                        yield AcquireAction(rid, mode, source)
                elif work.ready:
                    yield BatchAction(work.destination, (rid,))

    def observe(self):
        self._require_reset()
        s = self._sim
        requests = []
        for request in sorted(self._pending.values(), key=self._order):
            age = s.now - request.arrival_time_ns
            requests.append(RequestObservation(request, age_ns=age,
                            remaining_budget_ns=request.budget_ns-age if request.budget_ns else None))
        for queue in s.queues.values():
            for w in queue:
                age = s.now - w.request.arrival_time_ns
                requests.append(RequestObservation(w.request, w.destination, w.mode, w.source,
                                w.stage, w.allocated, w.ready, age,
                                w.request.budget_ns-age if w.request.budget_ns else None))
        copies = tuple(CopyObservation(c.key, c.location, c.size_bytes, c.ready,
                       s.directory.valid(c.key, c.location, s.now) is not None, c.pins, c.last_used_ns)
                       for _, c in sorted(s.directory.copies.items()))
        devices = tuple(DeviceObservation(loc, cap, s.directory.used(loc), s.directory.workspace[loc],
                        s.busy_until.get(loc, 0) > s.now) for loc, cap in s.directory.capacities.items())
        users = tuple(UserObservation(uid, u.observations, u.last_arrival_ns,
                      u.rate_at(s.now, s.directory.half_life_ns)) for uid, u in sorted(s.users.items()))
        actions = tuple(a for a in self._candidate_actions() if self._legal(a))
        if self._legal(WaitAction()):
            actions += (WaitAction(),)
        return Observation(s.now, tuple(requests), copies, devices, users, actions,
                           tuple(r['request_id'] for r in s.responses), self.done)

    def _drain_current(self):
        s = self._sim
        while s.events and s.events[0][0] == s.now:
            _, _, kind, payload = heapq.heappop(s.events)
            if kind == 'arrival':
                s.users.setdefault(payload.user_id, UserState()).observe(s.now)
                self._pending[payload.request_id] = payload
            elif kind == 'acquired':
                work, source = payload
                source.pins -= 1
                source.last_used_ns = s.now
                work.ready = True
                s._publish_prefix(work)
            elif kind == 'batch':
                s._finish_batch(*payload)
            elif kind == 'writeback':
                source, copy, request_id = payload
                source.pins -= 1
                copy.ready = True
                copy.last_used_ns = s.now
                s.records.append(dict(event='writeback_complete', time_ns=s.now, request_id=request_id))
        s.directory.expire(s.now)

    def step(self, action):
        self.validate(action)
        s = self._sim
        before = s.now
        if isinstance(action, RouteAction):
            request = self._pending.pop(action.request_id)
            s.queues[action.destination].append(Work(request, action.destination, 'unacquired'))
            s.records.append(dict(event='route', time_ns=s.now, request_id=action.request_id,
                                  destination=action.destination, mode='unacquired'))
        elif isinstance(action, AcquireAction):
            work = self._works()[action.request_id]
            if not s._allocate(work, (action.mode, action.source)):
                raise RuntimeError('validated acquisition unexpectedly failed')
        elif isinstance(action, BatchAction):
            works = self._works()
            batch = [works[i] for i in action.request_ids]
            duration = s._duration(action.destination, s._graph(batch[0].stage, batch))
            for work in batch:
                work.ready = False
                if work.first_start_ns is None:
                    work.first_start_ns = s.now
            s.busy_until[action.destination] = s.now + duration
            s._event(s.now + duration, 'batch', (action.destination, batch))
            s.records.append(dict(event='batch_start', time_ns=s.now, destination=action.destination,
                                  stage=batch[0].stage, request_ids=list(action.request_ids),
                                  remaining_budget_ns=[None if not w.request.budget_ns else
                                      w.request.budget_ns - (s.now - w.request.arrival_time_ns)
                                      for w in batch], duration_ns=duration))
        else:
            wakeups = [s.events[0][0]] if s.events else []
            wakeups.append(action.until_ns if action.until_ns is not None else s.now + self.max_wait_ns)
            # Never skip the point where currently feasible work becomes aged.
            for candidate in self._candidate_actions():
                if not self._legal(candidate):
                    continue
                ids = candidate.request_ids if isinstance(candidate, BatchAction) else (candidate.request_id,)
                for rid in ids:
                    request = self._pending.get(rid) or self._works()[rid].request
                    boundary = request.arrival_time_ns + s.aging_threshold_ns
                    if boundary > s.now:
                        wakeups.append(boundary)
            s.now = min(wakeups)
        self._drain_current()
        self._decisions.append(dict(index=len(self._decisions), time_ns=before,
                                   action=type(action).__name__, arguments=asdict(action), next_time_ns=s.now))
        observation = self.observe()
        return StepResult(observation, observation.done)

    def result(self):
        self._require_reset()
        if not self.done:
            raise RuntimeError('episode is not complete')
        s = self._sim
        return deepcopy(dict(timing_label=s.timing.label, responses=s.responses, events=s.records,
                             drained_time_ns=s.now, peak_bytes=s.directory.peak,
                             evictions=s.directory.evictions, seed=self.seed, decisions=self._decisions))


class FCFSReferencePolicy:
    """Deterministic legal-action baseline, not the legacy slack-aware policy.

    Routes to the least populated visible queue; prefers compatible local,
    remote then CPU prefixes; launches same-stage FCFS singleton batches. The
    singleton baseline deliberately makes no calibrated cost or optimality claim.
    """
    def __call__(self, observation):
        actions = observation.legal_actions
        routes = [a for a in actions if isinstance(a, RouteAction)]
        if routes:
            first = routes[0].request_id
            return min((a for a in routes if a.request_id == first),
                       key=lambda a: (sum(r.destination == a.destination for r in observation.requests),
                                      a.destination))
        acquisitions = [a for a in actions if isinstance(a, AcquireAction)]
        if acquisitions:
            first = acquisitions[0].request_id
            priority = {'local': 0, 'remote': 1, 'cpu_reload': 2, 'recompute': 3}
            return min((a for a in acquisitions if a.request_id == first),
                       key=lambda a: (priority[a.mode], a.source or ''))
        batches = [a for a in actions if isinstance(a, BatchAction)]
        if batches:
            return batches[0]
        if WaitAction() in actions:
            return WaitAction()
        raise RuntimeError('no legal action can make progress')
