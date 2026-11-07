"""Deterministic offline grid search over causal routing and batching policies.

Search selects on tuning episodes only; the selected policy is then evaluated on
an untouched holdout. All timings retain the evaluator's calibration caveat.
"""
import argparse
import hashlib
from dataclasses import asdict, dataclass
from itertools import product
import json
import math
from pathlib import Path

from .calibrated_timing import CalibratedTiming
from .evaluation import evaluate_policy
from .policy import AcquireAction, BatchAction, RouteAction, WaitAction, PolicyEnvironment
from .state import RecRequest, nonnegative_int
from .timing import SyntheticTiming


@dataclass(frozen=True)
class SearchParameters:
    queue_weight: float = 1.0
    locality_bonus: float = 0.0
    batch_size: int = 1
    acquisition: str = 'reuse'

    def __post_init__(self):
        for name in ('queue_weight', 'locality_bonus'):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
                raise ValueError(f'{name} must be finite and nonnegative')
        if not nonnegative_int(self.batch_size, 'batch_size'):
            raise ValueError('batch_size must be positive')
        if self.acquisition not in ('reuse', 'local_only', 'recompute'):
            raise ValueError('acquisition must be reuse, local_only or recompute')


class SearchPolicy:
    """Observation-only greedy policy; no future arrivals or simulator handle.

    Route oldest visible requests by queue population minus exact-prefix local
    affinity. Acquire before launch to expose same-stage batches. No deliberate
    waiting is introduced; aging and admission are enforced by legal actions.
    """
    def __init__(self, parameters):
        self.parameters = parameters

    def __call__(self, observation):
        p = self.parameters
        actions = observation.legal_actions
        requests = {r.request.request_id: r for r in observation.requests}
        routes = [a for a in actions if isinstance(a, RouteAction)]
        if routes:
            first = min((a.request_id for a in routes), key=lambda rid: (
                requests[rid].request.arrival_time_ns, rid))
            key = requests[first].request.prefix_key
            def score(action):
                queue = sum(r.destination == action.destination for r in observation.requests)
                local = any(c.key == key and c.location == action.destination and c.valid and c.ready
                            for c in observation.copies)
                return (p.queue_weight * queue - p.locality_bonus * local, action.destination)
            return min((a for a in routes if a.request_id == first), key=score)
        acquisitions = [a for a in actions if isinstance(a, AcquireAction)]
        if acquisitions:
            first = min((a.request_id for a in acquisitions), key=lambda rid: (
                requests[rid].request.arrival_time_ns, rid))
            priorities = {
                'reuse': ('local', 'remote', 'cpu_reload', 'recompute'),
                'local_only': ('local', 'recompute', 'remote', 'cpu_reload'),
                'recompute': ('recompute', 'local', 'remote', 'cpu_reload'),
            }[p.acquisition]
            return min((a for a in acquisitions if a.request_id == first),
                       key=lambda a: (priorities.index(a.mode), a.source or ''))
        batches = [a for a in actions if isinstance(a, BatchAction)]
        if batches:
            first = min(batches, key=lambda a: (
                requests[a.request_ids[0]].request.arrival_time_ns, a.request_ids[0], a.destination))
            anchor = requests[first.request_ids[0]]
            members = sorted((r for r in observation.requests if r.ready
                              and r.destination == first.destination
                              and (observation.execution_mode == 'wave' or r.stage == anchor.stage)),
                             key=lambda r: (r.request.arrival_time_ns, r.request.request_id))
            chosen = [anchor]
            def rows(member):
                return getattr(member.request, 'beam_rows', None) or observation.beam_width
            used_rows = rows(anchor)
            used_tokens = used_rows * (len(anchor.request.prefix_token_ids) + 3)
            for member in members:
                if member is anchor:
                    continue
                if len(chosen) >= min(p.batch_size, observation.max_batch_size):
                    break
                next_rows = used_rows + rows(member)
                next_tokens = used_tokens + rows(member) * (len(member.request.prefix_token_ids) + 3)
                if observation.max_batch_rows is not None and next_rows > observation.max_batch_rows:
                    continue
                if observation.max_batch_tokens is not None and next_tokens > observation.max_batch_tokens:
                    continue
                chosen.append(member)
                used_rows, used_tokens = next_rows, next_tokens
            return BatchAction(first.destination, tuple(r.request.request_id for r in chosen))
        if WaitAction() in actions:
            return WaitAction()
        raise RuntimeError('no legal action can make progress')


def parameter_grid(grid):
    """Canonical Cartesian product, rejecting empty, unknown and duplicate axes."""
    names = sorted(grid)
    if not names or any(name not in SearchParameters.__dataclass_fields__ for name in names):
        raise ValueError('grid must contain known SearchParameters fields')
    axes = []
    for name in names:
        values = list(grid[name])
        encoded = [json.dumps(v, sort_keys=True, allow_nan=False) for v in values]
        if not values or len(set(encoded)) != len(values):
            raise ValueError('grid axes must be nonempty and unique')
        axes.append([v for _, v in sorted(zip(encoded, values), key=lambda pair: pair[0])])
    return tuple(SearchParameters(**dict(zip(names, values))) for values in product(*axes))


_OBJECTIVES = {
    'latency_p99_ns': ('latency_ns', 'p99', False),
    'latency_p50_ns': ('latency_ns', 'p50', False),
    'slo_success_fraction': ('slo_success_fraction', None, True),
    'goodput_requests_per_second': ('goodput_requests_per_second', None, True),
}


def _score(evaluation, objective):
    metric, child, maximize = _OBJECTIVES[objective]
    values = []
    for run in evaluation['runs']:
        value = run['metrics'][metric]
        value = value[child] if child else value
        if value is None or not math.isfinite(value):
            raise ValueError(f'objective {objective} is undefined for this workload')
        values.append(value)
    mean = sum(values) / len(values)
    return mean, -mean if maximize else mean


def grid_search(environment_factory, tuning_requests, holdout_requests, *, grid,
                seeds, config, timing_metadata, objective='latency_p99_ns', max_steps=100000):
    """Evaluate every candidate on tuning, then only the winner on holdout.

    Each episode starts cold. Request IDs must be disjoint across splits; users
    may recur. Scores are arithmetic means across seeds (not pooled percentiles).
    All failures propagate, never silently discard a bad candidate. Ties use the
    canonical parameter JSON. Factories must return fresh environments, timing
    backends and reset external execution state on every call.
    """
    if objective not in _OBJECTIVES:
        raise ValueError(f'unknown objective: {objective}')
    tuning, holdout, seeds = tuple(tuning_requests), tuple(holdout_requests), tuple(seeds)
    if not tuning or not holdout:
        raise ValueError('tuning and holdout must both be nonempty')
    if {r.request_id for r in tuning} & {r.request_id for r in holdout}:
        raise ValueError('tuning and holdout request IDs must be disjoint')
    candidates = parameter_grid(grid)
    trials = []
    def evaluate(parameters, requests):
        return evaluate_policy(environment_factory, lambda seed: SearchPolicy(parameters), requests,
                               seeds=seeds, config=config,
                               policy_metadata={'name': 'causal-grid-policy-v1', 'parameters': asdict(parameters)},
                               timing_metadata=timing_metadata, max_steps=max_steps)
    for parameters in candidates:
        evaluation = evaluate(parameters, tuning)
        score, rank = _score(evaluation, objective)
        trials.append({'parameters': asdict(parameters), 'score': score,
                       'evaluation': evaluation, '_rank': rank})
    winner = min(trials, key=lambda trial: (trial['_rank'], json.dumps(trial['parameters'], sort_keys=True)))
    selected = SearchParameters(**winner['parameters'])
    held_out = evaluate(selected, holdout)
    for trial in trials:
        del trial['_rank']
    return {
        'format': 'openonerec-policy-search-v1', 'objective': objective,
        'aggregation': 'arithmetic_mean_of_per_seed_metrics',
        'selection_split': 'tuning', 'episode_initialization': 'cold_start_per_split',
        'selected_parameters': asdict(selected), 'selected_tuning_score': winner['score'],
        'trials': trials, 'holdout': held_out,
        'limitations': 'Finite offline grid; no global optimum or calibrated performance claim. '
                       'Holdout results never participate in selection. Deterministic policies ignore seeds.',
    }


def profile_timing_factory(path, *, interpolate=False):
    """Snapshot and validate selected profile once, retaining byte identity.

    The file is never reopened during search, so changing it cannot silently
    alter candidate comparisons. Labels and provenance remain producer claims.
    """
    payload = Path(path).read_bytes()
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f'duplicate JSON field {key}')
            result[key] = value
        return result
    def invalid(value):
        raise ValueError(f'nonfinite JSON value {value}')
    document = json.loads(payload, object_pairs_hook=unique, parse_constant=invalid)
    selected = CalibratedTiming(document, interpolate=interpolate)
    metadata = {
        'provider': 'CalibratedTiming', 'label': selected.label,
        'data_kind': selected.data_kind, 'provenance': selected.provenance,
        'profile_sha256': hashlib.sha256(payload).hexdigest(),
        'profile_schema_version': document['schema_version'],
        'interpolate': interpolate,
    }
    return lambda: CalibratedTiming(document, interpolate=interpolate), metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('config', 'grid', 'tuning-workload', 'holdout-workload', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    timing_group = parser.add_mutually_exclusive_group(required=True)
    timing_group.add_argument('--synthetic', action='store_true',
                              help='acknowledge uncalibrated arithmetic timings')
    timing_group.add_argument('--timing-profile', type=Path,
                              help='explicit measured or fixture timing profile JSON')
    parser.add_argument('--interpolate', action='store_true',
                        help='enable bounded interpolation within timing profile coverage')
    parser.add_argument('--seeds', nargs='+', type=int, default=[0])
    parser.add_argument('--objective', choices=tuple(_OBJECTIVES), default='latency_p99_ns')
    parser.add_argument('--max-steps', type=int, default=100000)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    environment_config = dict(config)
    timing_config = environment_config.pop('synthetic_timing', {})
    if args.interpolate and args.timing_profile is None:
        parser.error('--interpolate requires --timing-profile')
    if args.timing_profile is not None:
        if 'synthetic_timing' in config:
            parser.error('remove synthetic_timing from config when using --timing-profile')
        timing_factory, timing_metadata = profile_timing_factory(args.timing_profile,
                                                                interpolate=args.interpolate)
    else:
        timing_factory = lambda: SyntheticTiming(**timing_config)
        timing_metadata = {'label': 'synthetic-uncalibrated', 'parameters': timing_config}
    grid = json.loads(args.grid.read_text())
    if any(p.batch_size > environment_config.get('max_batch_size', 8) for p in parameter_grid(grid)):
        parser.error('grid batch_size exceeds environment max_batch_size')
    def workload(path):
        return [RecRequest(**json.loads(line)) for line in path.read_text().splitlines() if line.strip()]
    result = grid_search(
        lambda: PolicyEnvironment(timing=timing_factory(), **environment_config),
        workload(args.tuning_workload), workload(args.holdout_workload), grid=grid, seeds=args.seeds,
        config=config, timing_metadata=timing_metadata,
        objective=args.objective, max_steps=args.max_steps)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, sort_keys=True, indent=2, allow_nan=False) + '\n')
    print(json.dumps({'selected_parameters': result['selected_parameters'], 'output': str(args.output)}))


if __name__ == '__main__':
    main()
