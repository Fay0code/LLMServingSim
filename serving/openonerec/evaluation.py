"""Reproducible policy rollouts and explicitly defined, uncalibrated metrics.

Factories must return fresh environments/policies. Policy randomness belongs to
its factory's seed, never Python's process-global random state. Metadata is
caller-declared provenance, not proof of calibration or code identity.
"""
from collections import Counter
from dataclasses import asdict
import hashlib
import json
import math

from .state import nonnegative_int


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)


def _digest(value):
    return hashlib.sha256(_canonical(value).encode('utf-8')).hexdigest()


def _percentiles(values):
    """Linear interpolation at (n - 1) * p; empty samples are undefined."""
    values = sorted(values)
    def at(p):
        if not values:
            return None
        index = (len(values) - 1) * p
        lower = math.floor(index)
        upper = math.ceil(index)
        return values[lower] + (values[upper] - values[lower]) * (index - lower)
    return {name: at(p) for name, p in [('p50', .5), ('p95', .95), ('p99', .99)]}


def _jain(values):
    total = sum(values)
    squares = sum(value * value for value in values)
    return total * total / (len(values) * squares) if squares else None


def summarize_result(requests, result):
    """Summarize complete or partial responses without hiding unserved users.

    Goodput counts only completed requests having a positive budget and meeting
    it, over first offered arrival to last response. Unbudgeted requests are
    excluded from SLO denominators. No throughput is defined for zero duration.
    Jain indices compare per-user completion/on-time fractions, not raw counts.
    Cache modes describe actual completed acquisitions, not routing guesses.
    """
    requests = tuple(requests)
    by_id = {r.request_id: r for r in requests}
    if len(by_id) != len(requests):
        raise ValueError('request_id values must be unique')
    responses = result['responses']
    seen = set()
    completed = Counter()
    on_time = Counter()
    modes = Counter()
    latencies, waits = [], []
    last_response = None
    for response in responses:
        request_id = response['request_id']
        if request_id not in by_id or request_id in seen:
            raise ValueError('responses must have unique, offered request IDs')
        seen.add(request_id)
        request = by_id[request_id]
        response_time = nonnegative_int(response['response_time_ns'], 'response time')
        latency = response_time - request.arrival_time_ns
        if latency < 0 or response['latency_ns'] != latency:
            raise ValueError('response latency does not match external arrival')
        waiting = nonnegative_int(response['waiting_ns'], 'waiting time')
        if waiting > latency:
            raise ValueError('waiting time exceeds response latency')
        latencies.append(latency)
        waits.append(waiting)
        last_response = max(last_response or 0, response_time)
        completed[request.user_id] += 1
        if request.budget_ns and latency <= request.budget_ns:
            on_time[request.user_id] += 1
        modes[response['mode']] += 1
    offered = Counter(r.user_id for r in requests)
    budgeted = Counter(r.user_id for r in requests if r.budget_ns)
    start = min((r.arrival_time_ns for r in requests), default=None)
    elapsed = last_response - start if last_response is not None else None
    rate = lambda count: count * 1e9 / elapsed if elapsed else None
    users = {
        user: {
            'offered': count, 'completed': completed[user],
            'completion_fraction': completed[user] / count,
            'budgeted': budgeted[user], 'on_time': on_time[user],
            'slo_success_fraction': on_time[user] / budgeted[user] if budgeted[user] else None,
        }
        for user, count in sorted(offered.items())
    }
    return {
        'offered_requests': len(requests), 'completed_requests': len(responses),
        'incomplete_requests': len(requests) - len(responses),
        'window_start_ns': start, 'window_end_ns': last_response, 'window_ns': elapsed,
        'drained_time_ns': result.get('drained_time_ns'),
        'throughput_requests_per_second': rate(len(responses)),
        'budgeted_requests': sum(budgeted.values()),
        'on_time_requests': sum(on_time.values()),
        'slo_success_fraction': sum(on_time.values()) / sum(budgeted.values()) if budgeted else None,
        'goodput_requests_per_second': rate(sum(on_time.values())) if budgeted else None,
        'latency_ns': _percentiles(latencies), 'waiting_ns': _percentiles(waits),
        'per_user': users,
        'completion_fraction_jain_index': _jain([v['completion_fraction'] for v in users.values()]),
        'slo_success_fraction_jain_index': _jain([
            v['slo_success_fraction'] for v in users.values() if v['budgeted']]),
        'completed_acquisition_modes': dict(sorted(modes.items())),
        'completed_cache_reuse_fraction': sum(modes[m] for m in ('local', 'remote', 'cpu_reload')) / len(responses) if responses else None,
        'peak_bytes': dict(result.get('peak_bytes', {})),
        'evictions': result.get('evictions'),
    }


def evaluate_policy(environment_factory, policy_factory, requests, *, seeds,
                    config, policy_metadata, timing_metadata, max_steps=100000):
    """Run independently seeded policies against an identical immutable trace.

    ``environment_factory()`` returns a fresh PolicyEnvironment;
    ``policy_factory(seed)`` returns a fresh callable ``policy(observation)``.
    Factories receive neither future arrivals nor prior rollout results.
    Config/metadata must be JSON data and should include code revision, policy
    parameters, complete environment settings and timing/calibration identity.
    Exceptions propagate: an invalid action or step-limit failure is not a
    successful sample. No optimization or aggregate reward is silently chosen.
    """
    if not callable(environment_factory) or not callable(policy_factory):
        raise TypeError('environment_factory and policy_factory must be callable')
    if nonnegative_int(max_steps, 'max_steps') == 0:
        raise ValueError('max_steps must be positive')
    requests = tuple(requests)
    if len({r.request_id for r in requests}) != len(requests):
        raise ValueError('request_id values must be unique')
    seeds = tuple(seeds)
    if not seeds:
        raise ValueError('provide at least one seed')
    for seed in seeds:
        nonnegative_int(seed, 'seed')
    # Snapshot declared metadata before a factory can mutate caller-owned data.
    declared = json.loads(_canonical({
        'config': config, 'policy': policy_metadata, 'timing': timing_metadata,
    }))
    provenance = {
        'workload_sha256': _digest([asdict(r) for r in requests]),
        **{f'{name}_sha256': _digest(value) for name, value in declared.items()},
        **declared,
    }
    runs = []
    environments, policies = [], []
    for seed in seeds:
        environment = environment_factory()
        policy = policy_factory(seed)
        if any(environment is prior for prior in environments):
            raise ValueError('environment_factory must return a fresh environment')
        if any(policy is prior for prior in policies):
            raise ValueError('policy_factory must return a fresh policy')
        if not callable(policy):
            raise TypeError('policy_factory must return a callable policy')
        environments.append(environment)
        policies.append(policy)
        observation = environment.reset(requests, seed=seed)
        steps = 0
        while not observation.done:
            if steps >= max_steps:
                raise RuntimeError(f'policy exceeded max_steps={max_steps} for seed={seed}')
            transition = environment.step(policy(observation))
            observation = transition.observation
            steps += 1
        result = environment.result()
        # Snapshot outputs so later factory invocations cannot alter prior runs.
        result = json.loads(_canonical(result))
        runs.append({
            'seed': seed, 'steps': steps, 'timing_label': result['timing_label'],
            'metrics': summarize_result(requests, result), 'result': result,
        })
    return {
        'format': 'openonerec-policy-evaluation-v1',
        'provenance': provenance, 'seeds': list(seeds), 'max_steps': max_steps, 'runs': runs,
        'metric_contract': {
            'window': 'first_offered_arrival_to_last_response_excluding_writeback_drain',
            'goodput': 'completed_positive_budget_requests_meeting_budget_per_second',
            'percentiles': 'linear_interpolation_at_(n-1)*p',
            'fairness': 'jain_index_of_per_user_completion_or_slo_success_fractions',
            'calibration': 'timing_label_and_declared_metadata_are_not_accuracy_evidence',
        },
    }
