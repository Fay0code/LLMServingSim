"""Durable evaluation contract tests, independent of optional ASTRA tooling."""
from dataclasses import replace
from types import SimpleNamespace
import unittest

from serving.openonerec.evaluation import evaluate_policy, summarize_result
from serving.openonerec.policy import PolicyEnvironment
from serving.openonerec.simulator import Simulator
from serving.openonerec.state import RecRequest
from serving.openonerec.timing import SyntheticTiming


def request(name, user='u', arrival=100, budget=100):
    return RecRequest(name, user, 'm', 'v', (1, 2), arrival, budget)


def response(req, latency=10, waiting=1, mode='recompute'):
    return dict(request_id=req.request_id, response_time_ns=req.arrival_time_ns + latency,
                latency_ns=latency, waiting_ns=waiting, mode=mode)


class FakeEnvironment:
    def reset(self, requests, seed=0):
        self.requests = requests
        self.seed = seed
        return SimpleNamespace(done=not requests)

    def step(self, action):
        return SimpleNamespace(observation=SimpleNamespace(done=True))

    def result(self):
        return dict(timing_label='synthetic-uncalibrated',
                    responses=[response(req) for req in self.requests],
                    drained_time_ns=10000, peak_bytes={'gpu': 50}, evictions=2)


class EvaluationTests(unittest.TestCase):
    def evaluate(self, requests=None, **overrides):
        kwargs = dict(environment_factory=FakeEnvironment,
                      policy_factory=lambda seed: lambda obs: seed,
                      requests=[request('a')] if requests is None else requests,
                      seeds=[7, 8], config={'gpu_capacities': {'gpu': 100}},
                      policy_metadata={'name': 'test', 'revision': 'example'},
                      timing_metadata={'label': 'synthetic-uncalibrated'})
        kwargs.update(overrides)
        return evaluate_policy(**kwargs)

    def test_identical_runs_and_hashes_are_reproducible(self):
        self.assertEqual(self.evaluate(), self.evaluate())
        result = self.evaluate()
        self.assertEqual([run['seed'] for run in result['runs']], [7, 8])
        self.assertEqual(result['runs'][0]['timing_label'], 'synthetic-uncalibrated')
        self.assertEqual(result['runs'][0]['metrics']['window_ns'], 10)
        changed = self.evaluate(requests=[replace(request('a'), prefix_token_ids=(2, 1))])
        self.assertNotEqual(result['provenance']['workload_sha256'],
                            changed['provenance']['workload_sha256'])
        self.assertNotEqual(result['provenance']['policy_sha256'],
                            self.evaluate(policy_metadata={'name': 'other'})['provenance']['policy_sha256'])

    def test_real_environment_multi_seed_reproducibility(self):
        requests = [request('a'), request('b', 'cold', 100),
                    request('c', arrival=100000)]
        kwargs = dict(
            requests=requests,
            environment_factory=lambda: PolicyEnvironment({'gpu': 100000}, SyntheticTiming()),
            policy_factory=lambda seed: lambda obs: obs.legal_actions[0],
            seeds=[0, 42],
        )
        first = self.evaluate(**kwargs)
        self.assertEqual(first, self.evaluate(**kwargs))
        for run in first['runs']:
            self.assertEqual(run['metrics']['completed_requests'], len(requests))
            self.assertEqual(run['result']['seed'], run['seed'])
            self.assertEqual(len(run['result']['decisions']), run['steps'])

    def test_fresh_instances_required(self):
        env = FakeEnvironment()
        with self.assertRaisesRegex(ValueError, 'fresh environment'):
            self.evaluate(environment_factory=lambda: env)
        policy = lambda obs: 0
        with self.assertRaisesRegex(ValueError, 'fresh policy'):
            self.evaluate(policy_factory=lambda seed: policy)

    def test_step_limit_and_empty_workload(self):
        class Stuck(FakeEnvironment):
            def step(self, action):
                return SimpleNamespace(observation=SimpleNamespace(done=False))
        with self.assertRaisesRegex(RuntimeError, 'max_steps=2.*seed=7'):
            self.evaluate(environment_factory=Stuck, max_steps=2)
        result = self.evaluate(requests=[])
        self.assertEqual(result['runs'][0]['steps'], 0)
        self.assertIsNone(result['runs'][0]['metrics']['latency_ns']['p99'])
        self.assertIsNone(result['runs'][0]['metrics']['goodput_requests_per_second'])

    def test_invalid_provenance_and_seeds(self):
        for seed in [True, -1, 1.5]:
            with self.subTest(seed=seed), self.assertRaises(ValueError):
                self.evaluate(seeds=[seed])
        with self.assertRaises(ValueError):
            self.evaluate(seeds=[])
        with self.assertRaises(ValueError):
            self.evaluate(config={'bad': float('nan')})

    def test_metrics_slo_boundaries_unserved_users_and_drain(self):
        a, b = request('a', budget=10), request('b', 'cold', budget=20)
        c = request('c', 'unbudgeted', budget=0)
        result = dict(responses=[response(a, 10, mode='local'), response(c, 30)],
                      drained_time_ns=100000, peak_bytes={'gpu': 10}, evictions=3)
        metrics = summarize_result([a, b, c], result)
        self.assertEqual(metrics['window_ns'], 30)
        self.assertEqual(metrics['on_time_requests'], 1)
        self.assertEqual(metrics['slo_success_fraction'], .5)
        self.assertAlmostEqual(metrics['goodput_requests_per_second'], 1e9 / 30)
        self.assertEqual(metrics['latency_ns']['p50'], 20)
        self.assertEqual(metrics['latency_ns']['p95'], 29)
        self.assertEqual(metrics['completed_cache_reuse_fraction'], .5)
        self.assertEqual(metrics['per_user']['cold']['completed'], 0)
        self.assertEqual(metrics['slo_success_fraction_jain_index'], .5)
        self.assertAlmostEqual(metrics['completion_fraction_jain_index'], 2 / 3)
        self.assertEqual(metrics['incomplete_requests'], 1)

    def test_zero_time_and_no_budget_are_undefined(self):
        req = request('a', budget=0)
        metrics = summarize_result([req], {'responses': [response(req, 0, 0)]})
        self.assertIsNone(metrics['throughput_requests_per_second'])
        self.assertIsNone(metrics['slo_success_fraction'])
        self.assertIsNone(metrics['goodput_requests_per_second'])

    def test_invalid_response_ids_and_times_fail(self):
        req = request('a')
        for rows in [[response(req), response(req)], [response(request('other'))],
                     [response(req, -1)], [response(req, 1, 2)]]:
            with self.subTest(rows=rows), self.assertRaises(ValueError):
                summarize_result([req], {'responses': rows})

    def test_existing_simulator_results_have_supported_metrics(self):
        requests = [request('a'), request('b', arrival=100000)]
        result = Simulator({'gpu': 100000}, SyntheticTiming()).run(requests)
        metrics = summarize_result(requests, result)
        self.assertEqual(metrics['completed_requests'], 2)
        self.assertEqual(metrics['incomplete_requests'], 0)
        self.assertEqual(sum(metrics['completed_acquisition_modes'].values()), 2)
        self.assertEqual(metrics['completion_fraction_jain_index'], 1)


if __name__ == '__main__':
    unittest.main()
