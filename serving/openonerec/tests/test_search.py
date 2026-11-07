"""Durable optimizer contracts: causality, split isolation, and reproducibility."""
from dataclasses import replace
import unittest
import json
import hashlib
from pathlib import Path
import subprocess
import sys
import tempfile
from dataclasses import asdict

from serving.openonerec.policy import PolicyEnvironment, RouteAction, BatchAction
from serving.openonerec.search import SearchParameters, SearchPolicy, grid_search, parameter_grid, profile_timing_factory
from serving.openonerec.state import RecRequest
from serving.openonerec.timing import SyntheticTiming


def requests(prefix, arrival=0):
    return [RecRequest(f'{prefix}{i}', 'u', 'm', 'v', (1, 2), arrival, 1000000) for i in range(3)]


class SearchTests(unittest.TestCase):
    def search(self, **overrides):
        kwargs = dict(environment_factory=lambda: PolicyEnvironment({'g': 100000}, SyntheticTiming()),
                      tuning_requests=requests('t'), holdout_requests=requests('h'),
                      grid={'batch_size': [1, 3], 'locality_bonus': [0, 2]}, seeds=[0, 5],
                      config={'gpu_capacities': {'g': 100000}}, timing_metadata={'label': 'synthetic'})
        kwargs.update(overrides)
        return grid_search(**kwargs)

    def test_reproducible_grid_and_complete_rollouts(self):
        result = self.search()
        self.assertEqual(result, self.search())
        self.assertEqual(len(result['trials']), 4)
        for trial in result['trials']:
            for run in trial['evaluation']['runs']:
                self.assertEqual(run['metrics']['completed_requests'], 3)
        self.assertEqual(result['holdout']['runs'][0]['metrics']['completed_requests'], 3)
        self.assertEqual(result['selected_tuning_score'], min(t['score'] for t in result['trials']))

    def test_holdout_does_not_select_candidate(self):
        first = self.search()
        second = self.search(holdout_requests=[replace(r, prefix_token_ids=tuple(range(40)), arrival_time_ns=1000)
                                              for r in requests('different')])
        self.assertEqual(first['selected_parameters'], second['selected_parameters'])
        self.assertEqual(first['trials'], second['trials'])
        self.assertNotEqual(first['holdout']['provenance']['workload_sha256'],
                            second['holdout']['provenance']['workload_sha256'])

    def test_order_independent_ties_and_candidates(self):
        first = self.search(grid={'batch_size': [3, 1], 'queue_weight': [1, 2]})
        second = self.search(grid={'queue_weight': [2, 1], 'batch_size': [1, 3]})
        self.assertEqual(first, second)

    def test_policy_cannot_see_future_arrivals(self):
        env = lambda: PolicyEnvironment({'a': 100000, 'b': 100000}, SyntheticTiming())
        present = requests('now')
        later = requests('later', 10000000)
        a, b = env(), env()
        oa, ob = a.reset(present), b.reset(present + later)
        policy = SearchPolicy(SearchParameters(batch_size=3))
        for _ in range(8):
            self.assertEqual(oa, ob)
            aa, ab = policy(oa), policy(ob)
            self.assertEqual(aa, ab)
            oa, ob = a.step(aa).observation, b.step(ab).observation

    def test_batches_are_larger_than_singletons(self):
        result = self.search(grid={'batch_size': [3]})
        actions = result['trials'][0]['evaluation']['runs'][0]['result']['decisions']
        batches = [a for a in actions if a['action'] == 'BatchAction']
        self.assertTrue(any(len(a['arguments']['request_ids']) == 3 for a in batches))

    def test_execution_modes_and_resource_limits(self):
        for mode in ('stage', 'wave', 'fixed_step'):
            with self.subTest(mode=mode):
                result = self.search(
                    grid={'batch_size': [8]},
                    environment_factory=lambda: PolicyEnvironment(
                        {'g': 100000}, SyntheticTiming(), execution_mode=mode,
                        max_batch_size=3, max_batch_rows=16, max_batch_tokens=80))
                run = result['holdout']['runs'][0]
                self.assertEqual(run['metrics']['completed_requests'], 3)
                for action in run['result']['decisions']:
                    if action['action'] == 'BatchAction':
                        self.assertLessEqual(len(action['arguments']['request_ids']), 2)

    def test_cli_writes_reproducible_report(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'config.json').write_text(json.dumps({'gpu_capacities': {'g': 100000}}))
            (root / 'grid.json').write_text(json.dumps({'batch_size': [1, 3]}))
            for split in ('tuning', 'holdout'):
                (root / (split + '.jsonl')).write_text(''.join(
                    json.dumps(asdict(r)) + '\n' for r in requests(split)))
            command = [sys.executable, '-m', 'serving.openonerec.search', '--synthetic',
                       '--config', str(root / 'config.json'), '--grid', str(root / 'grid.json'),
                       '--tuning-workload', str(root / 'tuning.jsonl'),
                       '--holdout-workload', str(root / 'holdout.jsonl'),
                       '--output', str(root / 'result.json')]
            subprocess.run(command, check=True, capture_output=True, text=True)
            first = (root / 'result.json').read_text()
            subprocess.run(command, check=True, capture_output=True, text=True)
            self.assertEqual(first, (root / 'result.json').read_text())
            self.assertEqual(json.loads(first)['format'], 'openonerec-policy-search-v1')

    def test_cli_profile_identity_and_missing_coverage(self):
        # Fabricated test rows check declaration propagation, not calibration.
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'config.json').write_text(json.dumps({'gpu_capacities': {'g': 100000},
                                                         'writeback': False}))
            (root / 'grid.json').write_text(json.dumps({'batch_size': [1]}))
            for split in ('tuning', 'holdout'):
                (root / (split + '.jsonl')).write_text(json.dumps(asdict(requests(split)[0])) + '\n')
            command = [sys.executable, '-m', 'serving.openonerec.search',
                       '--config', str(root / 'config.json'), '--grid', str(root / 'grid.json'),
                       '--tuning-workload', str(root / 'tuning.jsonl'),
                       '--holdout-workload', str(root / 'holdout.jsonl'),
                       '--output', str(root / 'result.json'),
                       '--timing-profile', str(root / 'profile.json')]
            profile = dict(schema_version=1, data_kind='fixture',
                           provenance=dict(model='test', hardware='fake', software='test',
                                           measurement_source='fabricated test declaration'),
                           stages=[dict(destination='g', stage=stage, batch_size=1, beam_width=8,
                                        prefix_tokens=[2], kv_mode=['recompute' if stage == 0 else 'ready'],
                                        duration_ns=10) for stage in range(3)], transfers=[])
            for kind, label in [('fixture', 'synthetic-profile-fixture'), ('measured', 'measured-profile')]:
                profile['data_kind'] = kind
                (root / 'profile.json').write_text(json.dumps(profile))
                subprocess.run(command, check=True, capture_output=True, text=True)
                result = json.loads((root / 'result.json').read_text())
                for evaluation in [result['trials'][0]['evaluation'], result['holdout']]:
                    timing = evaluation['provenance']['timing']
                    self.assertEqual(timing['label'], label)
                    self.assertEqual(timing['data_kind'], kind)
                    self.assertEqual(timing['provenance'], profile['provenance'])
                    self.assertEqual(timing['profile_sha256'], hashlib.sha256(
                        (root / 'profile.json').read_bytes()).hexdigest())
                    self.assertEqual(evaluation['runs'][0]['timing_label'], label)
                    self.assertEqual(evaluation['runs'][0]['metrics']['latency_ns']['p99'], 30)
            factory, metadata = profile_timing_factory(root / 'profile.json')
            (root / 'profile.json').write_text('{}')
            self.assertEqual(factory().label, 'measured-profile')
            self.assertIsNot(factory(), factory())
            profile['stages'].pop()
            (root / 'profile.json').write_text(json.dumps(profile))
            (root / 'result.json').unlink()
            failed = subprocess.run(command, capture_output=True, text=True)
            self.assertNotEqual(failed.returncode, 0)
            self.assertIn('MissingProfileError', failed.stderr)
            self.assertFalse((root / 'result.json').exists())
            (root / 'profile.json').unlink()
            failed = subprocess.run(command, capture_output=True, text=True)
            self.assertNotEqual(failed.returncode, 0)
            self.assertIn('FileNotFoundError', failed.stderr)
            self.assertFalse((root / 'result.json').exists())

    def test_invalid_parameters_and_splits(self):
        for grid in ({}, {'bad': [1]}, {'batch_size': []}, {'batch_size': [1, 1]},
                     {'batch_size': [True]}, {'batch_size': [0]}, {'queue_weight': [float('nan')]},
                     {'acquisition': ['unknown']}):
            with self.subTest(grid=grid), self.assertRaises((ValueError, TypeError)):
                parameter_grid(grid)
        for kwargs in ({'holdout_requests': []}, {'holdout_requests': requests('t')},
                       {'objective': 'unknown'}, {'seeds': []}, {'max_steps': 1}):
            with self.subTest(kwargs=kwargs), self.assertRaises((ValueError, RuntimeError)):
                self.search(**kwargs)
        with self.assertRaisesRegex(ValueError, 'undefined'):
            self.search(objective='slo_success_fraction',
                        tuning_requests=[replace(r, budget_ns=0) for r in requests('t')])


if __name__ == '__main__':
    unittest.main()
