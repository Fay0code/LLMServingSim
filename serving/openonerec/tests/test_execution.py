"""Execution-plan contracts, not real-system performance evidence."""
import unittest

from serving.openonerec.simulator import Simulator, Work
from serving.openonerec.state import RecRequest
from serving.openonerec.timing import SyntheticTiming


def request(name, at=0, rows=None):
    return RecRequest(name, name, 'm', 'v', (1, 2), at, beam_rows=rows)


def simulator(**options):
    return Simulator({'g': 10000}, SyntheticTiming(stage_base_ns=10, query_ns=1,
                     prefix_token_ns=0, attention_token_ns=0),
                     beam_width=2, kv_bytes_per_token=1, writeback=False, **options)


class ExecutionContracts(unittest.TestCase):
    def test_request_rows_validation(self):
        for rows in (0, -1, True, 1.5):
            with self.assertRaises(ValueError):
                request('a', rows=rows)
        self.assertEqual(simulator().peak_workspace(request('a', rows=7)), 23)

    def test_heterogeneous_rows_timing(self):
        s = simulator()
        a, b = Work(request('a', rows=2), 'g', 'recompute'), Work(request('b', rows=4), 'g', 'recompute')
        a.stage = b.stage = 1
        self.assertEqual(s._batch_duration('g', [a, b]), 26)
        self.assertEqual(s._batch_duration('g', [a, b], estimate=True), 26)

    def test_row_cap_packing_and_memory_release(self):
        s = simulator(max_batch_rows=4)
        result = s.run([request('a', rows=3), request('b', rows=2), request('c', rows=1)])
        for event in result['events']:
            if event['event'] == 'batch_start':
                self.assertLessEqual(sum(event['beam_rows']), 4)
        self.assertEqual(len(result['responses']), 3)
        self.assertEqual(s.directory.workspace['g'], 0)

    def test_single_request_exceeding_resource_limit_fails(self):
        for options in (dict(max_batch_rows=1), dict(max_batch_tokens=9)):
            with self.assertRaises(ValueError):
                simulator(**options).run([request('a')])

    def test_wave_admits_at_boundary_and_mixes_stages(self):
        result = simulator(execution_mode='wave').run([request('a'), request('b', at=1)])
        starts = [e for e in result['events'] if e['event'] == 'batch_start']
        self.assertEqual(starts[0]['request_ids'], ['a'])
        self.assertTrue(any(len(set(e['stages'])) == 2 for e in starts))
        self.assertGreater(starts[1]['time_ns'], 1)
        self.assertEqual(len(result['responses']), 2)

    def test_fixed_step_cohort_is_nonpreemptive(self):
        result = simulator(execution_mode='fixed_step').run([request('a'), request('b', at=1)])
        starts = [e for e in result['events'] if e['event'] == 'batch_start']
        self.assertEqual([e['request_ids'] for e in starts], [['a'], ['b']])
        self.assertEqual(starts[0]['duration_ns'], 35)
        self.assertEqual(starts[1]['time_ns'], result['responses'][0]['response_time_ns'])
        self.assertEqual(len(result['responses']), 2)

    def test_active_request_cap(self):
        result = simulator(max_active_requests=1).run([request('a'), request('b')])
        acquire = [e for e in result['events'] if e['event'] == 'acquire']
        self.assertEqual(acquire[1]['time_ns'], result['responses'][0]['response_time_ns'])

    def test_mode_and_limits_validation(self):
        for options in (dict(execution_mode='unknown'), dict(max_batch_rows=0),
                        dict(max_batch_tokens=True), dict(max_active_requests=-1)):
            with self.assertRaises(ValueError):
                simulator(**options)


if __name__ == '__main__':
    unittest.main()
