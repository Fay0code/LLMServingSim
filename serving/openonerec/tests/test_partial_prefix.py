"""Exact partial-prefix identity, timing and lifetime contracts."""
import unittest

from serving.openonerec.simulator import Simulator, Work
from serving.openonerec.state import Directory, RecRequest, UserState
from serving.openonerec.timing import SyntheticTiming, build_stage_graph


def request(name, tokens, **kwargs):
    return RecRequest(name, kwargs.get('user', 'u'), kwargs.get('model', 'm'),
                      kwargs.get('version', 'v'), tokens, kwargs.get('arrival', 0))


class PartialPrefixContracts(unittest.TestCase):
    def directory(self):
        return Directory({'g': 1000, 'cpu': 1000}, {'u': UserState()}, 1000, 0)

    def retain(self, directory, req, location='g', ready=True):
        copy = directory.reserve(req.prefix_key, location, 10 * len(req.prefix_token_ids),
                                 0, req.prefix_token_ids)
        copy.ready = ready
        return copy

    def test_longest_exact_leading_match_and_namespace(self):
        d = self.directory()
        self.retain(d, request('short', (1, 2)))
        longest = self.retain(d, request('long', (1, 2, 3, 4)))
        for tokens, expected in [((1, 2, 3, 5), 3), ((1, 2), 2), ((1, 2, 3, 4, 5), 4)]:
            copy, count = d.best_prefix(request('new', tokens), 'g', 1)
            self.assertEqual(count, expected)
            if expected > 2:
                self.assertIs(copy, longest)
        for kwargs in [dict(user='other'), dict(model='other'), dict(version='other')]:
            self.assertEqual(d.best_prefix(request('new', (1, 2, 3), **kwargs), 'g', 1), (None, 0))
        self.assertEqual(d.best_prefix(request('new', (9, 2, 3)), 'g', 1), (None, 0))

    def test_pending_expired_or_digest_only_cannot_partial_match(self):
        d = self.directory()
        r = request('old', (1, 2))
        c = self.retain(d, r, ready=False)
        q = request('new', (1, 2, 3))
        self.assertEqual(d.best_prefix(q, 'g', 1), (None, 0))
        c.ready = True
        c.prefix_token_ids = None
        self.assertEqual(d.best_prefix(q, 'g', 1), (None, 0))
        c.prefix_token_ids = r.prefix_token_ids
        d.ttl_ns = 10
        self.assertEqual(d.best_prefix(q, 'g', 10), (None, 0))

    def test_incremental_prefill_preserves_full_attention_shape(self):
        graph = build_stage_graph(0, [5], 2, ['local'], cached_prefix_tokens=[3])
        self.assertEqual(graph.prefix_prefill_tokens, (2,))
        self.assertEqual(graph.prefix_tokens, (5,))
        timing = SyntheticTiming()
        full = build_stage_graph(0, [5], 2, ['recompute'])
        self.assertEqual(timing.duration_ns('g', full) - timing.duration_ns('g', graph), 30)
        self.assertEqual(build_stage_graph(1, [5], 2, ['local'], cached_prefix_tokens=[3]).prefix_prefill_tokens, (0,))
        for cached in [-1, 6, True]:
            with self.assertRaises(ValueError):
                build_stage_graph(0, [5], 2, ['local'], cached_prefix_tokens=[cached])

    def test_partial_local_retains_full_identity_after_prefill(self):
        s = Simulator({'g': 1000}, SyntheticTiming(), beam_width=2, kv_bytes_per_token=10)
        result = s.run([request('old', (1, 2)), request('new', (1, 2, 3, 4), arrival=10000)])
        acquisition = [e for e in result['events'] if e['event'] == 'acquire'][-1]
        self.assertEqual((acquisition['mode'], acquisition['reused_prefix_tokens'], acquisition['missing_prefix_tokens']), ('local', 2, 2))
        new = request('new', (1, 2, 3, 4))
        self.assertEqual(s.directory.valid(new.prefix_key, 'g', s.now).size_bytes, 40)
        self.assertTrue(all(c.pins == 0 for c in s.directory.copies.values()))
        self.assertEqual(s.directory.workspace['g'], 0)

    def test_remote_transfers_only_reused_tokens_and_publishes_after_compute(self):
        class RecordingTiming(SyntheticTiming):
            transfers = []
            def transfer_ns(self, source, destination, bytes_count):
                self.transfers.append(bytes_count)
                return 10
        timing = RecordingTiming()
        s = Simulator({'g': 1000}, timing, beam_width=2, kv_bytes_per_token=10,
                      cpu_capacity_bytes=1000)
        s.users['u'] = UserState()
        old = self.retain(s.directory, request('old', (1, 2)), 'cpu')
        work = Work(request('new', (1, 2, 3, 4)), 'g', 'unacquired')
        self.assertTrue(s._allocate(work, ('cpu_reload', 'cpu')))
        self.assertEqual(timing.transfers, [20])
        self.assertEqual(old.pins, 1)
        s._publish_prefix(work)
        self.assertIsNone(s.directory.valid(work.request.prefix_key, 'g', 0))
        # Emulate transfer completion and then completed stage-0 work.
        old.pins -= 1
        s._publish_prefix(work, completed=True)
        self.assertEqual(s.directory.valid(work.request.prefix_key, 'g', 0).size_bytes, 40)

    def test_memory_pressure_can_recompute_instead_of_pinning_parent(self):
        s = Simulator({'g': 100}, SyntheticTiming(), beam_width=2, kv_bytes_per_token=10)
        result = s.run([request('old', (1, 2)), request('new', (1, 2, 3, 4), arrival=10000)])
        self.assertEqual(result['responses'][-1]['mode'], 'recompute')
        self.assertLessEqual(result['peak_bytes']['g'], 100)


if __name__ == '__main__':
    unittest.main()
