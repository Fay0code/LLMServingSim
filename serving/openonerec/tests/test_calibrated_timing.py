"""Profile contract tests use fabricated numbers, never accuracy evidence."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

from serving.openonerec.calibrated_timing import CalibratedTiming, MissingProfileError
from serving.openonerec.timing import build_stage_graph, Timing


def row(history=10, batch=1, duration=100, stage=0, beam=2, mode='recompute'):
    return dict(destination='g0', stage=stage, batch_size=batch, beam_width=beam,
                prefix_tokens=[history] * batch, kv_mode=[mode] * batch,
                duration_ns=duration)


def profile(rows=None):
    return dict(schema_version=1, data_kind='fixture',
                provenance=dict(model='test-model', hardware='fake-device',
                                software='unit-test', measurement_source='fabricated unit-test values'),
                stages=rows or [row()],
                transfers=[dict(source='cpu', destination='g0', bytes_count=100, duration_ns=10),
                           dict(source='cpu', destination='g0', bytes_count=200, duration_ns=21)])


class CalibratedTimingContracts(unittest.TestCase):
    def test_exact_and_pure_protocol(self):
        timing = CalibratedTiming(profile())
        graph = build_stage_graph(0, [10], 2, ['recompute'])
        self.assertIsInstance(timing, Timing)
        self.assertEqual(timing.duration_ns('g0', graph), 100)
        self.assertEqual(timing.estimate_duration_ns('g0', graph), 100)
        self.assertEqual(timing.label, 'synthetic-profile-fixture')

    def test_bilinear_and_ceil(self):
        timing = CalibratedTiming(profile([row(10, 1, 100), row(20, 1, 200),
                                         row(10, 3, 300), row(20, 3, 400)]), interpolate=True)
        self.assertEqual(timing.duration_ns('g0', build_stage_graph(0, [15, 15], 2, ['recompute'] * 2)), 250)
        self.assertEqual(timing.transfer_ns('cpu', 'g0', 150), 16)

    def test_no_extrapolation_fallback_or_cross_direction(self):
        timing = CalibratedTiming(profile(), interpolate=True)
        for history, stage, beam, destination, mode in [(11, 0, 2, 'g0', 'recompute'),
                                                       (10, 1, 2, 'g0', 'local'),
                                                       (10, 0, 3, 'g0', 'recompute'),
                                                       (10, 0, 2, 'g1', 'recompute'),
                                                       (10, 0, 2, 'g0', 'local')]:
            with self.assertRaises(MissingProfileError):
                timing.duration_ns(destination, build_stage_graph(stage, [history], beam, [mode]))
        with self.assertRaises(MissingProfileError): timing.transfer_ns('g0', 'cpu', 100)
        with self.assertRaises(MissingProfileError): timing.transfer_ns('cpu', 'g0', 201)
        self.assertEqual(timing.transfer_ns('cpu', 'cpu', 100), 0)
        self.assertEqual(timing.transfer_ns('cpu', 'g0', 0), 0)

    def test_no_interpolation_by_default_or_missing_corner(self):
        document = profile([row(10, 1), row(20, 1), row(10, 3)])
        graph = build_stage_graph(0, [15, 15], 2, ['recompute'] * 2)
        for enabled in (False, True):
            with self.assertRaises(MissingProfileError):
                CalibratedTiming(document, interpolate=enabled).duration_ns('g0', graph)

    def test_exact_heterogeneous_no_average(self):
        sample = row(10, 2)
        sample['prefix_tokens'] = [10, 30]
        sample['kv_mode'] = ['recompute', 'ready']
        timing = CalibratedTiming(profile([sample]), interpolate=True)
        self.assertEqual(timing.duration_ns('g0', build_stage_graph(0, [10, 30], 2, ['recompute', 'local'])), 100)
        with self.assertRaises(MissingProfileError):
            timing.duration_ns('g0', build_stage_graph(0, [20, 20], 2, ['recompute', 'local']))

    def test_partial_cache_is_an_exact_conditioning_axis(self):
        samples = [row(10, mode='ready', duration=100), row(20, mode='ready', duration=200)]
        for sample in samples:
            sample['cached_prefix_tokens'] = [5]
        timing = CalibratedTiming(profile(samples), interpolate=True)
        graph = build_stage_graph(0, [15], 2, ['local'], cached_prefix_tokens=[5])
        self.assertEqual(timing.duration_ns('g0', graph), 150)
        for cached in (0, 6, 15):
            with self.assertRaises(MissingProfileError):
                timing.duration_ns('g0', build_stage_graph(0, [15], 2, ['local'],
                                                         cached_prefix_tokens=[cached]))

    def test_full_ready_cache_interpolates_history(self):
        timing = CalibratedTiming(profile([row(10, mode='ready'), row(20, mode='ready', duration=200)]),
                                  interpolate=True)
        self.assertEqual(timing.duration_ns('g0', build_stage_graph(0, [15], 2, ['cpu_reload'])), 150)

    def test_validation(self):
        changes = [lambda p: p.update(schema_version=True),
                   lambda p: p.update(data_kind='calibrated'),
                   lambda p: p['provenance'].update(model=''),
                   lambda p: p['stages'][0].update(duration_ns=float('nan')),
                   lambda p: p['stages'][0].update(duration_ns=-1),
                   lambda p: p['stages'][0].update(batch_size=True),
                   lambda p: p['stages'][0].update(prefix_tokens=[1.0]),
                   lambda p: p['stages'][0].update(kv_mode=['unknown']),
                   lambda p: p['stages'][0].update(cached_prefix_tokens=None),
                   lambda p: p['stages'][0].update(cached_prefix_tokens=[11]),
                   lambda p: p['stages'][0].update(cached_prefix_tokens=[1]),
                   lambda p: p['stages'].append(copy.deepcopy(p['stages'][0])),
                   lambda p: p['transfers'].append(copy.deepcopy(p['transfers'][0]))]
        for change in changes:
            document = profile()
            change(document)
            with self.subTest(document=document), self.assertRaises(ValueError):
                CalibratedTiming(document)

    def test_json_provenance_and_duplicates(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'profile.json'
            document = profile()
            document['data_kind'] = 'measured'
            path.write_text(json.dumps(document))
            self.assertEqual(CalibratedTiming.from_json(path).label, 'measured-profile')
            path.write_text('{"schema_version": 1, "schema_version": 1}')
            with self.assertRaises(ValueError): CalibratedTiming.from_json(path)
