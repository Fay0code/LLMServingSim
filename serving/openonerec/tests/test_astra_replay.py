"""Optional real-Chakra conversion and mocked IPC contracts, not backend accuracy."""
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from serving.openonerec.astra_replay import export_synthetic_replay, run_serial_replay


@unittest.skipUnless(importlib.util.find_spec('chakra') and importlib.util.find_spec('rich'),
                     'requires installed simulator and repository Chakra dependencies')
class AstraReplayContracts(unittest.TestCase):
    def test_real_chakra_nodes(self):
        from chakra.schema.protobuf.et_def_pb2 import GlobalMetadata, Node, COMP_NODE
        from chakra.src.third_party.utils.protolib import decodeMessage
        with tempfile.TemporaryDirectory() as directory:
            replay = export_synthetic_replay(directory, 8, 4)
            self.assertFalse(replay['calibrated'])
            self.assertFalse(replay['policy_loop_integrated'])
            self.assertEqual(len(replay['stages']), 3)
            for stage in replay['stages']:
                with open(stage['graph_path'], 'rb') as stream:
                    self.assertTrue(decodeMessage(stream, GlobalMetadata()))
                    node = Node()
                    self.assertTrue(decodeMessage(stream, node))
                    self.assertEqual(node.type, COMP_NODE)
                    self.assertEqual(node.duration_micros, stage['duration_ns'])
                    self.assertFalse(decodeMessage(stream, Node()))

    def test_controller_reports_and_serial_ipc(self):
        # This checks the real Controller parser and runner protocol, not C++.
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            replay = export_synthetic_replay(directory, 8, 4)
            (root / 'system.json').write_text('{}')
            (root / 'network.yml').write_text('npus_count: [1]\n')
            (root / 'memory.json').write_text('{}')
            output = '\n'.join(f'sys[0] iteration {i+1} finished, {clock} cycles, '
                               'exposed communication 0 cycles.'
                               for i, clock in enumerate([1189, 2629, 4073]))
            with patch('serving.openonerec.astra_replay.subprocess.run',
                       return_value=subprocess.CompletedProcess([], 0, output, '')) as run:
                result = run_serial_replay(replay, binary=sys.executable,
                                           system=root/'system.json', network=root/'network.yml',
                                           memory=root/'memory.json')
            self.assertEqual(result['observed_stage_ns'], [1189, 1440, 1444])
            self.assertEqual(run.call_args.kwargs['input'], '\n'.join(
                [s['workload'] for s in replay['stages'][1:]] + ['exit']) + '\n')
            self.assertEqual(run.call_args.kwargs['timeout'], 60)

    @unittest.skipUnless(os.environ.get('OPENONEREC_ASTRA_BINARY'),
                         'set OPENONEREC_ASTRA_BINARY to run the real backend smoke test')
    def test_real_cpp_serial_replay(self):
        repository = Path(__file__).resolve().parents[3]
        inputs = repository / 'astra-sim' / 'inputs'
        for prefix_tokens, beam_width in [(8, 4), (0, 1)]:
            with self.subTest(prefix_tokens=prefix_tokens, beam_width=beam_width):
                with tempfile.TemporaryDirectory() as directory:
                    replay = export_synthetic_replay(directory, prefix_tokens, beam_width)
                    result = run_serial_replay(
                        replay, binary=os.environ['OPENONEREC_ASTRA_BINARY'],
                        system=inputs / 'system/system.json',
                        network=inputs / 'network/network.yml',
                        memory=inputs / 'memory/no_memory_expansion.json')
                    self.assertTrue(result['backend_executed'])
                    self.assertFalse(result['calibrated'])
                    self.assertEqual(result['observed_stage_ns'],
                                     [stage['duration_ns'] for stage in replay['stages']])

    def test_non_single_rank_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            replay = export_synthetic_replay(directory, 8, 4)
            (root/'system.json').write_text('{}')
            (root/'network.yml').write_text('npus_count: [2]\n')
            (root/'memory.json').write_text('{}')
            with self.assertRaisesRegex(ValueError, 'single|npus_count'):
                run_serial_replay(replay, binary=sys.executable, system=root/'system.json',
                                  network=root/'network.yml', memory=root/'memory.json')
