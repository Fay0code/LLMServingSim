"""Explicit synthetic, single-rank SID duration replay through legacy graph tools.

This verifies the Chakra/backend plumbing, not OpenOneRec performance or the
policy simulator's concurrent execution. No generic LLM profile is substituted.
"""
import argparse
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace

from .state import nonnegative_int
from .timing import SyntheticTiming, build_stage_graph


def export_synthetic_replay(output_dir, prefix_tokens, beam_width):
    """Generate real .et files using existing TraceData and GraphGenerator.

    Each SID stage is one aggregate COMP node whose duration is explicitly
    synthetic. Keeping stages in separate graphs preserves their serial order;
    EVENT converter rows within one graph would otherwise be independent.
    """
    from ..core.trace_generator import TraceData
    from ..core.graph_generator import generate_graph
    from ..core.utils import get_workload

    root = Path(output_dir).resolve()
    root.mkdir(parents=True, exist_ok=True)
    timing = SyntheticTiming()
    stages = []
    for stage in range(3):
        graph = build_stage_graph(stage, [prefix_tokens], beam_width, ['recompute'])
        duration = timing.duration_ns('rank0', graph)
        batch = SimpleNamespace(model='OpenOneRec/synthetic', batch_id=stage)
        trace = TraceData('EVENT', [
            (f'synthetic_sid_stage_{stage}', str(duration), 'REMOTE', '0',
             'LOCAL', '0', 'REMOTE', '0', 'NONE', '0', 'NONE')
        ], str(root / 'trace' / f'stage_{stage}.txt'))
        generate_graph(batch, 'synthetic', 1, trace=trace, inputs_root=str(root))
        workload = get_workload(batch, 'synthetic', inputs_root=str(root))
        graph_path = Path(workload + '.0.et')
        if not graph_path.is_file() or not graph_path.stat().st_size:
            raise RuntimeError(f'Graph conversion produced no rank-zero graph: {graph_path}')
        stages.append(dict(stage=stage, duration_ns=duration, graph=graph.describe(),
                           workload=workload, graph_path=str(graph_path)))
    result = dict(timing_label='synthetic-uncalibrated', mode='serial-single-rank-duration-replay',
                  calibrated=False, policy_loop_integrated=False,
                  transfer_contention_modeled=False, stages=stages)
    (root / 'replay.json').write_text(json.dumps(result, indent=2) + '\n')
    return result


def run_serial_replay(replay, *, binary, system, network, memory, timeout_seconds=60):
    """Run real ASTRA IPC and parse its three reports with existing Controller.

    Requires a caller-supplied built backend and its single-rank configs. All
    three workloads are piped in order, followed by exit. subprocess.run gives
    an EOF/timeout boundary rather than the legacy read_wait infinite EOF loop.
    """
    import yaml
    from ..core.controller import Controller

    if nonnegative_int(timeout_seconds, 'timeout_seconds') == 0:
        raise ValueError('timeout_seconds must be positive')
    paths = {name: Path(value).resolve() for name, value in
             [('binary', binary), ('system', system), ('network', network), ('memory', memory)]}
    for name, path in paths.items():
        if not path.is_file():
            raise FileNotFoundError(f'{name} does not exist: {path}')
    topology = yaml.safe_load(paths['network'].read_text())
    if topology.get('npus_count') != [1]:
        raise ValueError('serial SID replay supports only npus_count: [1]')
    configuration = json.loads(paths['system'].read_text())
    if configuration.get('roofline-enabled', 0):
        raise ValueError('duration replay requires roofline-enabled to be false/zero')
    stages = replay['stages']
    if len(stages) != 3 or [stage['stage'] for stage in stages] != [0, 1, 2]:
        raise ValueError('replay must contain exactly three ordered SID stages')
    command = [str(paths['binary']), '--workload-configuration=' + stages[0]['workload'],
               '--system-configuration=' + str(paths['system']),
               '--network-configuration=' + str(paths['network']),
               '--memory-configuration=' + str(paths['memory']),
               '--start-npu-ids=0', '--end-npu-ids=0']
    process = subprocess.run(command, input='\n'.join([s['workload'] for s in stages[1:]] + ['exit']) + '\n',
                             text=True, capture_output=True, timeout=timeout_seconds, check=False,
                             cwd=str(Path(stages[0]['graph_path']).resolve().parent))
    if process.returncode:
        raise RuntimeError(f'ASTRA exited {process.returncode}: {process.stderr[-4000:]}')
    controller = Controller(1)
    reports = []
    for line in process.stdout.splitlines():
        report = controller.parse_output(line)
        if report is not None and (not reports or report['id'] != reports[-1]['id']):
            reports.append(report)
    if len(reports) != 3 or any(r['sys'] != 0 for r in reports):
        raise RuntimeError(f'Expected three rank-zero completion reports; observed {reports}')
    clocks = [r['cycle'] for r in reports]
    if clocks != sorted(clocks):
        raise RuntimeError('ASTRA completion clocks are not monotonic')
    return dict(replay, backend_executed=True, reports=reports,
                observed_stage_ns=[clocks[0]] + [b-a for a,b in zip(clocks, clocks[1:])],
                stdout=process.stdout, stderr=process.stderr)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--prefix-tokens', type=int, required=True)
    parser.add_argument('--beam-width', type=int, required=True)
    parser.add_argument('--synthetic', action='store_true', required=True)
    parser.add_argument('--binary', help='omit to export actual Chakra files without executing ASTRA')
    parser.add_argument('--system')
    parser.add_argument('--network')
    parser.add_argument('--memory')
    args = parser.parse_args()
    if not args.binary and any([args.system, args.network, args.memory]):
        parser.error('--system, --network and --memory require --binary')
    if args.binary and not all([args.system, args.network, args.memory]):
        parser.error('--binary requires --system, --network and --memory')
    replay = export_synthetic_replay(args.output_dir, args.prefix_tokens, args.beam_width)
    if args.binary:
        replay = run_serial_replay(replay, binary=args.binary, system=args.system,
                                   network=args.network, memory=args.memory)
        (Path(args.output_dir) / 'replay.json').write_text(json.dumps(replay, indent=2) + '\n')
    print(json.dumps({key: value for key, value in replay.items() if key != 'stages'}, indent=2))


if __name__ == '__main__':
    main()
