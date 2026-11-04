"""Run the explicit synthetic OpenOneRec policy example (not an accuracy run)."""
import argparse
import json
from pathlib import Path

from .simulator import Simulator
from .state import RecRequest
from .timing import SyntheticTiming


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workload', required=True, type=Path)
    parser.add_argument('--config', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--synthetic', action='store_true', required=True,
                        help='acknowledge uncalibrated demonstration timings')
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    timing = SyntheticTiming(**config.pop('synthetic_timing', {}))
    requests = [RecRequest(**json.loads(line)) for line in args.workload.read_text().splitlines()
                if line.strip()]
    result = Simulator(timing=timing, **config).run(requests)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(f'{len(result["responses"])} responses; timing={timing.label}; output={args.output}')


if __name__ == '__main__':
    main()
