"""Validate the current public data and reproduce its complete statistical summary."""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import argparse
import json
import math
import subprocess
import time
from scripts.environment_report import environment
from scripts.validate_source_data import validate


def compare_results(actual, expected, path='analysis'):
    """Compare statistical results while allowing ordinary floating-point roundoff."""
    if isinstance(expected, dict):
        if not isinstance(actual, dict) or set(actual) != set(expected):
            raise ValueError(f'Result keys differ at {path}')
        for key in expected:
            compare_results(actual[key], expected[key], f'{path}.{key}')
    elif isinstance(expected, list):
        if not isinstance(actual, list) or len(actual) != len(expected):
            raise ValueError(f'Result length differs at {path}')
        for i, (left, right) in enumerate(zip(actual, expected)):
            compare_results(left, right, f'{path}[{i}]')
    elif isinstance(expected, (int, float)) and not isinstance(expected, bool):
        if not isinstance(actual, (int, float)) or not math.isclose(actual, expected, rel_tol=1e-9, abs_tol=1e-12):
            raise ValueError(f'Numerical result differs at {path}: {actual!r} versus {expected!r}')
    elif actual != expected:
        raise ValueError(f'Result differs at {path}')


def run(output):
    start = time.perf_counter()
    out = Path(output).resolve()
    if out.exists() and any(out.iterdir()):
        raise FileExistsError('Choose an empty Quickstart output directory')
    out.mkdir(parents=True, exist_ok=True)
    validation = validate(ROOT / 'data/source_data')
    (out / 'validation.json').write_text(json.dumps(validation, indent=2) + '\n')
    command = [sys.executable, str(ROOT / 'scripts/analyze_predictions.py'),
               '--data', str(ROOT / 'data/source_data'), '--output', str(out / 'analysis.json')]
    subprocess.run(command, check=True, cwd=ROOT)
    actual = json.loads((out / 'analysis.json').read_text())
    reference = json.loads((ROOT / 'docs/source_data_summary.json').read_text())
    expected = {key: value for key, value in reference.items() if key not in ['runtime', 'public_source_data']}
    compare_results(actual, expected)
    (out / 'environment.json').write_text(json.dumps(environment(), indent=2) + '\n')
    verification = dict(passed=True, source_workbook_sha256=validation['source_workbook_sha256'],
                        checked_reference='docs/source_data_summary.json',
                        checked_fields='All analysis fields; runtime and release metadata excluded',
                        fixed_test=actual['models']['ArthroSkill-MC']['test'],
                        loso_pooled=actual['loso']['pooled'], elapsed_seconds=time.perf_counter() - start)
    (out / 'verification.json').write_text(json.dumps(verification, indent=2, allow_nan=False) + '\n')
    print(f'All current-source statistical results verified in {verification["elapsed_seconds"]:.1f} seconds.')
    return verification


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', default='outputs/quickstart')
    args = parser.parse_args()
    run(args.output)
