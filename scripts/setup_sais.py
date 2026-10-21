"""Fetch the pinned official SAIS checkout without copying it into this project."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import argparse
import hashlib
import json
import subprocess


def main(a):
    lock = json.loads((Path(__file__).resolve().parents[1]/'configs/sais_upstream.lock.json').read_text())
    target = Path(a.destination).resolve()
    if target.exists():
        raise FileExistsError(f'Refusing to replace an existing checkout: {target}')
    target.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(['git', 'clone', '--no-checkout', lock['repository'], str(target)], check=True)
    subprocess.run(['git', '-C', str(target), 'checkout', '--detach', lock['commit']], check=True)
    for path, digest in lock['source_sha256'].items():
        if hashlib.sha256((target/path).read_bytes()).hexdigest() != digest:
            raise ValueError(f'Upstream source verification failed: {path}')
    print(f'Pinned official SAIS checkout: {target}\nCommit: {lock["commit"]}')
    print('Upstream licensing and weight-access terms apply; see its README and LICENSE.')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--destination', default='external/SAIS')
    main(p.parse_args())
