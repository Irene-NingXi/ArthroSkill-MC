"""Record the runtime used by a reproducibility run without local identity paths."""
import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import platform

PACKAGES = ['torch', 'torchvision', 'timm', 'numpy', 'scipy', 'scikit-learn',
            'Pillow', 'PyYAML', 'openpyxl', 'tqdm']


def environment():
    packages = {}
    for name in PACKAGES:
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    result = dict(python=platform.python_version(), system=platform.system(),
                  system_release=platform.release(), machine=platform.machine(), packages=packages)
    try:
        result['physical_memory_bytes'] = os.sysconf('SC_PAGE_SIZE') * os.sysconf('SC_PHYS_PAGES')
    except (ValueError, OSError, AttributeError):
        result['physical_memory_bytes'] = None
    if packages['torch']:
        import torch
        result.update(cuda_build=torch.version.cuda, cuda_available=torch.cuda.is_available())
        result['cuda_devices'] = [dict(name=torch.cuda.get_device_name(i),
                                       memory_bytes=torch.cuda.get_device_properties(i).total_memory)
                                  for i in range(torch.cuda.device_count())]
    else:
        result.update(cuda_build=None, cuda_available=False, cuda_devices=[])
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(environment(), indent=2, allow_nan=False) + '\n')
