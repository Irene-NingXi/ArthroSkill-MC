# Reproducible software environments

`requirements.txt` keeps the broad development dependencies. `requirements-lock.txt` pins the complete dependency set for public-data statistics, model tests and the synthetic official SAIS training/evaluation cycle. Raw-video decoding, detector fitting and RAFT extraction are outside this CPU validation profile; their separate dependencies remain in the existing requirements files.

## Validated CPU profile

| Item | Value |
| --- | --- |
| Python | 3.12.14, also recorded in `.python-version` |
| PyTorch / torchvision | 2.8.0 / 0.23.0 |
| CUDA | Not installed; CPU wheels in Linux CI and native macOS wheels locally |
| NumPy / SciPy | 2.3.5 / 1.18.1 |
| scikit-learn / timm | 1.9.1 / 1.0.20 |
| Local hardware | Apple M1, 8 GB shared system/GPU memory; validation explicitly uses the CPU |
| Linux CI | Ubuntu 24.04, Python 3.12.14, PyTorch 2.8.0+cpu and torchvision 0.23.0+cpu |

The local validation uses a new virtual environment without system-site packages. Linux validation is executed by [the reproducibility workflow](../.github/workflows/reproducibility.yml); its run status, runtime JSON and numerical verification files are the evidence for that platform. Installing the CPU wheels before the lock file prevents a default Linux CUDA distribution from being selected. The `==2.8.0` requirement accepts the explicitly installed `2.8.0+cpu` build.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
# Linux only:
python -m pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cpu
# Linux and macOS:
python -m pip install -r requirements-lock.txt
python -m pip check
python scripts/setup_sais.py --destination external/SAIS
python -m unittest discover -s tests -v
python scripts/quickstart.py --output outputs/quickstart
python scripts/smoke_test.py --sais-root external/SAIS --output outputs/smoke
```

Use the exact Python patch version when recreating the validated profile. The CI runner starts without project dependencies; cached installed virtual environments are not used. The upstream checkout is pinned and checksum-verified. No private spreadsheet, videos, study weights or GPU are needed for these checks. For repeated runs, choose new empty output directories.

## GPU evidence and configuration boundary

The supplied MedIA manuscript specifies PyTorch and reports a single NVIDIA RTX 3090 for a 38 ms-per-clip forward-pass benchmark. The RTX 3090 has [24 GB GDDR6X memory](https://www.nvidia.com/en-us/geforce/graphics-cards/30-series/rtx-3090-3090ti/). This memory capacity is a manufacturer specification; the manuscript does not record measured peak allocation. The 38 ms figure excludes decoding, detection, tracking and feature extraction and is not a training-time measurement.

The manuscript does not give the historical PyTorch/torchvision patch versions, CUDA toolkit/runtime or driver version, or wall-clock times for the main model and SAIS. They cannot be recovered from the public prediction tables. No new full 100-epoch clinical-video run was performed for this environment profile.

For a future Linux CUDA validation, the [official PyTorch distribution](https://pytorch.org/get-started/previous-versions/) provides the matched 2.8.0/0.23.0 CUDA 12.6 wheels:

```bash
python -m pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu126
python -m pip install -r requirements-lock.txt
python scripts/environment_report.py --output outputs/cuda-environment.json
```

This is an explicit CUDA 12.6 configuration, not a claim that CUDA training has been validated on this machine or that these were the manuscript's historical package versions. A future GPU report must include the driver, actual GPU name, total and peak allocated memory, dependency check, data/weight checksums and elapsed fitting time. Raw-video extraction additionally requires its optional dependencies and authorised recordings.

## Measured demonstration times

The complete public-data Quickstart was measured at 95.2 seconds on the local CPU profile, including all 20,000 procedure-bootstrap draws and 99,999 procedure-level swaps. The synthetic two-epoch SAIS fitting/checkpoint/evaluation cycle was measured at 10.8 seconds. Times vary with CPU load and exclude dependency downloads and installation. They do not estimate the runtime of 100-epoch full-resolution model training.

Every Quickstart and smoke-test execution writes its own elapsed time. `environment.json` and the smoke report record the installed package versions, platform, CUDA build and visible GPU memory, without local account paths. GPU training time remains unrecorded rather than inferred from the forward-pass benchmark.
