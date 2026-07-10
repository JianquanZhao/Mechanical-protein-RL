# Conda Environment Conflict Fix Notes

## Environment

Target environment:

```bash
/mnt/data1/home/jianquanzhao/anaconda24/envs/mprl-vgpt
```

Project path:

```bash
/mnt/nas/jianquanzhao/gits/Mechanical-protein-RL
```

## Problem Summary

The `mprl-vgpt` environment looked inconsistent when checking dependencies with:

```bash
/mnt/data1/home/jianquanzhao/anaconda24/envs/mprl-vgpt/bin/python -m pip check
```

Initial conflict messages included:

```text
WARNING: Error parsing dependencies of grpcio:
No such file or directory:
/mnt/data4/home_bak/jianquanzhao/.local/lib/python3.10/site-packages/grpcio-1.66.1.dist-info/METADATA

tensorboard 2.20.0 requires grpcio, which is not installed.
proto-plus 1.24.0 has requirement protobuf<6.0.0dev,>=3.19.0, but you have protobuf 7.35.1.
```

After disabling user site packages, the remaining true environment-internal issues were missing Jupyter/IPython dependencies:

```text
ipykernel requires jupyter-core, matplotlib-inline, tornado, traitlets
ipython requires jedi, matplotlib-inline, pexpect, traitlets
jupyter-client/server/lab/nbclient/nbconvert/nbformat require jupyter-core, tornado, traitlets
prompt-toolkit requires wcwidth
terminado requires ptyprocess, tornado
```

## Root Cause

There were two separate issues.

### 1. User Site-Packages Pollution

The environment was loading packages from the user site directory:

```bash
/home/jianquanzhao/.local/lib/python3.10/site-packages
```

This was confirmed with:

```bash
/mnt/data1/home/jianquanzhao/anaconda24/envs/mprl-vgpt/bin/python -m site
```

The `sys.path` contained both the conda environment and the user site directory:

```text
/home/jianquanzhao/.local/lib/python3.10/site-packages
/mnt/data1/home/jianquanzhao/anaconda24/envs/mprl-vgpt/lib/python3.10/site-packages
```

The user site directory had broken metadata for `grpcio`, so `pip check` and `pip show` tried to parse a damaged package outside the conda environment.

This produced misleading dependency errors even though the conda environment itself contained working `grpc` and `tensorboard`.

### 2. Missing Environment-Internal Dependencies

After isolating the environment with:

```bash
PYTHONNOUSERSITE=1 /mnt/data1/home/jianquanzhao/anaconda24/envs/mprl-vgpt/bin/python -m pip check
```

the user-site pollution disappeared, and the real environment-internal issue became clear: several Jupyter/IPython runtime dependencies were missing from the conda environment.

## Diagnostic Commands

### Check Conda Environments

```bash
conda info --envs
```

### Inspect Packages in the Target Environment

```bash
conda list -p /mnt/data1/home/jianquanzhao/anaconda24/envs/mprl-vgpt
```

### Check Python Version

```bash
/mnt/data1/home/jianquanzhao/anaconda24/envs/mprl-vgpt/bin/python -V
```

Observed:

```text
Python 3.10.20
```

### Check Dependency Conflicts With User Site Enabled

```bash
/mnt/data1/home/jianquanzhao/anaconda24/envs/mprl-vgpt/bin/python -m pip check
```

This showed user-site pollution and misleading `grpcio`/`protobuf`-related issues.

### Check Dependency Conflicts With User Site Disabled

```bash
PYTHONNOUSERSITE=1 /mnt/data1/home/jianquanzhao/anaconda24/envs/mprl-vgpt/bin/python -m pip check
```

This is the cleaner check because it only evaluates the conda environment.

### Inspect Import Paths

```bash
/mnt/data1/home/jianquanzhao/anaconda24/envs/mprl-vgpt/bin/python - <<'PY'
import sys, site
print("ENABLE_USER_SITE", site.ENABLE_USER_SITE)
print("USER_SITE", site.getusersitepackages())
print("USER_SITE_IN_PATH", site.getusersitepackages() in sys.path)
print("sys.path:")
for path in sys.path:
    print(" ", path)
PY
```

## Fix

### Step 1. Install Missing Runtime Dependencies Into the Environment

The missing packages were installed directly into the conda environment:

```bash
/mnt/data1/home/jianquanzhao/anaconda24/envs/mprl-vgpt/bin/python -m pip install --no-user --force-reinstall \
  jupyter-core \
  traitlets \
  tornado \
  matplotlib-inline \
  jedi \
  pexpect \
  wcwidth \
  ptyprocess
```

Why:

1. `--no-user` prevents pip from installing to `~/.local`.
2. `--force-reinstall` ensures the packages are present inside the target environment, not only in user site-packages.
3. These packages are required by `ipykernel`, `ipython`, `jupyter-client`, `jupyter-server`, `jupyterlab`, `nbclient`, `nbconvert`, `nbformat`, `prompt-toolkit`, and `terminado`.

### Step 2. Isolate the Environment From Broken User Site-Packages

A `sitecustomize.py` file was added to the environment:

```bash
/mnt/data1/home/jianquanzhao/anaconda24/envs/mprl-vgpt/lib/python3.10/site-packages/sitecustomize.py
```

Content:

```python
"""Local startup fixes for the mprl-vgpt conda environment."""
from __future__ import annotations

import os
import site
import sys

# Keep this conda environment isolated from broken user-site packages.
_user_site = site.getusersitepackages()
if _user_site in sys.path:
    sys.path.remove(_user_site)

# Matplotlib/fontconfig caches under the home directory are not always writable
# on this machine, so default to a writable temporary cache.
os.environ.setdefault("PYTHONNOUSERSITE", "1")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/mprl-vgpt-matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/mprl-vgpt-cache")
```

Why:

1. Python automatically imports `sitecustomize` during startup if it exists in `site-packages`.
2. Removing `site.getusersitepackages()` from `sys.path` prevents damaged user-level packages from shadowing or confusing the conda environment.
3. Setting `MPLCONFIGDIR` and `XDG_CACHE_HOME` avoids Matplotlib/fontconfig cache warnings caused by non-writable home cache directories.

## Validation

### Dependency Check

After the fix:

```bash
/mnt/data1/home/jianquanzhao/anaconda24/envs/mprl-vgpt/bin/python -m pip check
```

Result:

```text
No broken requirements found.
```

### Import Smoke Test

Command:

```bash
/mnt/data1/home/jianquanzhao/anaconda24/envs/mprl-vgpt/bin/python - <<'PY'
import os, sys, site
print("ENABLE_USER_SITE", site.ENABLE_USER_SITE)
print("USER_SITE_IN_PATH", site.getusersitepackages() in sys.path)
print("MPLCONFIGDIR", os.environ.get("MPLCONFIGDIR"))
print("XDG_CACHE_HOME", os.environ.get("XDG_CACHE_HOME"))

mods = [
    "torch",
    "numpy",
    "pandas",
    "sklearn",
    "scipy",
    "esm",
    "tensorboard",
    "grpc",
    "jupyter_core",
    "ipykernel",
]

for name in mods:
    module = __import__(name)
    print(name, "OK", getattr(module, "__version__", "no_version"), getattr(module, "__file__", ""))
PY
```

Observed:

```text
USER_SITE_IN_PATH False
torch OK 2.12.0+cu130
numpy OK 2.2.6
pandas OK 2.3.3
sklearn OK 1.7.2
scipy OK 1.15.3
esm OK 2.0.0
tensorboard OK 2.20.0
grpc OK 1.81.1
jupyter_core OK 5.9.1
ipykernel OK 7.3.0
```

### Project-Relevant Smoke Tests

Matplotlib:

```bash
/mnt/data1/home/jianquanzhao/anaconda24/envs/mprl-vgpt/bin/python - <<'PY'
import matplotlib.pyplot as plt
plt.figure()
plt.plot([0, 1], [0, 1])
plt.close()
print("matplotlib smoke OK")
PY
```

TensorBoard:

```bash
/mnt/data1/home/jianquanzhao/anaconda24/envs/mprl-vgpt/bin/python - <<'PY'
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
p = Path("/tmp/mprl-vgpt-tb-smoke")
w = SummaryWriter(str(p))
w.add_scalar("smoke/value", 1.0, 0)
w.close()
print("tensorboard writer OK", p)
PY
```

PyRosetta:

```bash
/mnt/data1/home/jianquanzhao/anaconda24/envs/mprl-vgpt/bin/python - <<'PY'
import pyrosetta
print("pyrosetta import OK", getattr(pyrosetta, "__version__", "no_version"))
PY
```

All passed.

## Remaining Non-Conda Issue

The Python environment is now dependency-clean, but GPU was not available in the current session:

```bash
/mnt/data1/home/jianquanzhao/anaconda24/envs/mprl-vgpt/bin/python - <<'PY'
import torch
print(torch.__version__)
print(torch.cuda.is_available())
print(torch.version.cuda)
PY
```

Observed:

```text
torch 2.12.0+cu130
cuda_available False
cuda_version 13.0
```

`nvidia-smi` also failed:

```text
NVIDIA-SMI has failed because it couldn't communicate with the NVIDIA driver.
```

This is not a Python dependency conflict. It indicates that the current node/session cannot communicate with the NVIDIA driver, or no usable GPU driver is visible.

## Practical Rule for Future Debugging

When a conda environment looks broken, always compare:

```bash
python -m pip check
```

against:

```bash
PYTHONNOUSERSITE=1 python -m pip check
```

If the first command fails but the second command passes or shows fewer issues, the environment is being polluted by user site-packages.

For this project, the target environment should remain isolated from:

```bash
~/.local/lib/python3.10/site-packages
```

because damaged packages there can cause false conflict reports and import instability.
