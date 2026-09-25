#!/usr/bin/env bash
# Creates the project virtualenv on a Jetson, without touching the system /
# user (~/.local) site-packages.
#
# Why this isn't just `python3 -m venv .venv && pip install -r requirements.txt`:
#
#   1. --system-site-packages is REQUIRED. tensorrt ships with JetPack in
#      /usr/lib/python3.10/dist-packages, and torch/torchvision/ultralytics are
#      NVIDIA aarch64 CUDA builds. None of them can be reinstalled from PyPI —
#      a plain isolated venv would see no TensorRT and no CUDA-enabled torch.
#
#   2. Ubuntu's python3.10 has no `ensurepip`, so venv creation fails at the
#      pip step unless you `sudo apt install python3.10-venv`. We sidestep that
#      with --without-pip + the official get-pip.py bootstrap, so no sudo needed.
#
#   3. We deliberately do NOT `pip install -r requirements.txt`. That file lists
#      opencv-python and ultralytics; installing them would shadow the system
#      OpenCV (built with GStreamer) and risk pip pulling a non-Jetson torch
#      wheel over the working CUDA one. Only genuinely-missing packages are
#      installed here.
#
# Usage: bash scripts/setup_venv.sh
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV="$REPO_ROOT/.venv"
cd "$REPO_ROOT"

if [ -d "$VENV" ]; then
  echo "==> $VENV already exists; delete it first to rebuild from scratch."
  exit 0
fi

echo "==> Creating venv (inherits system tensorrt + Jetson torch)..."
python3 -m venv --system-site-packages --without-pip "$VENV"

echo "==> Bootstrapping pip (works around missing ensurepip)..."
curl -sSL https://bootstrap.pypa.io/get-pip.py -o /tmp/get-pip.py
"$VENV/bin/python" /tmp/get-pip.py
rm -f /tmp/get-pip.py

# Pin numpy to whatever the Jetson torch build already links against, so pip
# can never silently pull numpy 2.x in as a transitive dep and break torch.
NUMPY_VER="$("$VENV/bin/python" -c 'import numpy; print(numpy.__version__)')"
echo "numpy==$NUMPY_VER" > /tmp/constraints.txt
echo "==> Pinning numpy==$NUMPY_VER for all installs"

echo "==> Installing the packages JetPack does NOT provide..."
# pycuda compiles from source against the CUDA toolkit (~5 min on an Orin NX).
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}
export CUDA_ROOT=/usr/local/cuda
"$VENV/bin/python" -m pip install -c /tmp/constraints.txt pycuda pycocotools ruff
rm -f /tmp/constraints.txt

echo
echo "==> Verifying..."
"$VENV/bin/python" - <<'PY'
import tensorrt, torch, pycuda.autoinit, pycuda.driver as cuda
print(f"  tensorrt {tensorrt.__version__}")
print(f"  torch    {torch.__version__}  (CUDA available: {torch.cuda.is_available()})")
print(f"  pycuda   OK -> {cuda.Device(0).name()}")
PY

echo
echo "==> Done. Activate with:  source .venv/bin/activate"
