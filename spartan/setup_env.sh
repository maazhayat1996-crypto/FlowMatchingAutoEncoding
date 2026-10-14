#!/bin/bash
# One-time Python environment for the FMPE pipeline on Spartan.
# Run on the LOGIN node from the repository root:   bash spartan/setup_env.sh
#
# Edit before running:
#   FM_PROJECT   your project directory (<projectID>, e.g. punim1234)
#   TORCH_INDEX  PyTorch wheel index matching the GPU driver.  Run `nvidia-smi`
#                inside a GPU job (spartan/smoke.slurm prints it): the "CUDA
#                Version" in its header is the newest CUDA the driver supports.
#                cu126 wheels run on any 12.x driver >= 525 thanks to CUDA minor
#                version compatibility; if the smoke test reports
#                torch.cuda.is_available() == False, fall back to
#                TORCH_INDEX=https://download.pytorch.org/whl/cu124 with
#                TORCH_VERSION=2.6.0 (the newest torch built for cu124), and
#                change the torch pin in requirements.txt to match.
set -eo pipefail

FM_PROJECT=${FM_PROJECT:-/data/gpfs/projects/<projectID>}
case $FM_PROJECT in *"<projectID>"*) echo "set FM_PROJECT or edit <projectID> in spartan/setup_env.sh first" >&2; exit 1;; esac
VENV=${VENV:-$HOME/venvs/fmpe}
TORCH_INDEX=${TORCH_INDEX:-https://download.pytorch.org/whl/cu126}   # <-- PLACEHOLDER: match the driver (see top)
TORCH_VERSION=${TORCH_VERSION:-2.14.1}

module purge
# Python 3.11 is preferable if it loads: check `module spider Python/3.11.3`,
# load the toolchain it lists instead of foss/2022a, and use the same two
# modules in spartan/*.slurm.
module load foss/2022a Python/3.10.4
python --version

if [ ! -d "$VENV" ]; then
    virtualenv "$VENV" || python -m venv "$VENV"
fi
source "$VENV/bin/activate"
python -m pip install --upgrade pip

# torch first, from the CUDA-specific index (the default PyPI wheel may target
# a newer CUDA than the driver supports); requirements.txt then sees it as
# already satisfied.
python -m pip install "torch==$TORCH_VERSION" --index-url "$TORCH_INDEX"
python -m pip install -r requirements.txt
python -m pip check
python -c "import sys, sbi, torch, numpy, scipy, zuko; print('python', sys.version.split()[0], '| sbi', sbi.__version__, '| torch', torch.__version__, '| CUDA', torch.version.cuda, '| numpy', numpy.__version__, '| scipy', scipy.__version__, '| zuko', zuko.__version__)"

mkdir -p "$FM_PROJECT/fmpe/runs/logs"
echo "venv ready at $VENV; Slurm logs go to $FM_PROJECT/fmpe/runs/logs"
