#!/usr/bin/env bash
#
# setup_autodl.sh — One-shot setup for running OligoFormer inference on an AutoDL GPU instance.
#
# What it does (idempotent — safe to re-run):
#   1. Sanity-check that we're in the repo root and conda exists (AutoDL images ship Miniconda).
#   2. Download the 1.2GB RNA-FM pretrained weight if missing.
#   3. Create the `oligoformer` conda env (Python 3.8) and install requirements.
#   4. Create the `RNA-FM` conda env from RNA-FM/environment.yml.
#   5. Patch scripts/RNA-FM.sh so embedding generation runs in the RNA-FM env
#      (works around infer.py calling RNA-FM.sh with the oligoformer interpreter).
#   6. Optionally run a smoke-test inference on data/example.fa (RUN_INFER=1, default).
#
# Usage:
#   bash setup_autodl.sh            # full setup + smoke-test inference
#   RUN_INFER=0 bash setup_autodl.sh   # setup only, skip the test inference
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

RUN_INFER="${RUN_INFER:-1}"
RNAFM_WEIGHT="RNA-FM/redevelop/pretrained/RNA-FM_pretrained.pth"
RNAFM_URL="https://cloud.tsinghua.edu.cn/f/46d71884ee8848b3a958/?dl=1"

log()  { printf '\n\033[1;32m==> %s\033[0m\n' "$*"; }
warn() { printf '\n\033[1;33m[!] %s\033[0m\n' "$*"; }
die()  { printf '\n\033[1;31m[x] %s\033[0m\n' "$*" >&2; exit 1; }

# ---------------------------------------------------------------------------
# 1. Sanity checks
# ---------------------------------------------------------------------------
log "Checking environment"
[ -f scripts/main.py ] || die "Run this from the OligoFormer repo root (scripts/main.py not found)."
command -v conda >/dev/null 2>&1 || die "conda not found. On AutoDL it ships with the image; try: source ~/.bashrc"

# Make `conda activate` usable inside this non-interactive script.
CONDA_BASE="$(conda info --base)"
# shellcheck disable=SC1091
source "$CONDA_BASE/etc/profile.d/conda.sh"

# ---------------------------------------------------------------------------
# 2. RNA-FM pretrained weight (1.2GB)
# ---------------------------------------------------------------------------
if [ -f "$RNAFM_WEIGHT" ]; then
    log "RNA-FM weight already present, skipping download"
else
    log "Downloading RNA-FM pretrained weight + code (~1.2GB)"
    if [ ! -d RNA-FM/redevelop ]; then
        wget -c "$RNAFM_URL" -O RNA-FM.tar.gz
        tar -zxvf RNA-FM.tar.gz
    fi
    [ -f "$RNAFM_WEIGHT" ] || die "Expected weight at $RNAFM_WEIGHT after extraction — check the download."
fi
ls -lh "$RNAFM_WEIGHT"

# ---------------------------------------------------------------------------
# 3. oligoformer env (Python 3.8)
# ---------------------------------------------------------------------------
if conda env list | grep -qE '^oligoformer[[:space:]]'; then
    log "conda env 'oligoformer' already exists, skipping create"
else
    log "Creating conda env 'oligoformer' (python 3.8)"
    conda create -n oligoformer python=3.8 -y
fi
log "Installing oligoformer requirements"
conda activate oligoformer
pip install -r requirements.txt
log "Verifying GPU is visible to torch"
python -c "import torch; print('torch', torch.__version__, 'cuda?', torch.cuda.is_available())" \
    || warn "torch import failed — check requirements install."
conda deactivate

# ---------------------------------------------------------------------------
# 4. RNA-FM env
# ---------------------------------------------------------------------------
if conda env list | grep -qE '^RNA-FM[[:space:]]'; then
    log "conda env 'RNA-FM' already exists, skipping create"
else
    log "Creating conda env 'RNA-FM' from RNA-FM/environment.yml"
    if ! conda env create -n RNA-FM -f RNA-FM/environment.yml; then
        die "RNA-FM env creation failed (environment.yml pins exact build numbers and can fail across machines).
Fallback — run these manually, then re-run this script:
    conda create -n RNA-FM python=3.8 -y && conda activate RNA-FM
    sed 's/^- //' RNA-FM/requirements.txt > /tmp/rnafm_req.txt
    pip install -r /tmp/rnafm_req.txt
    pip install -e RNA-FM/ --no-deps"
    fi
fi
# Ensure the `fm` package is importable by RNA-FM/redevelop/launch/predict.py.
log "Installing the fm package into the RNA-FM env (no-deps)"
conda activate RNA-FM
pip install -e RNA-FM/ --no-deps || warn "pip install -e RNA-FM/ failed — if inference reports 'No module named fm', run it manually."
python -c "import torch; print('RNA-FM torch', torch.__version__, 'cuda?', torch.cuda.is_available())" \
    || warn "torch import failed in RNA-FM env."
conda deactivate

# ---------------------------------------------------------------------------
# 5. Patch scripts/RNA-FM.sh to use the RNA-FM env (idempotent)
# ---------------------------------------------------------------------------
if grep -q "conda run" scripts/RNA-FM.sh; then
    log "scripts/RNA-FM.sh already patched, skipping"
else
    log "Patching scripts/RNA-FM.sh to generate embeddings in the RNA-FM env"
    sed -i 's#^python launch/predict.py#conda run --no-capture-output -n RNA-FM python launch/predict.py#' scripts/RNA-FM.sh
fi
grep -n "predict.py" scripts/RNA-FM.sh

# ---------------------------------------------------------------------------
# 6. Smoke-test inference
# ---------------------------------------------------------------------------
if [ "$RUN_INFER" = "1" ]; then
    log "Running smoke-test inference on data/example.fa"
    conda activate oligoformer
    python scripts/main.py --infer 1 -i1 data/example.fa
    conda deactivate
    log "Done. Check the 'result/' directory for output."
else
    log "Setup complete. To run inference:"
    echo "    conda activate oligoformer"
    echo "    python scripts/main.py --infer 1 -i1 data/example.fa"
fi
