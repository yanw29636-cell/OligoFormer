#!/usr/bin/env bash
#
# setup_autodl_singleenv.sh — Single-environment AutoDL setup for OligoFormer.
#
# setup_autodl.sh creates two conda envs (`oligoformer` + `RNA-FM`) because the
# upstream RNA-FM code was pinned to torch 1.7.0/cu110. In practice RNA-FM's
# redevelop/launch/predict.py imports fine under a modern torch (verified with
# torch 2.2.1+cu121 on an RTX 4090), so on GPUs where the old cu110 build won't
# run (Ada/Hopper: no sm_89+ kernels) it's simpler to run everything — model
# training/inference AND RNA-FM embedding extraction — in ONE modern env.
#
# What it does (idempotent — safe to re-run):
#   1. Sanity-check repo root and conda.
#   2. Download the 1.2GB RNA-FM pretrained weight if missing.
#   3. Create the `oligoformer38` conda env (Python 3.8) if missing.
#   4. pip install -r requirements.txt, plus the extra packages RNA-FM's
#      redevelop code imports that aren't already covered by requirements.txt
#      (seaborn, Pillow, line_profiler).
#   5. pip install -e RNA-FM/ --no-deps to make the `fm` package importable.
#   6. Make sure scripts/RNA-FM.sh is in its plain (unpatched) form — it must
#      NOT `conda run -n RNA-FM`, since in single-env mode there is no
#      separate RNA-FM env; RNA-FM.sh should just inherit whichever env
#      called it.
#   7. Verify GPU + `fm` import work in the env.
#   8. Optionally set up the off-target toolchain (PITA + TargetScan), which
#      needs `apt-get install` privileges (SETUP_OFFTARGET=1, default off).
#      PITA bundles its own ViennaRNA-1.6 binaries under off-target/pita/Bin —
#      no system ViennaRNA needed, just `make install` to resolve the
#      EXE_BASE_DIR placeholder and chmod the binaries executable.
#      TargetScan needs the CPAN modules Bio::TreeIO + Statistics::Lite,
#      installed here via apt (libbio-perl-perl, libstatistics-lite-perl) —
#      no PERL5LIB needed, they land in perl's system @INC.
#   9. Optionally run a smoke-test inference on data/example.fa (RUN_INFER=1,
#      default). Includes -off -tox when SETUP_OFFTARGET=1.
#
# Usage:
#   bash setup_autodl_singleenv.sh                    # full setup + smoke-test inference
#   RUN_INFER=0 bash setup_autodl_singleenv.sh         # setup only, skip the test inference
#   ENV_NAME=myenv bash setup_autodl_singleenv.sh      # use a different env name
#   SETUP_OFFTARGET=1 bash setup_autodl_singleenv.sh   # also set up off-target (PITA+TargetScan)
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

RUN_INFER="${RUN_INFER:-1}"
SETUP_OFFTARGET="${SETUP_OFFTARGET:-0}"
ENV_NAME="${ENV_NAME:-oligoformer38}"
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
# 3. Single conda env (Python 3.8, modern torch)
# ---------------------------------------------------------------------------
if conda env list | grep -qE "^${ENV_NAME}[[:space:]]"; then
    log "conda env '$ENV_NAME' already exists, skipping create"
else
    log "Creating conda env '$ENV_NAME' (python 3.8)"
    conda create -n "$ENV_NAME" python=3.8 -y
fi
conda activate "$ENV_NAME"

# ---------------------------------------------------------------------------
# 4. Install OligoFormer + RNA-FM runtime deps into the one env
# ---------------------------------------------------------------------------
log "Installing OligoFormer requirements"
pip install -r requirements.txt

log "Installing extra packages RNA-FM's redevelop code needs (not in requirements.txt)"
pip install seaborn Pillow line_profiler

# ---------------------------------------------------------------------------
# 5. Install the `fm` package (RNA-FM) into the same env
# ---------------------------------------------------------------------------
log "Installing the fm package (no-deps, everything it needs is already installed above)"
pip install -e RNA-FM/ --no-deps
python -c "import fm; print('fm import OK')" \
    || die "import fm failed — check the output above for a missing module and pip install it into '$ENV_NAME'."

# ---------------------------------------------------------------------------
# 6. Make sure scripts/RNA-FM.sh is unpatched (plain, inherits caller's env)
# ---------------------------------------------------------------------------
if grep -q "conda run" scripts/RNA-FM.sh; then
    warn "scripts/RNA-FM.sh is patched to run in a separate 'RNA-FM' env — reverting for single-env mode"
    sed -i 's#^conda run --no-capture-output -n RNA-FM python launch/predict.py#python launch/predict.py#' scripts/RNA-FM.sh
fi
grep -n "predict.py" scripts/RNA-FM.sh

# ---------------------------------------------------------------------------
# 7. Verify GPU
# ---------------------------------------------------------------------------
log "Verifying GPU is visible to torch in '$ENV_NAME'"
python -c "import torch; print('torch', torch.__version__, 'cuda?', torch.cuda.is_available()); x = torch.randn(1024, 1024).cuda(); print('matmul on GPU OK:', float((x @ x).sum()))" \
    || die "torch GPU check failed — see error above."

# ---------------------------------------------------------------------------
# 8. Off-target toolchain (PITA + TargetScan) — optional, needs apt privileges
# ---------------------------------------------------------------------------
INFER_EXTRA_FLAGS=""
if [ "$SETUP_OFFTARGET" = "1" ]; then
    log "Setting up off-target toolchain (PITA + TargetScan)"

    log "Installing TargetScan's perl deps (Bio::TreeIO, Statistics::Lite) via apt"
    apt-get update -qq
    DEBIAN_FRONTEND=noninteractive apt-get install -y libbio-perl-perl libstatistics-lite-perl
    perl -MBio::TreeIO -e 1 || die "Bio::TreeIO still not importable after apt install"
    perl -MStatistics::Lite -e 1 || die "Statistics::Lite still not importable after apt install"

    log "Resolving PITA's EXE_BASE_DIR placeholder and chmod'ing its bundled ViennaRNA-1.6 binaries"
    ( cd off-target/pita && make install )

    log "Off-target toolchain ready"
    INFER_EXTRA_FLAGS="-off -tox"
fi

# ---------------------------------------------------------------------------
# 9. Smoke-test inference
# ---------------------------------------------------------------------------
if [ "$RUN_INFER" = "1" ]; then
    log "Running smoke-test inference on data/example.fa ${INFER_EXTRA_FLAGS}"
    python scripts/main.py --infer 1 -i1 data/example.fa $INFER_EXTRA_FLAGS
    log "Done. Check the 'result/' directory for output."
else
    log "Setup complete. To run inference:"
    echo "    conda activate $ENV_NAME"
    echo "    python scripts/main.py --infer 1 -i1 data/example.fa $INFER_EXTRA_FLAGS"
fi
conda deactivate
