#!/usr/bin/env bash
# Edit code locally, run it on the AutoDL GPU box.
#
#   ./remote.sh push                 send local code/config changes up
#   ./remote.sh run <cmd...>         run <cmd> in the remote repo with oligoformer38 active
#   ./remote.sh infer <args...>      shorthand for: run python scripts/main.py --infer <args>
#   ./remote.sh ls-result            list remote run outputs, newest last ('/' marks a dir)
#   ./remote.sh pull-result <name>   bring back one run dir or output file from result/
#   ./remote.sh pull-result --all    bring back all of result/ (~2.7G, asks first)
#   ./remote.sh shell                interactive shell in the remote repo
#
# Override the target with OLIGO_HOST / OLIGO_DIR / OLIGO_ENV.

set -euo pipefail

HOST="${OLIGO_HOST:-oligo}"
DIR="${OLIGO_DIR:-/root/autodl-tmp/OligoFormer-fixed}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_NAME="${OLIGO_ENV:-oligoformer38}"

# Artifacts that live only on the cloud box - never push over them, never pull wholesale.
CLOUD_ONLY=(
  --exclude='data/RNAFM/'
  --exclude='data/RiNALMo/'
  --exclude='data/RiNALMo650/'
  --exclude='data/infer/'
  --exclude='result/'
  --exclude='RNA-FM/redevelop/pretrained/'
  --exclude='__pycache__/'
  --exclude='*.pyc'
  --exclude='.git/'
)

# Activate conda in a non-interactive ssh session, then enter the repo.
PRELUDE="source \$HOME/miniconda3/etc/profile.d/conda.sh && conda activate ${ENV_NAME} && cd ${DIR}"

usage() { sed -n '2,13p' "${BASH_SOURCE[0]}" | sed 's/^# \?//'; }

cmd="${1:-}"; shift || true

case "$cmd" in
  push)
    rsync -az --info=stats1 "${CLOUD_ONLY[@]}" "$@" "${HERE}/" "${HOST}:${DIR}/"
    ;;

  run)
    [ $# -gt 0 ] || { echo "usage: ./remote.sh run <cmd...>" >&2; exit 2; }
    ssh -t "${HOST}" "${PRELUDE} && $*"
    ;;

  infer)
    ssh -t "${HOST}" "${PRELUDE} && python scripts/main.py --infer $*"
    ;;

  ls-result)
    ssh "${HOST}" "ls -1Ft ${DIR}/result | tac"
    ;;

  pull-result)
    name="${1:-}"
    mkdir -p "${HERE}/result"
    if [ -z "$name" ]; then
      echo "usage: ./remote.sh pull-result <name>   (see: ./remote.sh ls-result)" >&2
      echo "       ./remote.sh pull-result --all    (whole result/ tree, ~2.7G)" >&2
      exit 2
    elif [ "$name" = "--all" ]; then
      size="$(ssh "${HOST}" "du -sh ${DIR}/result | cut -f1")"
      if [ "${OLIGO_YES:-}" != "1" ]; then
        read -r -p "Pull the entire remote result/ (${size}) into $(basename "${HERE}")/result? [y/N] " ans
        [ "$ans" = "y" ] || [ "$ans" = "Y" ] || { echo "aborted"; exit 1; }
      fi
      rsync -az --info=stats1 "${HOST}:${DIR}/result/" "${HERE}/result/"
    else
      # no trailing slash: works for both a run directory and a loose output file
      rsync -az --info=stats1 "${HOST}:${DIR}/result/${name}" "${HERE}/result/"
    fi
    ;;

  shell)
    ssh -t "${HOST}" "${PRELUDE} && exec bash -l"
    ;;

  *)
    usage; exit 2
    ;;
esac
