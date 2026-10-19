#!/bin/bash
# Trajectory runs for the model-selection methodology comparison (Hu -> Mix).
#
# One training run per seed, logging every epoch's metrics on all three row sets; the three
# selection rules are then replayed post-hoc by scripts/compare_selection.py. Running the
# full --epoch budget with --no_early_stop is what makes that replay valid -- every rule has
# to see the same, untruncated trajectory.
#
# Hyperparameters are the shipped Hu->Mix production recipe (relabel_median + lr 2e-4), not
# the argparse defaults, so the result speaks to the checkpoint actually in model/.
set -euo pipefail

OUT=${OUT:-./selection_experiment}
SEEDS=${SEEDS:-"42 123 2024"}
VAL_SPLIT=${VAL_SPLIT:-0.4}
EPOCH=${EPOCH:-200}

mkdir -p "$OUT"
for seed in $SEEDS; do
    echo "=== seed $seed ==="
    python scripts/main.py \
        --datasets Hu Mix \
        --seed "$seed" \
        --relabel_median \
        --learning_rate 0.0002 \
        --epoch "$EPOCH" \
        --val_split "$VAL_SPLIT" \
        --split_seed 0 \
        --select_on all \
        --select_metric loss_and_auc \
        --no_early_stop \
        --no_checkpoint \
        --epoch_metrics_csv "$OUT/epochs_seed${seed}.csv" \
        --output_dir "$OUT/logs/" \
        2>&1 | tail -3
done

echo
python scripts/compare_selection.py "$OUT" --patience 100
