#!/bin/bash
PY=/root/miniconda3/envs/oligoformer38/bin/python
COMBOS=("Hu Mix" "Hu Taka" "Mix Hu" "Taka Mix" "Taka Hu")
for combo in "${COMBOS[@]}"; do
  read -r a b <<< "$combo"
  echo "=== Starting $a -> $b at $(date) ==="
  $PY scripts/main.py --datasets $a $b --cuda 0 --learning_rate 0.0001 --batch_size 16 --epoch 200 --early_stopping 100 --seed 42 > train_${a}${b}_patience100.log 2>&1
  echo "=== Finished $a -> $b at $(date) ==="
done
echo ALL_DONE
