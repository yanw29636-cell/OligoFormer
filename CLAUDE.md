# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

OligoFormer is a transformer-based model for predicting siRNA knockdown efficacy against a target mRNA. Beyond raw efficacy prediction, the inference pipeline can layer on functional filtering, off-target risk (PITA + TargetScan), and toxicity prediction to rank candidate siRNAs.

## Environments

Two **separate** conda environments are required:

- `oligoformer` (python 3.8, see `environment.yml` / `requirements.txt`) — runs all training/inference in `scripts/`.
- `RNA-FM` — runs the pretrained RNA foundation model in `RNA-FM/redevelop/` to extract sequence embeddings. The off-target perl tools additionally need ViennaRNA and the `PerlLib` (`Statistics::Lite`, `Bio::TreeIO`) on `PERL5LIB`.

A GPU + driver is expected for training and inference.

## Common commands

All commands are run from the repo root with the `oligoformer` env active.

```bash
# Train (inter-dataset): --datasets [Train Test] chosen from Hu, Mix, Taka
python scripts/main.py --datasets Hu Mix --cuda 0 --learning_rate 0.0001 --batch_size 16 --epoch 200 --early_stopping 30
python scripts/main.py --datasets Hu Mix -s          # intra-dataset (single) training / k-fold
python scripts/main.py --datasets Hu Mix -t          # test inter-dataset
python scripts/main.py --datasets Hu Mix -s -t       # test intra-dataset

# Inference
python scripts/main.py --infer 1 -i1 data/example.fa                              # scan mRNA with 19nt window
python scripts/main.py --infer 1 -i1 data/example.fa -i2 data/example_siRNA.fa    # score only the given siRNAs
python scripts/main.py --infer 2                                                  # paste mRNA at the prompt
python scripts/main.py --infer 1 -i1 data/example.fa -off -tox -a                 # + off-target + toxicity, scan all human UTR/ORF
python scripts/main.py --infer 1 -i1 data/example.fa -off -tox -a -top 100        # off-target only for top 100 ranked siRNAs

# Mismatch scanning (efficacy of mismatched variants)
python scripts/main.py -m 2   # transverse 1-2 mismatches (3 = up to 3)
```

`main.py` is pure dispatch: `--mismatch` > `--infer` > `--test` > train, with `-s/--single` selecting the `*_single` variant in each case. There is no test framework — "test" here means model evaluation, not unit tests.

Embeddings must be precomputed before training. `scripts/RNA-FM-features.sh` runs RNA-FM over the dataset fastas in `data/fasta/`; during inference `infer.py` calls `scripts/RNA-FM.sh` on the fly to embed each input mRNA/siRNA.

## Architecture

### Model (`scripts/model.py`)
`Oligo` is the top-level `nn.Module`. It fuses five feature streams in a final MLP `classifier` (2-class softmax output = inefficient/efficient):

1. **siRNA encoder** — Conv2d → avg/max pool → biLSTM → custom transformer `Encoder`.
2. **mRNA encoder** — same shape, over the 19nt site plus `lm1`/`lm2` flanking nucleotides (default 19+19 each → 57nt window).
3. **siRNA RNA-FM embedding** (avg-pooled).
4. **mRNA RNA-FM embedding** (avg-pooled).
5. **Thermodynamic (`td`) features** — base-pairing ΔG/ΔH and positional nucleotide-composition features.

The transformer stack (`MultiHeadAttention`, `EncoderLayer`, `PositionalEncoding`, `LayerNorm`, etc.) is implemented from scratch rather than using `torch.nn.Transformer`. The classifier's first `nn.Linear` input dim is hard-wired to the concatenated feature width — if you change `lm1`/`lm2`, embedding dims, or pooling, that dimension (`1216 + (19 + lm1 + lm2 - 4) * 64 + 256 + 24`) must be kept consistent.

### Inference pipeline (`scripts/infer.py`)
This is the largest and most orchestration-heavy file:
- Generates candidate siRNAs (sliding 19nt window over the mRNA, or uses supplied siRNAs), computes `antiRNA` / thermodynamic `td` features (`calculate_td`, `Calculate_DGH`, `Calculate_end_diff`).
- Writes per-input fastas under `data/infer/<name>/`, shells out to `scripts/RNA-FM.sh` for embeddings, then runs the model.
- **Off-target** (`-off`): shells out to `scripts/pita.sh` and `scripts/targetscan.sh` (perl tools in `off-target/`), reads `pita.tab` / `targetscan.tab`, and derives an `off_target_filter` column from `--pita_threshold` / `--targetscan_threshold`. With `-a/--all_human` it scans the full human UTR/ORF references in `off-target/ref/`.
- **Toxicity** (`-tox`): scores against `toxicity/cell_viability.txt`, filtered by `--toxicity_threshold`.
- Results (efficacy score + filters) are written to `result/`.

### Data loading (`scripts/loader.py`, `loader_infer.py`)
Sequences are integer-embedded (ACGU→1234, `Embed`) and one-hot encoded (`sequence_OneHot`). `data_process_loader` joins a dataset CSV (`data/<name>.csv`) with its `data/fasta/<name>_siRNA.fa` and `<name>_mRNA.fa` reference fastas and the precomputed RNA-FM embeddings. Datasets are referenced by short name (`Hu`, `Mix`, `Taka`); `unnorm/` holds the un-normalized source CSVs.

### Training (`scripts/train.py` / `train_single.py`)
Binary classification with k-fold CV (`--kfold`, default 5; `get_kfold_data_2`). `find_metrics_best_for_shuffle` picks the operating point at a fixed specificity. Best checkpoint is saved to `--best_model` (default `model/best_model.pth`); `model/mismatch_model.pth` backs the mismatch module.

## Directory map
- `scripts/` — all python entry points and modules (the actual codebase).
- `data/` — dataset CSVs, `fasta/` references, `infer/` (generated per-run), `process_demo/` (dataset prep example).
- `model/` — pretrained checkpoints (`best_model.pth`, `mismatch_model.pth`).
- `off-target/` — perl PITA & TargetScan tools (`pita/`, `targetscan/`) and reference UTR/ORF (`ref/`).
- `toxicity/` — cell-viability reference data.
- `RNA-FM/` — vendored RNA foundation model; embeddings generated via its `redevelop/launch/predict.py`.
- `Comparison methods/` — baseline models (Monopoli-RF, siRNAPred) for benchmarking, not part of the main pipeline.

## Docker
A prebuilt image (`yilanbai/oligoformer:v1.0`) exposes an `oligoformer` CLI wrapping `scripts/main.py` with the same flags (see README for the flag-to-command mapping).
