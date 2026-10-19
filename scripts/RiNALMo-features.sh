#!/bin/bash
# Extract RiNALMo-650M embeddings for all datasets (--embedding_source rinalmo).
#
# Must run in a separate conda env with `multimolecule` + `transformers==5.8.0`
# installed (see the comment at the top of scripts/rinalmo_extract.py for why the
# transformers version is pinned) — this conflicts with the main `oligoformer`
# env's pinned torch, the same reason RNA-FM.sh/RNA-FM-features.sh need their own
# `RNA-FM` env. One env setup:
#   conda create -n rinalmo python=3.10 -y && conda activate rinalmo
#   pip install torch --index-url https://download.pytorch.org/whl/cu121
#   pip install multimolecule && pip install 'transformers==5.8.0'
python scripts/rinalmo_extract.py . Hu_siRNA Hu_mRNA Mix_siRNA Mix_mRNA Taka_siRNA Taka_mRNA
