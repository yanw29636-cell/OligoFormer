#!/usr/bin/env python3
"""
Extract RiNALMo-650M embeddings for OligoFormer fasta files, matching the
(seq_len, hidden_size) .npy convention data_process_loader expects under
data/RiNALMo/<name>/representations/<id>.npy (see loader.py, EMBEDDING_DIRS).

RiNALMo-150M was benchmarked too and consistently underperformed both RNA-FM and
RiNALMo-650M (see the "扩展实验" section of the residual-fix report), so only the
650M variant is wired in here.

Needs the `multimolecule` package (pip install multimolecule) with
transformers==5.8.0 pinned — newer transformers registers its own native "esmc"
model type that collides with multimolecule's, older ones are missing symbols
multimolecule's RiNALMo wrapper imports. This conflicts with the main
`oligoformer` env's pinned torch, so run this in a separate conda env (see
scripts/RiNALMo-features.sh).

Usage: python scripts/rinalmo_extract.py <repo_root> <fasta_basename ...>
e.g.   python scripts/rinalmo_extract.py . Hu_siRNA Hu_mRNA Mix_siRNA Mix_mRNA
"""
import sys
import os
import time
import numpy as np
import torch
from multimolecule import RnaTokenizer, RiNALMoModel

MODEL_NAME = "multimolecule/rinalmo-giga"

def read_fasta(path):
    records = []
    name = None
    seq = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if name is not None:
                    records.append((name, "".join(seq)))
                name = line[1:]
                seq = []
            else:
                seq.append(line)
    if name is not None:
        records.append((name, "".join(seq)))
    return records

def main():
    repo_root = sys.argv[1]
    fasta_bases = sys.argv[2:]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}", flush=True)

    tokenizer = RnaTokenizer.from_pretrained(MODEL_NAME)
    model = RiNALMoModel.from_pretrained(MODEL_NAME).to(device).eval()

    # Sanity check: confirm exactly 1 leading + 1 trailing special token, since we
    # slice them off below assuming that layout.
    probe = tokenizer("ACGU", return_tensors="pt")
    probe_len = probe["input_ids"].shape[1]
    assert probe_len == 4 + 2, f"unexpected tokenizer special-token count: seq_len=4 -> input_ids len={probe_len}"

    for base in fasta_bases:
        fasta_path = os.path.join(repo_root, "data", "fasta", f"{base}.fa")
        out_dir = os.path.join(repo_root, "data", "RiNALMo", base, "representations")
        os.makedirs(out_dir, exist_ok=True)
        records = read_fasta(fasta_path)
        print(f"[{base}] {len(records)} sequences -> {out_dir}", flush=True)
        t0 = time.time()
        with torch.no_grad():
            for i, (name, seq) in enumerate(records):
                out_path = os.path.join(out_dir, f"{name}.npy")
                if os.path.exists(out_path):
                    continue
                inputs = tokenizer(seq, return_tensors="pt").to(device)
                out = model(**inputs)
                hidden = out.last_hidden_state[0]  # (seq_len+2, hidden_size)
                emb = hidden[1:-1, :].detach().cpu().numpy().astype(np.float32)
                assert emb.shape[0] == len(seq), f"{base}/{name}: emb rows {emb.shape[0]} != seq len {len(seq)}"
                np.save(out_path, emb)
                if (i + 1) % 500 == 0:
                    dt = time.time() - t0
                    print(f"[{base}] {i+1}/{len(records)} ({dt:.1f}s, {dt/(i+1):.3f}s/seq)", flush=True)
        print(f"[{base}] done in {time.time()-t0:.1f}s", flush=True)

if __name__ == "__main__":
    main()
