#!/usr/bin/env python3
"""
Extract per-token RiNALMo-mega (150M) embeddings for OligoFormer fasta files,
matching the (seq_len, 640) .npy convention used by data/RNAFM/<name>/representations/<id>.npy
(see loader.py:71-74 for the read side).

Usage: python rinalmo_extract.py <repo_root> <fasta_basename ...>
e.g.   python rinalmo_extract.py /root/autodl-tmp/OligoFormer-fixed Hu_siRNA Hu_mRNA Mix_siRNA Mix_mRNA
"""
import sys
import os
import time
import numpy as np
import torch
from multimolecule import RnaTokenizer, RiNALMoModel

MODEL_NAME = "multimolecule/rinalmo-mega"

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

    # Sanity check: confirm special-token layout (expect exactly 1 leading + 1 trailing token).
    probe = tokenizer("ACGU", return_tensors="pt")
    probe_len = probe["input_ids"].shape[1]
    assert probe_len == 4 + 2, f"unexpected tokenizer special-token count: seq_len=4 -> input_ids len={probe_len}"
    print(f"tokenizer special-token check OK (len 4 -> {probe_len})", flush=True)

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
                hidden = out.last_hidden_state[0]  # (seq_len+2, 640)
                emb = hidden[1:-1, :].detach().cpu().numpy().astype(np.float32)
                assert emb.shape[0] == len(seq), f"{base}/{name}: emb rows {emb.shape[0]} != seq len {len(seq)}"
                np.save(out_path, emb)
                if (i + 1) % 500 == 0:
                    dt = time.time() - t0
                    print(f"[{base}] {i+1}/{len(records)} ({dt:.1f}s, {dt/(i+1):.3f}s/seq)", flush=True)
        print(f"[{base}] done in {time.time()-t0:.1f}s", flush=True)

if __name__ == "__main__":
    main()
