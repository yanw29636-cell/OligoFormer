import glob
import torch

def load_ckpt(dirpath, prefix):
    matches = glob.glob(f"{dirpath}/{prefix}*.pth")
    assert len(matches) == 1, matches
    path = matches[0]
    sd = torch.load(path, map_location="cpu")
    if isinstance(sd, dict) and "state_dict" in sd:
        sd = sd["state_dict"]
    return path, sd

fixed_path, fixed_sd = load_ckpt("result/20260828_094051_Taka/model", "epoch-53,")
base_path, base_sd = load_ckpt("result/20260828_101214_Taka/model", "epoch-13,")

print("fixed  :", fixed_path)
print("baseline:", base_path)
print()

fixed_keys = set(fixed_sd.keys())
base_keys = set(base_sd.keys())
print(f"# params fixed={len(fixed_keys)} baseline={len(base_keys)}")
only_fixed = fixed_keys - base_keys
only_base = base_keys - fixed_keys
if only_fixed:
    print("keys only in fixed:", sorted(only_fixed))
if only_base:
    print("keys only in baseline:", sorted(only_base))

common = sorted(fixed_keys & base_keys)

total_params = 0
total_sq_diff = 0.0
total_sq_fixed = 0.0
total_sq_base = 0.0

rows = []
for k in common:
    a = fixed_sd[k].float()
    b = base_sd[k].float()
    if a.shape != b.shape:
        rows.append((k, "SHAPE MISMATCH", a.shape, b.shape))
        continue
    diff = (a - b)
    l2_diff = diff.norm().item()
    l2_a = a.norm().item()
    l2_b = b.norm().item()
    denom = max(l2_a, l2_b, 1e-12)
    rel = l2_diff / denom
    cos = torch.nn.functional.cosine_similarity(a.flatten().unsqueeze(0), b.flatten().unsqueeze(0)).item()
    n = a.numel()
    total_params += n
    total_sq_diff += (diff**2).sum().item()
    total_sq_fixed += (a**2).sum().item()
    total_sq_base += (b**2).sum().item()
    rows.append((k, n, l2_diff, rel, cos))

print()
print(f"{'param':60s} {'numel':>9s} {'||Δ||':>10s} {'rel L2':>8s} {'cos_sim':>8s}")
for r in rows:
    if r[1] == "SHAPE MISMATCH":
        print(f"{r[0]:60s} SHAPE MISMATCH fixed={r[2]} baseline={r[3]}")
        continue
    k, n, l2_diff, rel, cos = r
    print(f"{k:60s} {n:9d} {l2_diff:10.4f} {rel:8.4f} {cos:8.4f}")

import math
overall_rel = math.sqrt(total_sq_diff) / max(math.sqrt(total_sq_fixed), math.sqrt(total_sq_base), 1e-12)
print()
print(f"TOTAL trainable params compared: {total_params}")
print(f"Overall ||Δ||_2 (all params concatenated): {math.sqrt(total_sq_diff):.4f}")
print(f"Overall ||fixed||_2: {math.sqrt(total_sq_fixed):.4f}   ||baseline||_2: {math.sqrt(total_sq_base):.4f}")
print(f"Overall relative L2 difference: {overall_rel:.4f}")

# average cosine sim over params weighted by numel, only for weight matrices (ndim>=2) to avoid biases dominating
weighted_cos = []
for r in rows:
    if r[1] == "SHAPE MISMATCH":
        continue
    k, n, l2_diff, rel, cos = r
    weighted_cos.append((n, cos))
wsum = sum(n for n, _ in weighted_cos)
wavg_cos = sum(n*c for n, c in weighted_cos) / wsum
print(f"Numel-weighted average per-tensor cosine similarity: {wavg_cos:.4f}")
