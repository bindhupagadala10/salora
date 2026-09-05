"""
Validate Linear CKA for domain comparison.

Author:
Bindhu Pagadala
"""

import torch
from pathlib import Path

from src.analysis.drift_metrics import linear_cka

N = 500
LAYER = 12
SEED = 42

torch.manual_seed(SEED)

BASE = Path(f"results/representations/n{N}")

mnli = torch.load(BASE / "mnli.pt", map_location="cpu")["representations"][LAYER]
wanli = torch.load(BASE / "wanli.pt", map_location="cpu")["representations"][LAYER]

# --------------------------
# MNLI A vs MNLI B
# --------------------------

perm = torch.randperm(len(mnli))

mnli_a = mnli[perm[:250]]
mnli_b = mnli[perm[250:]]

cka_mnli = linear_cka(mnli_a, mnli_b)

# --------------------------
# WANLI A vs WANLI B
# --------------------------

perm = torch.randperm(len(wanli))

wanli_a = wanli[perm[:250]]
wanli_b = wanli[perm[250:]]

cka_wanli = linear_cka(wanli_a, wanli_b)

# --------------------------
# Cross Domain
# --------------------------

cka_cross = linear_cka(mnli, wanli)

print("=" * 60)
print("CKA VALIDATION")
print("=" * 60)
print(f"MNLI  vs MNLI  : {cka_mnli:.4f}")
print(f"WANLI vs WANLI : {cka_wanli:.4f}")
print(f"MNLI  vs WANLI : {cka_cross:.4f}")
print("=" * 60)

if cka_mnli > cka_cross and cka_wanli > cka_cross:
    print("PASS ✓")
else:
    print("FAIL ✗")