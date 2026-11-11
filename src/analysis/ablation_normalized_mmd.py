"""
Ablation: does the MMD/Sinkhorn disagreement survive matched normalization?

Reviewer concern: Sinkhorn is computed on L2-normalized representations
while MMD is computed on raw (unnormalized) representations, so the
disagreement reported in Table II could be an artifact of this
normalization mismatch rather than a genuine MMD-vs-Sinkhorn disagreement.

This script recomputes MMD on the same L2-normalized representations
Sinkhorn already uses (reusing the cached representations at N=950, no
new extraction), and reports the Spearman correlation between this
normalized-MMD profile and the existing Sinkhorn profile, alongside the
original raw-MMD-vs-Sinkhorn correlation already reported in Table II,
for direct comparison.

Author:
Bindhu Pagadala
"""

from pathlib import Path

import pandas as pd
import torch
from scipy.stats import pearsonr, spearmanr

from src.analysis.drift_metrics import mmd_rbf

N = 950
REP_DIR = Path(f"results/representations/n{N}")
DRIFT_DIR = Path(f"results/drift_analysis/n{N}")

TARGETS = {
    "WANLI": "wanli",
    "ANLI_R1": "anli_r1",
    "ANLI_R2": "anli_r2",
    "ANLI_R3": "anli_r3",
}

source = torch.load(REP_DIR / "mnli.pt", map_location="cpu")["representations"]

existing_corr = pd.read_csv(DRIFT_DIR / "metric_correlation.csv").set_index("Dataset")

results = []

print("=" * 78)
print("Ablation: MMD recomputed on L2-normalized representations (N=950)")
print("=" * 78)

for dataset, filestem in TARGETS.items():
    target = torch.load(REP_DIR / f"{filestem}.pt", map_location="cpu")["representations"]

    drift_csv = pd.read_csv(DRIFT_DIR / f"mnli_vs_{filestem}.csv")
    sinkhorn_profile = drift_csv["Sinkhorn"].tolist()
    raw_mmd_profile = drift_csv["MMD"].tolist()

    normalized_mmd_profile = []
    for layer in range(source.shape[0]):
        X = torch.nn.functional.normalize(source[layer], dim=1)
        Y = torch.nn.functional.normalize(target[layer], dim=1)
        normalized_mmd_profile.append(mmd_rbf(X, Y))

    pearson_raw, p_raw = pearsonr(raw_mmd_profile, sinkhorn_profile)
    spearman_raw, ps_raw = spearmanr(raw_mmd_profile, sinkhorn_profile)

    pearson_norm, p_norm = pearsonr(normalized_mmd_profile, sinkhorn_profile)
    spearman_norm, ps_norm = spearmanr(normalized_mmd_profile, sinkhorn_profile)

    print(f"\n{dataset}")
    print(f"  raw-MMD   vs Sinkhorn: Spearman={spearman_raw:7.4f} (p={ps_raw:.4f})  [existing, Table II]")
    print(f"  norm-MMD  vs Sinkhorn: Spearman={spearman_norm:7.4f} (p={ps_norm:.4f})  [new, this ablation]")

    results.append({
        "Dataset": dataset,
        "Spearman_rawMMD_vs_Sinkhorn": spearman_raw,
        "Spearman_rawMMD_vs_Sinkhorn_p": ps_raw,
        "Spearman_normMMD_vs_Sinkhorn": spearman_norm,
        "Spearman_normMMD_vs_Sinkhorn_p": ps_norm,
        "Pearson_rawMMD_vs_Sinkhorn": pearson_raw,
        "Pearson_normMMD_vs_Sinkhorn": pearson_norm,
    })

df = pd.DataFrame(results)
out_path = DRIFT_DIR / "metric_correlation_normalization_ablation.csv"
df.to_csv(out_path, index=False)

print()
print("=" * 78)
print(f"Saved : {out_path}")
print("=" * 78)
