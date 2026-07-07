"""
Correlation Analysis

Computes Pearson and Spearman correlation
between MMD and Sinkhorn layer-wise drift profiles.

Author:
Bindhu Pagadala
"""

from pathlib import Path
import argparse

import pandas as pd
from scipy.stats import pearsonr, spearmanr

# ----------------------------------------------------------

parser = argparse.ArgumentParser()

parser.add_argument(
    "--sample_size",
    type=int,
    required=True,
)

args = parser.parse_args()

ROOT = Path(f"results/drift_analysis/n{args.sample_size}")

FILES = {
    "WANLI": ROOT / "mnli_vs_wanli.csv",
    "ANLI_R1": ROOT / "mnli_vs_anli_r1.csv",
    "ANLI_R2": ROOT / "mnli_vs_anli_r2.csv",
    "ANLI_R3": ROOT / "mnli_vs_anli_r3.csv",
}

results = []

print("=" * 70)
print(f"Correlation Analysis (N={args.sample_size})")
print("=" * 70)

for dataset, path in FILES.items():

    df = pd.read_csv(path)

    pearson, p1 = pearsonr(
        df["MMD"],
        df["Sinkhorn"],
    )

    spearman, p2 = spearmanr(
        df["MMD"],
        df["Sinkhorn"],
    )

    print(
        f"{dataset:8s} | "
        f"Pearson={pearson:7.4f} "
        f"(p={p1:.4e}) | "
        f"Spearman={spearman:7.4f} "
        f"(p={p2:.4e})"
    )

    results.append(
        {
            "Dataset": dataset,
            "Pearson": pearson,
            "Pearson_p": p1,
            "Spearman": spearman,
            "Spearman_p": p2,
        }
    )

results = pd.DataFrame(results)

OUT = ROOT / "metric_correlation.csv"
results.to_csv(
    OUT,
    index=False,
)

print()
print("=" * 70)
print(f"Saved : {OUT}")
print("=" * 70)