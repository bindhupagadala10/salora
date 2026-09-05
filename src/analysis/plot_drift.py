"""
Plot Layer-wise Domain Drift

Produces

- MMD line plot
- Sinkhorn line plot
- MMD heatmap
- Sinkhorn heatmap

Author:
Bindhu Pagadala
"""

from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

import argparse 

# --------------------------------------------------

parser = argparse.ArgumentParser()

parser.add_argument(
    "--sample_size",
    type=int,
    required=True,
)

args = parser.parse_args()

RESULTS = Path(f"results/drift_analysis/n{args.sample_size}")

OUT = Path(f"results/plots/n{args.sample_size}")
OUT.mkdir(parents=True, exist_ok=True)

FILES = {
    "WANLI": RESULTS / "mnli_vs_wanli.csv",
    "ANLI R1": RESULTS / "mnli_vs_anli_r1.csv",
    "ANLI R2": RESULTS / "mnli_vs_anli_r2.csv",
    "ANLI R3": RESULTS / "mnli_vs_anli_r3.csv",
}

# --------------------------------------------------

dfs = {}

for name, path in FILES.items():
    dfs[name] = pd.read_csv(path)

layers = dfs["WANLI"]["Layer"]

# ==================================================
# MMD
# ==================================================

plt.figure(figsize=(8,5))

for name, df in dfs.items():
    plt.plot(
        layers,
        df["MMD"],
        marker="o",
        linewidth=2,
        label=name,
    )

plt.xlabel("Transformer Layer")
plt.ylabel("MMD")
plt.title("Layer-wise Distribution Drift (MMD)")
plt.grid(alpha=0.3)
plt.legend()

plt.tight_layout()

plt.savefig(
    OUT / "mmd_layers.png",
    dpi=300,
)

plt.close()

# ==================================================
# Sinkhorn
# ==================================================

plt.figure(figsize=(8,5))

for name, df in dfs.items():
    plt.plot(
        layers,
        df["Sinkhorn"],
        marker="o",
        linewidth=2,
        label=name,
    )

plt.xlabel("Transformer Layer")
plt.ylabel("Sinkhorn Distance")
plt.title("Layer-wise Distribution Drift (Sinkhorn)")
plt.grid(alpha=0.3)
plt.legend()

plt.tight_layout()

plt.savefig(
    OUT / "sinkhorn_layers.png",
    dpi=300,
)

plt.close()

# ==================================================
# MMD Heatmap
# ==================================================

mmd = np.vstack(
    [
        dfs["WANLI"]["MMD"],
        dfs["ANLI R1"]["MMD"],
        dfs["ANLI R2"]["MMD"],
        dfs["ANLI R3"]["MMD"],
    ]
)

plt.figure(figsize=(9,3))

plt.imshow(
    mmd,
    aspect="auto",
)

plt.colorbar(label="MMD")

plt.yticks(
    [0,1,2,3],
    [
        "WANLI",
        "ANLI R1",
        "ANLI R2",
        "ANLI R3",
    ],
)

plt.xticks(
    range(13),
    range(13),
)

plt.xlabel("Layer")
plt.title("MMD Heatmap")

plt.tight_layout()

plt.savefig(
    OUT / "mmd_heatmap.png",
    dpi=300,
)

plt.close()

# ==================================================
# Sinkhorn Heatmap
# ==================================================

sinkhorn = np.vstack(
    [
        dfs["WANLI"]["Sinkhorn"],
        dfs["ANLI R1"]["Sinkhorn"],
        dfs["ANLI R2"]["Sinkhorn"],
        dfs["ANLI R3"]["Sinkhorn"],
    ]
)

plt.figure(figsize=(9,3))

plt.imshow(
    sinkhorn,
    aspect="auto",
)

plt.colorbar(label="Sinkhorn")

plt.yticks(
    [0,1,2,3],
    [
        "WANLI",
        "ANLI R1",
        "ANLI R2",
        "ANLI R3",
    ],
)

plt.xticks(
    range(13),
    range(13),
)

plt.xlabel("Layer")
plt.title("Sinkhorn Heatmap")

plt.tight_layout()

plt.savefig(
    OUT / "sinkhorn_heatmap.png",
    dpi=300,
)

plt.close()

print("=" * 60)
print("Plots saved to")
print(OUT)
print("=" * 60)