"""
Layer-wise Representation Drift

Computes

- Linear CKA
- Gaussian MMD

between source and target domains.

Author:
Bindhu Pagadala
"""

import argparse
from pathlib import Path

import pandas as pd
import torch

from src.analysis.drift_metrics import (
    mmd_rbf,
    sinkhorn_distance,
)

parser = argparse.ArgumentParser()

parser.add_argument("--source", required=True)
parser.add_argument("--target", required=True)
parser.add_argument("--sample_size", type=int, required=True)

args = parser.parse_args()

SOURCE = args.source
TARGET = args.target
N = args.sample_size

SOURCE_FILE = Path(
    f"results/representations/n{N}/{SOURCE}.pt"
)

TARGET_FILE = Path(
    f"results/representations/n{N}/{TARGET}.pt"
)

OUTPUT_DIR = Path(
    f"results/drift_analysis/n{N}"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

print("=" * 70)
print("Layer-wise Drift Computation")
print("=" * 70)
print(f"Source : {SOURCE}")
print(f"Target : {TARGET}")

source = torch.load(
    SOURCE_FILE,
    map_location="cpu",
)

target = torch.load(
    TARGET_FILE,
    map_location="cpu",
)

source_rep = source["representations"]
target_rep = target["representations"]

if source_rep.shape != target_rep.shape:
    raise ValueError(
        "Representation tensors must have identical shapes."
    )

results = []

print()

for layer in range(source_rep.shape[0]):

    X = source_rep[layer]
    Y = target_rep[layer]

    mmd = mmd_rbf(X, Y)

    wasserstein = sinkhorn_distance(
        X,
        Y,
    )

    results.append(
        {
            "Layer": layer,
            "Layer_Name": (
                "Embedding"
                if layer == 0
                else f"Encoder_{layer}"
            ),
            "MMD": mmd,
            "Sinkhorn": wasserstein,
        }
    )

    print(
        f"Layer {layer:2d} | "
        f"MMD={mmd:.5f} | "
        f"Sinkhorn={wasserstein:.5f}"
    )
df = pd.DataFrame(results)

save_path = OUTPUT_DIR / f"{SOURCE}_vs_{TARGET}.csv"

df.to_csv(
    save_path,
    index=False,
)

print()
print("=" * 70)
print(f"Saved : {save_path}")
print("=" * 70)