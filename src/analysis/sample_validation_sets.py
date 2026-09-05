"""
Balanced Validation Sampling

Creates reproducible, stratified analysis subsets
for MNLI, WANLI and ANLI.

Author:
Bindhu Pagadala
"""

import argparse
from pathlib import Path

from datasets import load_dataset
from sklearn.model_selection import train_test_split

SEED = 42

parser = argparse.ArgumentParser()
parser.add_argument(
    "--sample_size",
    type=int,
    required=True,
    help="Samples per dataset",
)

args = parser.parse_args()
N = args.sample_size

ROOT = Path("data")

INPUTS = {
    "mnli": ROOT / "processed/mnli/validation_matched.parquet",
    "wanli": ROOT / "processed/wanli/train.parquet",
    "anli_r1": ROOT / "processed/anli/dev_r1.parquet",
    "anli_r2": ROOT / "processed/anli/dev_r2.parquet",
    "anli_r3": ROOT / "processed/anli/dev_r3.parquet",
}

OUTPUT_DIR = ROOT / "analysis" / f"n{N}"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 65)
print("Balanced Validation Sampling")
print("=" * 65)

for dataset_name, parquet_file in INPUTS.items():

    dataset = load_dataset(
        "parquet",
        data_files=str(parquet_file),
        split="train",
    )

    if N > len(dataset):
        raise ValueError(
            f"{dataset_name} has only {len(dataset)} samples."
        )

    df = dataset.to_pandas()

    sampled, _ = train_test_split(
        df,
        train_size=N,
        random_state=SEED,
        stratify=df["label"],
    )
    if N == len(df):
        sampled = df.copy()

    else:
        sampled, _ = train_test_split(
            df,
            train_size=N,
            random_state=SEED,
            stratify=df["label"],
        )
    output_path = OUTPUT_DIR / f"{dataset_name}.parquet"

    sampled.to_parquet(
        output_path,
        index=False,
    )

    print(
        f"{dataset_name:<10}"
        f"{len(sampled):>8} samples"
        f"  ->  {output_path}"
    )

print("=" * 65)
print("Finished.")