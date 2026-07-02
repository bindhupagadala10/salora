"""
SALoRA Dataset Audit

Purpose
-------
Automatically audit benchmark datasets before preprocessing.

Outputs
-------
results/audit/

    dataset_statistics.csv
    audit_report.txt

Author:
Bindhu Pagadala

Project:
SALoRA
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

RAW_DATA = Path("data/raw")
RESULTS = Path("results/audit")


def ensure_directory(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def audit_dataset(dataset_name: str) -> dict:

    dataset_dir = RAW_DATA / dataset_name

    statistics = {
        "Dataset": dataset_name,
        "Samples": 0,
        "Classes": 0,
        "Missing": 0,
        "Duplicates": 0,
        "Premise Avg Length": 0,
        "Hypothesis Avg Length": 0,
    }

    train_file = dataset_dir / "train.parquet"

    df = pd.read_parquet(train_file)

    statistics["Samples"] = len(df)

    statistics["Classes"] = df["label"].nunique()

    statistics["Missing"] = df.isna().sum().sum()

    statistics["Duplicates"] = df.duplicated().sum()

    statistics["Premise Avg Length"] = (
        df["premise"]
        .astype(str)
        .str.split()
        .str.len()
        .mean()
    )

    statistics["Hypothesis Avg Length"] = (
        df["hypothesis"]
        .astype(str)
        .str.split()
        .str.len()
        .mean()
    )

    return statistics


def main():

    ensure_directory(RESULTS)

    datasets = [
        "mnli",
    ]

    rows = []

    for dataset in datasets:

        print(f"Auditing {dataset}")

        rows.append(audit_dataset(dataset))

    report = pd.DataFrame(rows)

    report.to_csv(
        RESULTS / "dataset_statistics.csv",
        index=False,
    )

    with open(
        RESULTS / "audit_report.txt",
        "w",
    ) as f:

        f.write(report.to_string(index=False))

    print()

    print(report)

    print()

    print("Audit complete.")


if __name__ == "__main__":
    main()