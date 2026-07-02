"""
SALoRA Benchmark Profiler

Purpose
-------
Profiles all candidate benchmark datasets prior to benchmark selection.

Outputs
-------
results/benchmark_profile/

    dataset_summary.csv
    label_distribution.csv
    schema_report.txt

Author:
Bindhu Pagadala

Project:
SALoRA
"""

from pathlib import Path

import pandas as pd

RAW_DIR = Path("data/raw")
OUTPUT_DIR = Path("results/benchmark_profile")


def ensure_directory(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def detect_text_columns(df):
    """
    Automatically detect premise and hypothesis columns.
    """

    premise_candidates = [
        "premise",
        "sentence1",
        "context",
    ]

    hypothesis_candidates = [
        "hypothesis",
        "sentence2",
        "question",
    ]

    premise = None
    hypothesis = None

    for col in premise_candidates:
        if col in df.columns:
            premise = col
            break

    for col in hypothesis_candidates:
        if col in df.columns:
            hypothesis = col
            break

    return premise, hypothesis


def detect_label_column(df):

    candidates = [
        "label",
        "gold",
        "labels",
    ]

    for col in candidates:
        if col in df.columns:
            return col

    return None


def profile_file(dataset_name, split_name, parquet_file):

    df = pd.read_parquet(parquet_file)

    premise_col, hypothesis_col = detect_text_columns(df)

    label_col = detect_label_column(df)

    summary = {
        "Dataset": dataset_name,
        "Split": split_name,
        "Samples": len(df),
        "Columns": len(df.columns),
        "Premise Column": premise_col,
        "Hypothesis Column": hypothesis_col,
        "Label Column": label_col,
        "Classes": None,
        "Missing Values": int(df.isna().sum().sum()),
        "Duplicate Rows": int(df.duplicated().sum()),
        "Avg Premise Length": None,
        "Avg Hypothesis Length": None,
    }

    label_distribution = []

    if label_col is not None:

        summary["Classes"] = df[label_col].nunique()

        counts = df[label_col].value_counts()

        for label, count in counts.items():

            label_distribution.append(
                {
                    "Dataset": dataset_name,
                    "Split": split_name,
                    "Label": label,
                    "Count": count,
                    "Percentage": round(
                        100 * count / len(df),
                        2,
                    ),
                }
            )

    if premise_col:

        summary["Avg Premise Length"] = round(
            df[premise_col]
            .astype(str)
            .str.split()
            .str.len()
            .mean(),
            2,
        )

    if hypothesis_col:

        summary["Avg Hypothesis Length"] = round(
            df[hypothesis_col]
            .astype(str)
            .str.split()
            .str.len()
            .mean(),
            2,
        )

    schema = {
        "Dataset": dataset_name,
        "Split": split_name,
        "Columns": ", ".join(df.columns),
    }

    return summary, label_distribution, schema


def main():

    ensure_directory(OUTPUT_DIR)

    summaries = []
    labels = []
    schemas = []

    for dataset_dir in sorted(RAW_DIR.iterdir()):

        if not dataset_dir.is_dir():
            continue

        dataset_name = dataset_dir.name

        print(f"\nProfiling {dataset_name}")

        for parquet_file in sorted(dataset_dir.glob("*.parquet")):

            split = parquet_file.stem

            summary, label_dist, schema = profile_file(
                dataset_name,
                split,
                parquet_file,
            )

            summaries.append(summary)

            labels.extend(label_dist)

            schemas.append(schema)

    summary_df = pd.DataFrame(summaries)
    label_df = pd.DataFrame(labels)
    schema_df = pd.DataFrame(schemas)

    summary_df.to_csv(
        OUTPUT_DIR / "dataset_summary.csv",
        index=False,
    )

    label_df.to_csv(
        OUTPUT_DIR / "label_distribution.csv",
        index=False,
    )

    with open(
        OUTPUT_DIR / "schema_report.txt",
        "w",
        encoding="utf-8",
    ) as f:

        for _, row in schema_df.iterrows():

            f.write("=" * 70 + "\n")

            f.write(
                f"{row['Dataset']} | {row['Split']}\n"
            )

            f.write("=" * 70 + "\n")

            f.write(row["Columns"])

            f.write("\n\n")

    print("\n")
    print("=" * 70)
    print("Benchmark profiling complete.")
    print("=" * 70)

    print(summary_df)


if __name__ == "__main__":
    main()