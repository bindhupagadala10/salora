"""
Dataset Statistics
"""

from pathlib import Path

import pandas as pd

DATA_DIR = Path("data/processed")

rows = []

for dataset in DATA_DIR.iterdir():

    for file in dataset.glob("*.parquet"):

        df = pd.read_parquet(file)

        rows.append(
            {
                "Dataset": dataset.name,
                "Split": file.stem,
                "Samples": len(df),
                "Avg Premise Length":
                    df.premise.str.split().str.len().mean(),
                "Avg Hypothesis Length":
                    df.hypothesis.str.split().str.len().mean(),
            }
        )

summary = pd.DataFrame(rows)

print(summary)

summary.to_csv(
    "results/dataset_statistics.csv",
    index=False,
)