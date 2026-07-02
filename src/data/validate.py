"""
Dataset Validation
"""

from pathlib import Path

import pandas as pd

DATA_DIR = Path("data/processed")

for dataset in DATA_DIR.iterdir():

    print("=" * 70)
    print(dataset.name.upper())
    print("=" * 70)

    for file in dataset.glob("*.parquet"):

        df = pd.read_parquet(file)

        print(file.stem)

        print("Samples :", len(df))
        print("Duplicates :", df.duplicated(
            subset=["premise", "hypothesis"]
        ).sum())

        print("Empty premise :", (df.premise == "").sum())
        print("Empty hypothesis :", (df.hypothesis == "").sum())

        print("Labels")

        print(df.label.value_counts().sort_index())

        print()