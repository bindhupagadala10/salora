"""
MNLI Genre Statistics
"""

import pandas as pd

df = pd.read_parquet("data/processed/mnli/train.parquet")

print("=" * 60)
print("MNLI Genre Distribution")
print("=" * 60)

summary = (
    df.groupby("genre")
      .size()
      .sort_values(ascending=False)
      .rename("Samples")
)

print(summary)

summary.to_csv(
    "results/mnli_genre_distribution.csv"
)