"""
SALoRA Dataset Preprocessing

Purpose
-------
Convert all benchmark datasets into a unified schema.

Output
------
data/processed/
    mnli/
    wanli/
    anli/

Author:
Bindhu Pagadala
"""

from pathlib import Path

from datasets import load_dataset

OUTPUT_DIR = Path("data/processed")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

LABEL_MAP = {
    "entailment": 0,
    "neutral": 1,
    "contradiction": 2,
}


def save_split(dataset, dataset_name, split_name):

    rows = []

    for sample in dataset:

        label = sample["label"] if "label" in sample else sample["gold"]

        if isinstance(label, str):
            label = LABEL_MAP[label.lower()]

        rows.append(
            {
                "premise": sample["premise"].strip(),
                "hypothesis": sample["hypothesis"].strip(),
                "label": label,
                "dataset": dataset_name,
                "split": split_name,
                "genre": sample.get("genre", "NA"),
            }
        )

    from datasets import Dataset

    ds = Dataset.from_list(rows)

    save_path = OUTPUT_DIR / dataset_name
    save_path.mkdir(exist_ok=True)

    ds.to_parquet(str(save_path / f"{split_name}.parquet"))

    print(f"Saved {dataset_name}/{split_name}")


print("=" * 70)
print("Processing MNLI")
print("=" * 70)

mnli = load_dataset("glue", "mnli")

for split in [
    "train",
    "validation_matched",
    "validation_mismatched",
]:
    save_split(mnli[split], "mnli", split)

print("=" * 70)
print("Processing WANLI")
print("=" * 70)

wanli = load_dataset("alisawuffles/WANLI")

for split in ["train", "test"]:
    save_split(wanli[split], "wanli", split)

print("=" * 70)
print("Processing ANLI")
print("=" * 70)

anli = load_dataset("facebook/anli")

for split in [
    "train_r1",
    "dev_r1",
    "train_r2",
    "dev_r2",
    "train_r3",
    "dev_r3",
]:
    save_split(anli[split], "anli", split)

print("\nDone.")