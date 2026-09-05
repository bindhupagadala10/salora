"""
SALoRA Dataset Downloader

Purpose
-------
Downloads all benchmark datasets required for SALoRA experiments.

This script is intentionally isolated from preprocessing and
training to ensure reproducibility.

Author:
Bindhu Pagadala

Project:
SALoRA
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict

from datasets import DatasetDict
from datasets import load_dataset

RAW_DATA_DIR = Path("data/raw")

DATASETS = {
    "mnli": ("glue", "mnli"),
    "wanli": ("alisawuffles/WANLI", None),
    "anli": ("facebook/anli", None),
    "mednli": ("bigbio/med_qa", "mednli"),
}

def ensure_directory(path: Path) -> None:
    """
    Create a directory if it does not already exist.
    """
    path.mkdir(parents=True, exist_ok=True)

def save_dataset(dataset: DatasetDict, output_dir: Path) -> None:
    """
    Save every dataset split as an Apache Parquet file.
    """

    for split_name, split in dataset.items():

        output_file = output_dir / f"{split_name}.parquet"

        split.to_parquet(output_file)

        print(f"Saved {output_file.name}")

def download_dataset(
    dataset_name: str,
    hf_name: str,
    subset: str | None,
) -> None:
    """
    Download a dataset from Hugging Face and save all splits.
    """

    print("\n" + "=" * 70)
    print(f"Downloading: {dataset_name}")
    print("=" * 70)

    dataset = load_dataset(
        path=hf_name,
        name=subset,
    )

    output_dir = RAW_DATA_DIR / dataset_name

    ensure_directory(output_dir)

    save_dataset(dataset, output_dir)

    print(f"{dataset_name} download complete.")

def main() -> None:
    """
    Download all datasets defined in the configuration.
    """

    ensure_directory(RAW_DATA_DIR)

    print("=" * 70)
    print("SALoRA Dataset Acquisition")
    print("=" * 70)

    for dataset_name, (hf_name, subset) in DATASETS.items():

        try:

            download_dataset(
                dataset_name=dataset_name,
                hf_name=hf_name,
                subset=subset,
            )

        except Exception as error:

            print(f"\nERROR: Failed to download '{dataset_name}'")
            print(error)

    print("\n" + "=" * 70)
    print("Dataset acquisition finished.")
    print("=" * 70)

if __name__ == "__main__":
    main()