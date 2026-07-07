"""
Layer-wise Representation Extraction

Extracts mean-pooled hidden representations from the
frozen MNLI source model.

Author:
Bindhu Pagadala
"""

import argparse
from pathlib import Path

import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
)

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

MODEL_PATH = "models/source_roberta"

parser = argparse.ArgumentParser()

parser.add_argument(
    "--dataset",
    required=True,
)

parser.add_argument(
    "--sample_size",
    type=int,
    required=True,
)

parser.add_argument(
    "--batch_size",
    type=int,
    default=32,
)

args = parser.parse_args()

DATASET = args.dataset
N = args.sample_size
BATCH = args.batch_size

DATA_FILE = Path(
    f"data/analysis/n{N}/{DATASET}.parquet"
)

OUTPUT_DIR = Path(
    f"results/representations/n{N}"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

print("=" * 70)
print("Representation Extraction")
print("=" * 70)
print(f"Dataset     : {DATASET}")
print(f"Sample Size : {N}")

dataset = load_dataset(
    "parquet",
    data_files=str(DATA_FILE),
    split="train",
)

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_PATH
)

def tokenize(batch):
    return tokenizer(
        batch["premise"],
        batch["hypothesis"],
        truncation=True,
        max_length=128,
    )

dataset = dataset.map(
    tokenize,
    batched=True,
)

columns = [
    "input_ids",
    "attention_mask",
    "label",
]

dataset = dataset.remove_columns(
    [
        c
        for c in dataset.column_names
        if c not in columns
    ]
)

dataset = dataset.rename_column("label", "labels")
dataset.set_format("torch")

collator = DataCollatorWithPadding(
    tokenizer
)

loader = DataLoader(
    dataset,
    batch_size=BATCH,
    shuffle=False,
    collate_fn=collator,
)

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_PATH,
    output_hidden_states=True,
)

model.to(DEVICE)
model.eval()

layer_outputs = [[] for _ in range(13)]
labels = []

print("\nExtracting representations...\n")

with torch.no_grad():

    for batch in loader:

        labels.append(
            batch["label"]
        )

        batch = {
            k: v.to(DEVICE)
            for k, v in batch.items()
            if k != "label"
        }

        outputs = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            output_hidden_states=True,
        )

        hidden_states = outputs.hidden_states

        for i, hidden in enumerate(hidden_states):

            pooled = hidden.mean(dim=1)

            layer_outputs[i].append(
                pooled.cpu()
            )

representations = torch.stack(
    [
        torch.cat(layer, dim=0)
        for layer in layer_outputs
    ]
)

labels = torch.cat(labels)

save_path = OUTPUT_DIR / f"{DATASET}.pt"

torch.save(
    {
        "dataset": DATASET,
        "sample_size": N,
        "model": "source_roberta",
        "pooling": "mean",
        "representations": representations,
        "labels": labels,
    },
    save_path,
)

print("\nSaved:", save_path)
print("Representation Shape :", representations.shape)
print("Labels Shape         :", labels.shape)
print("=" * 70)