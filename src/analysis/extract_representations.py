"""
Layer-wise Representation Extraction

Extracts mean-pooled hidden representations from the
frozen MNLI source model.

Pooling method: attention-mask-weighted mean pooling.
    - For every layer, each token position is summed only if its
      attention_mask == 1 (i.e. it is a real, non-padding token).
    - The sum is divided by the per-example count of non-padding
      tokens, not by the batch's (padded) sequence length.
    - CLS pooling is NOT used anywhere in this project. The pooled
      vector is the mean of all non-padding token representations
      for the premise+hypothesis pair at that layer.

This replaces an earlier implementation that used a naive
`hidden.mean(dim=1)`, which averaged over the full padded sequence
length and therefore diluted every pooled vector by however much
padding happened to be present in that batch (dynamic padding means
this varied per batch, not just per dataset). See
docs/design/Repository_Audit_Report.md, Section 5, for the original
finding.

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
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
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

# Standardize column naming to 'labels'
dataset = dataset.rename_column("label", "labels")

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
    "labels",
]

dataset = dataset.remove_columns(
    [
        c
        for c in dataset.column_names
        if c not in columns
    ]
)

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
labels_list = []

print("\nExtracting representations...\n")

with torch.no_grad():

    for batch in loader:

        labels_list.append(
            batch["labels"]
        )

        input_ids = batch["input_ids"].to(DEVICE)
        attention_mask = batch["attention_mask"].to(DEVICE)

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )

        hidden_states = outputs.hidden_states

        # --- Masked mean pooling ---
        # mask: (B, T, 1), broadcastable over the hidden dimension.
        # Padding positions (attention_mask == 0) contribute exactly
        # zero to the numerator AND are excluded from the token count
        # used in the denominator, so they cannot dilute the pooled
        # vector regardless of how much padding this batch added.
        mask = attention_mask.unsqueeze(-1).to(
            dtype=hidden_states[0].dtype
        )

        # Per-example non-padding token count. Clamp guards against a
        # division by zero, which should not occur in practice (every
        # example has at least one real token) but costs nothing to
        # guard against explicitly.
        token_counts = mask.sum(dim=1).clamp(min=1e-9)

        for i, hidden in enumerate(hidden_states):

            pooled = (hidden * mask).sum(dim=1) / token_counts

            layer_outputs[i].append(
                pooled.cpu()
            )

representations = torch.stack(
    [
        torch.cat(layer, dim=0)
        for layer in layer_outputs
    ]
)

labels = torch.cat(labels_list)

save_path = OUTPUT_DIR / f"{DATASET}.pt"

torch.save(
    {
        "dataset": DATASET,
        "sample_size": N,
        "model": "source_roberta",
        "pooling": "masked_mean",
        "representations": representations,
        "labels": labels,
    },
    save_path,
)

print("\nSaved:", save_path)
print("Representation Shape :", representations.shape)
print("Labels Shape         :", labels.shape)
print("=" * 70)