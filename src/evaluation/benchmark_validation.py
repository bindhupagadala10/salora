"""
SALoRA Benchmark Validation

Purpose
-------
Evaluate a pretrained MNLI model on all candidate datasets to determine
whether meaningful domain shift exists BEFORE training any models.

Outputs
-------
results/benchmark_validation/

    benchmark_results.csv

Author:
Bindhu Pagadala
"""

from pathlib import Path

import pandas as pd
import torch
from datasets import load_dataset
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm
from transformers import pipeline

# -------------------------------------------------------
# Configuration
# -------------------------------------------------------

MODEL_NAME = "FacebookAI/roberta-large-mnli"

OUTPUT_DIR = Path("results/benchmark_validation")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = 0 if torch.cuda.is_available() else -1

# -------------------------------------------------------
# Load Model
# -------------------------------------------------------

print("=" * 70)
print("Loading MNLI model...")
print("=" * 70)

classifier = pipeline(
    "text-classification",
    model=MODEL_NAME,
    tokenizer=MODEL_NAME,
    return_all_scores=False,
    device=DEVICE,
)

print("Model Loaded.\n")

# -------------------------------------------------------
# Dataset Definitions
# -------------------------------------------------------

DATASETS = {

    "MNLI": (
        load_dataset("glue", "mnli")["validation_matched"],
        "premise",
        "hypothesis",
        "label",
    ),

    "WANLI": (
        load_dataset("alisawuffles/WANLI")["test"],
        "premise",
        "hypothesis",
        "gold",
    ),

    "ANLI_R1": (
        load_dataset("facebook/anli")["dev_r1"],
        "premise",
        "hypothesis",
        "label",
    ),

    "ANLI_R2": (
        load_dataset("facebook/anli")["dev_r2"],
        "premise",
        "hypothesis",
        "label",
    ),

    "ANLI_R3": (
        load_dataset("facebook/anli")["dev_r3"],
        "premise",
        "hypothesis",
        "label",
    ),
}

# -------------------------------------------------------
# Dynamic Label Mapping
# -------------------------------------------------------

model_labels = classifier.model.config.label2id

print("\nModel Label Mapping")
print(model_labels)
print()

reverse_map = {}

for label_name, idx in model_labels.items():

    label_name = label_name.lower()

    if "entail" in label_name:
        reverse_map[idx] = 0

    elif "neutral" in label_name:
        reverse_map[idx] = 1

    elif "contrad" in label_name:
        reverse_map[idx] = 2

print("Internal Mapping:", reverse_map)
print()

# -------------------------------------------------------
# Evaluation
# -------------------------------------------------------

results = []

for dataset_name, (
    dataset,
    premise_col,
    hypothesis_col,
    label_col,
) in DATASETS.items():

    print("=" * 70)
    print(dataset_name)
    print("=" * 70)

    predictions = []
    labels = []

    for sample in tqdm(dataset):

        text = sample[premise_col]

        result = classifier(
            text,
            text_pair=sample[hypothesis_col],
            truncation=True,
        )[0]

        predicted = classifier.model.config.label2id[
            result["label"]
        ]

        predicted = reverse_map[predicted]

        predictions.append(predicted)

        if dataset_name == "WANLI":

            label_map = {
                "entailment": 0,
                "neutral": 1,
                "contradiction": 2,
            }

            labels.append(label_map[sample[label_col]])

        else:

            labels.append(sample[label_col])

    acc = accuracy_score(labels, predictions)

    macro = f1_score(
        labels,
        predictions,
        average="macro",
    )

    print(f"Accuracy : {acc:.4f}")
    print(f"Macro F1 : {macro:.4f}")
    print()

    results.append(
        {
            "Dataset": dataset_name,
            "Samples": len(dataset),
            "Accuracy": round(acc, 4),
            "Macro F1": round(macro, 4),
        }
    )

# -------------------------------------------------------
# Save
# -------------------------------------------------------

results_df = pd.DataFrame(results)

results_df.to_csv(
    OUTPUT_DIR / "benchmark_results.csv",
    index=False,
)

print("=" * 70)
print("Benchmark Validation Complete")
print("=" * 70)
print(results_df)