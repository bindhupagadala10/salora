"""
Evaluate Source Model by MNLI Genre

Author:
Bindhu Pagadala
"""

import os
os.environ["WANDB_DISABLED"] = "true"

from pathlib import Path
import numpy as np
import pandas as pd
import evaluate

from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding,
)

# ============================================================
# Configuration
# ============================================================

MODEL_PATH = "models/source_roberta"
DATA_PATH = "data/processed/mnli/validation_matched.parquet"

# ============================================================
# Metrics
# ============================================================

accuracy = evaluate.load("accuracy")
f1 = evaluate.load("f1")


def compute_metrics(eval_pred):

    logits, labels = eval_pred

    preds = np.argmax(logits, axis=-1)

    return {
        "accuracy": accuracy.compute(
            predictions=preds,
            references=labels,
        )["accuracy"],

        "macro_f1": f1.compute(
            predictions=preds,
            references=labels,
            average="macro",
        )["f1"],
    }


# ============================================================
# Load Dataset
# ============================================================

dataset = load_dataset(
    "parquet",
    data_files={"validation": DATA_PATH},
)["validation"]

print(f"\nTotal Samples : {len(dataset):,}")

print("\nGenres")

print(pd.Series(dataset["genre"]).value_counts().sort_index())

# ============================================================
# Load Model
# ============================================================

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_PATH
)

data_collator = DataCollatorWithPadding(tokenizer)

args = TrainingArguments(
    output_dir="tmp_eval",
    per_device_eval_batch_size=32,
    report_to="none",
)

trainer = Trainer(
    model=model,
    args=args,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

# ============================================================
# Evaluate Each Genre
# ============================================================

results = []

genres = sorted(set(dataset["genre"]))

for genre in genres:

    print("\n" + "=" * 70)
    print(genre.upper())
    print("=" * 70)

    subset = dataset.filter(
        lambda x: x["genre"] == genre
    )

    subset = subset.map(
        lambda batch: tokenizer(
            batch["premise"],
            batch["hypothesis"],
            truncation=True,
            max_length=128,
        ),
        batched=True,
    )

    metrics = trainer.evaluate(subset)

    results.append({

        "Genre": genre,

        "Samples": len(subset),

        "Accuracy": round(metrics["eval_accuracy"],4),

        "Macro F1": round(metrics["eval_macro_f1"],4),

    })

# ============================================================
# Save
# ============================================================

results = pd.DataFrame(results)

print("\n")
print(results)

Path("results").mkdir(exist_ok=True)

results.to_csv(
    "results/mnli_genre_baseline.csv",
    index=False,
)

print("\nSaved results/mnli_genre_baseline.csv")