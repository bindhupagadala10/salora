"""
Evaluate the trained MNLI source model on all benchmarks.

Author:
Bindhu Pagadala
"""

from pathlib import Path
import numpy as np
import pandas as pd
import evaluate

from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    DataCollatorWithPadding,
)

# ============================================================
# Configuration
# ============================================================

MODEL_PATH = "models/source_roberta"

DATA_DIR = Path("data/processed")

BATCH_SIZE = 32

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
# Model
# ============================================================

print("=" * 70)
print("Loading trained source model")
print("=" * 70)

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_PATH
)

data_collator = DataCollatorWithPadding(tokenizer)

trainer = Trainer(
    model=model,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

# ============================================================
# Evaluation Sets
# ============================================================

DATASETS = {

    "MNLI_Matched":
        DATA_DIR / "mnli" / "validation_matched.parquet",

    "MNLI_Mismatched":
        DATA_DIR / "mnli" / "validation_mismatched.parquet",

    "WANLI":
        DATA_DIR / "wanli" / "test.parquet",

    "ANLI_R1":
        DATA_DIR / "anli" / "dev_r1.parquet",

    "ANLI_R2":
        DATA_DIR / "anli" / "dev_r2.parquet",

    "ANLI_R3":
        DATA_DIR / "anli" / "dev_r3.parquet",
}

# ============================================================
# Evaluate
# ============================================================

results = []

for name, file in DATASETS.items():

    print("\n" + "=" * 70)
    print(name)
    print("=" * 70)

    dataset = load_dataset(
        "parquet",
        data_files={"test": str(file)},
    )["test"]

    def tokenize(batch):

        return tokenizer(
            batch["premise"],
            batch["hypothesis"],
            truncation=True,
            max_length=128,
        )

    dataset = dataset.map(tokenize, batched=True)

    metrics = trainer.evaluate(
        eval_dataset=dataset,
        metric_key_prefix="eval",
    )

    print(metrics)

    results.append({

        "Dataset": name,

        "Samples": len(dataset),

        "Accuracy": round(metrics["eval_accuracy"], 4),

        "Macro F1": round(metrics["eval_macro_f1"], 4),

    })

# ============================================================
# Save
# ============================================================

results = pd.DataFrame(results)

print("\n")
print(results)

Path("results").mkdir(exist_ok=True)

results.to_csv(
    "results/source_baseline.csv",
    index=False,
)

print("\nSaved results/source_baseline.csv")