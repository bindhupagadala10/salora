"""
SALoRA Source Model Training

Train RoBERTa-base on the processed MNLI dataset.

Author:
Bindhu Pagadala
"""

from pathlib import Path
import numpy as np
import evaluate
import random
import torch

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
from datasets import load_dataset
from src.utils.experiment_logger import log_experiment

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

# ============================================================
# Configuration
# ============================================================

MODEL_NAME = "roberta-base"

DATA_DIR = Path("data/processed/mnli")

OUTPUT_DIR = "models/source_roberta"

CHECKPOINT_DIR = "checkpoints/source"

# ============================================================
# Load Dataset
# ============================================================

print("=" * 70)
print("Loading processed MNLI")
print("=" * 70)

dataset = load_dataset(
    "parquet",
    data_files={
        "train": str(DATA_DIR / "train.parquet"),
        "validation": str(DATA_DIR / "validation_matched.parquet"),
    },
)

train = dataset["train"]
validation = dataset["validation"]
print(dataset)

print("\nLabel Distribution")

from collections import Counter

print("\nLabel Distribution")
print(Counter(train["label"]))

# ============================================================
# Tokenizer
# ============================================================

print("=" * 70)
print("Loading tokenizer")
print("=" * 70)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def tokenize(batch):

    return tokenizer(
        batch["premise"],
        batch["hypothesis"],
        truncation=True,
        max_length=128,
        padding=False
    )

train = train.map(tokenize, batched=True)
validation = validation.map(tokenize, batched=True)
columns = [
    "input_ids",
    "attention_mask",
    "label",
]

train.set_format("torch", columns=columns)
validation.set_format("torch", columns=columns)
data_collator = DataCollatorWithPadding(tokenizer)

# ============================================================
# Model
# ============================================================

print("=" * 70)
print("Loading model")
print("=" * 70)

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=3,
)
device = "cuda" if torch.cuda.is_available() else "cpu"

print(f"\nDevice : {device}")

print(f"GPU : {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
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
# Training Arguments
# ============================================================

args = TrainingArguments(

    output_dir=CHECKPOINT_DIR,

    overwrite_output_dir=True,

    num_train_epochs=3,

    learning_rate=2e-5,

    weight_decay=0.01,

    warmup_ratio=0.10,

    per_device_train_batch_size=16,

    per_device_eval_batch_size=32,

    evaluation_strategy="epoch",

    save_strategy="no",

    logging_strategy="steps",

    logging_steps=250,

    fp16=True,

    report_to="none",

    seed=42,

    logging_dir="logs/source",
)

# ============================================================
# Trainer
# ============================================================

trainer = Trainer(

    model=model,

    args=args,

    train_dataset=train,

    eval_dataset=validation,

    tokenizer=tokenizer,

    data_collator=data_collator,

    compute_metrics=compute_metrics,
)

# ============================================================
# Train
# ============================================================

print("=" * 70)
print("Training Source Model")
print("=" * 70)
print(f"Training samples : {len(train):,}")
print(f"Validation samples : {len(validation):,}")
trainer.train()

print("=" * 70)
print("Evaluating")
print("=" * 70)

metrics = trainer.evaluate(eval_dataset=validation)
log_experiment({

    "Experiment": "Source Baseline",

    "Model": "RoBERTa-base",

    "Dataset": "MNLI",

    "Epochs": 3,

    "Learning Rate": 2e-5,

    "Batch Size": 16,

    "Accuracy": metrics["eval_accuracy"],

    "Macro F1": metrics["eval_macro_f1"],

    "Checkpoint": OUTPUT_DIR,

})
print(metrics)

# ============================================================
# Save
# ============================================================

trainer.save_model(OUTPUT_DIR)

tokenizer.save_pretrained(OUTPUT_DIR)

print("=" * 70)
print("Training Complete")
print("=" * 70)
