"""
Standard LoRA Baseline (WANLI) - Part 1

Author:
Bindhu Pagadala

This script establishes the foundational baseline for RoBERTa LoRA fine-tuning.
It focuses on environment setup, dataset preparation, and LoRA injection
targeted at query and value modules to maintain a standard parameter budget.
"""

import os
# Ensure W&B is disabled for baseline reproducibility
os.environ["WANDB_DISABLED"] = "true"

import random
import numpy as np
import torch
from pathlib import Path
import argparse

from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

from peft import (
    LoraConfig,
    TaskType,
    get_peft_model,
)

from src.utils.experiment_logger import log_experiment

# --- Reproducibility ---
SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# --- Configuration ---
MODEL_PATH = "models/source_roberta"

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", required=True)
args = parser.parse_args()

DATASET = args.dataset
if DATASET == "wanli":
    TRAIN_FILE = "data/processed/wanli/train.parquet"
    TEST_FILE = "data/processed/wanli/test.parquet"

elif DATASET == "anli_r1":
    TRAIN_FILE = "data/processed/anli/train_r1.parquet"
    TEST_FILE = "data/processed/anli/dev_r1.parquet"

elif DATASET == "anli_r2":
    TRAIN_FILE = "data/processed/anli/train_r2.parquet"
    TEST_FILE = "data/processed/anli/dev_r2.parquet"

elif DATASET == "anli_r3":
    TRAIN_FILE = "data/processed/anli/train_r3.parquet"
    TEST_FILE = "data/processed/anli/dev_r3.parquet"

else:
    raise ValueError(f"Unknown dataset: {DATASET}")

OUTPUT_DIR = f"models/lora_{DATASET}"
CHECKPOINT_DIR = f"checkpoints/lora_{DATASET}"

MAX_LEN = 128
BATCH = 16
LR = 2e-4
EPOCHS = 3

# --- Dataset Loading ---
dataset = load_dataset(
    "parquet",
    data_files={
        "train": TRAIN_FILE,
        "test": TEST_FILE,
    },
)

train = dataset["train"]
test = dataset["test"]

# --- Tokenizer Setup ---
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

def tokenize(batch):
    return tokenizer(
        batch["premise"],
        batch["hypothesis"],
        truncation=True,
        max_length=MAX_LEN,
        padding=False,
    )

train = train.map(tokenize, batched=True)
test = test.map(tokenize, batched=True)

collator = DataCollatorWithPadding(tokenizer)

# --- Load Source Model ---
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_PATH,
    num_labels=3,
)

columns = [
    "input_ids",
    "attention_mask",
    "label",
]

train = train.remove_columns(
    [c for c in train.column_names if c not in columns]
)

test = test.remove_columns(
    [c for c in test.column_names if c not in columns]
)

train.set_format("torch")
test.set_format("torch")

# --- LoRA Configuration ---
# Targeting only ["query", "value"] as per standard RoBERTa LoRA best practices
lora_config = LoraConfig(
    task_type=TaskType.SEQ_CLS,
    r=8,
    lora_alpha=16,
    lora_dropout=0.1,
    bias="none",
    target_modules=[
        "query",
        "value",
    ],
)

# --- Inject LoRA ---
model = get_peft_model(
    model,
    lora_config,
)

# --- Training Arguments ---
training_args = TrainingArguments(
    output_dir=CHECKPOINT_DIR,
    learning_rate=LR,
    per_device_train_batch_size=BATCH,
    per_device_eval_batch_size=BATCH,
    num_train_epochs=EPOCHS,
    weight_decay=0.01,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    save_total_limit=1,
    # Gated on cuda specifically, not mps: verified empirically (2026-09-05) that both
    # fp16=True and bf16=True raise ValueError via accelerate's Accelerator on MPS
    # ("fp16 mixed precision requires a GPU" / "doesn't support bf16/gpu") with
    # accelerate==1.9.0 + transformers==4.55.0. Do not widen this to
    # `torch.cuda.is_available() or torch.backends.mps.is_available()` -- that crashes
    # on this hardware. MPS runs in fp32; Trainer already places the model on "mps"
    # automatically without any device_map/device argument here.
    fp16=torch.cuda.is_available(),
    logging_steps=100,
    report_to="none",
    seed=SEED,
    remove_unused_columns=False,
)

# --- Metrics ---
import evaluate
accuracy_metric = evaluate.load("accuracy")
f1_metric = evaluate.load("f1")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    acc = accuracy_metric.compute(predictions=predictions, references=labels)
    f1 = f1_metric.compute(predictions=predictions, references=labels, average="macro")
    return {
        "accuracy": acc["accuracy"],
        "macro_f1": f1["f1"]
    }

# --- Trainer ---
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train,
    eval_dataset=test,
    tokenizer=tokenizer,
    data_collator=collator,
    compute_metrics=compute_metrics,
)

# --- Training Execution ---
if __name__ == "__main__":
    print("======================================================================")
    print("LoRA Training")
    print("======================================================================")
    print(f"Dataset : {DATASET}")
    print(f"Train Samples : {len(train)}")
    print(f"Test Samples  : {len(test)}")
    model.print_trainable_parameters()
    
    print("Starting training...")
    trainer.train()

    # --- Final Evaluation ---
    print("Running final evaluation...")
    eval_results = trainer.evaluate()
    print(f"Evaluation Results: {eval_results}")

    # --- Saving ---
    # We save the adapter weights to the specified output directory
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"Model adapters saved to {OUTPUT_DIR}")

    # --- Experiment Logging ---
    # Log the final results for baseline tracking
    log_experiment({
        "Experiment": "LoRA Baseline",
        "Base Model": "Source RoBERTa",
        "Target": DATASET,
        "Rank": lora_config.r,
        "Alpha": lora_config.lora_alpha,
        "Dropout": lora_config.lora_dropout,
        "Target Modules": ",".join(lora_config.target_modules),
        "Learning Rate": LR,
        "Epochs": EPOCHS,
        "Accuracy": eval_results["eval_accuracy"],
        "Macro F1": eval_results["eval_macro_f1"],
    })