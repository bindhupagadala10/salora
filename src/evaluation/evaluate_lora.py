"""
Evaluate LoRA Baselines

Author:
Bindhu Pagadala

This script evaluates LoRA-finetuned adapters on WANLI and ANLI benchmarks.
It uses the Hugging Face Trainer for evaluation, computes Accuracy and Macro-F1,
and logs results to a CSV file.
"""

import os
# Ensure W&B is disabled for reproducibility
os.environ["WANDB_DISABLED"] = "true"

import random
import numpy as np
import torch
import pandas as pd
import evaluate
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)
from peft import PeftModel

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
RESULTS_FILE = "results/lora_baseline.csv"
MAX_LEN = 128
BATCH = 32

DATASETS = {
    "WANLI": {
        "file": "data/processed/wanli/test.parquet",
        "adapter": "models/lora_wanli",
    },
    "ANLI_R1": {
        "file": "data/processed/anli/dev_r1.parquet",
        "adapter": "models/lora_anli_r1",
    },
    "ANLI_R2": {
        "file": "data/processed/anli/dev_r2.parquet",
        "adapter": "models/lora_anli_r2",
    },
    "ANLI_R3": {
        "file": "data/processed/anli/dev_r3.parquet",
        "adapter": "models/lora_anli_r3",
    },
}

# --- Metrics ---
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

# --- Evaluation Loop ---
results = []
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

def tokenize(batch):
    return tokenizer(
        batch["premise"],
        batch["hypothesis"],
        truncation=True,
        max_length=MAX_LEN,
        padding=False,
    )

for name, info in DATASETS.items():
    print(f"\n{'='*54}")
    print(f"Dataset : {name}")
    print(f"Adapter : {info['adapter']}")
    print(f"{'='*54}")

    # Load and Prepare Data
    ds = load_dataset("parquet", data_files={"test": info["file"]})["test"]
    ds = ds.map(tokenize, batched=True)
    
    # Load Model and Adapter
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH, num_labels=3)
    model = PeftModel.from_pretrained(model, info["adapter"])
    
    trainer = Trainer(
        model=model,
        args=TrainingArguments(
            output_dir="tmp_eval",
            per_device_eval_batch_size=BATCH,
            report_to="none",
        ),
        eval_dataset=ds,
        tokenizer=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer),
        compute_metrics=compute_metrics,
    )

    metrics = trainer.evaluate()
    
    results.append({
        "Dataset": name,
        "Samples": len(ds),
        "Accuracy": metrics["eval_accuracy"],
        "Macro F1": metrics["eval_macro_f1"]
    })

# --- Saving and Summary ---
df = pd.DataFrame(results)
os.makedirs("results", exist_ok=True)
df.to_csv(RESULTS_FILE, index=False)

print("\nEvaluation Summary:")
print(df.to_string(index=False))
print(f"\nResults saved to {RESULTS_FILE}")