"""
SALoRA / control-baseline training.

Reuses the exact same data pipeline, hyperparameters, and TrainingArguments
as src/training/train_lora.py (the uniform-LoRA baseline) -- model, data,
optimizer, LR, epochs, batch size, and target_modules are all identical
across every method, per the project's non-negotiable comparison rule.
The only thing that varies between methods is the LoRA rank_pattern.

Methods (Spec Section 8, docs/design/SALoRA_Allocation_Spec.md):
  uniform            -- r=8 every layer (same as train_lora.py; included
                         here too so a matched-seed uniform run can be
                         produced with the identical script/config path).
  salora_mmd         -- allocation from the MMD drift profile.
  salora_sinkhorn    -- allocation from the Sinkhorn drift profile.
  random             -- reproducible random partition (allocation_seed,
                         NOT the training seed).
  inverse_mmd        -- 1/(D_l+eps) on the MMD profile (paired control for
                         salora_mmd, per the 2026-09-05 deviation log entry).
  inverse_sinkhorn   -- 1/(D_l+eps) on the Sinkhorn profile (paired control
                         for salora_sinkhorn).

Every run asserts its actual total trainable-parameter count (LoRA + the
classifier head that peft auto-attaches for TaskType.SEQ_CLS, per the
2026-09-05 deviation log entry) matches the expected constant before
training starts, and logs it with the result -- per the project's rule
that parameter-count equality must be verified programmatically, not
assumed.

Author:
Bindhu Pagadala
"""

import os
os.environ["WANDB_DISABLED"] = "true"

import random as pyrandom
import argparse
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)
from peft import LoraConfig, TaskType, get_peft_model
import evaluate

from src.utils.experiment_logger import log_experiment
from src.salora.allocator import (
    Allocation,
    allocate_uniform,
    to_peft_rank_pattern,
)

MODEL_PATH = "models/source_roberta"
N = 950  # drift sample size used for every serialized allocation (Deviation Log, 2026-09-06)

# Verified empirically in Step 3 (src/salora/verify_peft_mechanism.py, 2026-09-05):
# get_peft_model(..., task_type=TaskType.SEQ_CLS) auto-attaches the classifier head
# via modules_to_save, adding this many trainable params, constant regardless of
# rank_pattern. Every method's expected total is LoRA-only params + this constant.
HEAD_TRAINABLE_PARAMS = 592_899

DATASET_FILES = {
    "wanli": ("data/processed/wanli/train.parquet", "data/processed/wanli/test.parquet"),
    "anli_r1": ("data/processed/anli/train_r1.parquet", "data/processed/anli/dev_r1.parquet"),
    "anli_r2": ("data/processed/anli/train_r2.parquet", "data/processed/anli/dev_r2.parquet"),
    "anli_r3": ("data/processed/anli/train_r3.parquet", "data/processed/anli/dev_r3.parquet"),
}

METHODS = [
    "uniform",
    "salora_mmd",
    "salora_sinkhorn",
    "random",
    "inverse_mmd",
    "inverse_sinkhorn",
]


def count_trainable(model) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=list(DATASET_FILES))
    parser.add_argument("--method", required=True, choices=METHODS)
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()

    DATASET = args.dataset
    METHOD = args.method
    SEED = args.seed

    pyrandom.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    TRAIN_FILE, TEST_FILE = DATASET_FILES[DATASET]
    MAX_LEN = 128
    BATCH = 16
    LR = 2e-4
    EPOCHS = 3

    OUTPUT_DIR = f"models/salora_{METHOD}_{DATASET}_seed{SEED}"
    CHECKPOINT_DIR = f"checkpoints/salora_{METHOD}_{DATASET}_seed{SEED}"

    print("=" * 70)
    print("SALoRA / control-baseline training")
    print("=" * 70)
    print(f"Dataset : {DATASET}")
    print(f"Method  : {METHOD}")
    print(f"Seed    : {SEED}")

    # --- Allocation ---
    allocation_seed = None
    metric = METHOD
    if METHOD == "uniform":
        allocation = allocate_uniform()
    else:
        alloc_path = Path(f"results/salora/allocations/{DATASET}_{METHOD}_n{N}.csv")
        allocation = Allocation.load_csv(alloc_path)
        allocation_seed = allocation.allocation_seed
        metric = allocation.metric

    lora_only_expected = allocation.trainable_params()
    total_expected = lora_only_expected + HEAD_TRAINABLE_PARAMS
    ranks = [allocation.layer_ranks[l] for l in sorted(allocation.layer_ranks)]
    print(f"Per-layer ranks (1..12): {ranks}")
    print(f"Expected LoRA-only params: {lora_only_expected}")
    print(f"Expected total trainable params (LoRA + head): {total_expected}")

    # --- Dataset Loading ---
    dataset = load_dataset("parquet", data_files={"train": TRAIN_FILE, "test": TEST_FILE})
    train = dataset["train"]
    test = dataset["test"]

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

    def tokenize(batch):
        return tokenizer(
            batch["premise"], batch["hypothesis"],
            truncation=True, max_length=MAX_LEN, padding=False,
        )

    train = train.map(tokenize, batched=True)
    test = test.map(tokenize, batched=True)

    collator = DataCollatorWithPadding(tokenizer)

    # --- Load Source Model ---
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH, num_labels=3)

    columns = ["input_ids", "attention_mask", "label"]
    train = train.remove_columns([c for c in train.column_names if c not in columns])
    test = test.remove_columns([c for c in test.column_names if c not in columns])
    train.set_format("torch")
    test.set_format("torch")

    # --- LoRA Configuration ---
    lora_kwargs = dict(
        task_type=TaskType.SEQ_CLS,
        r=8,
        lora_alpha=16,
        lora_dropout=0.1,
        bias="none",
        target_modules=["query", "value"],
    )
    if METHOD != "uniform":
        lora_kwargs["rank_pattern"] = to_peft_rank_pattern(allocation)

    lora_config = LoraConfig(**lora_kwargs)
    model = get_peft_model(model, lora_config)

    # --- Parameter-count verification (must pass before training starts) ---
    actual_total = count_trainable(model)
    print(f"Actual total trainable params: {actual_total}")
    assert actual_total == total_expected, (
        f"Parameter-count mismatch for {DATASET}/{METHOD}/seed{SEED}: "
        f"expected {total_expected} (LoRA {lora_only_expected} + head {HEAD_TRAINABLE_PARAMS}), "
        f"got {actual_total}. Do not proceed with training until this is understood."
    )
    print("Parameter-count check: PASS")

    # --- Training Arguments (identical to train_lora.py) ---
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
        # See train_lora.py: gated on cuda specifically, not mps -- fp16/bf16
        # both raise via accelerate's Accelerator on MPS with this
        # transformers/accelerate combo (verified empirically 2026-09-05).
        fp16=torch.cuda.is_available(),
        logging_steps=100,
        report_to="none",
        seed=SEED,
        remove_unused_columns=False,
    )

    accuracy_metric = evaluate.load("accuracy")
    f1_metric = evaluate.load("f1")

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        acc = accuracy_metric.compute(predictions=predictions, references=labels)
        f1 = f1_metric.compute(predictions=predictions, references=labels, average="macro")
        return {"accuracy": acc["accuracy"], "macro_f1": f1["f1"]}

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train,
        eval_dataset=test,
        tokenizer=tokenizer,
        data_collator=collator,
        compute_metrics=compute_metrics,
    )

    print(f"Train Samples : {len(train)}")
    print(f"Test Samples  : {len(test)}")
    model.print_trainable_parameters()

    print("Starting training...")
    trainer.train()

    print("Running final evaluation...")
    eval_results = trainer.evaluate()
    print(f"Evaluation Results: {eval_results}")

    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"Model adapters saved to {OUTPUT_DIR}")

    log_experiment({
        "Experiment": "SALoRA Matrix",
        "Base Model": "Source RoBERTa",
        "Target": DATASET,
        "Method": METHOD,
        "Metric": metric,
        "Seed": SEED,
        "Allocation_Seed": allocation_seed,
        "Layer_Ranks": ",".join(str(r) for r in ranks),
        "LoRA_Only_Params": lora_only_expected,
        "Head_Params": HEAD_TRAINABLE_PARAMS,
        "Total_Trainable_Params": actual_total,
        "Alpha": lora_config.lora_alpha,
        "Dropout": lora_config.lora_dropout,
        "Target Modules": ",".join(lora_config.target_modules),
        "Learning Rate": LR,
        "Epochs": EPOCHS,
        "Accuracy": eval_results["eval_accuracy"],
        "Macro F1": eval_results["eval_macro_f1"],
    })


if __name__ == "__main__":
    main()
