"""
AdaLoRA baseline (Zhang et al., 2023), under the closest achievable match
to this project's 887,811-trainable-parameter budget.

Reuses the exact same data pipeline, hyperparameters, and TrainingArguments
as src/salora/train_salora.py -- model, data, optimizer, LR, epochs, batch
size, and target_modules are all identical, per the project's non-negotiable
comparison rule. What differs is the rank-allocation mechanism itself:
AdaLoRA prunes rank *during* training from a gradient-derived importance
score, rather than from an ante-hoc drift signal or a fixed uniform rank.

IMPORTANT, verified empirically before writing this script (see chat, and
docs/design/Research_Design_Document.md): AdaLoRA's SVD-style
parameterization (P*E*Q, with a learned per-rank-unit singular value E)
means its backing parameter tensors are sized at `init_r`, not `target_r`,
and this project's own counting convention --
`sum(p.numel() for p in model.parameters() if p.requires_grad)` -- does
NOT shrink as AdaLoRA prunes down to target_r over training. Rank pruning
is enforced by zeroing values in the rank_pattern, not by resizing tensors
(confirmed directly in peft's AdaLoraModel.update_and_allocate, which
explicitly notes resizing "freezes the trainable parameters and nothing
gets updates" and masks instead). Consequently:

  - RAW_EXPECTED_TRAINABLE (asserted below, constant for the entire run):
    24 modules * (2*768*INIT_R + INIT_R) + 592,899 head params.
    With INIT_R=16 (this project's own r_max ceiling): 1,183,107.
    This is NOT directly comparable to the other methods' 887,811.

  - EFFECTIVE_BUDGET_AT_TARGET_R (computed, not asserted, from the final
    rank_pattern logged after training): 24 * target_r * (2*768) +
    24 * target_r (E-vectors) + 592,899 head params.
    With TARGET_R=8 (matching the uniform baseline's rank): 888,003 --
    192 parameters (0.0216%) more than 887,811, an unavoidable, disclosed
    structural overhead from AdaLoRA's per-rank-unit E scalar, which plain
    LoRA does not have.

Both numbers are logged per run. The paper reports this honestly as the
closest achievable match, not a bit-for-bit identical budget -- see
docs/design/Research_Design_Document.md and the paper's Limitations
section for the full disclosure.

Author:
Bindhu Pagadala
"""

import os
os.environ["WANDB_DISABLED"] = "true"

import math
import random as pyrandom
import argparse

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
from peft import AdaLoraConfig, TaskType, get_peft_model
import evaluate

from src.utils.experiment_logger import log_experiment

MODEL_PATH = "models/source_roberta"

HEAD_TRAINABLE_PARAMS = 592_899  # same constant verified in src/salora/verify_peft_mechanism.py
HIDDEN = 768
TARGET_MODULES = ["query", "value"]
NUM_MODULES = 12 * len(TARGET_MODULES)  # 12 encoder layers x {query, value} = 24

TARGET_R = 8   # matches the uniform-LoRA baseline's rank, for the closest achievable budget match
INIT_R = 16    # matches this project's own r_max clamp (SALoRA_Allocation_Spec.md), giving AdaLoRA
               # genuine headroom to prune, not a no-op target_r==init_r setup

RAW_EXPECTED_TRAINABLE = NUM_MODULES * (2 * HIDDEN * INIT_R + INIT_R) + HEAD_TRAINABLE_PARAMS
EFFECTIVE_BUDGET_AT_TARGET_R = NUM_MODULES * (2 * HIDDEN * TARGET_R + TARGET_R) + HEAD_TRAINABLE_PARAMS

DATASET_FILES = {
    "wanli": ("data/processed/wanli/train.parquet", "data/processed/wanli/test.parquet"),
    "anli_r1": ("data/processed/anli/train_r1.parquet", "data/processed/anli/dev_r1.parquet"),
    "anli_r2": ("data/processed/anli/train_r2.parquet", "data/processed/anli/dev_r2.parquet"),
    "anli_r3": ("data/processed/anli/train_r3.parquet", "data/processed/anli/dev_r3.parquet"),
}


def count_trainable(model) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class AdaLoraTrainer(Trainer):
    """
    peft's AdaLoRA rank pruning is not wired into Trainer automatically --
    model.base_model.update_and_allocate(global_step) must be called once
    per optimizer step, after loss.backward() and before zero_grad(), while
    gradients are still populated (its importance-score update reads
    p.grad directly). A Trainer.training_step override is used rather than
    an on_step_end callback: empirically verified (see chat/smoke test)
    that by the time on_step_end fires in this transformers version,
    zero_grad() has already run and p.grad is None, crashing
    update_and_allocate's importance-score computation.
    """

    def training_step(self, model, inputs, num_items_in_batch=None):
        loss = super().training_step(model, inputs, num_items_in_batch)
        model.base_model.update_and_allocate(self.state.global_step)
        return loss


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=list(DATASET_FILES))
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()

    DATASET = args.dataset
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

    OUTPUT_DIR = f"models/adalora_{DATASET}_seed{SEED}"
    CHECKPOINT_DIR = f"checkpoints/adalora_{DATASET}_seed{SEED}"

    print("=" * 70)
    print("AdaLoRA baseline training")
    print("=" * 70)
    print(f"Dataset : {DATASET}")
    print(f"Seed    : {SEED}")

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

    columns = ["input_ids", "attention_mask", "label"]
    train = train.remove_columns([c for c in train.column_names if c not in columns])
    test = test.remove_columns([c for c in test.column_names if c not in columns])
    train.set_format("torch")
    test.set_format("torch")

    # --- AdaLoRA schedule, computed from the real number of optimizer steps ---
    # (total_step must match reality for AdaLoRA's tinit/tfinal/deltaT bookkeeping
    # to behave sensibly -- it is NOT a free-standing hyperparameter here.)
    steps_per_epoch = math.ceil(len(train) / BATCH)
    total_step = steps_per_epoch * EPOCHS
    tinit = max(1, round(0.1 * total_step))
    tfinal = max(1, round(0.1 * total_step))
    deltaT = max(1, round(total_step / 100))

    print(f"Total optimizer steps: {total_step} (tinit={tinit}, tfinal={tfinal}, deltaT={deltaT})")

    # --- Model + AdaLoRA injection ---
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH, num_labels=3)

    adalora_config = AdaLoraConfig(
        task_type=TaskType.SEQ_CLS,
        target_modules=TARGET_MODULES,
        target_r=TARGET_R,
        init_r=INIT_R,
        tinit=tinit,
        tfinal=tfinal,
        deltaT=deltaT,
        total_step=total_step,
        lora_alpha=16,
        lora_dropout=0.1,
    )
    model = get_peft_model(model, adalora_config)

    # --- Parameter-count verification (raw backing-tensor count; see module
    # docstring for why this is NOT 887,811 and is not expected to be) ---
    actual_total = count_trainable(model)
    print(f"Raw trainable params (init_r={INIT_R}, constant throughout training): {actual_total}")
    print(f"Expected (precomputed): {RAW_EXPECTED_TRAINABLE}")
    assert actual_total == RAW_EXPECTED_TRAINABLE, (
        f"Raw trainable-parameter count mismatch for adalora/{DATASET}/seed{SEED}: "
        f"expected {RAW_EXPECTED_TRAINABLE}, got {actual_total}. This would indicate a "
        f"peft-version behavior change in AdaLoRA's parameterization -- do not proceed "
        f"with training until this is understood."
    )
    print("Raw parameter-count check: PASS (matches precomputed expectation)")
    print(f"Effective budget AdaLoRA will converge to at target_r={TARGET_R}: "
          f"{EFFECTIVE_BUDGET_AT_TARGET_R} (887,811 + 192-param E-vector overhead)")

    # --- Training Arguments (identical to train_salora.py) ---
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

    trainer = AdaLoraTrainer(
        model=model,
        args=training_args,
        train_dataset=train,
        eval_dataset=test,
        processing_class=tokenizer,
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

    # --- Final effective rank pattern (post-pruning), for honest accounting ---
    final_rank_pattern = adalora_config.rank_pattern
    if final_rank_pattern:
        final_ranks = [int(torch.as_tensor(v).sum().item()) if not isinstance(v, int) else v
                       for v in final_rank_pattern.values()]
        effective_lora_params = sum(r * (2 * HIDDEN + 1) for r in final_ranks)
        effective_total = effective_lora_params + HEAD_TRAINABLE_PARAMS
    else:
        # force_mask step never triggered (e.g. tfinal reached exactly at total_step);
        # fall back to the designed target_r for every module.
        final_ranks = [TARGET_R] * NUM_MODULES
        effective_total = EFFECTIVE_BUDGET_AT_TARGET_R

    print(f"Final per-module ranks (post-pruning): {final_ranks}")
    print(f"Effective (post-pruning) trainable-parameter budget: {effective_total}")

    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"Model adapters saved to {OUTPUT_DIR}")

    log_experiment({
        "Experiment": "AdaLoRA Baseline",
        "Base Model": "Source RoBERTa",
        "Target": DATASET,
        "Method": "adalora",
        "Metric": "adalora_importance",
        "Seed": SEED,
        "Allocation_Seed": None,
        "Layer_Ranks": ",".join(str(r) for r in final_ranks),
        "LoRA_Only_Params_Raw": RAW_EXPECTED_TRAINABLE - HEAD_TRAINABLE_PARAMS,
        "Head_Params": HEAD_TRAINABLE_PARAMS,
        "Total_Trainable_Params_Raw": actual_total,
        "Total_Trainable_Params_Effective": effective_total,
        "Alpha": adalora_config.lora_alpha,
        "Dropout": adalora_config.lora_dropout,
        "Target Modules": ",".join(adalora_config.target_modules),
        "Learning Rate": LR,
        "Epochs": EPOCHS,
        "Init_R": INIT_R,
        "Target_R": TARGET_R,
        "Accuracy": eval_results["eval_accuracy"],
        "Macro F1": eval_results["eval_macro_f1"],
    })


if __name__ == "__main__":
    main()
