"""
Step 5 executable verification: confirms, against the REAL source
checkpoint and the REAL installed peft/transformers versions, that:

  1. LoraConfig(rank_pattern=...) actually applies different ranks to
     different encoder layers (not silently falling back to a single
     default rank everywhere).
  2. The resulting trainable-parameter count matches what
     src/salora/allocator.compute_trainable_params() predicts.
  3. Only query/value modules in the 12 encoder layers receive LoRA --
     no embedding, output-dense, FFN, or classifier-head modules do.
  4. The classification head stays frozen under the current LoraConfig
     (no modules_to_save), confirming the parameter-budget comparison
     across methods is clean (identical head across all methods).

This does NOT train anything -- it only instantiates the model,
applies a rank pattern, and inspects parameter counts/requires_grad.
Safe to run on CPU, takes seconds.

Run from repo root:
    python -m src.salora.verify_peft_mechanism

Author:
Bindhu Pagadala
"""

import re

import torch
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoModelForSequenceClassification

from src.salora.allocator import (
    allocate_ranks,
    compute_trainable_params,
    to_peft_rank_pattern,
)

MODEL_PATH = "models/source_roberta"


def count_trainable(model) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def main():
    print("=" * 70)
    print("Step 5: PEFT rank_pattern executable verification")
    print("=" * 70)

    # A deliberately non-uniform, non-trivial drift profile so the test
    # actually exercises varying ranks per layer (not all-8 by accident).
    drift = {
        1: 0.10, 2: 0.12, 3: 0.15, 4: 0.17, 5: 0.20, 6: 0.27,
        7: 0.31, 8: 0.42, 9: 0.55, 10: 0.61, 11: 0.72, 12: 0.91,
    }
    allocation = allocate_ranks(drift, total_budget=96, metric="mmd_smoketest")
    print("\nAllocation under test:")
    for l in sorted(allocation.layer_ranks):
        print(f"  layer {l:2d} -> r={allocation.layer_ranks[l]}")
    expected_params = allocation.trainable_params()
    print(f"\nAllocator-predicted trainable LoRA params: {expected_params}")

    print("\nLoading base model from:", MODEL_PATH)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_PATH, num_labels=3
    )

    rank_pattern = to_peft_rank_pattern(allocation)
    print("\nrank_pattern passed to LoraConfig:")
    for k, v in rank_pattern.items():
        print(f"  {k!r}: {v}")

    config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=8,  # default/fallback rank; every layer should be overridden below
        lora_alpha=16,
        lora_dropout=0.1,
        bias="none",
        target_modules=["query", "value"],
        rank_pattern=rank_pattern,
    )

    peft_model = get_peft_model(model, config)

    print("\n--- Check 1: per-layer ranks actually applied ---")
    mismatches = []
    for name, module in peft_model.named_modules():
        if hasattr(module, "lora_A") and "default" in getattr(module, "lora_A", {}):
            actual_r = module.lora_A["default"].out_features
            m = re.search(r"layer\.(\d+)\.attention\.self\.(query|value)", name)
            if not m:
                print(f"  UNEXPECTED LoRA module (does not match query/value pattern): {name}")
                mismatches.append(name)
                continue
            layer0 = int(m.group(1))
            expected_r = allocation.layer_ranks[layer0 + 1]
            status = "OK" if actual_r == expected_r else "MISMATCH"
            if status == "MISMATCH":
                mismatches.append(name)
            print(f"  {name:55s} r={actual_r:3d} (expected {expected_r:3d}) [{status}]")

    if mismatches:
        print(f"\nFAIL: {len(mismatches)} module(s) did not get the expected rank.")
    else:
        print("\nPASS: every LoRA module has exactly the rank the allocator assigned.")

    print("\n--- Check 2: trainable parameter count matches allocator prediction ---")
    actual_trainable = count_trainable(peft_model)
    print(f"  peft model trainable params : {actual_trainable}")
    print(f"  allocator predicted params  : {expected_params}")
    if actual_trainable == expected_params:
        print("  PASS: exact match.")
    else:
        print(
            "  FAIL or NEEDS INVESTIGATION: mismatch. If the classifier head "
            "is unexpectedly trainable (see Check 4 below), that would "
            "explain a difference of the head's own parameter count -- "
            "subtract it and re-check before concluding the allocator is wrong."
        )

    print("\n--- Check 3: only query/value in the 12 encoder layers got LoRA ---")
    lora_module_names = [
        name for name, module in peft_model.named_modules()
        if hasattr(module, "lora_A") and "default" in getattr(module, "lora_A", {})
    ]
    unexpected = [
        n for n in lora_module_names
        if not re.search(r"layer\.\d+\.attention\.self\.(query|value)$", n)
    ]
    print(f"  total LoRA modules found: {len(lora_module_names)} (expected 24 = 12 layers x 2)")
    if len(lora_module_names) != 24 or unexpected:
        print(f"  FAIL: unexpected count or modules: {unexpected}")
    else:
        print("  PASS: exactly 24 LoRA modules, all query/value in encoder layers.")

    print("\n--- Check 4: classification head is frozen (no modules_to_save) ---")
    head_trainable = [
        (n, p.numel()) for n, p in peft_model.named_parameters()
        if ("classifier" in n or "score" in n) and p.requires_grad
    ]
    if head_trainable:
        print(f"  Head IS trainable ({sum(c for _, c in head_trainable)} params): {head_trainable}")
        print("  This means train_lora.py's baseline WAS updating the classifier "
              "head after all -- re-check the parameter-equality assumption in "
              "docs/design/SALoRA_Allocation_Spec.md Section 3/9.")
    else:
        print("  PASS: classifier head has zero trainable parameters (frozen), "
              "confirming only LoRA adapters vary across methods.")

    print("\n" + "=" * 70)
    print("Done. Report all four check results back before proceeding to Step 6.")
    print("=" * 70)


if __name__ == "__main__":
    main()
