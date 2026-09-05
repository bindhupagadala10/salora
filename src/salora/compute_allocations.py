"""
Computes and serializes every allocation needed for the Step 4 experimental
matrix (SALoRA-MMD, SALoRA-Sinkhorn, Random, Inverse-MMD, Inverse-Sinkhorn)
for all four target domains, from the N=950 drift profiles (Deviation Log,
SALoRA_Allocation_Spec.md, 2026-09-06: N=950 chosen as the lowest-variance
sample size now that Step 2 confirmed stability across N).

Inverse-drift is computed separately per metric (Inverse-MMD, Inverse-Sinkhorn),
per the 2026-09-05 deviation log entry resolving the spec's ambiguity on
which metric feeds D_l' = 1/(D_l+eps).

Does not train anything. Safe to run anywhere; takes seconds.

Run from repo root:
    python -m src.salora.compute_allocations
"""

from pathlib import Path

import pandas as pd

from src.salora.allocator import (
    allocate_ranks,
    allocate_inverse_drift,
    allocate_random,
    compute_trainable_params,
)

N = 950
TARGETS = ["wanli", "anli_r1", "anli_r2", "anli_r3"]

# Fixed, logged allocation seeds for the Random baseline -- one per target,
# distinct from any training seed (Spec Section 8, method E).
RANDOM_ALLOCATION_SEEDS = {
    "wanli": 101,
    "anli_r1": 102,
    "anli_r2": 103,
    "anli_r3": 104,
}

OUT_DIR = Path("results/salora/allocations")


def load_drift(target: str) -> tuple[dict, dict]:
    df = pd.read_csv(f"results/drift_analysis/n{N}/mnli_vs_{target}.csv")
    df = df[df["Layer"] != 0]  # exclude embedding (Spec Section 2)
    mmd = dict(zip(df["Layer"], df["MMD"]))
    sinkhorn = dict(zip(df["Layer"], df["Sinkhorn"]))
    return mmd, sinkhorn


def main():
    print("=" * 70)
    print(f"Computing all Step 4 allocations from N={N} drift profiles")
    print("=" * 70)

    all_params = {}

    for target in TARGETS:
        mmd, sinkhorn = load_drift(target)

        allocations = {
            "salora_mmd": allocate_ranks(mmd, metric="mmd", sample_size=N),
            "salora_sinkhorn": allocate_ranks(sinkhorn, metric="sinkhorn", sample_size=N),
            "inverse_mmd": allocate_inverse_drift(mmd, metric="inverse_mmd", sample_size=N),
            "inverse_sinkhorn": allocate_inverse_drift(sinkhorn, metric="inverse_sinkhorn", sample_size=N),
            "random": allocate_random(
                allocation_seed=RANDOM_ALLOCATION_SEEDS[target], metric="random"
            ),
        }

        print(f"\n--- {target} ---")
        for method, alloc in allocations.items():
            path = OUT_DIR / f"{target}_{method}_n{N}.csv"
            alloc.save_csv(path)
            params = alloc.trainable_params()
            all_params[(target, method)] = params
            ranks = [alloc.layer_ranks[l] for l in sorted(alloc.layer_ranks)]
            print(f"  {method:18s} params={params:7d} ranks={ranks}")

    print("\n" + "=" * 70)
    print("Parameter-count equality check across ALL target/method combinations")
    print("=" * 70)
    unique_params = set(all_params.values())
    if len(unique_params) == 1:
        print(f"PASS: every allocation yields exactly {unique_params.pop()} LoRA-only trainable params.")
    else:
        print(f"FAIL: multiple distinct trainable-param counts found: {sorted(unique_params)}")
        for k, v in all_params.items():
            print(f"  {k}: {v}")

    print(f"\nSaved allocations to {OUT_DIR}/")


if __name__ == "__main__":
    main()
