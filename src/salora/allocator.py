"""
SALoRA Rank Allocator

Implements the allocation algorithm specified in
docs/design/SALoRA_Allocation_Spec.md. Do not modify the constraint
or rounding logic here without updating that spec's Deviation Log
first -- this module must never silently diverge from the frozen
design.

Author:
Bindhu Pagadala
"""

from __future__ import annotations

import csv
import json
import math
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence

# RoBERTa-base: hidden dim 768, LoRA on query+value, bias="none".
# Parameters per targeted module at rank r: r * (d_in + d_out).
HIDDEN_DIM = 768
PARAMS_PER_MODULE_PER_RANK = 2 * HIDDEN_DIM  # r * (d_in + d_out) = r * 1536
MODULES_PER_LAYER = 2  # query + value, always coupled to one r_l (Spec Section 2)


@dataclass
class Allocation:
    """
    Result of allocating a fixed rank budget across layers.

    layer_ranks: mapping of eligible layer index (1..12) -> allocated
        integer rank. Layer 0 (embedding) is never a key here -- it is
        not allocation-eligible (Spec Section 2).
    """

    metric: str
    total_budget: int
    r_min: int
    r_max: int
    drift: Dict[int, float]
    normalized_weight: Dict[int, float]
    raw_rank: Dict[int, float]
    layer_ranks: Dict[int, int]
    allocation_seed: Optional[int] = None  # only meaningful for "random"
    sample_size: Optional[int] = None
    metadata: Dict[str, object] = field(default_factory=dict)

    def total_rank(self) -> int:
        return sum(self.layer_ranks.values())

    def trainable_params(self) -> int:
        return compute_trainable_params(self.layer_ranks)

    def to_rows(self) -> List[dict]:
        layers = sorted(self.layer_ranks.keys())
        return [
            {
                "layer": l,
                "drift": self.drift.get(l, ""),
                "normalized_weight": self.normalized_weight.get(l, ""),
                "raw_rank": self.raw_rank.get(l, ""),
                "allocated_rank": self.layer_ranks[l],
            }
            for l in layers
        ]

    def save_csv(self, path: str | Path) -> None:
        """
        Serialize per Spec Section 12: one row per eligible layer, plus
        a JSON sidecar carrying budget/constraint/provenance metadata
        that doesn't fit naturally into a per-row CSV column.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        rows = self.to_rows()
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=["layer", "drift", "normalized_weight", "raw_rank", "allocated_rank"],
            )
            writer.writeheader()
            writer.writerows(rows)

        sidecar = path.with_suffix(".meta.json")
        with open(sidecar, "w") as f:
            json.dump(
                {
                    "metric": self.metric,
                    "total_budget": self.total_budget,
                    "r_min": self.r_min,
                    "r_max": self.r_max,
                    "allocation_seed": self.allocation_seed,
                    "sample_size": self.sample_size,
                    "trainable_params": self.trainable_params(),
                    **self.metadata,
                },
                f,
                indent=2,
            )

    @staticmethod
    def load_csv(path: str | Path) -> "Allocation":
        """Round-trip counterpart to save_csv (Spec Section 11, test 9)."""
        path = Path(path)
        sidecar = path.with_suffix(".meta.json")
        meta = json.loads(sidecar.read_text()) if sidecar.exists() else {}

        drift, weight, raw, ranks = {}, {}, {}, {}
        with open(path, newline="") as f:
            for row in csv.DictReader(f):
                l = int(row["layer"])
                drift[l] = float(row["drift"]) if row["drift"] != "" else 0.0
                weight[l] = float(row["normalized_weight"]) if row["normalized_weight"] != "" else 0.0
                raw[l] = float(row["raw_rank"]) if row["raw_rank"] != "" else 0.0
                ranks[l] = int(row["allocated_rank"])

        return Allocation(
            metric=meta.get("metric", "unknown"),
            total_budget=meta.get("total_budget", sum(ranks.values())),
            r_min=meta.get("r_min", min(ranks.values())),
            r_max=meta.get("r_max", max(ranks.values())),
            drift=drift,
            normalized_weight=weight,
            raw_rank=raw,
            layer_ranks=ranks,
            allocation_seed=meta.get("allocation_seed"),
            sample_size=meta.get("sample_size"),
        )


def compute_trainable_params(layer_ranks: Dict[int, int]) -> int:
    """
    Spec Section 9: total trainable LoRA parameters implied by a
    layer -> rank mapping, assuming query+value both at rank r_l,
    RoBERTa-base hidden_dim=768, bias="none" (baseline configuration).
    """
    return sum(
        MODULES_PER_LAYER * PARAMS_PER_MODULE_PER_RANK * r
        for r in layer_ranks.values()
    )


def _largest_remainder_apportion(
    raw_ranks: Dict[int, float],
    total_budget: int,
    r_min: int,
    r_max: int,
) -> Dict[int, int]:
    """
    Spec Section 6: largest-remainder (Hamilton) apportionment with
    min/max clamping and iterative rebalancing so Sigma r_l == total_budget
    exactly, deterministically, regardless of input.
    """
    layers = sorted(raw_ranks.keys())

    if total_budget < r_min * len(layers):
        raise ValueError(
            f"total_budget={total_budget} cannot satisfy r_min={r_min} "
            f"across {len(layers)} layers (needs >= {r_min * len(layers)})."
        )
    if total_budget > r_max * len(layers):
        raise ValueError(
            f"total_budget={total_budget} cannot satisfy r_max={r_max} "
            f"across {len(layers)} layers (allows <= {r_max * len(layers)})."
        )

    base = {l: math.floor(raw_ranks[l]) for l in layers}
    remainder = {l: raw_ranks[l] - base[l] for l in layers}

    leftover = total_budget - sum(base.values())

    # Step 6.6: distribute +1 to the `leftover` layers with the largest
    # remainder; ties broken by ascending layer index for determinism.
    order = sorted(layers, key=lambda l: (-remainder[l], l))
    ranks = dict(base)
    for l in order[: max(leftover, 0)]:
        ranks[l] += 1
    # If leftover is negative (shouldn't happen given floor semantics,
    # but guarded for safety/determinism under adversarial input),
    # remove from the smallest-remainder layers instead.
    if leftover < 0:
        for l in sorted(layers, key=lambda l: (remainder[l], l))[: -leftover]:
            ranks[l] -= 1

    # Step 6.7: clamp into [r_min, r_max].
    for l in layers:
        ranks[l] = min(max(ranks[l], r_min), r_max)

    # Step 6.8: iteratively rebalance if clamping moved the sum away
    # from total_budget. Deterministic tie-breaking by layer index.
    def current_sum() -> int:
        return sum(ranks.values())

    guard = 0
    while current_sum() != total_budget:
        guard += 1
        if guard > 10_000:
            raise RuntimeError(
                "Apportionment failed to converge -- this indicates a "
                "logic error, not a data issue (budget/min/max feasibility "
                "is already checked above)."
            )

        deficit = total_budget - current_sum()

        if deficit > 0:
            candidates = [l for l in layers if ranks[l] < r_max]
            if not candidates:
                raise RuntimeError("No layer below r_max to absorb remaining budget.")
            candidates.sort(key=lambda l: (-(raw_ranks[l] - ranks[l]), l))
            ranks[candidates[0]] += 1
        else:
            candidates = [l for l in layers if ranks[l] > r_min]
            if not candidates:
                raise RuntimeError("No layer above r_min to give up budget.")
            candidates.sort(key=lambda l: ((raw_ranks[l] - ranks[l]), l))
            ranks[candidates[0]] -= 1

    return ranks


def allocate_ranks(
    drift: Dict[int, float],
    total_budget: int = 96,
    r_min: int = 1,
    r_max: int = 16,
    metric: str = "unknown",
    sample_size: Optional[int] = None,
) -> Allocation:
    """
    Spec Sections 4-6: proportional drift-based allocation.

    drift: {layer_index (1..12): drift_value}. Layer 0 (embedding) must
        NOT be included -- it is not allocation-eligible.
    """
    layers = sorted(drift.keys())
    if layers != list(range(1, 13)):
        raise ValueError(
            f"Expected drift for exactly layers 1..12 (embedding excluded), "
            f"got layers={layers}."
        )

    total_drift = sum(drift.values())

    if total_drift <= 0:
        # Spec Section 11, test 4: zero/near-zero drift falls back to a
        # uniform allocation rather than raising or dividing by zero.
        weight = {l: 1.0 / len(layers) for l in layers}
    else:
        weight = {l: drift[l] / total_drift for l in layers}

    raw = {l: total_budget * weight[l] for l in layers}
    ranks = _largest_remainder_apportion(raw, total_budget, r_min, r_max)

    return Allocation(
        metric=metric,
        total_budget=total_budget,
        r_min=r_min,
        r_max=r_max,
        drift=dict(drift),
        normalized_weight=weight,
        raw_rank=raw,
        layer_ranks=ranks,
        sample_size=sample_size,
    )


def allocate_uniform(total_budget: int = 96, num_layers: int = 12) -> Allocation:
    """Baseline B. Not drift-based; every layer gets total_budget/num_layers."""
    if total_budget % num_layers != 0:
        raise ValueError(
            f"Uniform allocation requires total_budget divisible by num_layers "
            f"(got {total_budget}/{num_layers})."
        )
    r = total_budget // num_layers
    layers = list(range(1, num_layers + 1))
    ranks = {l: r for l in layers}
    return Allocation(
        metric="uniform",
        total_budget=total_budget,
        r_min=r,
        r_max=r,
        drift={l: 0.0 for l in layers},
        normalized_weight={l: 1.0 / num_layers for l in layers},
        raw_rank={l: float(r) for l in layers},
        layer_ranks=ranks,
    )


def allocate_inverse_drift(
    drift: Dict[int, float],
    total_budget: int = 96,
    r_min: int = 1,
    r_max: int = 16,
    metric: str = "inverse",
    sample_size: Optional[int] = None,
    epsilon: float = 1e-8,
) -> Allocation:
    """Baseline F (Spec Section 8): r_l ~ 1/(D_l + eps), same apportionment path."""
    inverse = {l: 1.0 / (d + epsilon) for l, d in drift.items()}
    alloc = allocate_ranks(
        inverse, total_budget=total_budget, r_min=r_min, r_max=r_max,
        metric=metric, sample_size=sample_size,
    )
    # Report the *original* drift for readability/plotting, not the
    # inverted values, since "drift" should mean drift everywhere else.
    alloc.drift = dict(drift)
    return alloc


def allocate_random(
    total_budget: int = 96,
    r_min: int = 1,
    r_max: int = 16,
    num_layers: int = 12,
    allocation_seed: int = 0,
    metric: str = "random",
) -> Allocation:
    """
    Baseline E (Spec Section 8): a reproducible random partition, run
    through the *same* apportionment code path as the drift-based
    methods (via uniform random weights), so a result difference can't
    be attributed to different rounding logic.
    """
    rng = random.Random(allocation_seed)
    layers = list(range(1, num_layers + 1))
    raw_weights = {l: rng.random() for l in layers}
    total = sum(raw_weights.values())
    weight = {l: w / total for l, w in raw_weights.items()}
    raw = {l: total_budget * weight[l] for l in layers}
    ranks = _largest_remainder_apportion(raw, total_budget, r_min, r_max)

    return Allocation(
        metric=metric,
        total_budget=total_budget,
        r_min=r_min,
        r_max=r_max,
        drift={l: 0.0 for l in layers},
        normalized_weight=weight,
        raw_rank=raw,
        layer_ranks=ranks,
        allocation_seed=allocation_seed,
    )


def to_peft_rank_pattern(
    allocation: Allocation,
    module_name_template: str = r"encoder\.layer\.{layer0}\.attention\.self\.(query|value)$",
) -> Dict[str, int]:
    """
    Converts a layer -> rank mapping into the regex-keyed dict PEFT's
    LoraConfig(rank_pattern=...) expects (verified mechanism: Step 5).

    Layer indices here are 1..12 (Spec Section 2 convention); RoBERTa's
    actual module names are 0-indexed (encoder.layer.0 .. encoder.layer.11),
    hence layer0 = layer_index - 1.

    NOTE: the exact module path prefix (e.g. whether it's
    "roberta.encoder.layer.N..." vs "base_model.model.roberta.encoder...")
    depends on how the model is wrapped and MUST be confirmed against a
    real model.named_modules() dump in Step 5 before this is used for
    actual training -- this function only encodes the *pattern shape*
    agreed in the design spec, not a verified-correct module prefix.
    """
    return {
        module_name_template.format(layer0=layer - 1): rank
        for layer, rank in allocation.layer_ranks.items()
    }
