# SALoRA Rank-Allocation Specification (Frozen Design)

Status: **DESIGN FROZEN, PENDING STEP 5 EMPIRICAL CONFIRMATION** of `R_total` and the PEFT mechanism.
This document is the single source of truth for the allocation algorithm. `src/salora/` must implement exactly what's specified here; if implementation forces a change, this file gets updated first and the change is logged in the "Deviation Log" at the bottom — the code must never silently diverge from this spec.

---

## 1. Scope

This spec covers **only** the mapping from a layer-wise drift profile to a set of per-layer integer LoRA ranks under a fixed total budget. It does not cover training hyperparameters (those match the existing uniform-LoRA baseline exactly, per `train_lora.py`) or the drift metrics themselves (frozen separately, pending Step 2's masked-pooling recompute).

## 2. What Gets a Rank

- Allocation-eligible layers: the 12 transformer encoder layers (indices 1–12 in the drift profile). **Layer 0 (embedding) is excluded** — there is no LoRA module at the embedding layer in any method (baseline or SALoRA), so its drift value is descriptive only (feeds RQ1 discussion) and never enters the allocation math.
- Within an eligible layer, **query and value share one rank value `r_l`** — they are not allocated independently. This matches how the existing uniform baseline defines its budget (`12 layers × r=8`, not `24 modules × r=8`) and keeps the method's claim honestly "layer-wise" (matching what the drift signal actually measures) rather than implying an unevidenced query-vs-value distinction.
- No other modules (attention output, FFN, embeddings, classifier head) receive LoRA in any method, matching the current baseline's `target_modules=["query","value"]`.

## 3. Total Rank Budget

`R_total = 96` (= 12 layers × uniform baseline's r=8), **provisional** until Step 5 confirms it via `model.print_trainable_parameters()` on the real checkpoint. If Step 5 finds the actual trainable-parameter count for the uniform baseline doesn't match the hand-computed value below, this number gets corrected before any SALoRA training runs, not after.

Hand-computed expectation (RoBERTa-base, hidden=768, no bias since baseline uses `bias="none"`):
- Per targeted module: `r × (d_in + d_out) = r × 1536` trainable parameters.
- Per layer (query + value): `2 × r_l × 1536 = 3072 × r_l`.
- Total for uniform (`r_l=8` ∀l): `12 × 3072 × 8 = 294,912` trainable LoRA parameters.
- This must be identical across every method in Section 8 (Baseline A–F) for the comparison to be fair — Section 9 defines the automated check.

## 4. Allocation Formula (SALoRA-MMD, SALoRA-Sinkhorn)

For a given metric (MMD or Sinkhorn — computed and applied **completely separately**, see Section 7):

```
D_l          = drift value at layer l, l = 1..12   (embedding excluded)
w_l          = D_l / Σ_{j=1..12} D_j
raw_r_l      = R_total × w_l
```

`raw_r_l` is a real number; Sections 5–6 define how it becomes a valid integer allocation.

## 5. Constraints

- **Minimum rank: `r_min = 1`.** Rank 0 is not allowed anywhere. Rationale: allowing 0 changes *which* layers get adapted at all across methods, not just *how much* — that conflates "presence of capacity" with "amount of capacity" and muddies the causal claim in Section 12 of the engineering plan (does more rank at high-drift layers help). It also sidesteps a PEFT edge case (rank-0 LoRA modules are not a well-defined/tested configuration).
- **Maximum rank: `r_max = 16`** (= 2× the uniform baseline's r=8). Directly addresses the degenerate case where one dominant layer's drift would otherwise absorb nearly the entire budget (e.g. drift = [.001, .001, .002, .005, .99, ...]) and starve every other layer to near-zero. This cap is chosen **before** seeing any SALoRA results, specifically so it can't look like post-hoc tuning later. If a future revision of this spec loosens or tightens it, that revision must be dated and justified in the Deviation Log, not silently changed.
- `Σ_{l=1}^{12} r_l = R_total` exactly, always, for every method.

## 6. Integer Rounding: Largest-Remainder (Hamilton) Apportionment

Chosen because it's a standard, named, deterministic method (used for real-world seat apportionment) rather than an ad hoc "round and hope it sums right" rule — easy to defend to a reviewer in one sentence, and trivially reproducible.

```
1. raw_r_l = R_total * w_l                         for l = 1..12
2. base_l  = floor(raw_r_l)
3. remainder_l = raw_r_l - base_l
4. leftover = R_total - Σ base_l
5. Sort layers by remainder_l descending (ties broken by layer index ascending,
   for determinism).
6. Give +1 to each of the top `leftover` layers by remainder.
7. Clamp every r_l into [r_min, r_max].
8. If clamping changed Σ r_l away from R_total:
   - If Σ r_l < R_total: distribute the shortfall one unit at a time to the
     layer(s) with the largest (raw_r_l - r_l) among layers not yet at r_max,
     breaking ties by layer index ascending.
   - If Σ r_l > R_total: remove one unit at a time from the layer(s) with the
     smallest (raw_r_l - r_l) among layers not yet at r_min, breaking ties by
     layer index ascending.
   - Repeat until Σ r_l == R_total exactly.
9. Return {r_l for l in 1..12}, guaranteed: Σ r_l = R_total, r_min ≤ r_l ≤ r_max,
   fully deterministic given (D, R_total, r_min, r_max).
```

This is deliberately specified as pseudocode here, before any Python exists, so Step 4's implementation is a direct translation, not a design decision made while coding.

## 7. MMD and Sinkhorn Are Never Combined

`SALoRA-MMD` and `SALoRA-Sinkhorn` are two fully independent methods, each running Sections 4–6 with its own `D_l` vector. There is no blended/averaged drift score anywhere in this project. Given the audit's finding that MMD and Sinkhorn are significantly *anti*-correlated on ANLI R1/R2, forcing them into one score would actively hide the most interesting empirical result available — the research question this creates ("does the choice of drift metric change whether drift-guided allocation helps?") is treated as a first-class result, not an inconvenience to average away.

## 8. All Methods in the Experimental Matrix, Defined Precisely

| Method | `D_l` source | Same Σr_l constraint? |
|---|---|---|
| A — Zero-shot | n/a (no LoRA) | n/a |
| B — Uniform LoRA | n/a | `r_l = 8` ∀l (already R_total=96 by construction) |
| C — SALoRA-MMD | MMD drift profile | yes, via Sections 4–6 |
| D — SALoRA-Sinkhorn | Sinkhorn drift profile | yes, via Sections 4–6 |
| E — Random allocation | a uniform-random partition of R_total into 12 parts, each in [r_min, r_max], generated with a **fixed, logged seed** (not the training seed — a separate `allocation_seed` so the random *allocation itself* is reproducible independent of which training seed later uses it) | yes |
| F — Inverse-drift | `D_l' = 1 / (D_l + ε)`, `ε = 1e-8`, then Sections 4–6 unchanged | yes |
| G — AdaLoRA (conditional) | n/a (dynamic, not ante-hoc) | matched via AdaLoRA's own target-budget parameter, verified empirically to equal 294,912 trainable params at convergence — only reported if this equivalence actually holds |

Random allocation (E) generation method: draw 12 uniform integers ≥ `r_min`, then apply the largest-remainder correction from Section 6 exactly as for the drift-based methods, so E and F go through the identical constraint-handling code path as C/D — the *only* thing that differs between methods is the input weight vector `w_l`, never the apportionment logic. This is important for a fair comparison: if E or F used a different rounding path than C/D, a difference in results could be an artifact of the apportionment method rather than the allocation signal.

## 9. Parameter-Equality Verification (Automated, Not Just Asserted)

Even though `Σ r_l = R_total` guarantees equal parameter count by construction (given identical target-module shapes across methods, true here), the allocator must still expose a `compute_trainable_params(ranks: dict[int, int]) -> int` function computing `Σ_l 2 × r_l × 1536`, and every method's run must log this number alongside its results. A test (Section 11, #7) asserts all methods produce the identical value. This is defense-in-depth, not reliance on the arithmetic argument alone.

## 10. Alpha Policy

`lora_alpha = 16`, constant across every layer and every method, including all SALoRA variants. **Not** scaled with rank (i.e., `alpha/r` is *not* held constant). Rationale: LoRA's update magnitude scales as `alpha/r`; if alpha scaled proportionally with `r`, every layer would receive the same effective step size regardless of rank, which would partially cancel out the very capacity difference SALoRA is testing. Holding alpha fixed keeps rank as the *only* manipulated variable between uniform and SALoRA — necessary for the causal claim in Section 12 of the engineering plan to mean anything. "Scale alpha with rank" is a documented, deliberately-rejected alternative and a candidate ablation, not the default.

## 11. Allocator Unit Tests (to implement in Step 4)

1. `Σ r_l == R_total` for every valid input.
2. No `r_l` violates `[r_min, r_max]`.
3. Deterministic: same `(D, R_total, r_min, r_max)` → byte-identical output across repeated calls.
4. Zero/near-zero drift handling: if all `D_l ≈ 0` (or exactly 0, guarded against div-by-zero in `w_l`), falls back to uniform allocation rather than raising or producing NaN.
5. Tied drift values: two or more layers with identical `D_l` → deterministic tie-break (layer index ascending), not dependent on dict/set ordering.
6. Floating-point rounding edge cases: `raw_r_l` exactly at a `.5` boundary; `leftover` computed as 0; `leftover` computed as 12 (all layers tied).
7. Parameter-count equivalence: `compute_trainable_params()` returns the same value for uniform, SALoRA-MMD, SALoRA-Sinkhorn, random, and inverse-drift allocations given the same `R_total`.
8. `r_max` clamp actually triggers and redistributes correctly for the handoff's own pathological example (`D = [.001,.001,.002,.005,.99, ...]`, padded to 12 layers).
9. Serialization round-trip: allocation written to CSV and read back reproduces the identical rank dict.

## 12. Serialization Format

`results/salora/allocations/{target}_{metric}_n{N}.csv`, columns:

```
layer, drift, normalized_weight, raw_rank, allocated_rank
```

One row per eligible layer (1–12). A header comment line (or a paired `.json` sidecar) records `R_total`, `r_min`, `r_max`, the metric name, the sample size `N` the drift was computed from, and (for Random) the `allocation_seed` used.

## 13. Layer-Capacity Sensitivity Experiment (Causal Probe) — Design

Goal: distinguish "drift correlates with adaptation demand" from "the drift metric produces an interesting-looking allocation."

- **Layer selection:** pick 4 layers per target metric from the *actual* recomputed (Step 2) profile: the eligible layer with max drift, the eligible layer with min drift, plus 2 layers near the middle of the ranking — not chosen post hoc after seeing SALoRA results.
- **Perturbation:** for a chosen layer `l*`, construct two perturbed allocations relative to the **uniform** baseline (not relative to SALoRA's own allocation, to keep the manipulation isolated and interpretable):
  - **Boost:** `r_{l*} = 16` (the max allowed), with the extra `+8` taken evenly from the other 11 layers (each loses `8/11`, apportioned via the same Section 6 rounding, so it's `Σ = 96` exactly, no scapegoat layer, no different apportionment logic).
  - **Cut:** `r_{l*} = 4`, with the freed `+4` distributed evenly to the other 11 layers by the same method.
- **Scope:** run initially only on ANLI R1 and ANLI R2 (where the MMD/Sinkhorn anti-correlation is statistically significant, making "does the metric's specific allocation matter" most informative), 1 seed for the initial exploratory pass. Only expand to WANLI/ANLI R3 or add seeds if the initial pass shows a signal worth confirming — this keeps the probe's compute cost bounded given it's the least essential part of the matrix.
- **Read-out:** for each perturbed layer, compare accuracy/macro-F1 delta vs. uniform baseline. A layer whose drift-based rank claim is causally meaningful should show boost > uniform and cut < uniform in a consistent direction; a layer where drift is a coincidental correlate should show no consistent pattern.

## 14. Deviation Log

*(empty — record any change to Sections 3–10 here, dated, with the reason, once implementation or Step 5's real numbers force a revision.)*
