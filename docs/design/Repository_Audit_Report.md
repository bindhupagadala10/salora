# SALoRA — Repository Audit Report

Scope of this pass (agreed with Bee, 2026-09-05): inspect the `hons_journal` repository against the two handoff/audit PDFs, cross-check documentation vs. code vs. results, and run lightweight CPU checks on existing CSV/parquet outputs. **No model training or GPU experiments were run** — none of the claims below required training, and the sandbox has no GPU. Where a claim can only be settled by running code, it is marked `NOT VERIFIED — requires execution`.

---

## 1. Executive Verdict

The project is **further along on infrastructure than the handoff PDF implies, but the core SALoRA mechanism (rank allocation) does not exist in code at all** — `src/salora/`, `src/baselines/`, `src/models/`, `results/salora/`, and `experiments/` are all empty directories. Everything currently in the repo is the *setup* for SALoRA (source model, uniform-LoRA baseline, drift measurement), not SALoRA itself.

The single most important new finding — **not mentioned anywhere in either PDF** — is that the project's own `results/drift_analysis/*/metric_correlation.csv` files already show MMD and Sinkhorn layer-wise drift rankings are **negatively correlated** for every target domain, and **significantly** anti-correlated (Spearman ρ ≈ −0.61 to −0.66, p<0.05) for ANLI R1 and R2, stable across all four sample sizes (250/500/750/950). The handoff's framing ("if they disagree, that's still informative") undersells this — the two metrics don't just fail to agree, they point in close to opposite directions on two of four target domains. This has to be addressed head-on in the paper, and it changes what "Baseline B: SALoRA" even means, since the handoff assumed MMD and Sinkhorn would give a similar allocation.

Also new: the handoff explicitly warned not to claim "Sinkhorn monotonically increases through the network" without checking the full profile. I checked it — **it is not monotonic**. Sinkhorn drift is high at the embedding layer (~0.45), dips to a minimum around layer 2 (~0.27), then rises steadily to a maximum at layer 12 (~0.97). MMD shows a different, roughly opposite shape (high at embedding, low in the middle, mild uptick at layer 12 but nowhere near its embedding-layer value). This is good ammunition for RQ1/RQ2 but the "increases monotonically" framing must be dropped.

---

## 2. What Definitely Exists and Works

Verified directly from code + data, not just docs:

- **Source model training** (`src/training/train_source.py`): RoBERTa-base, MNLI, epochs=3, lr=2e-5, batch=16, weight_decay=0.01, warmup_ratio=0.10, AdamW (via `TrainingArguments` defaults) — **matches the handoff exactly**.
- **Source/zero-shot evaluation** (`src/evaluation/evaluate_source.py`): evaluates the frozen source model on MNLI matched/mismatched, WANLI, ANLI R1–R3. Numbers in `results/baselines/source_baseline.csv` (87.65 / 87.59 / 60.88 / 31.30 / 30.10 / 28.58) match the handoff's Section 7 table exactly, and match what the model would plausibly produce (a non-random, monotonically-degrading-with-difficulty pattern), which is indirect evidence the evaluation pipeline is wired correctly.
- **Uniform LoRA baseline** (`src/training/train_lora.py`): r=8, alpha=16, dropout=0.1, target_modules=`["query","value"]`, lr=2e-4, epochs=3, batch=16 — matches the handoff. `results/baselines/lora_baseline.csv` (72.98/44.40/40.90/42.67) matches Section 8 exactly.
- **MMD** (`src/analysis/drift_metrics.py::mmd_rbf`): Gaussian RBF kernel, median-heuristic bandwidth excluding zero (diagonal) distances, biased estimator (`Kxx.mean()+Kyy.mean()-2*Kxy.mean()`). Matches the handoff's description.
- **Sinkhorn** (`drift_metrics.py::sinkhorn_distance`): L2-normalizes representations, Euclidean cost matrix via `ot.dist`, `ot.bregman.sinkhorn_stabilized`, `reg=1.0` hard-coded default, uniform marginals via `ot.unif`. Matches the handoff.
- **Drift profiles are stable across sample size.** I compared `n250/500/750/950` for `mnli_vs_wanli.csv`: both MMD and Sinkhorn layer-wise shapes are nearly identical at every N (values differ by <15% relative, same qualitative shape). This genuinely supports RQ5 ("is the drift profile stable w.r.t. sample size?") — this part of the project is in good shape.
- **CKA has already been removed from the production drift pipeline.** `src/analysis/compute_drift.py` (the script that actually generates the `results/drift_analysis/*.csv` files used everywhere else) only imports and calls `mmd_rbf` and `sinkhorn_distance` — it never touches `linear_cka`. Git history confirms this was a deliberate change (`26a129a changing cka with sinkhorn`). CKA now exists only in `drift_metrics.py::linear_cka` (the function) and `src/analysis/validate_cka.py` (a standalone face-validity check, not a production metric). **The handoff's framing that CKA-vs-unpaired-domains is still an open, unresolved decision is out of date — it's already been resolved in code, just not in the docs.**

---

## 3. Documentation Is Stale and Contradicts the Handoff in Several Places

The handoff PDF was generated from a mix of old and new project state and inherited several stale claims:

| Doc | Says | Reality (from code/git) |
|---|---|---|
| `docs/design/Project_Timeline.md` | "Current Phase: Research Design" (Phase 0 of 6) | Repo is actually well into Phase 3 (representation/drift analysis); Phase 4 (SALoRA) hasn't started. |
| `docs/design/Decision_Log.md` (Decision 005) | "Benchmark: Not yet locked... awaiting systematic evaluation" | MNLI/WANLI/ANLI have been used in every experiment since. `docs/design/Benchmark_Scoring.md` and `Dataset_Evaluation_Table.md` are empty templates — the formal scoring rubric described in the docs was **never actually filled in**. The benchmark was chosen ad hoc (via the `roberta-large-mnli` feasibility study in `docs/benchmark_selection.md`), not through the documented scoring process. |
| `README.md` | Documents only EXP-001–EXP-003A: CKA + MMD, no Sinkhorn | Git log shows Sinkhorn was added and CKA dropped several commits ago (`654c31c`, `530b5a7`, `26a129a`). README was never updated after that. |
| `docs/design/NEXT.md` | LoRA target_modules = `["query","key","value"]`, lr=2e-4 | Actual `train_lora.py` uses `["query","value"]` (two modules, matches handoff) — NEXT.md was an earlier draft plan, superseded but left in place, creating a real (if easily-resolved) contradiction if someone reads NEXT.md instead of the code. |
| Handoff §43, "Code Repository Currently Observed" | Cites `experiments/experiments.csv` | That path doesn't exist (`experiments/` is empty). The real ledger is `results/experiments.csv`, and it contains **exactly one row** (the WANLI LoRA run) — the source-model training run and the three ANLI LoRA runs were never logged via `log_experiment()`, even though `train_source.py` and `train_lora.py` both call it. This means the experiment ledger the handoff describes as a completed artifact is in fact ~80% missing. |
| `docs/benchmark_selection.md` | Zero-shot numbers: MNLI 90.60 / WANLI 61.38 / ANLI R1 45.60 / R2 27.10 / R3 26.83 | These are **not** the same experiment as `source_baseline.csv` — they come from a *different, earlier* feasibility study using the public `FacebookAI/roberta-large-mnli` checkpoint (RoBERTa-**large**, not the project's own fine-tuned RoBERTa-**base**). The doc says as much ("Not used as a training baseline in later experiments"), but the handoff PDF doesn't distinguish these two studies anywhere, which invites confusing the two in the paper. Worth stating explicitly in the methods section which numbers come from which model. |
| `docs/design/Paper_Contributions.md` | Contribution 5, status "Locked": *"Extensive comparison against state-of-the-art PEFT methods"* | No AdaLoRA (or any other PEFT baseline) code exists anywhere in the repo. This contribution is not just unproven, it's unstarted. Recommend un-locking/downgrading this claim until at least one additional PEFT baseline is implemented, or dropping it. |
| `docs/benchmark_selection.md` | "MNLI genres are retained to evaluate controlled stylistic distribution shifts... Genre-level evaluation helps distinguish..." | `src/analysis/genre_evaluation.py` is a 9-line stub: `print("To be implemented after source model training.")`. The source model has been trained; this was never implemented. |
| `docs/literature/Related_Work.md`, `docs/design/Implementation_Checklist.md` | Files exist in the tree | Both are **completely empty** (0 lines). No literature review has been written yet, contradicting the handoff's implicit framing that positioning against AdaLoRA etc. is a documented/settled matter. |

---

## 4. Critical Finding: The Rank-Allocation Algorithm Does Not Exist

This is the handoff's own biggest flagged risk, and the audit confirms it in the strongest possible terms: it isn't partially done, it isn't stubbed, it's **empty directories**:

```
src/salora/          (empty)
src/baselines/       (empty)
src/models/          (empty)
results/salora/      (empty)
experiments/         (empty)
```

There is no code anywhere in the repository that:
- reads a drift profile and produces per-layer ranks,
- rounds fractional ranks to integers,
- enforces a fixed total budget,
- applies min/max rank constraints,
- trains a model with non-uniform LoRA ranks per layer.

`train_lora.py` hard-codes `r=8` for every layer with no per-layer rank parameter at all — the PEFT `LoraConfig` API it uses (`peft==0.17.0`) would need a different config mechanism (e.g. `rank_pattern`/`alpha_pattern` per-module, which does exist in recent `peft` versions) to support non-uniform ranks; this hasn't been explored yet. **Everything the handoff describes in Sections 24–29 (rank allocation formula, integer rounding, min/max rank, inverse-drift/random baselines) is still 100% design, not implementation.** Any claim that SALoRA "has been evaluated" or that adapters like `adapters/lora_wanli.zip` represent SALoRA runs is wrong — those four adapters are all uniform-rank (r=8) baselines, confirmed by `train_lora.py`'s hard-coded config and by `experiments.csv`'s single logged row showing `Rank=8`.

---

## 5. Methodological Issue Found in Code: Mean Pooling Does Not Exclude Padding

`src/analysis/extract_representations.py`, line 160:

```python
pooled = hidden.mean(dim=1)
```

This averages over the full sequence dimension **without masking out padded positions**. Batches are built with `DataCollatorWithPadding` (dynamic padding to the longest sequence *in that batch*), so:
- the amount of pad-token contamination in a given example's pooled vector depends on which other examples happen to share its batch (batches are not shuffled — `shuffle=False` — so this is at least deterministic per run, but it's still not "the average of non-padding token representations" as both PDFs claim/require),
- shorter examples in a batch with long neighbors get proportionally more pad-token contribution than the same example would get in a different batch.

This is a real, verifiable-from-code discrepancy between the documented method ("mean pooling = average of non-padding token representations," handoff §11–12) and the actual implementation. It doesn't necessarily invalidate the drift results (pad-token hidden states are generally small/attention-masked-out during self-attention, so the effect is likely a minor noise source rather than a dominant confound), but it should either be (a) fixed with `attention_mask`-weighted mean pooling before any final numbers go in the paper, or (b) explicitly disclosed and justified as-is. Given how cheap the fix is, I'd fix it and re-extract — it's the kind of thing a reviewer would catch immediately if they read the code.

**Not checked / requires execution:** whether re-extracting with masked mean pooling meaningfully changes the drift profiles above. This needs a GPU run of `extract_representations.py` + `compute_drift.py`, which is out of scope for this pass.

---

## 6. Other Findings

- **`data.gitignore` excludes `models/`, `adapters/`, `checkpoints/`, and `*.zip`.** The actual trained weights (`source_roberta.zip`, 460MB; the four LoRA adapter zips) exist only as local files, never committed to git, with no external storage/versioning strategy documented anywhere. If this repo is the sole copy, that 460MB source checkpoint is a single point of failure for reproducibility. Worth deciding now (e.g. push to a Hugging Face Hub private repo, or a release artifact) rather than after the paper is drafted.
- **Drift-analysis sample for WANLI is drawn from the WANLI *train* split** (`src/analysis/sample_validation_sets.py`), while WANLI's zero-shot/LoRA *evaluation* uses the WANLI *test* split (`evaluate_source.py`, `evaluate_lora.py`, `train_lora.py`). Not necessarily wrong — the drift signal is meant to characterize the domain, not the eval set — but this should be an explicit, stated design choice in the methods section, not something a reviewer discovers themselves.
- **`requirements.txt` was saved as UTF-16** (with a BOM) rather than plain UTF-8. Harmless for `pip install -r`, but worth normalizing — some tools/CI choke on this.
- **Duplicate/dead code**: `train_source.py` calls `train.map(tokenize, batched=True)` twice in a row (lines 92 and 94) — harmless (idempotent) but signals the file hasn't been cleaned up since drafting.
- **`src/evaluation/evaluate.py` is a 0-byte empty file** sitting alongside `evaluate_source.py` and `evaluate_lora.py` — looks like an abandoned stub; safe to delete once someone confirms nothing imports it (nothing does, per a repo-wide grep).
- The **`docs/design/Benchmark_Scoring.md`** and **`Dataset_Evaluation_Table.md`** rubric templates were never filled in, so the "why these datasets" justification the paper will need (per `Benchmark_Requirements.md`'s own R7/R8: "different semantic domains," "natural justification for domain adaptation") doesn't formally exist yet — WANLI/ANLI are stylistic/adversarial *NLI* shifts, not the cross-domain shifts (legal/biomedical/financial) the requirements doc itself lists as examples. This is worth a deliberate decision (documented, not silently glossed over) about whether "domain adaptation" is the right term versus "distribution shift" / "robustness adaptation" — the second PDF (§21) raises exactly this concern and it's still open.
- **`label` vs `labels` column naming**: none of `train_source.py`, `train_lora.py`, `evaluate_source.py`, `evaluate_lora.py` rename the dataset's `label` column to `labels` before handing it to a HF `Trainer`. This looked like a potential bug at first, but `DataCollatorWithPadding` in `transformers==4.55.0` auto-renames `label`→`labels` internally, and the resulting accuracy numbers (87.65% on MNLI, sensible degradation on WANLI/ANLI) are far too coherent to be an artifact of a label-alignment bug (a broken label mapping would produce ~33% chance accuracy on 3-way MNLI). I could not install `transformers` in this sandbox to confirm line-by-line (no network access to the full package in time), so this is **verified with high confidence from behavioral evidence, not from re-running the exact library code** — flagging the residual uncertainty rather than asserting it outright.

---

## 7. Truth Table

| Item | Handoff says | Docs say | Code says | Results say | Assessment | Action |
|---|---|---|---|---|---|---|
| Source model | RoBERTa-base, MNLI, 3ep, lr2e-5 | Matches (`NEXT.md`, outline) | Matches exactly | `source_baseline.csv` matches | Verified | Keep |
| Zero-shot baseline | 87.65/87.59/60.88/31.30/30.10/28.58 | Matches in `NEXT.md` | evaluate_source.py implements it | CSV matches exactly | Verified, reproducible in principle (needs checkpoint to actually re-run) | Keep |
| Uniform LoRA baseline | r=8, α=16, query+value | `NEXT.md` says query+key+value (stale) | Code says query+value | CSV matches handoff | Code is correct; NEXT.md is the outlier | Fix/annotate NEXT.md as superseded |
| CKA | "still needs resolving, don't retain by default" | README still lists CKA as current | Already removed from production pipeline (`compute_drift.py`); kept only as a face-validity check (`validate_cka.py`) | No CKA column in any drift CSV | Already resolved in code, not in docs | Update README + handoff framing |
| MMD | RBF, median heuristic | matches | matches | profiles stable across N | Verified | Keep |
| Sinkhorn | L2-norm, Euclidean cost, reg=1.0 | matches | matches | profiles stable across N; **not monotonic** (dip then rise) | Partially contradicts handoff's monotonicity caveat — now resolved with data | Use exact profile shape in paper, drop "monotonic" language |
| MMD vs Sinkhorn agreement | "may agree or disagree, complementary" | not discussed quantitatively anywhere | `correlation_analysis.py` computes it | **Significant negative** correlation for ANLI R1/R2 (Spearman ≈ −0.65, p<0.05), near-zero/unstable-sign for WANLI/ANLI R3, stable across all 4 sample sizes | New, important, unaddressed finding | Must be a named result/figure in the paper, not glossed over |
| Mean pooling | "average of non-padding tokens" | matches | **Does not mask padding** (`hidden.mean(dim=1)`) | Unknown effect size on current numbers | Discrepancy, likely minor but unverified | Fix pooling, re-extract, or explicitly disclose as a limitation |
| Rank allocation / SALoRA | Core formula + rounding + budget described in detail | Described in design docs as still "not yet scientifically finalized" | **Not implemented at all** (empty `src/salora/`) | No SALoRA results exist anywhere | Confirmed: 0% implemented | This is the actual next engineering task |
| Baselines (inverse-drift, random, AdaLoRA) | Recommended | Not mentioned as started | Not implemented | N/A | Confirmed: 0% implemented | Prioritize inverse-drift + random (cheap); AdaLoRA is a bigger lift |
| Experiment ledger | Implied to be a working artifact at `experiments/experiments.csv` | — | `log_experiment()` exists and is called from both training scripts | Real file is `results/experiments.csv` with **1 of ~5 expected rows** | Partial/broken in practice | Backfill missing rows or accept the ledger as informal and rebuild going forward |
| Model checkpoints | Implied available for reproduction | — | `.gitignore` excludes them; only local `.zip` files exist | `source_roberta.zip` = 460MB, untracked | Reproducibility risk | Decide on and document a checkpoint storage/versioning plan |

---

## 8. Ranked Scientific/Process Concerns

**Critical**
1. SALoRA's actual mechanism (rank allocation) is unimplemented — the project cannot currently test its own central hypothesis.
2. MMD and Sinkhorn drift rankings are significantly *anti*-correlated on 2 of 4 target domains. If SALoRA-MMD and SALoRA-Sinkhorn are both implemented as planned, they will likely produce very different, possibly opposite, rank allocations for ANLI — the paper needs a clear story for this, not a footnote.

**High**
3. Mean pooling includes padding tokens, contradicting the documented method; effect size unverified.
4. Experiment ledger is ~80% incomplete relative to what's actually been run; reproducibility claims in a paper can't rely on it as-is.
5. Trained checkpoints (460MB+ source model, 4 LoRA adapters) are git-ignored with no backup/versioning strategy.

**Medium**
6. Formal benchmark-selection scoring (docs/design/Benchmark_Scoring.md, Dataset_Evaluation_Table.md) was never filled in; the "why MNLI→WANLI/ANLI counts as domain adaptation" justification the project's own requirements doc demands doesn't formally exist.
7. Several docs (Project_Timeline, Decision_Log, README, NEXT.md) are stale relative to git history and will mislead anyone (including a future AI session) who reads docs instead of code.
8. WANLI drift-analysis sample (train split) vs. evaluation sample (test split) mismatch is undocumented.
9. "Extensive comparison against SOTA PEFT methods" is marked "Locked" as a contribution with zero supporting code (no AdaLoRA, no other PEFT baseline).

**Low**
10. Dead/empty files (`evaluate.py`, `genre_evaluation.py` stub, `Implementation_Checklist.md`, `Related_Work.md`) should be either filled in or removed so the tree reflects real state.
11. `requirements.txt` saved as UTF-16; minor tooling annoyance.
12. Minor duplicate code in `train_source.py` (double `.map()` call).

---

## 9. What Was NOT Done in This Pass (by agreement)

- No model was trained, fine-tuned, or loaded onto a GPU.
- The exact `label`→`labels` Trainer behavior was inferred from documented `transformers` behavior + circumstantial evidence (sensible accuracy numbers), not from executing `transformers==4.55.0` directly in this sandbox (no network access to install it in time).
- Whether fixing the padding/mean-pooling bug changes the drift profiles or the eventual SALoRA results — this requires re-running `extract_representations.py` and `compute_drift.py`, which needs the actual model checkpoint and (ideally) a GPU.
- Literature verification of SALoRA's novelty vs. AdaLoRA/other drift-guided PEFT work (§24/§37 of the audit PDF) — this needs a web-search pass, not just repo inspection, and wasn't part of the agreed scope for this turn.

---

## 10. Suggested Next Steps (not yet executed — for your decision)

1. Decide how to handle the MMD-vs-Sinkhorn anti-correlation before building anything else — it changes what "the SALoRA allocation" even means (pick one primary metric? report both as separate methods? treat the disagreement as a finding in itself?).
2. Fix mean pooling to mask padding, re-extract representations, and check whether the drift profiles / RQ1–RQ2 conclusions change.
3. Implement the rank-allocation module (`src/salora/`) — this is genuinely 0% done and is the actual bottleneck to any SALoRA result.
4. Backfill or accept-and-rebuild the experiment ledger before it grows further.
5. Decide on a checkpoint storage strategy (Hugging Face Hub, cloud bucket, or similar) before continuing.
6. Update the stale docs (README, NEXT.md, Project_Timeline, Decision_Log) to reflect actual current state, or archive them clearly as historical.

I have not made any of these changes — flagging them for your call before touching anything.
