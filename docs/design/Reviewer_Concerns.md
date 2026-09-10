# Reviewer Concerns

*(Updated 2026-09-10 — full reasoning for each in
docs/design/Research_Design_Document.md, "Reviewer Concerns Crosswalk".)*

## Concern 1

Why should representation drift imply adaptation demand?

Status

Open, but with a grounded partial answer: ANLI's adversarial-targeting
construction vs. WANLI's pattern-based construction explains why drift and
difficulty dissociate (ANLI R3 is the hardest target but shows the lowest
Sinkhorn drift of all four). Genuinely resolved only once RQ4/H3 are tested.

---

## Concern 2

Is SALoRA sufficiently different from AdaLoRA?

Status

Answered. No surveyed adaptive-rank method (AdaLoRA + 5 successors) uses a
pre-computed, ante-hoc domain-shift signal — all are in-training or static-
weight-derived. Caveat: literature scan, not a systematic review.

---

## Concern 3

Is the benchmark appropriate?

Status

Still open. Benchmark_Requirements.md R7 ("different semantic domains") is
not satisfied by WANLI/ANLI as originally written — they're stylistic/
adversarial NLI shifts, not cross-domain shifts. Needs an explicit reframing
decision ("distribution shift" vs. "domain adaptation"), not an assertion
that it's fine as-is.

---

## Concern 4

Does the pre-analysis cost outweigh the training savings?

Status

Open, not yet quantified. Drift analysis is a one-time frozen-model forward
pass (no backprop) per target; needs actual wall-clock numbers reported
alongside training time, not just an assumption that it's cheap.

---

## Concern 5

Are improvements statistically significant?

Status

Blocked on the full 3-seed matrix and an explicit significance-testing
method (paired vs. independent comparison, which test) -- not yet decided.