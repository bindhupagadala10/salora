# Related Work

Status: initial pass (2026-09-06), based on a web-search literature scan — not a
systematic review. Scope: parameter-efficient fine-tuning (PEFT) with non-uniform
rank allocation, and the use of domain-shift distance metrics (MMD, Wasserstein/
Sinkhorn) in transfer/domain adaptation. Every claim below is sourced; update this
file as the actual experiments (Step 4) proceed and the framing sharpens.

## 1. LoRA and the uniform-rank problem

LoRA freezes the pre-trained weights and injects trainable low-rank update matrices
into selected linear layers, at a fixed rank `r` applied identically everywhere. This
project's own uniform-LoRA baseline (`r=8`, query+value, RoBERTa-base) follows this
standard recipe. The core limitation motivating this project — and essentially every
paper cited below — is that a single fixed rank across all layers "fails to account
for the heterogeneous importance of different layers in contributing to task
performance" [Layer-Aware Adaptive Rank Allocation, LAARA].

## 2. Adaptive / non-uniform rank allocation for LoRA

This is the space SALoRA must be positioned against directly. All of the following
allocate a non-uniform rank budget across layers, matching SALoRA's high-level goal —
but every one of them derives its allocation signal from **something computed during
fine-tuning on the target task itself** (gradients, Fisher information, SVD of the
update matrices, or properties of the pre-trained weights), not from a pre-computed,
task-agnostic measurement of how much the input distribution has shifted.

- **AdaLoRA** (Zhang et al., ICLR 2023) is the foundational and most-cited method in
  this space. It parameterizes LoRA updates via SVD and adaptively prunes singular
  values of "unimportant" updates based on an importance score computed *during*
  training, avoiding the cost of exact SVD. It is the canonical baseline this project
  already scopes as method G, conditional on a genuinely matched parameter budget
  (`docs/design/SALoRA_Allocation_Spec.md`, Section 8).
  [arXiv:2303.10512](https://arxiv.org/abs/2303.10512),
  [GitHub](https://github.com/QingruZhang/AdaLoRA)

- **La-LoRA** (2025) allocates rank per layer using a "Dynamic Contribution-Driven
  Parameter Budget" and "Truncated Norm Weighted Dynamic Rank Allocation," both
  computed during training.
  [ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S089360802500975X),
  [PubMed](https://pubmed.ncbi.nlm.nih.gov/40953549/)

- **LAARA** (2026) uses lightweight diagonal Fisher-information estimates accumulated
  during training, plus projection-wise normalization and a "vote-to-change" dampening
  mechanism, to reallocate rank without an explicit search. Reports Fisher traces are
  "highly heterogeneous across layers... nearly two orders of magnitude" apart — a
  finding structurally similar to this project's own layer-wise drift heterogeneity,
  but derived from a completely different (training-time, task-loss-coupled) signal.
  [arXiv:2607.19391](https://arxiv.org/abs/2607.19391)

- **ElaLoRA** (2025) elastically prunes and expands ranks during training; the authors
  report that "layers receiving higher rank allocations contribute more significantly
  to model performance," offered as post-hoc justification for adaptive allocation in
  general. [arXiv:2504.00254](https://arxiv.org/html/2504.00254)

- **BaRA** (2026), Bayesian Adaptive Rank Allocation, treats rank allocation as
  Bayesian inference over layer importance.
  [arXiv:2606.29184](https://arxiv.org/pdf/2606.29184)

- **SR-LoRA / "Beyond Low-Rank Tuning"** (Zhang et al., 2025) is the closest match in
  *motivation* to this project: it explicitly targets "large-gap" transfer regimes
  (large domain gaps between source and target) and argues fixed low rank is
  particularly inadequate there. Its signal, however, is the **stable rank of the
  pre-trained weight matrices themselves** — a static property of the frozen model,
  not a measurement of how the target domain's data distribution differs from the
  source domain's. It does not use the target data's representations at all.
  [arXiv:2507.00327](https://arxiv.org/abs/2507.00327)

**Summary of the gap:** every non-uniform rank-allocation method found allocates
based on (a) gradient/Fisher signals collected *during* fine-tuning, (b) SVD/weight
properties of the *update* matrices, or (c) static properties of the *pre-trained
weights*. None allocate rank from a **pre-training, ante-hoc measurement of
distributional drift between frozen source-model representations of the source vs.
target domain's actual data** — which is exactly what SALoRA does (MMD/Sinkhorn on
masked-mean-pooled hidden states, computed once, before any LoRA training begins).
This ante-hoc/in-hoc distinction is already how this project's own spec frames
AdaLoRA's exclusion from a fully matched comparison (Section 8, method G: "dynamic,
not ante-hoc") — the literature scan supports that framing rather than requiring a
change to it.

## 3. Distance metrics for domain shift (MMD, Wasserstein/Sinkhorn)

MMD and the Wasserstein/optimal-transport distance are both well-established as
domain-divergence measures in the domain-adaptation literature, typically used as a
*loss term to minimize* (e.g., domain-invariant representation learning) rather than
as a *diagnostic signal to allocate capacity*:

- Wasserstein-distance-guided representation learning for domain adaptation predates
  this project by nearly a decade and treats the distance as an adversarial training
  objective, not a diagnostic.
  [arXiv:1707.01217](https://arxiv.org/pdf/1707.01217),
  [arXiv:1910.07676](https://arxiv.org/pdf/1910.07676)
- MMD as a domain-adaptation loss/divergence measure is likewise long-established and
  still actively revisited (e.g. "Rethink Maximum Mean Discrepancy for Domain
  Adaptation," 2020). [arXiv:2007.00689](https://arxiv.org/html/2007.00689)

A general web search targeting the specific combination "MMD/Wasserstein-guided LoRA
rank allocation" returned no matching prior work — the search tool's own summary
noted the combination "does not [appear in] information... indexed by search
engines... may be a very recent development." This is not proof of novelty (a search
engine scan is not a systematic review, and a match could exist in a venue/format the
search missed), but it's consistent with the gap identified in Section 2: distributional
drift metrics are well-established for measuring domain shift, but not previously
reported as an *allocation signal* for non-uniform LoRA rank.

## 4. Naming collision — needs a decision before submission

**SaLoRA** ("Safety-Alignment Preserved Low-Rank Adaptation," Li et al., ICLR 2025) is
an existing, published method with a name one character of capitalization away from
this project's "SALoRA." It addresses an unrelated problem (preserving LLM safety
alignment during LoRA fine-tuning via a fixed safety module derived from ~300 harmful
prompts), so there's no conceptual overlap — but the name collision is close enough
that a reviewer or search engine could easily conflate the two.
[arXiv:2501.01765](https://arxiv.org/abs/2501.01765),
[OpenReview](https://openreview.net/forum?id=GOoVzE9nSj)

**This needs an explicit decision, flagged here rather than silently worked around:**
either rename the method (e.g. something that foregrounds "drift" or "shift" rather
than the generic "SA-" prefix) or keep the name and explicitly disambiguate from
SaLoRA in the paper's introduction/related work. Not resolved in this pass.

## 5. Implication for `docs/design/Paper_Contributions.md`

Contribution 5 currently claims (per the Repository Audit Report) "Extensive
comparison against state-of-the-art PEFT methods," marked "Locked," with zero
supporting code. Given how crowded this space actually is (at least six adaptive-rank
LoRA variants found from 2023–2026 alone), "extensive" is not a credible claim for a
single-author project on a laptop with no GPU. The comparison this project can
actually support is: uniform LoRA (done), and AdaLoRA *if and only if* a genuinely
matched parameter budget can be verified (already scoped conditionally in the spec).
The newer 2025–2026 methods (La-LoRA, LAARA, ElaLoRA, BaRA, SR-LoRA) should be cited
as related/contemporaneous work in the related-work section, not experimentally
compared against — recommend downgrading Contribution 5's claim accordingly rather
than leaving it "Locked" as currently written.

---

Sources (all consulted 2026-09-06 via web search; URLs as returned):

- [AdaLoRA (arXiv:2303.10512)](https://arxiv.org/abs/2303.10512)
- [AdaLoRA GitHub](https://github.com/QingruZhang/AdaLoRA)
- [La-LoRA (ScienceDirect)](https://www.sciencedirect.com/science/article/abs/pii/S089360802500975X)
- [La-LoRA (PubMed)](https://pubmed.ncbi.nlm.nih.gov/40953549/)
- [LAARA (arXiv:2607.19391)](https://arxiv.org/abs/2607.19391)
- [ElaLoRA (arXiv:2504.00254)](https://arxiv.org/html/2504.00254)
- [BaRA (arXiv:2606.29184)](https://arxiv.org/pdf/2606.29184)
- [SR-LoRA / "Beyond Low-Rank Tuning" (arXiv:2507.00327)](https://arxiv.org/abs/2507.00327)
- [Wasserstein Distance Guided Representation Learning (arXiv:1707.01217)](https://arxiv.org/pdf/1707.01217)
- [Wasserstein Distance Guided Cross-Domain Learning (arXiv:1910.07676)](https://arxiv.org/pdf/1910.07676)
- [Rethink MMD for Domain Adaptation (arXiv:2007.00689)](https://arxiv.org/html/2007.00689)
- [SaLoRA (arXiv:2501.01765)](https://arxiv.org/abs/2501.01765)
- [SaLoRA (OpenReview)](https://openreview.net/forum?id=GOoVzE9nSj)
