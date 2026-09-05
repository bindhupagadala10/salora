# SALoRA Paper Outline

## 1. Introduction

(To be written after benchmark selection is finalized.)

---

## 2. Related Work

*(Initial pass, 2026-09-06 -- full source discussion and open decisions in
docs/literature/Related_Work.md; full entries in paper/references.bib. Cite
keys below with `\cite{}` if this becomes a LaTeX draft.)*

**Low-rank adaptation.** LoRA \cite{hu2021lora} freezes pre-trained weights and
injects trainable low-rank update matrices at a fixed rank applied uniformly
across layers -- the recipe this project's own uniform-LoRA baseline follows
(`r=8`, query+value, RoBERTa-base).

**Adaptive rank allocation for LoRA.** A uniform rank ignores that layers
differ in how much adaptation they need. AdaLoRA \cite{zhang2023adalora} is
the foundational method here: it parameterizes updates via SVD and prunes
singular values by an importance score computed during training. A wave of
2025--2026 methods extends this idea with different in-training signals --
La-LoRA \cite{gu2025lalora} (a dynamic contribution-driven budget), LAARA
\cite{tripathi2026laara} (diagonal Fisher-information estimates), ElaLoRA
\cite{chang2025elalora} (elastic pruning/expansion during training), BaRA
\cite{duan2026bara} (Bayesian inference over layer importance), and SR-LoRA
\cite{zhang2025srlora} (the stable rank of the frozen pre-trained weights,
explicitly targeting large source-target domain gaps). Every one of these
allocates rank from a signal computed during fine-tuning or from a static
property of the model itself -- **none allocate rank from a pre-computed,
ante-hoc measurement of how much the target domain's own data distribution
has shifted from the source domain's**, which is what this project's
drift-guided allocation (SALoRA-MMD, SALoRA-Sinkhorn) does.

**Distributional distance metrics for domain shift.** MMD and the
Wasserstein/optimal-transport distance are both established tools for
measuring and minimizing domain divergence \cite{shen2017wasserstein,
su2019wasserstein, wang2020rethinkmmd}, but in that literature the distance is
almost always an adversarial *training objective* to be minimized, not a
*diagnostic* used to decide where to spend adaptation capacity. No prior work
found (via a web-search literature scan, not a systematic review) uses MMD or
Sinkhorn/Wasserstein distance as the allocation signal for non-uniform LoRA
rank.

**Open item -- naming collision.** \citet{li2025salora} is an existing,
unrelated, published method ("Safety-Alignment Preserved Low-Rank
Adaptation") whose acronym differs from this project's "SALoRA" only in
capitalization. Needs a decision (rename vs. explicit disambiguation) before
submission -- see docs/literature/Related_Work.md Section 4.

---

## 3. Methodology

- Representation extraction
- Layer-wise drift estimation
- Drift-guided rank allocation
- LoRA adaptation

---

## 4. Experimental Setup

### 4.1 Benchmarks

Automatically acquired benchmark datasets using the Hugging Face Datasets library.

### 4.2 Dataset Quality Assurance

Benchmark datasets are verified through an automated audit pipeline that reports dataset size, class distribution, missing values, duplicate samples, and sequence statistics prior to preprocessing.

---

## 5. Results

(To be completed after experiments.)

---

## 6. Discussion

(To be completed after experiments.)

---

## 7. Conclusion

## Motivation

A strong MNLI model achieves over 90% accuracy on in-domain evaluation but degrades substantially under distribution shift, falling to 61.38% on WANLI and below 30% on the hardest ANLI rounds. These results motivate adaptive parameter-efficient domain adaptation.

## Experimental Setup

### Source Model

Backbone: RoBERTa-base

Training Dataset: MNLI

Epochs: 3

Learning Rate: 2e-5

Batch Size: 16

Maximum Sequence Length: 128

Optimizer: AdamW

Weight Decay: 0.01

Warmup Ratio: 0.1