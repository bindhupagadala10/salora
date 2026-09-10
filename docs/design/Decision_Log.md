# Decision Log

## Decision 001

Project Goal

Publish one Q3 journal paper.

Status

Locked.

---

## Decision 002

Method

Ante-hoc representation-guided adaptation planning.

Status

Locked.

---

## Decision 003

Implementation

LoRA is the first implementation of the framework.

Status

Locked.

---

## Decision 004

Research Philosophy

Empirical investigation rather than claiming theoretical proof.

Status

Locked.

---

## Decision 005

Benchmark

De facto locked as MNLI (source) -> WANLI, ANLI R1/R2/R3 (targets) -- these
have been used in every experiment for months; this entry was stale.

Reason

Not from the systematic scoring process Benchmark_Scoring.md /
Dataset_Evaluation_Table.md were meant to produce (both were never filled
in) -- chosen ad hoc. Still open: Benchmark_Requirements.md R7 ("different
semantic domains") is not satisfied by this benchmark as literally written.
See docs/design/Research_Design_Document.md, "Benchmark Justification", for
the unresolved reframing decision ("domain adaptation" vs. "distribution
shift adaptation") this creates.

Status

Updated 2026-09-10 -- not re-locked pending the reframing decision above.