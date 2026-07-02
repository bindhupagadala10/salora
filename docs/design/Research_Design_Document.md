# SALoRA: Shift-Aware Layer-wise Parameter Allocation for Parameter-Efficient Domain Adaptation

---

# Version

v1.0

---

# Objective

Develop a parameter-efficient domain adaptation framework that performs adaptation planning before fine-tuning by analyzing representation geometry between source and target domains.

Instead of allocating LoRA parameters uniformly or dynamically during training, SALoRA investigates whether layer-wise representation drift can be used as a practical signal for allocating adaptation capacity under a fixed parameter budget.

The goal is to improve adaptation efficiency while maintaining competitive downstream performance.

---

# Target Publication

Primary Target:
Q3 Journal

Stretch Goal:
Q2 Journal

Publication is prioritized over novelty.

The project emphasizes:

- rigorous experimentation
- reproducibility
- careful evaluation
- practical contribution

rather than attempting an overly ambitious theoretical breakthrough.

---

# Research Problem

Current parameter-efficient fine-tuning methods generally determine adaptation capacity in one of two ways:

1. Fixed allocation
   - Same LoRA rank across all layers.

2. Dynamic allocation
   - Rank changes during training based on gradients or parameter importance.

Neither strategy explicitly considers how different domains affect internal transformer representations before training begins.

This motivates the following question:

Can representation drift measured between source and target domains before fine-tuning be used as a practical signal for planning parameter-efficient adaptation?

---

# Research Questions

RQ1

Can layer-wise representation drift be measured reliably between source and target domains?

RQ2

Do different domains exhibit distinct layer-wise drift profiles?

RQ3

Can representation drift serve as a useful proxy for adaptation planning?

RQ4

Can representation-guided allocation outperform fixed LoRA allocation under an equivalent parameter budget?

---

# Hypotheses

H1

Different target domains produce different representation drift patterns across transformer layers.

H2

Representation drift contains useful information for adaptation planning.

H3

Layer-wise allocation based on representation drift performs competitively against fixed-rank LoRA while using the same overall adaptation budget.

---

# Scope

This project does NOT attempt to prove that representation drift is theoretically equivalent to adaptation capacity.

Instead, the project empirically investigates whether representation drift is a useful engineering signal for parameter allocation.

Any observed relationship will be validated experimentally rather than assumed mathematically.