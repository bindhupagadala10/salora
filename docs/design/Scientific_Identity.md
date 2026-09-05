# Scientific Identity

## One Sentence Contribution

SALoRA investigates whether layer-wise representation drift measured before fine-tuning can serve as a practical signal for planning parameter-efficient adaptation.

---

## We ARE

- A parameter-efficient adaptation planning framework.
- An empirical study of representation geometry.
- A lightweight engineering method.
- A reproducible PEFT framework.

---

## We are NOT

- A new transformer architecture.
- A replacement for LoRA.
- A new optimization algorithm.
- A new representation learning method.
- A new domain adaptation theory.

---

## Reviewer Elevator Pitch

Current PEFT methods allocate adaptation capacity using fixed rules or during-training importance scores.

SALoRA instead investigates whether domain-specific representation geometry can be analyzed before fine-tuning to produce a better adaptation plan under the same parameter budget.

---

## Success Criterion

If SALoRA consistently performs competitively with strong PEFT baselines while requiring no online allocation mechanism, the research hypothesis is supported.

Absolute SOTA is NOT required.