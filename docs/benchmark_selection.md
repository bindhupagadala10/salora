# Benchmark Selection

## Objective

Before developing SALoRA, we evaluated whether a strong pretrained NLI model already generalized across candidate target benchmarks. This experiment serves as a feasibility study to justify the need for parameter-efficient domain adaptation.

## Validation Model

Model:
FacebookAI/roberta-large-mnli

Reason:
- Official MNLI checkpoint from Meta.
- Strong zero-shot NLI baseline.
- Used only to validate the existence of cross-domain degradation.
- Not used as a training baseline in later experiments.

## Datasets

| Dataset | Domain |
|----------|--------|
| MNLI | General |
| WANLI | Natural distribution shift |
| ANLI R1 | Adversarial |
| ANLI R2 | Hard adversarial |
| ANLI R3 | Hardest adversarial |

## Zero-Shot Results

| Dataset | Accuracy | Macro F1 |
|----------|----------|----------|
| MNLI | 90.60 | 90.51 |
| WANLI | 61.38 | 60.26 |
| ANLI R1 | 45.60 | 45.32 |
| ANLI R2 | 27.10 | 27.09 |
| ANLI R3 | 26.83 | 26.87 |

## Observation

Performance degrades consistently as evaluation moves farther from the MNLI training distribution. This validates the existence of a significant domain adaptation problem and motivates the development of SALoRA.

## Dataset Engineering

All benchmark datasets were transformed into a unified schema:

- premise
- hypothesis
- label
- dataset
- split

Labels were normalized to:

0 → Entailment

1 → Neutral

2 → Contradiction

This preprocessing guarantees identical downstream training and evaluation across benchmarks.

## Data Engineering

All benchmark datasets were converted into a unified schema.

Validation Results

- Empty premise: 0
- Empty hypothesis: 0
- Label normalization successful
- Duplicate rate <0.05% across all datasets
- Original benchmark distributions preserved

No preprocessing errors were detected.