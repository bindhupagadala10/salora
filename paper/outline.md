# SALoRA Paper Outline

## 1. Introduction

(To be written after benchmark selection is finalized.)

---

## 2. Related Work

(To be populated during literature review.)

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