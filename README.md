### EXP-001 — Balanced Validation Sampling

Generated stratified analysis subsets from:

- MNLI validation_matched
- WANLI train
- ANLI dev

Multiple sample sizes are supported to perform a stability study before selecting the final analysis set.

### EXP-002 – Representation Extraction

Extracts mean-pooled hidden representations from all 13 hidden states of the frozen MNLI source model using `output_hidden_states=True`. Representations are stored as stacked tensors `(13, N, 768)` together with labels and metadata for downstream drift analysis.