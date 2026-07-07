### EXP-001 — Balanced Validation Sampling

Generated stratified analysis subsets from:

- MNLI validation_matched
- WANLI train
- ANLI dev

Multiple sample sizes are supported to perform a stability study before selecting the final analysis set.

### EXP-002 – Representation Extraction

Extracts mean-pooled hidden representations from all 13 hidden states of the frozen MNLI source model using `output_hidden_states=True`. Representations are stored as stacked tensors `(13, N, 768)` together with labels and metadata for downstream drift analysis.
Hidden representations are extracted from the frozen MNLI source model using Hugging Face's `output_hidden_states=True` interface. Mean pooling is applied over the sequence dimension, producing a tensor of shape `(13, N, 768)` containing the embedding layer and all 12 transformer encoder layers. These representations are used for downstream CKA and MMD-based domain drift analysis.
Representations are extracted from the frozen MNLI source model using `output_hidden_states=True`. Mean pooling is applied across the sequence dimension to obtain a 768-dimensional vector for the embedding layer and each of the 12 transformer encoder layers. The resulting tensor has shape `(13, N, 768)` and is stored together with labels and metadata for downstream drift analysis.

### EXP-003A – Drift Metrics

Implemented Linear CKA and Gaussian RBF MMD with the median heuristic. These metrics quantify structural similarity and distributional divergence between layer representations extracted from the frozen source model.
Layer-wise representation drift is computed between the frozen MNLI source model and target domains using Linear CKA and Gaussian RBF MMD. CKA is additionally converted to `CKA_Drift = 1 - CKA` so that all drift measures share a common interpretation where larger values indicate greater domain shift.