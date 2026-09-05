# SALoRA Next Session

## Completed
- Source model trained (RoBERTa-base on MNLI)
- Cross-domain evaluation completed
- Source model archived
- Baseline results frozen

## Source Results

MNLI Matched      87.65
MNLI Mismatched   87.59
WANLI             60.88
ANLI R1           31.30
ANLI R2           30.10
ANLI R3           28.58

## Next Task

Implement Standard LoRA Baseline

Base Model:
models/source_roberta

Target:
WANLI

Configuration
- r = 8
- alpha = 16
- dropout = 0.1
- target_modules = ["query","key","value"]
- epochs = 3
- lr = 2e-4

Deliverables
- train_lora.py
- evaluate_lora.py
- adapter_model.safetensors
- lora_wanli_results.csv