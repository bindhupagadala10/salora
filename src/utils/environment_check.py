import importlib

packages = [
    "torch",
    "transformers",
    "datasets",
    "peft",
    "accelerate",
    "evaluate",
    "sentencepiece",
    "sklearn",
    "pandas",
    "numpy",
    "matplotlib",
    "scipy",
    "tqdm",
    "huggingface_hub",
]

print("=" * 60)
print("SALoRA Environment Verification")
print("=" * 60)

for pkg in packages:
    try:
        module = importlib.import_module(pkg)
        version = getattr(module, "__version__", "Unknown")
        print(f"✓ {pkg:<20} {version}")
    except Exception as e:
        print(f"✗ {pkg:<20} FAILED")
        print(e)

print("=" * 60)