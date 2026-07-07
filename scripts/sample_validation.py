from datasets import load_dataset
import pandas as pd
from pathlib import Path

SEED = 42
N_PER_CLASS = 250

OUTPUT_DIR = Path("outputs/samples")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def save_balanced_sample(dataset_name,
                         split,
                         text_column,
                         label_column,
                         output_name):

    print(f"\nLoading {dataset_name} ({split})")

    ds = load_dataset(dataset_name, split=split)

    df = ds.to_pandas()

    df0 = df[df[label_column] == 0].sample(
        N_PER_CLASS,
        random_state=SEED
    )

    df1 = df[df[label_column] == 1].sample(
        N_PER_CLASS,
        random_state=SEED
    )

    balanced = pd.concat([df0, df1])

    balanced = balanced.sample(
        frac=1,
        random_state=SEED
    ).reset_index(drop=True)

    balanced.to_csv(
        OUTPUT_DIR / f"{output_name}.csv",
        index=False
    )

    print(f"Saved {len(balanced)} samples.")


save_balanced_sample(
    dataset_name="stanfordnlp/imdb",
    split="test",
    text_column="text",
    label_column="label",
    output_name="imdb_validation"
)

save_balanced_sample(
    dataset_name="sentiment140",
    split="test",
    text_column="text",
    label_column="sentiment",
    output_name="twitter_validation"
)