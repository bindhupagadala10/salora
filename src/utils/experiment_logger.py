from pathlib import Path
import pandas as pd

RESULTS = Path("results")
RESULTS.mkdir(exist_ok=True)

CSV = RESULTS / "experiments.csv"


def log_experiment(record: dict):

    if CSV.exists():
        df = pd.read_csv(CSV)
    else:
        df = pd.DataFrame()

    df = pd.concat(
        [df, pd.DataFrame([record])],
        ignore_index=True,
    )

    df.to_csv(CSV, index=False)

    print(f"Experiment logged -> {CSV}")