from pathlib import Path
from time import perf_counter

import pandas as pd

from src.cross_validation import run_cross_validation

ROOT = Path(__file__).resolve().parents[1]
TRAIN_PATH = ROOT / "data" / "processed" / "sms_train.csv"
RESULTS_PATH = ROOT / "results" / "config_comparison.csv"

CONFIGURATIONS = {
    "baseline": {
        "d_model": 64,
        "n_heads": 4,
        "n_layers": 2,
        "dim_feedforward": 128,
        "dropout": 0.1,
    },
    "narrow": {
        "d_model": 32,
        "n_heads": 4,
        "n_layers": 2,
        "dim_feedforward": 64,
        "dropout": 0.1,
    },
    "wide": {
        "d_model": 96,
        "n_heads": 4,
        "n_layers": 2,
        "dim_feedforward": 192,
        "dropout": 0.1,
    },
    "shallow": {
        "d_model": 64,
        "n_heads": 4,
        "n_layers": 1,
        "dim_feedforward": 128,
        "dropout": 0.1,
    },
    "deep": {
        "d_model": 64,
        "n_heads": 4,
        "n_layers": 3,
        "dim_feedforward": 128,
        "dropout": 0.1,
    },
}


def compare_configurations(
    frame: pd.DataFrame,
    configurations: dict[str, dict[str, int | float]] | None = None,
    n_splits: int = 5,
    epochs: int = 3,
    batch_size: int = 32,
    learning_rate: float = 0.001,
    max_length: int = 128,
    random_seed: int = 42,
    progress: bool = False,
) -> pd.DataFrame:
    if configurations is None:
        configurations = CONFIGURATIONS
    if not configurations:
        raise ValueError("Potrebna je bar jedna konfiguracija")

    results = []
    for name, model_params in configurations.items():
        start = perf_counter()
        if progress:
            print(f"Konfiguracija: {name}", flush=True)
        history = run_cross_validation(
            frame,
            n_splits=n_splits,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            max_length=max_length,
            random_seed=random_seed,
            model_params=model_params,
            progress=progress,
        )
        history.insert(0, "configuration", name)
        for key, value in model_params.items():
            history[key] = value
        results.append(history)
        if progress:
            elapsed = perf_counter() - start
            print(f"{name} završen za {elapsed:.1f} s", flush=True)

    return pd.concat(results, ignore_index=True)


def main() -> None:
    train = pd.read_csv(TRAIN_PATH)
    results = compare_configurations(train, progress=True)
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(RESULTS_PATH, index=False)
    print(f"Rezultati sačuvani u {RESULTS_PATH}")


if __name__ == "__main__":
    main()
