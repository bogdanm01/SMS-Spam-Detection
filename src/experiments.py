import os
from pathlib import Path
from time import perf_counter

import mlflow
import pandas as pd

from src.cross_validation import run_cross_validation

ROOT = Path(__file__).resolve().parents[1]
TRAIN_PATH = ROOT / "data" / "processed" / "sms_train.csv"
RESULTS_PATH = ROOT / "results" / "config_comparison.csv"
EXPERIMENT_NAME = "sms-spam-transformer"

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
    tracking_uri: str | None = None,
) -> pd.DataFrame:
    if configurations is None:
        configurations = CONFIGURATIONS
    if not configurations:
        raise ValueError("Potrebna je bar jedna konfiguracija")

    mlflow.set_tracking_uri(
        tracking_uri or os.environ.get("MLFLOW_TRACKING_URI") or f"sqlite:///{ROOT / 'mlflow.db'}"
    )
    mlflow.set_experiment(EXPERIMENT_NAME)

    results = []
    for name, model_params in configurations.items():
        start = perf_counter()
        if progress:
            print(f"Konfiguracija: {name}", flush=True)
        with mlflow.start_run(run_name=name):
            mlflow.log_params({
                **model_params,
                "n_splits": n_splits,
                "epochs": epochs,
                "batch_size": batch_size,
                "learning_rate": learning_rate,
                "max_length": max_length,
                "random_seed": random_seed,
            })

            def log_epoch(row: dict[str, int | float]) -> None:
                fold = int(row["fold"])
                mlflow.log_metrics(
                    {
                        f"fold_{fold}_{metric}": float(row[metric])
                        for metric in ("train_loss", "validation_loss", "accuracy", "precision", "recall", "f1")
                    },
                    step=int(row["epoch"]),
                )

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
                on_epoch_end=log_epoch,
            )
            history.insert(0, "configuration", name)
            for key, value in model_params.items():
                history[key] = value
            final = history.loc[history["epoch"] == epochs]
            mlflow.log_metrics({
                "mean_f1": float(final["f1"].mean()),
                "std_f1": float(final["f1"].std()),
                "mean_accuracy": float(final["accuracy"].mean()),
                "mean_precision": float(final["precision"].mean()),
                "mean_recall": float(final["recall"].mean()),
                "mean_parameter_count": float(final["parameter_count"].mean()),
                "elapsed_seconds": perf_counter() - start,
            })
            mlflow.log_text(history.to_csv(index=False), "fold_history.csv")
        results.append(history)
        if progress:
            print(f"{name} završen za {perf_counter() - start:.1f} s", flush=True)

    return pd.concat(results, ignore_index=True)


def main() -> None:
    train = pd.read_csv(TRAIN_PATH)
    results = compare_configurations(train, progress=True)
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(RESULTS_PATH, index=False)
    print(f"Rezultati sačuvani u {RESULTS_PATH}")


if __name__ == "__main__":
    main()
