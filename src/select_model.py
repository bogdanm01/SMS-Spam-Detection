import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
COMPARISON_PATH = ROOT / "results" / "config_comparison.csv"
SELECTION_PATH = ROOT / "results" / "selected_model.json"
MODEL_COLUMNS = ("d_model", "n_heads", "n_layers", "dim_feedforward", "dropout")


def select_configuration(history: pd.DataFrame) -> dict:
    required = {"configuration", "fold", "epoch", "f1", "precision", "recall", "validation_loss", "parameter_count", *MODEL_COLUMNS}
    if not required.issubset(history.columns) or history.empty:
        raise ValueError("Nedostaju podaci za izbor modela")
    if history[list(required)].isna().any().any():
        raise ValueError("Rezultati sadrže nedostajuće vrednosti")
    if history.duplicated(["configuration", "fold", "epoch"]).any():
        raise ValueError("Svaki fold i epoha moraju imati tačno jedan rezultat")

    folds = set(history["fold"])
    epochs = set(history["epoch"])
    if not folds or not epochs or epochs != set(range(1, max(epochs) + 1)):
        raise ValueError("Epohe moraju biti uzastopne i početi od 1")
    for _, group in history.groupby("configuration"):
        if set(group["fold"]) != folds:
            raise ValueError("Konfiguracije nemaju iste fold-ove")
        for _, fold_history in group.groupby("fold"):
            if set(fold_history["epoch"]) != epochs:
                raise ValueError("Nedostaje rezultat neke epohe")
        if any(group[column].nunique() != 1 for column in MODEL_COLUMNS):
            raise ValueError("Parametri konfiguracije nisu dosledni")

    comparison_epoch = max(epochs)
    architecture_scores = (
        history.loc[history["epoch"] == comparison_epoch]
        .groupby("configuration")
        .agg(mean_f1=("f1", "mean"), mean_parameter_count=("parameter_count", "mean"))
        .sort_values(["mean_f1", "mean_parameter_count"], ascending=[False, True])
    )
    name = str(architecture_scores.index[0])
    selected_history = history.loc[history["configuration"] == name]
    epoch_scores = selected_history.groupby("epoch")["f1"].mean().sort_values(ascending=False, kind="stable")
    selected_epoch = int(epoch_scores.index[0])
    selected_rows = selected_history.loc[selected_history["epoch"] == selected_epoch]
    parameters = selected_history.iloc[0]

    return {
        "configuration": name,
        "epochs": selected_epoch,
        "model_params": {
            column: float(parameters[column]) if column == "dropout" else int(parameters[column])
            for column in MODEL_COLUMNS
        },
        "selection": {
            "source": "results/config_comparison.csv",
            "folds": len(folds),
            "architecture_comparison_epoch": int(comparison_epoch),
            "architecture_rule": "Najveći prosečan F1 za spam u unapred izabranoj završnoj epohi",
            "epoch_rule": "Najveći prosečan F1 za spam izabrane arhitekture; u slučaju izjednačenja ranija epoha",
            "mean_f1_at_comparison_epoch": float(architecture_scores.loc[name, "mean_f1"]),
        },
        "validation": {
            "mean_f1": float(selected_rows["f1"].mean()),
            "std_f1": float(selected_rows["f1"].std()),
            "mean_precision": float(selected_rows["precision"].mean()),
            "mean_recall": float(selected_rows["recall"].mean()),
            "mean_loss": float(selected_rows["validation_loss"].mean()),
        },
    }


def main() -> None:
    history = pd.read_csv(COMPARISON_PATH)
    selection = select_configuration(history)
    SELECTION_PATH.write_text(json.dumps(selection, ensure_ascii=False, indent=2) + "\n")
    print(f"Izabrano: {selection['configuration']}, {selection['epochs']} epohe")
    print(f"Odluka sačuvana u {SELECTION_PATH}")


if __name__ == "__main__":
    main()
