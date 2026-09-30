import json
import os
from pathlib import Path

import mlflow
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

from src.model import SMSClassifier
from src.model_io import load_final_model, save_final_model
from src.sms_dataset import SMSDataset
from src.training import train_one_epoch
from src.vocabulary import build_vocabulary

ROOT = Path(__file__).resolve().parents[1]
TRAIN_PATH = ROOT / "data" / "processed" / "sms_train.csv"
SELECTION_PATH = ROOT / "results" / "selected_model.json"
HISTORY_PATH = ROOT / "results" / "final_training.csv"
MODEL_PATH = ROOT / "models" / "final_model.pt"
EXPERIMENT_NAME = "sms-spam-final-training"


def train_final_model(
    frame: pd.DataFrame,
    selection: dict,
    model_path: Path,
    batch_size: int = 32,
    learning_rate: float = 0.001,
    max_length: int = 128,
    random_seed: int = 42,
    tracking_uri: str | None = None,
) -> pd.DataFrame:
    if list(frame.columns) != ["label", "text"] or frame.empty:
        raise ValueError("Očekivane su neprazne kolone label i text")
    if frame["text"].duplicated().any():
        raise ValueError("Trening skup ne sme sadržati duplikate")
    if not isinstance(selection.get("epochs"), int) or selection["epochs"] < 1:
        raise ValueError("Broj epoha mora biti pozitivan ceo broj")
    if batch_size < 1 or learning_rate <= 0:
        raise ValueError("Batch veličina i stopa učenja moraju biti pozitivni")

    torch.manual_seed(random_seed)
    vocabulary = build_vocabulary(frame["text"])
    dataset = SMSDataset(frame, vocabulary, max_length=max_length)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(random_seed),
    )
    model_params = selection["model_params"]
    model = SMSClassifier(
        vocab_size=len(vocabulary), max_length=max_length, **model_params
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    loss_fn = nn.CrossEntropyLoss()
    parameter_count = sum(parameter.numel() for parameter in model.parameters())

    mlflow.set_tracking_uri(
        tracking_uri or os.environ.get("MLFLOW_TRACKING_URI") or f"sqlite:///{ROOT / 'mlflow.db'}"
    )
    mlflow.set_experiment(EXPERIMENT_NAME)

    history = []
    with mlflow.start_run(run_name=f"{selection['configuration']}-{selection['epochs']}-epochs"):
        mlflow.log_params({
            **model_params,
            "configuration": selection["configuration"],
            "epochs": selection["epochs"],
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "max_length": max_length,
            "random_seed": random_seed,
            "train_examples": len(dataset),
            "vocabulary_size": len(vocabulary),
            "parameter_count": parameter_count,
        })
        for epoch in range(1, selection["epochs"] + 1):
            loss = train_one_epoch(model, loader, loss_fn, optimizer)
            history.append({"epoch": epoch, "train_loss": loss})
            mlflow.log_metric("train_loss", loss, step=epoch)
            print(f"Epoha {epoch}/{selection['epochs']}: trening gubitak {loss:.4f}", flush=True)

        training = {
            "configuration": selection["configuration"],
            "epochs": selection["epochs"],
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "random_seed": random_seed,
            "train_examples": len(dataset),
            "parameter_count": parameter_count,
        }
        save_final_model(model_path, model, vocabulary, model_params, max_length, training)
        loaded_model, _ = load_final_model(model_path)
        sample = dataset[0]
        model.eval()
        with torch.inference_mode():
            inputs = sample["input_ids"].unsqueeze(0)
            mask = sample["padding_mask"].unsqueeze(0)
            if not torch.allclose(model(inputs, mask), loaded_model(inputs, mask)):
                raise RuntimeError("Ponovo učitani model ne daje isti izlaz")

        result = pd.DataFrame(history)
        mlflow.log_text(result.to_csv(index=False), "training_history.csv")
    return result


def main() -> None:
    train = pd.read_csv(TRAIN_PATH)
    selection = json.loads(SELECTION_PATH.read_text())
    history = train_final_model(train, selection, MODEL_PATH)
    HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)
    history.to_csv(HISTORY_PATH, index=False)
    print(f"Model sačuvan u {MODEL_PATH}")
    print(f"Istorija treniranja sačuvana u {HISTORY_PATH}")


if __name__ == "__main__":
    main()
