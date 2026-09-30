import json
from pathlib import Path

import pandas as pd
from torch import nn
from torch.utils.data import DataLoader

from src.evaluation import evaluate_model
from src.model_io import load_final_model
from src.sms_dataset import SMSDataset

ROOT = Path(__file__).resolve().parents[1]
TRAIN_PATH = ROOT / "data" / "processed" / "sms_train.csv"
TEST_PATH = ROOT / "data" / "processed" / "sms_test.csv"
MODEL_PATH = ROOT / "models" / "final_model.pt"
RESULT_PATH = ROOT / "results" / "test_metrics.json"


def evaluate_final_model(
    train: pd.DataFrame,
    test: pd.DataFrame,
    model_path: Path,
    batch_size: int = 32,
) -> dict:
    if batch_size < 1:
        raise ValueError("Batch veličina mora biti pozitivna")
    if test["text"].duplicated().any():
        raise ValueError("Test skup ne sme sadržati duplikate")
    if set(train["text"]) & set(test["text"]):
        raise ValueError("Trening i test skup imaju zajedničke poruke")

    model, checkpoint = load_final_model(model_path)
    if checkpoint["training"]["train_examples"] != len(train):
        raise ValueError("Veličina trening skupa ne odgovara sačuvanom modelu")

    dataset = SMSDataset(test, checkpoint["vocabulary"], checkpoint["max_length"])
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    metrics = evaluate_model(model, loader, nn.CrossEntropyLoss())
    return {
        "configuration": checkpoint["training"]["configuration"],
        "epochs": checkpoint["training"]["epochs"],
        "test_examples": len(dataset),
        "positive_class": "spam",
        "metrics": metrics,
    }


def main() -> None:
    train = pd.read_csv(TRAIN_PATH)
    test = pd.read_csv(TEST_PATH)
    result = evaluate_final_model(train, test, MODEL_PATH)
    RESULT_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULT_PATH.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(f"Test rezultat sačuvan u {RESULT_PATH}")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
