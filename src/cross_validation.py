from random import Random

import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

from src.evaluation import evaluate_model
from src.model import SMSClassifier
from src.sms_dataset import SMSDataset
from src.training import train_one_epoch
from src.vocabulary import build_vocabulary


def stratified_folds(
    frame: pd.DataFrame,
    n_splits: int = 5,
    random_seed: int = 42,
) -> list[tuple[pd.DataFrame, pd.DataFrame]]:
    if list(frame.columns) != ["label", "text"]:
        raise ValueError("Očekivane kolone su label i text")
    if frame.empty or frame.isna().any().any():
        raise ValueError("Skup ne sme biti prazan niti sadržati nedostajuće vrednosti")
    if not frame["text"].map(lambda text: isinstance(text, str) and bool(text.strip())).all():
        raise ValueError("Svaka poruka mora imati tekst")
    if set(frame["label"]) != {"ham", "spam"}:
        raise ValueError("Očekivane su obe klase: ham i spam")
    if frame["text"].duplicated().any():
        raise ValueError("Ukloni duplikate pre podele")
    if not isinstance(n_splits, int) or n_splits < 2:
        raise ValueError("Broj fold-ova mora biti najmanje 2")
    if frame["label"].value_counts().min() < n_splits:
        raise ValueError("Svaka klasa mora imati bar po jednu poruku u svakom foldu")

    data = frame.reset_index(drop=True)
    buckets = [[] for _ in range(n_splits)]
    random = Random(random_seed)
    for label in ("ham", "spam"):
        positions = data.index[data["label"] == label].tolist()
        random.shuffle(positions)
        for offset, position in enumerate(positions):
            buckets[offset % n_splits].append(position)

    return [
        (
            data.drop(index=positions).reset_index(drop=True),
            data.iloc[sorted(positions)].reset_index(drop=True),
        )
        for positions in buckets
    ]


def run_cross_validation(
    frame: pd.DataFrame,
    n_splits: int = 5,
    epochs: int = 3,
    batch_size: int = 32,
    learning_rate: float = 0.001,
    max_length: int = 128,
    random_seed: int = 42,
) -> pd.DataFrame:
    if epochs < 1 or batch_size < 1 or learning_rate <= 0:
        raise ValueError("Broj epoha, veličina batch-a i stopa učenja moraju biti pozitivni")

    results = []
    folds = stratified_folds(frame, n_splits=n_splits, random_seed=random_seed)
    for fold_number, (fit_frame, validation_frame) in enumerate(folds, start=1):
        vocabulary = build_vocabulary(fit_frame["text"])
        fit_loader = DataLoader(
            SMSDataset(fit_frame, vocabulary, max_length=max_length),
            batch_size=batch_size,
            shuffle=True,
            generator=torch.Generator().manual_seed(random_seed + fold_number),
        )
        validation_loader = DataLoader(
            SMSDataset(validation_frame, vocabulary, max_length=max_length),
            batch_size=batch_size,
            shuffle=False,
        )
        torch.manual_seed(random_seed + fold_number)
        model = SMSClassifier(vocab_size=len(vocabulary), max_length=max_length)
        loss_fn = nn.CrossEntropyLoss()
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

        for epoch in range(1, epochs + 1):
            train_loss = train_one_epoch(model, fit_loader, loss_fn, optimizer)
            metrics = evaluate_model(model, validation_loader, loss_fn)
            results.append({
                "fold": fold_number,
                "epoch": epoch,
                "train_size": len(fit_frame),
                "validation_size": len(validation_frame),
                "vocabulary_size": len(vocabulary),
                "train_loss": train_loss,
                "validation_loss": metrics["loss"],
                "accuracy": metrics["accuracy"],
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "f1": metrics["f1"],
                "fp": metrics["fp"],
                "fn": metrics["fn"],
            })

    return pd.DataFrame(results)
