from pathlib import Path

import torch

from src.model import SMSClassifier
from src.sms_dataset import LABEL_TO_ID
from src.vocabulary import PAD_ID, PAD_TOKEN, UNK_ID, UNK_TOKEN

CHECKPOINT_VERSION = 1


def save_final_model(
    path: Path,
    model: SMSClassifier,
    vocabulary: dict[str, int],
    model_params: dict[str, int | float],
    max_length: int,
    training: dict[str, int | float | str],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "format_version": CHECKPOINT_VERSION,
            "model_state_dict": model.state_dict(),
            "vocabulary": vocabulary,
            "model_params": model_params,
            "max_length": max_length,
            "label_to_id": LABEL_TO_ID,
            "training": training,
        },
        path,
    )


def load_final_model(path: Path) -> tuple[SMSClassifier, dict]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if checkpoint.get("format_version") != CHECKPOINT_VERSION:
        raise ValueError("Nepoznat format sačuvanog modela")
    vocabulary = checkpoint["vocabulary"]
    if vocabulary.get(PAD_TOKEN) != PAD_ID or vocabulary.get(UNK_TOKEN) != UNK_ID:
        raise ValueError("Sačuvani rečnik nema očekivane posebne tokene")
    if set(vocabulary.values()) != set(range(len(vocabulary))):
        raise ValueError("ID-jevi sačuvanog rečnika nisu uzastopni")
    if checkpoint["label_to_id"] != LABEL_TO_ID:
        raise ValueError("Mapiranje klasa u checkpoint-u nije očekivano")

    model = SMSClassifier(
        vocab_size=len(vocabulary),
        max_length=checkpoint["max_length"],
        **checkpoint["model_params"],
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, checkpoint
