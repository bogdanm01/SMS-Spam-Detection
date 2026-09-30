import torch
from torch import nn
from torch.utils.data import DataLoader


def evaluate_model(
    model: nn.Module,
    loader: DataLoader,
    loss_fn: nn.Module,
) -> dict[str, float | int]:
    model.eval()
    total_loss = 0.0
    total_examples = 0
    tn = fp = fn = tp = 0

    with torch.inference_mode():
        for batch in loader:
            logits = model(batch["input_ids"], batch["padding_mask"])
            labels = batch["label"]
            predictions = logits.argmax(dim=1)
            batch_size = labels.shape[0]
            total_loss += loss_fn(logits, labels).item() * batch_size
            total_examples += batch_size
            tn += ((labels == 0) & (predictions == 0)).sum().item()
            fp += ((labels == 0) & (predictions == 1)).sum().item()
            fn += ((labels == 1) & (predictions == 0)).sum().item()
            tp += ((labels == 1) & (predictions == 1)).sum().item()

    if total_examples == 0:
        raise ValueError("Validacioni skup ne sme biti prazan")

    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0
    return {
        "loss": total_loss / total_examples,
        "accuracy": (tp + tn) / total_examples,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
    }
