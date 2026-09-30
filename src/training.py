import torch
from torch import nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader


def train_one_batch(
    model: nn.Module,
    batch: dict[str, torch.Tensor],
    loss_fn: nn.Module,
    optimizer: Optimizer,
) -> float:
    model.train()
    optimizer.zero_grad()
    logits = model(batch["input_ids"], batch["padding_mask"])
    loss = loss_fn(logits, batch["label"])
    loss.backward()
    optimizer.step()
    return loss.item()


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    loss_fn: nn.Module,
    optimizer: Optimizer,
) -> float:
    total_loss = 0.0
    total_examples = 0
    for batch in loader:
        batch_size = batch["label"].shape[0]
        batch_loss = train_one_batch(model, batch, loss_fn, optimizer)
        total_loss += batch_loss * batch_size
        total_examples += batch_size
    if total_examples == 0:
        raise ValueError("Trening skup ne sme biti prazan")
    return total_loss / total_examples
