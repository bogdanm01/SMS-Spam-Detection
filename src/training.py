import torch
from torch import nn
from torch.optim import Optimizer


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
