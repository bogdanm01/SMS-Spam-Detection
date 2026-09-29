import pandas as pd
import torch
from torch.utils.data import Dataset

from src.sequences import DEFAULT_MAX_LENGTH, encode_and_pad

LABEL_TO_ID = {"ham": 0, "spam": 1}


class SMSDataset(Dataset):
    def __init__(
        self,
        frame: pd.DataFrame,
        vocabulary: dict[str, int],
        max_length: int = DEFAULT_MAX_LENGTH,
    ) -> None:
        if list(frame.columns) != ["label", "text"]:
            raise ValueError("Očekivane kolone su label i text")
        if frame.empty or frame.isna().any().any():
            raise ValueError("Skup ne sme biti prazan niti sadržati nedostajuće vrednosti")
        if not frame["text"].map(lambda text: isinstance(text, str) and bool(text.strip())).all():
            raise ValueError("Svaka poruka mora imati tekst")
        if not frame["label"].isin(LABEL_TO_ID).all():
            raise ValueError("Nepoznata oznaka klase")

        self.messages = frame["text"].tolist()
        self.labels = [LABEL_TO_ID[label] for label in frame["label"]]
        self.vocabulary = vocabulary
        self.max_length = max_length
        encode_and_pad(self.messages[0], vocabulary, max_length)

    def __len__(self) -> int:
        return len(self.messages)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        input_ids, padding_mask = encode_and_pad(
            self.messages[index], self.vocabulary, self.max_length
        )
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "padding_mask": torch.tensor(padding_mask, dtype=torch.bool),
            "label": torch.tensor(self.labels[index], dtype=torch.long),
        }
