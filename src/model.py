import torch
from torch import nn

from src.sequences import DEFAULT_MAX_LENGTH
from src.vocabulary import PAD_ID


class SMSClassifier(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 64,
        n_heads: int = 4,
        n_layers: int = 2,
        dim_feedforward: int = 128,
        dropout: float = 0.1,
        max_length: int = DEFAULT_MAX_LENGTH,
    ) -> None:
        super().__init__()
        if vocab_size < 3:
            raise ValueError("Rečnik mora sadržati posebne tokene i bar jedan običan token")
        if d_model < 1 or n_heads < 1 or d_model % n_heads != 0:
            raise ValueError("Dimenzija reprezentacije mora biti deljiva brojem glava")
        if n_layers < 1 or dim_feedforward < 1 or max_length < 1:
            raise ValueError("Broj slojeva, feed-forward dimenzija i dužina moraju biti pozitivni")
        if not 0 <= dropout < 1:
            raise ValueError("Dropout mora biti između 0 i 1")

        self.max_length = max_length
        self.token_embedding = nn.Embedding(vocab_size, d_model, padding_idx=PAD_ID)
        self.position_embedding = nn.Embedding(max_length, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=n_layers, enable_nested_tensor=False
        )
        self.classifier = nn.Linear(d_model, 2)

    def forward(self, input_ids: torch.Tensor, padding_mask: torch.Tensor) -> torch.Tensor:
        if input_ids.ndim != 2 or padding_mask.shape != input_ids.shape:
            raise ValueError("ID-jevi i maska moraju imati oblik (batch, dužina)")
        if input_ids.shape[0] < 1 or not 1 <= input_ids.shape[1] <= self.max_length:
            raise ValueError("Batch mora sadržati poruke dozvoljene dužine")
        if input_ids.dtype != torch.long or padding_mask.dtype != torch.bool:
            raise TypeError("ID-jevi moraju biti long, a maska bool")
        if padding_mask.all(dim=1).any():
            raise ValueError("Svaka poruka mora imati bar jedan stvarni token")

        positions = torch.arange(input_ids.shape[1], device=input_ids.device).unsqueeze(0)
        embeddings = self.token_embedding(input_ids) + self.position_embedding(positions)
        encoded = self.encoder(embeddings, src_key_padding_mask=padding_mask)
        valid_tokens = (~padding_mask).unsqueeze(-1)
        message_vectors = (encoded * valid_tokens).sum(dim=1) / valid_tokens.sum(dim=1)
        return self.classifier(message_vectors)
