from collections import Counter
from collections.abc import Iterable

from src.tokenization import tokenize_sms

PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"
PAD_ID = 0
UNK_ID = 1


def build_vocabulary(messages: Iterable[str]) -> dict[str, int]:
    frequencies = Counter(
        token for message in messages for token in tokenize_sms(message)
    )
    if not frequencies:
        raise ValueError("Potreban je bar jedan trening primer")

    ordered_tokens = sorted(
        frequencies,
        key=lambda token: (-frequencies[token], token),
    )
    vocabulary = {PAD_TOKEN: PAD_ID, UNK_TOKEN: UNK_ID}
    vocabulary.update(
        {token: index for index, token in enumerate(ordered_tokens, start=2)}
    )
    return vocabulary


def encode_sms(message: str, vocabulary: dict[str, int]) -> list[int]:
    if vocabulary.get(PAD_TOKEN) != PAD_ID or vocabulary.get(UNK_TOKEN) != UNK_ID:
        raise ValueError("Rečnik mora imati <PAD>=0 i <UNK>=1")
    return [vocabulary.get(token, UNK_ID) for token in tokenize_sms(message)]
