from src.vocabulary import PAD_ID, encode_sms

DEFAULT_MAX_LENGTH = 128


def encode_and_pad(
    message: str,
    vocabulary: dict[str, int],
    max_length: int = DEFAULT_MAX_LENGTH,
) -> tuple[list[int], list[bool]]:
    if not isinstance(max_length, int) or isinstance(max_length, bool) or max_length < 1:
        raise ValueError("Maksimalna dužina mora biti pozitivan ceo broj")

    token_ids = encode_sms(message, vocabulary)[:max_length]
    padding_length = max_length - len(token_ids)
    input_ids = token_ids + [PAD_ID] * padding_length
    padding_mask = [False] * len(token_ids) + [True] * padding_length
    return input_ids, padding_mask
