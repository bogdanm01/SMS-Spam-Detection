import re

TOKEN_PATTERN = re.compile(r"\w+|[^\w\s]", re.UNICODE)


def tokenize_sms(text: str) -> list[str]:
    if not isinstance(text, str):
        raise TypeError("SMS poruka mora biti tekst")
    tokens = TOKEN_PATTERN.findall(text.lower())
    if not tokens:
        raise ValueError("SMS poruka ne sme biti prazna")
    return tokens
