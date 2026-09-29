from pathlib import Path

import pandas as pd

from src.data_loading import load_raw_sms
from src.preprocessing import prepare_sms

PROCESSED_DIR = Path(__file__).resolve().parents[1] / "data" / "processed"
TEST_FRACTION = 0.2
RANDOM_SEED = 42


def split_sms(
    cleaned: pd.DataFrame,
    test_fraction: float = TEST_FRACTION,
    random_seed: int = RANDOM_SEED,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if list(cleaned.columns) != ["label", "text"]:
        raise ValueError("Očekivane kolone su label i text")
    if cleaned.empty or cleaned.isna().any().any() or cleaned["text"].eq("").any():
        raise ValueError("Skup sadrži prazne ili nedostajuće podatke")
    if set(cleaned["label"]) != {"ham", "spam"}:
        raise ValueError("Očekivane su obe klase: ham i spam")
    if cleaned["text"].duplicated().any():
        raise ValueError("Ukloni duplikate pre podele")
    if not 0 < test_fraction < 1:
        raise ValueError("Udeo test skupa mora biti između 0 i 1")
    if cleaned.groupby("label").size().lt(2).any():
        raise ValueError("Svaka klasa mora imati bar dve poruke")

    test = cleaned.groupby("label", group_keys=False).sample(
        frac=test_fraction,
        random_state=random_seed,
    )
    train = cleaned.drop(index=test.index)
    if train.empty or test.empty or set(train["label"]) != set(test["label"]):
        raise ValueError("Obe podele moraju sadržati obe klase")

    return train.reset_index(drop=True), test.reset_index(drop=True)


def main() -> None:
    cleaned = prepare_sms(load_raw_sms())
    train, test = split_sms(cleaned)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    train.to_csv(PROCESSED_DIR / "sms_train.csv", index=False)
    test.to_csv(PROCESSED_DIR / "sms_test.csv", index=False)
    print(f"Trening: {len(train)} poruka")
    print(train["label"].value_counts().to_string())
    print(f"Test: {len(test)} poruka")
    print(test["label"].value_counts().to_string())


if __name__ == "__main__":
    main()
