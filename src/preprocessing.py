from pathlib import Path

import pandas as pd

from src.data_loading import EXPECTED_COLUMNS, load_raw_sms

TEXT_COLUMNS = EXPECTED_COLUMNS[1:]
PROCESSED_PATH = Path(__file__).resolve().parents[1] / "data" / "processed" / "sms_clean.csv"


def prepare_sms(raw: pd.DataFrame) -> pd.DataFrame:
    if tuple(raw.columns) != EXPECTED_COLUMNS:
        raise ValueError(f"Neočekivane kolone: {list(raw.columns)}")
    if raw["v1"].isna().any() or not raw["v1"].isin(("ham", "spam")).all():
        raise ValueError("Nedostaje oznaka ili postoji nepoznata klasa")
    if raw["v2"].isna().any():
        raise ValueError("Nedostaje početak teksta poruke")

    text = raw[list(TEXT_COLUMNS)].apply(
        lambda row: ",".join(str(part) for part in row if pd.notna(part) and str(part) != ""),
        axis=1,
    ).str.strip()
    if text.eq("").any():
        raise ValueError("Postoji prazna poruka")

    cleaned = pd.DataFrame({"label": raw["v1"], "text": text})
    if cleaned.groupby("text")["label"].nunique().gt(1).any():
        raise ValueError("Ista poruka ima različite oznake")

    return cleaned.drop_duplicates(subset="text").reset_index(drop=True)


def main() -> None:
    raw = load_raw_sms()
    cleaned = prepare_sms(raw)
    PROCESSED_PATH.parent.mkdir(parents=True, exist_ok=True)
    cleaned.to_csv(PROCESSED_PATH, index=False)
    print(f"Sačuvano {len(cleaned)} poruka u {PROCESSED_PATH}")
    print(f"Uklonjeno {len(raw) - len(cleaned)} tačnih duplikata")
    print(cleaned["label"].value_counts().to_string())


if __name__ == "__main__":
    main()
