from pathlib import Path

import pandas as pd

from src.download_data import DATA_PATH

EXPECTED_COLUMNS = ("v1", "v2", "Unnamed: 2", "Unnamed: 3", "Unnamed: 4")


def load_raw_sms(path: Path = DATA_PATH) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"Nedostaje {path}. Pokreni python -m src.download_data")

    frame = pd.read_csv(path, encoding="latin-1", compression="zip")
    if tuple(frame.columns) != EXPECTED_COLUMNS:
        raise ValueError(f"Neočekivane kolone: {list(frame.columns)}")
    if frame[["v1", "v2"]].isna().any().any():
        raise ValueError("Nedostaje oznaka ili tekst poruke")
    if not frame["v1"].isin(("ham", "spam")).all():
        raise ValueError("Nepoznata oznaka klase")

    return frame
