from hashlib import sha256
from io import BytesIO
from pathlib import Path
from urllib.request import urlopen
from zipfile import ZipFile

DATA_URL = "https://www.kaggle.com/api/v1/datasets/download/uciml/sms-spam-collection-dataset"
DATA_SHA256 = "3e05b8e6e1e8fc9aef3ca69399a1bf3849a22084c8401d5d5d592e6c9a0e422b"
DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "raw" / "sms-spam-collection.zip"


def download_data(path: Path = DATA_PATH) -> Path:
    if path.exists():
        digest = sha256(path.read_bytes()).hexdigest()
        if digest != DATA_SHA256:
            raise ValueError(f"Postojeći fajl ima drugačiji SHA-256: {digest}")
        return path

    with urlopen(DATA_URL, timeout=60) as response:
        content = response.read()

    digest = sha256(content).hexdigest()
    if digest != DATA_SHA256:
        raise ValueError(f"Preuzeti fajl ima drugačiji SHA-256: {digest}")

    with ZipFile(BytesIO(content)) as archive:
        if archive.namelist() != ["spam.csv"]:
            raise ValueError(f"Neočekivan sadržaj arhive: {archive.namelist()}")
        archive.testzip()

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return path


if __name__ == "__main__":
    print(download_data())
