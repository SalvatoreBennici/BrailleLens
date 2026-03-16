import sys
import zipfile
from pathlib import Path
import gdown

def download_and_extract(url: str) -> None:
    current_dir: Path = Path(__file__).parent.resolve()
    zip_path: Path = current_dir / "dataset.zip"

    gdown.download(url, str(zip_path), quiet=False, fuzzy=True)

    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(current_dir)

    zip_path.unlink()

if __name__ == "__main__":
    dataset_url = "https://drive.google.com/drive/folders/15Q-8RnFknXmIN9CqX_6urYNd3jXJfNev?usp=sharing"
    download_and_extract(dataset_url)