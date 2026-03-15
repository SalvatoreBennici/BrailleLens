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
    if len(sys.argv) != 2:
        print("Usage: python download_datasets.py <url>")
        sys.exit(1)
    
    download_and_extract(sys.argv[1])