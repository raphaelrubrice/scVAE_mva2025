import tarfile
import tempfile
import os
import tarfile
import requests
from tqdm import tqdm
from pathlib import Path
from data_pipeline.src.config import DATASETS


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "pbmc_raw"

def extract_tar_gz(temp_path, out_dir):
    """Extract a .tar.gz archive to out_dir."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with tarfile.open(temp_path, "r:gz") as tar:
        tar.extractall(out_dir)
    print(f"✓ Extracted to {out_dir}")


def download_and_extract(url: str, out_dir: Path):
    """
    Stream‑download a tar.gz and extract directly to out_dir,
    deleting the temporary archive afterward.
    """
    out_dir = Path(out_dir)
    if out_dir.exists() and any(out_dir.iterdir()):
        print(f"✓ Already downloaded/extracted: {out_dir}")
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"⇩ Downloading and extracting to {out_dir}")

    with tempfile.NamedTemporaryFile(suffix=".tar.gz", delete=True) as tmp:
        response = requests.get(url, stream=True, timeout=120)
        response.raise_for_status()
        total = int(response.headers.get("content-length", 0))

        with tqdm(total=total, unit="B", unit_scale=True,
                  desc=out_dir.name, colour="cyan") as bar:
            for chunk in response.iter_content(chunk_size=1024 * 64):
                if chunk:
                    tmp.write(chunk)
                    bar.update(len(chunk))
        tmp.flush()

        extract_tar_gz(tmp.name, out_dir)

    print(f"✓ Finished {out_dir.name}")


def run_downloads(raw_dir: Path = RAW_DIR, meta: dict = None):
    """
    Ensure each dataset is present under data/pbmc_raw/<dataset_name>/.
    No archives are permanently stored.
    """
    raw_path = Path(raw_dir)
    raw_path.mkdir(parents=True, exist_ok=True)
    meta = meta or DATASETS

    print(f"\nSaving extracted PBMC datasets to: {raw_path.resolve()}\n")

    for name, info in meta.items():
        target_folder = raw_path / name
        try:
            download_and_extract(info["url"], target_folder)
        except Exception as e:
            print(f"✗ Problem with {name}: {e}")
        print("-" * 60)

    print("All PBMC datasets ready in:", raw_path.resolve())