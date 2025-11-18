# data_pipeline/utils.py

import os
import tarfile
import urllib.request
from pathlib import Path
from typing import Union
import numpy as np

def download_and_extract(
    url: str,
    dest_dir: Union[str, Path],
    verbose: bool = True
) -> Path:
    """
    Download a tar.gz file and extract it.
    Returns the path to the extracted folder.
    """
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)

    filename = url.split("/")[-1]
    filepath = dest_dir / filename

    # Download if missing
    if not filepath.exists():
        if verbose:
            print(f"Downloading {filename} ...")
        urllib.request.urlretrieve(url, filepath)
    else:
        if verbose:
            print(f"✓ {filename} already exists")

    # Extract if not already done
    extract_folder = dest_dir / filename.replace(".tar.gz", "")
    if not extract_folder.exists():
        if verbose:
            print(f"Extracting {filename} ...")
        with tarfile.open(filepath, "r:gz") as tar:
            tar.extractall(path=dest_dir)
    else:
        if verbose:
            print(f"✓ Already extracted to {extract_folder.name}")

    return extract_folder


def generate_toy_data(shape=(100, 50), random_seed=42):
    """Generates simple random data for testing."""
    np.random.seed(random_seed)
    X = np.random.rand(*shape)
    y = np.random.randint(0, 3, size=shape[0]) # 3 classes
    return X, y