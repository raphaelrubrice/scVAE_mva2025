# data_pipeline/data_loader.py

import scanpy as sc
from pathlib import Path
from typing import Dict
from .config import DATASETS, DATA_DIR
from .utils import download_and_extract



def load_dataset_raw(name: str, verbose: bool = True):
    """
    Load raw dataset by name.
    """
    name = name.lower()
    if name == "pbmc3k":
        if verbose:
            print("Loading PBMC3k dataset directly from Scanpy cloud mirror …")
        adata = sc.datasets.pbmc3k()  # built‑in scanpy copy of 10x data
        adata.obs["cell_type"] = "unknown"
        return adata

    # fall back to our manual loader (10x_combine etc.)
    from .config import DATASETS
    if name not in DATASETS:
        raise ValueError(f"Unknown dataset {name}")
    cfg = DATASETS[name]
    if cfg["format"] == "10x_combine":
        return _load_10x_combine(name, cfg, verbose)
    raise NotImplementedError(f"Format {cfg['format']} not implemented")

def _load_10x_combine(
    name: str,
    config: Dict,
    verbose: bool = True
) -> "sc.AnnData":
    """
    Load and combine multiple 10x datasets.
    Each one is a separate cell type; they get merged with labels.
    """
    if verbose:
        print(f"\nLoading dataset: {name}")
        print(f"Source: {config['source']}")

    adata_list = []
    cell_type_labels = []

    for cell_type, url in config["urls"].items():
        if verbose:
            print(f"\n   → {cell_type}")

        # Download and extract
        folder = download_and_extract(url, DATA_DIR, verbose=verbose)

        # The tar.gz extracts to a folder structure like:
        # cd56_nk_filtered_gene_bc_matrices/
        #   └── hg19/
        #       ├── matrix.mtx
        #       ├── genes.tsv
        #       └── barcodes.tsv
        # We need to find the inner 'hg19' or similar folder
        inner_folders = list(folder.glob("**/matrix.mtx"))
        if not inner_folders:
            raise FileNotFoundError(
                f"Could not find matrix.mtx inside {folder}"
            )
        mtx_path = inner_folders[0].parent

        # Load with Scanpy
        adata = sc.read_10x_mtx(
            mtx_path,
            var_names="gene_symbols",
            cache=False
        )

        # Add cell type label to metadata
        adata.obs["cell_type"] = cell_type

        adata_list.append(adata)
        cell_type_labels.append(cell_type)
        if verbose:
            print(f"      Loaded: {adata.n_obs} cells × {adata.n_vars} genes")

    # Concatenate all
    adata_combined = adata_list[0].concatenate(
        adata_list[1:],
        batch_key="cell_type"
    )

    if verbose:
        print(f"\n✓ Combined dataset:")
        print(f"   Total: {adata_combined.n_obs} cells × {adata_combined.n_vars} genes")
        print(f"   Cell types: {', '.join(cell_type_labels)}")

    return adata_combined