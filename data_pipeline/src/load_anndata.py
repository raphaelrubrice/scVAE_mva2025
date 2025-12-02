# src/load_anndata.py
"""
Load a single 10x Genomics dataset as AnnData and annotate it
with hierarchical lineage labels from config.py.

Expected folder structure (created by download_script.py):
    data/pbmc_raw/<dataset_name>/
        matrix.mtx
        barcodes.tsv
        genes.tsv or features.tsv

Returned AnnData has:
    obs columns: cell_type_lvl1..lvl4 + corresponding __code columns + 'dataset'
"""

from pathlib import Path
import scanpy as sc
import anndata as ad
import pandas as pd
import numpy as np
import os 

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR  = DATA_DIR / "pbmc_raw"


def _find_matrix_dir(folder: Path) -> Path:
    """
    Look inside `folder` for a 10x directory that actually contains matrix.mtx.
    Handles nested paths like filtered_matrices_mex/hg19/.
    """
    for root, dirs, files in os.walk(folder):
        if "matrix.mtx" in files:
            return Path(root)
    raise FileNotFoundError(f"Could not locate matrix.mtx below {folder}")


def load_anndata(folder: str | Path, meta: dict) -> ad.AnnData:
    """
    Load a single 10x dataset (possibly nested) and annotate with hierarchical labels.
    """
    folder = Path(folder)

    # NEW — automatically find the real matrix directory
    matrix_dir = _find_matrix_dir(folder)

    print(f"Loading 10x data from: {matrix_dir}")
    adata = sc.read_10x_mtx(matrix_dir, var_names="gene_symbols", cache=False)
    adata.var_names_make_unique()

    # annotate hierarchy
    for key in ["lvl1", "lvl2", "lvl3", "lvl4"]:
        adata.obs[f"cell_type_{key}"] = meta[key]

    adata.obs["dataset"] = folder.name

    # convert to categorical and numeric codes
    for lvl in ["lvl1", "lvl2", "lvl3", "lvl4"]:
        col = f"cell_type_{lvl}"
        adata.obs[col] = adata.obs[col].astype("category")
        adata.obs[f"{col}__code"] = adata.obs[col].cat.codes.astype(np.int64)

    print(f"✓ Loaded {adata.shape[0]} cells × {adata.shape[1]} genes from {folder.name}")
    return adata