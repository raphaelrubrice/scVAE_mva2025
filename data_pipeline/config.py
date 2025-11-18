
from pathlib import Path

# Root directory where all raw/processed data gets stored
DATA_ROOT_DIR = Path("./data") # Renamed from DATA_DIR for clarity
DATA_ROOT_DIR.mkdir(parents=True, exist_ok=True)
(DATA_ROOT_DIR / "raw").mkdir(exist_ok=True)
(DATA_ROOT_DIR / "processed").mkdir(exist_ok=True)


DATASETS_CONFIG = {
    "pbmc3k": {
        "type": "scanpy_builtin",
        "description": "3k PBMCs from a Healthy Donor (Scanpy mirror)",
        "output_h5ad": DATA_ROOT_DIR / "processed" / "pbmc3k_preprocessed.h5ad",
    },
    "pbmc_full": {
        "type": "scvae_api", # Indicates this dataset comes via the scvae package's loader
        "description": "Full PBMC (Zheng et al., 2017) purified populations (92k cells) from SCVAE paper",
        "output_h5ad": DATA_ROOT_DIR / "processed" / "pbmc_zheng2017_full.h5ad",
        # Original SCVAE dataset name to pass to its loader
        "scvae_name": "10x-PBMC-PP",
    },
    "tcga": {
        "type": "tcga_xena",
        "description": "TCGA Pan-Cancer RSEM Expected Counts (Bulk RNA-seq)",
        "output_h5ad": DATA_ROOT_DIR / "processed" / "tcga_rsem_preprocessed.h5ad",
        "xena_name": "tcga_gene_expected_count", # For potential Xena API/downloader
    },
    "toy": {
        "type": "synthetic",
        "description": "Synthetic NumPy array for rapid testing",
        "shape": (100, 50), # Example shape
    },
    # Add other datasets here following their 'type' and specific keys
}

# (can still be overridden by get_dataloaders arguments)
DEFAULT_BATCH_SIZE = 128
DEFAULT_VAL_SPLIT = 0.1