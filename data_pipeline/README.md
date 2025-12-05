# PBMC Pipeline (Download → Combine → Prepare → Train)

It standardizes raw data, loads each dataset into AnnData, harmonizes labels, builds a combined AnnCollection, and exposes PyTorch-ready DataLoaders.

The pipeline is fully modular:

- `downloader.py` handles **downloads + extraction only**
- `load_anndata.py` handles **loading + annotation only**
- `combine.py` handles **combining + freezing categories + saving shards**
- `dataloader.py` handles **building AnnCollection + AnnLoader for PyTorch**
- `AnnDatasetWrapper` (in `DatasetWrapper.py`) provides a **PyTorch Dataset-compatible wrapper** so each sample returns `X` + the 4 hierarchical labels
- `config.py` stores URLs + label metadata

You can see examples of workflow in **test_pipeline.ipynb**
You can see the hierarchy levels I've defined in config.py:
```text
Stem_cell
└── CD34_HSC
    

Non_stem
├── B_cell
│   └── B_cell
│       └── CD19_B_cell
├── NK_cell
│   └── NK_cell
│       └── CD56_NK_cell
└── T_cell
    ├── CD4_T
    │   ├── CD4_CD3_T_helper
    │   ├── CD4_CD25_Treg
    │   ├── CD4_CD45RA_CD25neg_naive_T
    │   └── CD4_CD45RO_memory_T
    └── CD8_T
        ├── CD8_cytotoxic_T
        └── CD8_CD45RA_naive_T
```
###Extracting Labels in Experiments (MOST IMPORTANT) 

```python
for batch in train_loader:
    X  = batch["X"]       # gene matrix
    y1 = batch["y1"]      # lvl1 (broad)
    y2 = batch["y2"]      # lvl2
    y3 = batch["y3"]      # lvl3
    y4 = batch["y4"]      # lvl4 (specific 9)
    break
```


```text
data_pipeline/
│── config.py # metadata for all 9 datasets
│
├── data/
│ ├── pbmc_raw/ # downloaded + extracted raw files
│ └── pbmc_processed/
│ ├── pbmc_combined.h5ad # full combined dataset
│ └── shards/ # per-dataset .h5ad shards
│
├── src/
│ ├── download_script.py # download + extract only
│ ├── load_anndata.py # load 10X folders → AnnData with labels
│ ├── combine.py # combine + write shards + freeze labels
│ ├── dataloader.py # AnnCollection + AnnLoader for PyTorch
│ └── utils_io.py # shared helpers (var cleanup, logging, etc.)
```
