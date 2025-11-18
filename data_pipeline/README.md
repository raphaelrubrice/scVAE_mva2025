# Data Pipeline Module (SCVAE_MVA2025)

This directory contains the modular data loading and preprocessing system 
for all VAE variants (SCVAE, MoM‑VAE, Hierarchical VAE, etc.). 

Its goal is to provide a **single, consistent interface** for accessing data as 
PyTorch `DataLoader` objects, regardless of the dataset source 
(toy, Scanpy `.h5ad`, or CSV).

---

## 🧩 Overview

The pipeline converts raw single‑cell datasets → processed `torch.Tensor`s 
in three main stages:

```text
Raw Data (Scanpy)
   ↓
Preprocessing (normalization, log transforms, gene selection)
   ↓
PyTorch Dataset → DataLoader
```

This ensures that all model scripts can call the same high‑level function:

```python
from data_pipeline.dataset_registry import get_dataloaders
train_loader, val_loader = get_dataloaders("pbmc10x", batch_size=128)
```

and train without worrying about data format differences.

---

## File structure and responsibilities

| File | Purpose |
|------|----------|
| **`__init__.py`** | Declares this folder as a Python package. |
| **`data_loader.py`** | Defines how to load raw data sources (toy arrays, CSV, or Scanpy/AnnData objects). Provides a `load_dataset()` factory that returns raw numeric matrices and optional labels. |
| **`preprocess.py`** | Contains reusable preprocessing functions — normalization, log-transform, gene selection, etc. These are applied *before* converting to PyTorch tensors. |
| **`dataset_registry.py`** | The public entry point for the pipeline. Combines loading + preprocessing to output ready-to-use PyTorch `DataLoader` objects for training, validation, and testing. |
| **`converters.py`** | Utilities for converting between single‑cell data formats (e.g., `AnnData` ↔ NumPy arrays ↔ SCVAE `DataSet` class). Stubbed now; will be filled in once Scanpy data are integrated. |
| **`config.py`** | Central place to define paths, default batch sizes, data splits, and dataset registry options. Keeps magic numbers and paths out of code. |
| **`utils.py`** | Miscellaneous helper functions — e.g., dataset splitting, summary logging, file checks, or timing wrappers. |

---

## Typical workflow

1. **Choose dataset**
   - Datasets: (example, features, classes, sparsity)
      - `pmbc-3k` smaller dataset with 3K cells
      - `pbmc` (92 043, 32 738, 9, 98%), 
      - `tcga` (10 830, 58 581, 29, 52%)
      - `"toy"` generates synthetic NumPy arrays for rapid testing.


2. **How to get Pytorch dataloaders**
   ```python
   from data_pipeline.dataset_registry import get_dataloaders

   train_loader, val_loader = get_dataloaders("pmbc", batch_size=128)
   ```

It will:

- Load raw data,

- Preprocess it using Scanpy’s exact pipeline,

- Serve ready PyTorch minibatches.

---

## Example test

You can quickly test the stub version with:

```python
from data_pipeline.dataset_registry import get_dataloaders

train_loader, val_loader = get_dataloaders("toy")

for x_batch, y_batch in train_loader:
    print(x_batch.shape, y_batch.shape)
    break
```

Expected output:
``` torch.Size([128, 50]) torch.Size([128]) ```