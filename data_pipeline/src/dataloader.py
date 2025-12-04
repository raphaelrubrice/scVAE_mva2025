# src/dataloader.py
"""
Build train/val/test AnnData Loaders for PBMC data
--------------------------------------------------
Uses .h5ad shards -> AnnCollection -> AnnLoader
Supports train/val/test split via AnnCollectionView.
"""

from pathlib import Path
import numpy as np
import torch
import scipy
import json
import anndata as ad
from anndata.experimental import AnnCollection, AnnLoader
from torch.utils.data import DataLoader
from data_pipeline.src.DatasetWrapper import AnnDatasetWrapper


def one_hot(idx: np.ndarray, num_classes: int) -> torch.Tensor:
    t = torch.zeros((idx.shape[0], num_classes), dtype=torch.float32)
    t.scatter_(1, torch.from_numpy(idx).view(-1, 1), 1.0)
    return t


#converter builder
def make_converter(label_maps_path: str | Path, one_hot_labels=False):
    """Return a mini‑batch converter function that AnnLoader can use."""
    with open(label_maps_path) as f:
        label_maps = json.load(f)
    n1, n2, n3, n4 = (
        len(label_maps["cell_type_lvl1"]),
        len(label_maps["cell_type_lvl2"]),
        len(label_maps["cell_type_lvl3"]),
        len(label_maps["cell_type_lvl4"]),
    )

    def convert_fn(batch):
        X = batch.X.toarray().astype(np.float32) if hasattr(batch.X, "toarray") \
            else np.asarray(batch.X, dtype=np.float32)

        y1 = batch.obs["cell_type_lvl1__code"].to_numpy(dtype=np.int64)
        y2 = batch.obs["cell_type_lvl2__code"].to_numpy(dtype=np.int64)
        y3 = batch.obs["cell_type_lvl3__code"].to_numpy(dtype=np.int64)
        y4 = batch.obs["cell_type_lvl4__code"].to_numpy(dtype=np.int64)

        out = {"X": torch.from_numpy(X)}
        if one_hot_labels:
            out["y1"], out["y2"], out["y3"], out["y4"] = (
                one_hot(y1, n1),
                one_hot(y2, n2),
                one_hot(y3, n3),
                one_hot(y4, n4),
            )
        else:
            out["y1"], out["y2"], out["y3"], out["y4"] = (
                torch.from_numpy(y1),
                torch.from_numpy(y2),
                torch.from_numpy(y3),
                torch.from_numpy(y4),
            )
        return out

    return convert_fn


#  build collection from shards
def build_collection_from_shards(
    shard_dir: str | Path = "data/pbmc_processed/shards",
    filter_genes: bool = False,
    max_genes: int = 5000,
) -> AnnCollection:
    """
    Load .h5ad shards and join into an AnnCollection.

    - Ensures *global* uniqueness of obs_names by prefixing with shard index.
    - If `filter_genes=True`, materializes a unified AnnData to compute per-gene
      std and keep the top `max_genes` most variable genes.
    """
    shard_dir = Path(shard_dir)
    paths = sorted(shard_dir.glob("*.h5ad"))

    if not paths:
        raise FileNotFoundError(f"No .h5ad shards found in {shard_dir}")

    # ------------------------------------------------------------------
    # 1) Load shards and enforce *global* unique obs_names
    # ------------------------------------------------------------------
    adatas = []
    for shard_idx, p in enumerate(paths):
        # For gene filtering we need in-memory AnnData anyway.
        # For no filtering you can switch to `backed="r"` if memory is tight.
        a = ad.read_h5ad(p) if filter_genes else ad.read_h5ad(p, backed="r")

        # Save original names (barcodes) so they are not lost
        # (in backed mode, this is allowed and will be written to disk if changed)
        if "orig_obs_name" not in a.obs:
            # `a.obs_names` is an Index; cast to str to be safe
            a.obs["orig_obs_name"] = np.asarray(a.obs_names.astype(str))

        # Build globally unique obs_names: "<shard_idx>_<original_name>"
        new_obs_names = [f"{shard_idx}_{name}" for name in a.obs["orig_obs_name"].astype(str)]
        a.obs_names = new_obs_names

        adatas.append(a)

    collection = AnnCollection(
        adatas,
        join_vars="outer",
        join_obs="outer",
        label="dataset",
    )

    # Sanity check: now this MUST be True
    if not collection.obs_names.is_unique:
        raise RuntimeError("obs_names are still not unique after prefixing; check shard loading logic.")

    # ------------------------------------------------------------------
    # 2) Optional gene filtering: requires materializing to AnnData
    # ------------------------------------------------------------------
    if filter_genes:
        # This is where it used to crash; now obs_names are globally unique
        collection.to_adata()
        print(collection)
        X = collection.X
        if scipy.sparse.issparse(X):
            stds = np.asarray(X.std(axis=0)).ravel()
        else:
            stds = X.std(axis=0)

        if max_genes > collection.n_vars:
            max_genes = collection.n_vars

        # Take highest-variance genes
        kept_idx = np.argsort(stds)[-max_genes:]
        kept_gene_names = collection.var_names[kept_idx]

        print("Before filtering:", collection)
        collection = collection[:, kept_gene_names]
        print("After filtering:", collection)

    print(
        f"✓ Built AnnCollection with {collection.n_obs} total cells "
        f"and {collection.n_vars} genes."
    )
    return collection

# split collection to train/val/test
def split_collection(collection: AnnCollection, train_frac=0.81, val_frac=0.09, seed=42):
    n = collection.n_obs
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)

    n_train = int(n * train_frac)
    n_val = int(n * val_frac)
    n_test = n - n_train - n_val

    train_idx = idx[:n_train]
    val_idx = idx[n_train:n_train + n_val]
    test_idx = idx[n_train + n_val:]

    train_view, val_view, test_view = (
        collection[train_idx, :],
        collection[val_idx, :],
        collection[test_idx, :],
    )

    print(f"Split: train={n_train}, val={n_val}, test={n_test}")
    return train_view, val_view, test_view


#  build dataloaders
def build_dataloaders(
    shard_dir="data/pbmc_processed/shards",
    label_maps_path="data/pbmc_processed/label_maps.json",
    batch_size=512,
    one_hot=False,
    shuffle=True,
    num_workers=0,
    train_frac=0.81,
    val_frac=0.09,
    seed=42,
    **kwargs
):
    """Return (train_loader, val_loader, test_loader) PyTorch DataLoaders."""

    # Build collection and converter
    collection = build_collection_from_shards(shard_dir, **kwargs)
    converter = make_converter(label_maps_path, one_hot_labels=one_hot)

    # Split into train/val/test AnnCollectionViews
    train_view, val_view, test_view = split_collection(collection, train_frac, val_frac, seed)

    # Wrap each subset in our custom Dataset
    datasets = {
        "train": AnnDatasetWrapper(train_view, converter),
        "val":   AnnDatasetWrapper(val_view, converter),
        "test":  AnnDatasetWrapper(test_view, converter),
    }

    # Standard PyTorch DataLoaders
    loaders = {
        split: DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=(split == "train" and shuffle),
            num_workers=num_workers,
            pin_memory=False,
        )
        for split, ds in datasets.items()
    }

    print(
        f"✓ Train={len(datasets['train'])}, Val={len(datasets['val'])}, Test={len(datasets['test'])}"
    )
    return loaders["train"], loaders["val"], loaders["test"]


def create_cv_loaders(train_dataset, 
                    val_dataset, 
                    n_folds=5, 
                    batch_size=32, 
                    shuffle=True, 
                    seed=1234,
                    pin_m=False):
    """
    Create cross-validation dataloaders from train and validation datasets.
    """
    # Combine train + val datasets for CV
    full_dataset = torch.utils.data.ConcatDataset([train_dataset, 
                                                    val_dataset])
    dataset_size = len(full_dataset)
    indices = list(range(dataset_size))

    if shuffle:
        np.random.seed(seed)
        np.random.shuffle(indices)

    fold_sizes = dataset_size // n_folds
    folds = []

    for fold in range(n_folds):
        val_start = fold * fold_sizes
        val_end = val_start + fold_sizes if fold < n_folds - 1 else dataset_size

        val_indices = indices[val_start:val_end]
        train_indices = indices[:val_start] + indices[val_end:]

        train_sampler = SubsetRandomSampler(train_indices)
        val_sampler = SubsetRandomSampler(val_indices)

        train_loader = DataLoader(full_dataset, 
                                    batch_size=batch_size, 
                                    sampler=train_sampler,
                                    pin_memory=pin_m)
        val_loader = DataLoader(full_dataset, 
                                batch_size=batch_size, 
                                sampler=val_sampler,
                                pin_memory=pin_m)

        folds.append((train_loader, val_loader))

    return folds

def build_cv_dataloaders(
    shard_dir="data/pbmc_processed/shards",
    label_maps_path="data/pbmc_processed/label_maps.json",
    batch_size=256,
    n_folds=5,
    one_hot=False,
    shuffle=True,
    num_workers=0,
    train_frac=0.81,
    val_frac=0.09,
    seed=1234,
    **kwargs
):
    """Return (train_loader, val_loader, test_loader) PyTorch DataLoaders."""

    # Build collection and converter
    collection = build_collection_from_shards(shard_dir, **kwargs)
    converter = make_converter(label_maps_path, one_hot_labels=one_hot)

    # Split into train/val/test AnnCollectionViews
    train_view, val_view, test_view = split_collection(collection, train_frac, val_frac, seed)

    # Wrap each subset in our custom Dataset
    datasets = {
        "train": AnnDatasetWrapper(train_view, converter),
        "val":   AnnDatasetWrapper(val_view, converter),
        "test":  AnnDatasetWrapper(test_view, converter),
    }

    # Create pairs of train and val loaders for CV
    folds = create_cv_loaders(datasets["train"], 
                                dataset["val"], 
                                n_folds=n_folds, 
                                batch_size=batch_size, 
                                shuffle=shuffle, 
                                seed=seed,
                                num_workers=num_workers,
                                pin_m=False)
    # Test
    test_loader = DataLoader(datasets["test"],
                            batch_size=batch_size,
                            shuffle=False,
                            num_workers=num_workers,
                            pin_memory=False,
                            )

    print(
        f"✓ Train={len(datasets['train'])}, Val={len(datasets['val'])}, Test={len(datasets['test'])}"
    )
    print(f"✓ Train/Val CV Folds: {[(len(train),len(val)) for train,val in folds]}")
    return folds, test_loader