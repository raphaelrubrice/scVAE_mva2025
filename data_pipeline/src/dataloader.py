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
from torch.utils.data import DataLoader, SubsetRandomSampler
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
):
    shard_dir = Path(shard_dir)
    paths = sorted(shard_dir.glob("*.h5ad"))
    if not paths:
        raise FileNotFoundError(f"No .h5ad shards found in {shard_dir}")

    # Load all shards in memory (needed if we want X)
    adatas = [ad.read_h5ad(p) for p in paths]

    if filter_genes:
        # 1) Concatenate into a single AnnData WITH .X
        big = ad.concat(
            adatas,
            join="outer",
            label="dataset",
            keys=[str(i) for i in range(len(adatas))],
            index_unique="shard-",
        )

        X = big.X
        if scipy.sparse.issparse(X):
            # mean of x
            means = np.asarray(X.mean(axis=0)).ravel()
            # mean of x^2
            means_sq = np.asarray(X.power(2).mean(axis=0)).ravel()
            # std = sqrt(E[x^2] - (E[x])^2)
            stds = np.sqrt(means_sq - means**2)
        else:
            stds = X.std(axis=0)

        if max_genes > big.n_vars:
            max_genes = big.n_vars

        kept_idx = np.argsort(stds)[-max_genes:]  # highest-variance genes
        print("\nKEPT IDX:", kept_idx)
        kept_genes = big.var_names[kept_idx]
        print("\nKEPT GENES:", kept_genes)

        # 2) Filter each shard individually to those genes
        adatas = [a[:, kept_genes] for a in adatas]

    # 3) Build AnnCollection from the (optionally filtered) shards
    collection = AnnCollection(
        adatas,
        join_vars="outer",
        join_obs="outer",
        label="dataset",
    )

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
                    num_workers=1,
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
                                    num_workers=num_workers,
                                    pin_memory=pin_m)
        val_loader = DataLoader(full_dataset, 
                                batch_size=batch_size, 
                                sampler=val_sampler,
                                num_workers=num_workers,
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
                                datasets["val"], 
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
    print(f"✓ Train/Val CV Folds (n°batches): {[(len(train),len(val)) for train,val in folds]}")
    return folds, test_loader