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
from scanpy.preprocessing import sample
from collections import Counter
from typing import Iterable

def _format_counts(counts: dict[int, int]) -> str:
    total = int(sum(counts.values()))
    if total == 0:
        return "total=0"
    items = sorted(counts.items(), key=lambda kv: kv[0])
    parts = [f"{k}:{v} ({(v/total)*100:5.2f}%)" for k, v in items]
    return " | ".join(parts) + f" | total={total}"


def debug_print_loader_stratification(
    loader,
    name: str,
    levels: tuple[str, ...] = ("y1", "y2", "y3", "y4"),
    max_batches: int | None = None,
):
    """
    Iterates through a DataLoader and prints the observed label distribution per level.

    Robust to label shapes:
      - [B]                     (int labels)
      - [B, C]                  (one-hot or probs)
      - [B, 1, C] / [B, *, C]   (extra singleton / wrapper dims)
      - Python lists nested similarly

    Strategy:
      - Convert to torch tensor if needed
      - If last dim looks like "classes" (>=2), take argmax over last dim
      - Squeeze singleton dims
      - Flatten to 1D and update Counter
    """
    totals = {lvl: Counter() for lvl in levels}
    seen_batches = 0
    seen_samples = 0

    for b_idx, batch in enumerate(loader):
        if max_batches is not None and b_idx >= max_batches:
            break

        seen_batches += 1

        # approximate sample count
        if "X" in batch and isinstance(batch["X"], torch.Tensor):
            seen_samples += int(batch["X"].shape[0])

        for lvl in levels:
            if lvl not in batch:
                continue

            y = batch[lvl]

            # Ensure torch tensor
            if not isinstance(y, torch.Tensor):
                y = torch.as_tensor(y)

            # Move to CPU for safe numpy ops
            y = y.detach().cpu()

            # If it looks like one-hot/probs: take argmax along last dim
            # (works for [B,C], [B,1,C], [B,*,C], etc.)
            if y.ndim >= 2 and y.shape[-1] >= 2:
                y = torch.argmax(y, dim=-1)

            # Remove singleton dims and flatten
            y = y.squeeze()

            # Make sure it's 1D (flatten any remaining dims, e.g., [B,1] -> [B])
            y = y.reshape(-1).to(torch.int64).numpy()

            totals[lvl].update(y.tolist())

    print(f"\n[STRAT DEBUG] {name}")
    print(f"  batches_seen={seen_batches}, samples_seen={seen_samples}")
    for lvl in levels:
        print(f"  {lvl}: {_format_counts(dict(totals[lvl]))}")



def _count_obs_codes(adatas: list[ad.AnnData], obs_key: str) -> dict[int, int]:
    out = Counter()
    for a in adatas:
        vals = a.obs[obs_key].to_numpy(dtype=np.int64)
        out.update(vals.tolist())
    return dict(out)


def debug_print_pre_post_downsample(adatas_before: list[ad.AnnData], adatas_after: list[ad.AnnData]):
    """
    Prints distributions across all shards before vs after downsampling, for y1..y4.
    """
    keys = [
        ("y1", "cell_type_lvl1__code"),
        ("y2", "cell_type_lvl2__code"),
        ("y3", "cell_type_lvl3__code"),
        ("y4", "cell_type_lvl4__code"),
    ]

    print("\n[STRAT DEBUG] BEFORE downsampling (across all shards)")
    for lvl, obs_key in keys:
        counts = _count_obs_codes(adatas_before, obs_key)
        print(f"  {lvl}: {_format_counts(counts)}")

    print("\n[STRAT DEBUG] AFTER downsampling (across all shards)")
    for lvl, obs_key in keys:
        counts = _count_obs_codes(adatas_after, obs_key)
        print(f"  {lvl}: {_format_counts(counts)}")


def _stratified_sample_indices(
    labels: np.ndarray,
    n_select: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Stratified sampling without replacement preserving proportions as closely as possible.
    Uses Hamilton / largest-remainder method to hit exactly n_select.
    """
    labels = np.asarray(labels, dtype=np.int64)
    n = labels.shape[0]
    assert 0 < n_select <= n, f"n_select must be in (0, {n}], got {n_select}"

    # Group indices per class
    classes, counts = np.unique(labels, return_counts=True)
    class_to_indices = {c: np.where(labels == c)[0] for c in classes}

    # Ideal (fractional) allocations
    frac = counts / counts.sum()
    raw = frac * n_select
    base = np.floor(raw).astype(int)
    base = np.minimum(base, counts)  # cannot allocate more than available

    remainder = n_select - int(base.sum())

    # Distribute remaining by largest fractional part, while respecting availability
    fractional = raw - np.floor(raw)
    order = np.argsort(-fractional)

    alloc = base.copy()
    if remainder > 0:
        for idx in order:
            c = classes[idx]
            if remainder == 0:
                break
            if alloc[idx] < counts[idx]:
                alloc[idx] += 1
                remainder -= 1

    # If still remainder (due to caps), fill from any class with spare capacity
    if remainder > 0:
        for idx in range(len(classes)):
            if remainder == 0:
                break
            if alloc[idx] < counts[idx]:
                take = min(remainder, counts[idx] - alloc[idx])
                alloc[idx] += take
                remainder -= take

    assert int(alloc.sum()) == n_select, f"Allocation error: {int(alloc.sum())} != {n_select}"

    # Sample within each class
    picked = []
    for c, k in zip(classes, alloc):
        if k == 0:
            continue
        idxs = class_to_indices[c]
        # permute indices for this class deterministically from rng
        perm = rng.permutation(idxs)
        picked.append(perm[:k])

    picked = np.concatenate(picked, axis=0)
    picked = rng.permutation(picked)  # shuffle final selection
    return picked


def split_collection_test_holdout_stratified(
    collection: AnnCollection,
    test_frac: float = 0.09,
    seed: int = 42,
    stratify_obs_key: str = "cell_type_lvl4__code",
):
    """
    Stratified holdout split:
      - test: test_frac of full data (stratified by stratify_obs_key)
      - trainval: remaining
    Returns (trainval_view, test_view).
    """
    assert 0.0 < test_frac < 1.0, f"test_frac must be in (0,1), got {test_frac}"
    n = int(collection.n_obs)
    n_test = int(round(n * test_frac))
    n_test = max(1, min(n_test, n - 1))

    labels = collection.obs[stratify_obs_key].to_numpy(dtype=np.int64)
    rng = np.random.default_rng(seed)

    test_idx = _stratified_sample_indices(labels, n_test, rng)
    mask = np.ones(n, dtype=bool)
    mask[test_idx] = False
    trainval_idx = np.where(mask)[0]

    trainval_view = collection[trainval_idx, :]
    test_view = collection[test_idx, :]

    print(f"Split (stratified): trainval={trainval_view.n_obs}, test={test_view.n_obs}")
    return trainval_view, test_view


def make_repeated_stratified_trainval_splits(
    trainval_view,
    n_splits: int = 5,
    val_frac_within_trainval: float = 0.08 / 0.91,
    seed: int = 42,
    stratify_obs_key: str = "cell_type_lvl4__code",
):
    """
    Creates 'n_splits' independent stratified shuffle splits over trainval_view.
    Each split yields (train_view, val_view) where:
      val size ≈ val_frac_within_trainval * trainval size,
      stratified by stratify_obs_key.

    This is the correct mechanism to achieve an overall 81/8/9 when test is 9%.
    """
    assert n_splits >= 1
    assert 0.0 < val_frac_within_trainval < 1.0

    n = int(trainval_view.n_obs)
    n_val = int(round(n * val_frac_within_trainval))
    n_val = max(1, min(n_val, n - 1))

    labels = trainval_view.obs[stratify_obs_key].to_numpy(dtype=np.int64)

    splits = []
    for k in range(n_splits):
        rng = np.random.default_rng(seed + 1000 * k + 17)
        val_idx = _stratified_sample_indices(labels, n_val, rng)

        mask = np.ones(n, dtype=bool)
        mask[val_idx] = False
        train_idx = np.where(mask)[0]

        splits.append((trainval_view[train_idx, :], trainval_view[val_idx, :]))

    return splits


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
    downsample: bool = False,
    frac: float = 0.1,
    debug_stratification: bool = True,
):
    shard_dir = Path(shard_dir)
    paths = sorted(shard_dir.glob("*.h5ad"))
    if not paths:
        raise FileNotFoundError(f"No .h5ad shards found in {shard_dir}")

    # Load all shards in memory
    adatas = [ad.read_h5ad(p) for p in paths]

    if filter_genes:
        big = ad.concat(
            adatas,
            join="outer",
            label="dataset",
            keys=[str(i) for i in range(len(adatas))],
            index_unique="shard-",
        )

        X = big.X
        if scipy.sparse.issparse(X):
            means = np.asarray(X.mean(axis=0)).ravel()
            means_sq = np.asarray(X.power(2).mean(axis=0)).ravel()
            stds = np.sqrt(means_sq - means**2)
        else:
            stds = X.std(axis=0)

        if max_genes > big.n_vars:
            max_genes = big.n_vars

        kept_idx = np.argsort(stds)[-max_genes:]
        kept_genes = big.var_names[kept_idx]

        adatas = [a[:, kept_genes] for a in adatas]

    if downsample:
        adatas_before = adatas
        adatas_after = [a.copy() for a in adatas]

        for adata in adatas_after:
            sample(adata, fraction=frac)

        if debug_stratification:
            debug_print_pre_post_downsample(adatas_before, adatas_after)

        adatas = adatas_after

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
    if filter_genes:
        return collection, kept_idx
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
    pin_m=False,
    **kwargs
):
    """Return (train_loader, val_loader, test_loader) PyTorch DataLoaders."""

    # Build collection and converter
    if "filter_genes" in kwargs.keys():
        collection, kept_idx = build_collection_from_shards(shard_dir, **kwargs)
    else:
        collection = build_collection_from_shards(shard_dir, **kwargs)
        kept_idx = None
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
            pin_memory=pin_m,
        )
        for split, ds in datasets.items()
    }

    print(
        f"✓ Train={len(datasets['train'])}, Val={len(datasets['val'])}, Test={len(datasets['test'])}"
    )
    if kept_idx is not None:
        return kept_idx, loaders["train"], loaders["val"], loaders["test"]
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
    num_workers=0,
    seed=1234,
    pin_m=False,
    stratify_level: str = "y4",   # y1/y2/y3/y4
    debug_stratification: bool = True,
    **kwargs
):
    """
    Implements:
      1) (Optional) downsampling (handled in build_collection_from_shards)
      2) Stratified 9% test holdout
      3) On remaining 91%: n_folds repeated stratified shuffle splits producing ~81/8 overall
    Returns: (kept_idx?, folds, test_loader)
    """
    level_to_obs_key = {
        "y1": "cell_type_lvl1__code",
        "y2": "cell_type_lvl2__code",
        "y3": "cell_type_lvl3__code",
        "y4": "cell_type_lvl4__code",
    }
    if stratify_level not in level_to_obs_key:
        raise ValueError(f"stratify_level must be one of {list(level_to_obs_key)}, got {stratify_level}")
    stratify_obs_key = level_to_obs_key[stratify_level]

    # Build collection and converter
    if "filter_genes" in kwargs.keys():
        collection, kept_idx = build_collection_from_shards(
            shard_dir, debug_stratification=debug_stratification, **kwargs
        )
    else:
        collection = build_collection_from_shards(
            shard_dir, debug_stratification=debug_stratification, **kwargs
        )
        kept_idx = None

    converter = make_converter(label_maps_path, one_hot_labels=one_hot)

    # --- 1) Stratified 9% test holdout ---
    trainval_view, test_view = split_collection_test_holdout_stratified(
        collection,
        test_frac=0.09,
        seed=seed,
        stratify_obs_key=stratify_obs_key,
    )

    # --- 2) Repeated stratified train/val splits on trainval ---
    # val fraction within trainval so that overall you get ~81/8/9
    val_frac_within_trainval = 0.08 / 0.91

    split_views = make_repeated_stratified_trainval_splits(
        trainval_view,
        n_splits=n_folds,
        val_frac_within_trainval=val_frac_within_trainval,
        seed=seed,
        stratify_obs_key=stratify_obs_key,
    )

    folds = []
    for i, (tr_view, va_view) in enumerate(split_views):
        tr_ds = AnnDatasetWrapper(tr_view, converter)
        va_ds = AnnDatasetWrapper(va_view, converter)

        tr_loader = DataLoader(
            tr_ds,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_m,
        )
        va_loader = DataLoader(
            va_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_m,
        )
        folds.append((tr_loader, va_loader))

    # Test loader
    test_ds = AnnDatasetWrapper(test_view, converter)
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_m,
    )

    # Print sizes
    total_n = int(collection.n_obs)
    test_n = int(test_view.n_obs)
    trainval_n = int(trainval_view.n_obs)
    # For reporting: the first split sizes are representative
    rep_train_n = int(split_views[0][0].n_obs)
    rep_val_n = int(split_views[0][1].n_obs)

    print(f"Split: train~{rep_train_n}, val~{rep_val_n}, test={test_n} (total={total_n})")
    print(f"✓ Train/Val repeated stratified splits (n°batches): {[(len(t), len(v)) for t, v in folds]}")

    # Debug per-loader stratification (optional)
    if debug_stratification:
        debug_print_loader_stratification(test_loader, name="TEST", max_batches=None)
        for i, (tr, va) in enumerate(folds):
            debug_print_loader_stratification(tr, name=f"SPLIT {i} TRAIN", max_batches=None)
            debug_print_loader_stratification(va, name=f"SPLIT {i} VAL", max_batches=None)

    if kept_idx is not None:
        return kept_idx, folds, test_loader
    return folds, test_loader
