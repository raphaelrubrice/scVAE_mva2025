import torch
from torch.utils.data import SubsetRandomSampler, DataLoader
from collections.abc import Iterable
from typing import Any
from sklearn.metrics import confusion_matrix
from sklearn.metrics.cluster import adjusted_rand_score
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import numpy as np
from scipy.optimize import linear_sum_assignment
from tqdm.auto import tqdm

# data type utils
def check_nonbatch(x: torch.Tensor) -> bool:
    """
    Check if a tensor is 1D or 2D with one row
    """
    if x.ndim == 1:
        return True
    elif x.ndim == 2 and x.shape[0] == 1:
        return True
    return False

def safe_T(x):
    """
    A replacement for x.T that works across dimensions.

    - 1D: returns the same tensor (no transpose).
    -2D: returns the usual matrix transpose.
    - higher-D tensors: reverses all dimensions (like .permute with reversed order).
    """
    if x.ndim == 1:
        return x
    elif x.ndim == 2:
        return x.t()
    else:
        return x.permute(*torch.arange(x.ndim - 1, -1, -1))

def split_or_validate_features(T, dims):
    """
    Utility to split a tensor into features or 
    validate an iterable of tensors.
    """
    # Case 1: single tensor
    if isinstance(T, torch.Tensor):
        shape = T.size()
        if len(shape) == 2:
            B, D = shape
            N = None
        elif len(shape) == 3:
            B, N, D = shape
        if D != sum(dims):
            raise ValueError(f"Tensor has {D} features, expected {sum(dims)}")
        out = []
        start = 0
        for d in dims:
            if N is None:
                out.append(T[:, start:start+d])
            else:
                out.append(T[:, :, start:start+d])
            start += d
        return tuple(out)

    # Case 2: iterable of tensors
    elif isinstance(T, Iterable):
        T = tuple(T)
        if len(T) != len(dims):
            raise ValueError(f"Iterable length {len(T)} does not match dims length {len(dims)}")
        for t, d in zip(T, dims):
            if not isinstance(t, torch.Tensor):
                raise TypeError("All elements must be torch.Tensors")
            if t.ndim != 2 or t.shape[1] != d:
                raise ValueError(f"Tensor has shape {t.shape}, expected (B,{d})")
        return T

    else:
        raise TypeError("Input must be a torch.Tensor or an iterable of tensors")
    
def get_dim(x):
    """
    Return the 'size' of an element:
      - If x is int, or a 0‑D/1‑element tensor, return 1
      - else return its dimension along axis 1
    """
    if isinstance(x, int):
        return 1
    if isinstance(x, torch.Tensor):
        # scalar tensor (0‑D) or tensor with a single element
        if x.ndim == 0 or x.numel() == 1:
            return 1
        return x.size(1)
    # fallback: try to get shape attribute
    if hasattr(x, "shape"):
        return x.shape[1]
    raise TypeError(f"Unsupported type {type(x)}")

def _to_tensor_dict(param_dict: dict[str, Any]
                    ) -> dict[str, torch.Tensor]:
    tensor_dict = {}
    for key, value in param_dict.items():
        if isinstance(value, torch.Tensor):
            tensor_dict[key] = value.clone()
        else:
            try:
                tensor_dict[key] = torch.tensor(value, dtype=torch.float32)
            except Exception as e:
                raise ValueError(f"Cannot convert key '{key}' to tensor: {e}")
    return tensor_dict


# evaluation utils 
def match_labels(label_true: np.ndarray[int], label_pred: np.ndarray[int]) -> np.ndarray[int]:
    """
    Hungarian algo to identify a mapping between two clustering naming conventions
    """
    cm: np.ndarray[int] = confusion_matrix(label_true, label_pred)
    row_ind, col_ind = linear_sum_assignment(-cm)

    mapping: dict[int: int] = {col: row for row, col in zip(row_ind, col_ind)}

    label_pred_matched: np.ndarray[int] = np.vectorize(lambda x: mapping.get(x, x))(label_pred)

    return label_pred_matched

def _to_ordinal_labels(y: torch.Tensor) -> torch.Tensor:
    """
    Accept y as:
      - (B,) or (B,1) integer labels
      - (B,K) one-hot or probabilities
    Return: (B,) long labels
    """
    y = y.detach()
    if y.ndim == 0:
        # scalar (should not happen for batches, but be safe)
        return y.view(1).long()
    if y.ndim == 1:
        return y.long()
    if y.ndim == 2 and y.size(1) == 1:
        return y[:, 0].long()
    # (B,K) or (...,K) => argmax on last dim
    return torch.argmax(y, dim=-1).long()


def _infer_pbmc_loader(first_batch) -> bool:
    """
    PBMC loader is dict-based and has y1..y4 keys (at least y1).
    """
    if not isinstance(first_batch, dict):
        return False
    return any(k.startswith("y") for k in first_batch.keys())


def _count_dataset_levels(first_batch) -> int:
    """
    - PBMC: count y* keys (y1..y4)
    - classic: batch is tuple/list -> labels are batch[1:], so K = len(batch)-1
    """
    if _infer_pbmc_loader(first_batch):
        ys = [k for k in first_batch.keys() if k.startswith("y")]
        # Expect y1..y4 but allow missing intermediate
        # We count the keys present.
        return len(ys)
    # classic
    if isinstance(first_batch, (tuple, list)):
        return max(0, len(first_batch) - 1)
    raise ValueError(f"Unsupported batch type: {type(first_batch)}")


def _map_dataset_level_to_model_level(dataset_lvl_1idx: int, model_n_levels: int, dataset_K: int) -> int:
    """
    Map dataset level in [1..K] onto model level in [1..N], aligning to the last K levels.

    Example:
      N=2, K=1: dataset lvl 1 -> model lvl 2
      N=3, K=2: dataset lvl 1 -> model lvl 2, dataset lvl 2 -> model lvl 3
      N=3, K=3: identity mapping
    """
    assert 1 <= dataset_lvl_1idx <= dataset_K
    offset = model_n_levels - dataset_K
    if offset < 0:
        # dataset has more levels than model; best effort: clip to model levels
        offset = 0
    model_lvl = offset + dataset_lvl_1idx
    # clip just in case
    return int(max(1, min(model_n_levels, model_lvl)))

def _dbg(msg: str, enabled: bool):
    if enabled:
        print(msg)

def compute_radj_pbmc(
    model,
    loader,
    dataset_K: int,
    model_n_levels: int,
    debug: bool = False,
):
    """
    PBMC loader: batch is dict with X and labels y1..y4 (present subset).
    dataset_K: number of y* keys present (levels)
    model_n_levels: model.n_levels
    """
    model.eval()
    device = next(model.parameters()).device

    true_by_lvl = {lvl: [] for lvl in range(1, dataset_K + 1)}
    pred_by_lvl = {lvl: [] for lvl in range(1, dataset_K + 1)}

    # ---- debug: print dataset ↔ model level mapping
    _dbg(
        f"[PBMC][DEBUG] dataset_K={dataset_K}, model_n_levels={model_n_levels}",
        debug,
    )
    for d_lvl in range(1, dataset_K + 1):
        m_lvl = _map_dataset_level_to_model_level(d_lvl, model_n_levels, dataset_K)
        _dbg(
            f"[PBMC][DEBUG] dataset level {d_lvl} -> model level {m_lvl}",
            debug,
        )

    with torch.no_grad():
        for b_idx, batch in enumerate(loader):
            x = batch["X"][:, 0, :].to(device)

            if b_idx == 0:
                _dbg(
                    f"[PBMC][DEBUG] batch keys = {list(batch.keys())}",
                    debug,
                )
                _dbg(
                    f"[PBMC][DEBUG] X.shape = {tuple(x.shape)}",
                    debug,
                )

            for d_lvl in range(1, dataset_K + 1):
                y_key = f"y{d_lvl}"
                y = batch.get(y_key, None)

                if b_idx == 0:
                    if y is None:
                        _dbg(
                            f"[PBMC][DEBUG] y{d_lvl} is None",
                            debug,
                        )
                    else:
                        _dbg(
                            f"[PBMC][DEBUG] y{d_lvl}.shape = {tuple(y.shape)}",
                            debug,
                        )

                if y is None:
                    true_by_lvl[d_lvl].append(None)
                else:
                    true_by_lvl[d_lvl].append(_to_ordinal_labels(y))

                m_lvl = _map_dataset_level_to_model_level(
                    d_lvl, model_n_levels, dataset_K
                )

                try:
                    pred = model.cluster_input(x, at_level=m_lvl - 1)
                except TypeError:
                    pred = model.cluster_input(x)

                if b_idx == 0:
                    _dbg(
                        f"[PBMC][DEBUG] pred clusters (dataset lvl {d_lvl}, model lvl {m_lvl}) "
                        f"shape = {tuple(pred.shape)} | unique={pred.unique().tolist()}",
                        debug,
                    )

                pred_by_lvl[d_lvl].append(pred)

    radj = {}
    for d_lvl in range(1, dataset_K + 1):
        labels_list = [t for t in true_by_lvl[d_lvl] if t is not None]
        preds_list = [t for t in pred_by_lvl[d_lvl] if t is not None]

        if len(labels_list) == 0 or len(preds_list) == 0:
            _dbg(
                f"[PBMC][DEBUG] Dropping dataset level {d_lvl} (missing labels or preds)",
                debug,
            )
            continue

        y_true = torch.cat(labels_list, dim=0).cpu().numpy().ravel()
        y_pred = torch.cat(preds_list, dim=0).cpu().numpy().ravel()

        _dbg(
            f"[PBMC][DEBUG] Final ARI eval lvl {d_lvl}: "
            f"y_true unique={np.unique(y_true)}, "
            f"y_pred unique={np.unique(y_pred)}",
            debug,
        )

        radj[d_lvl] = [adjusted_rand_score(y_true, y_pred)]

    return radj


def compute_radj_classic(
    model,
    loader,
    dataset_K: int,
    model_n_levels: int,
    debug: bool = False,
):
    """
    Classic loader: batch is (X, y1, y2, ..., yK)
    """
    model.eval()
    device = next(model.parameters()).device

    true_by_lvl = {lvl: [] for lvl in range(1, dataset_K + 1)}
    pred_by_lvl = {lvl: [] for lvl in range(1, dataset_K + 1)}

    _dbg(
        f"[CLASSIC][DEBUG] dataset_K={dataset_K}, model_n_levels={model_n_levels}",
        debug,
    )
    for d_lvl in range(1, dataset_K + 1):
        m_lvl = _map_dataset_level_to_model_level(d_lvl, model_n_levels, dataset_K)
        _dbg(
            f"[CLASSIC][DEBUG] dataset level {d_lvl} -> model level {m_lvl}",
            debug,
        )

    with torch.no_grad():
        for b_idx, batch in enumerate(loader):
            x = batch[0].to(device)

            if b_idx == 0:
                _dbg(
                    f"[CLASSIC][DEBUG] batch len = {len(batch)}",
                    debug,
                )
                _dbg(
                    f"[CLASSIC][DEBUG] X.shape = {tuple(x.shape)}",
                    debug,
                )

            for d_lvl in range(1, dataset_K + 1):
                y = batch[d_lvl]

                if b_idx == 0:
                    if y is None:
                        _dbg(
                            f"[CLASSIC][DEBUG] y{d_lvl} is None",
                            debug,
                        )
                    else:
                        _dbg(
                            f"[CLASSIC][DEBUG] y{d_lvl}.shape = {tuple(y.shape)}",
                            debug,
                        )

                if y is None:
                    true_by_lvl[d_lvl].append(None)
                else:
                    true_by_lvl[d_lvl].append(_to_ordinal_labels(y))

                m_lvl = _map_dataset_level_to_model_level(
                    d_lvl, model_n_levels, dataset_K
                )

                try:
                    pred = model.cluster_input(x, at_level=m_lvl - 1)
                except TypeError:
                    pred = model.cluster_input(x)

                if b_idx == 0:
                    _dbg(
                        f"[CLASSIC][DEBUG] pred clusters (dataset lvl {d_lvl}, model lvl {m_lvl}) "
                        f"shape={tuple(pred.shape)} | unique={pred.unique().tolist()}",
                        debug,
                    )

                pred_by_lvl[d_lvl].append(pred)

    radj = {}
    for d_lvl in range(1, dataset_K + 1):
        labels_list = [t for t in true_by_lvl[d_lvl] if t is not None]
        preds_list = [t for t in pred_by_lvl[d_lvl] if t is not None]

        if len(labels_list) == 0 or len(preds_list) == 0:
            _dbg(
                f"[CLASSIC][DEBUG] Dropping dataset level {d_lvl} (missing labels or preds)",
                debug,
            )
            continue

        y_true = torch.cat(labels_list, dim=0).cpu().numpy().ravel()
        y_pred = torch.cat(preds_list, dim=0).cpu().numpy().ravel()

        _dbg(
            f"[CLASSIC][DEBUG] Final ARI eval lvl {d_lvl}: "
            f"y_true unique={np.unique(y_true)}, "
            f"y_pred unique={np.unique(y_pred)}",
            debug,
        )

        radj[d_lvl] = [adjusted_rand_score(y_true, y_pred)]

    return radj



# def compute_CV_radj(cv_models: list, 
#                     test_loader: DataLoader,
#                     cv_val_loaders: list = None,
#                    ):
#     """
#     Computes Cross-Validated ARI per hierarchical level.

#     Returns:
#       - if cv_val_loaders is None:
#           test_radj: dict[level -> list of ARIs across folds]
#       - else:
#           overall_radj: dict[level -> (mean, std) over (val + test)]
#           val_radj_stats: dict[level -> (mean, std) over val folds]
#           test_radj: dict[level -> list of ARIs across folds]
#     """
#     # Infer number of levels
#     batch = next(iter(test_loader))
#     if isinstance(batch, dict):
#         n_levels = len([key for key in batch.keys() if 'y' in key])
#     else:
#         n_levels = cv_models[0].n_levels
#     level_list = list(range(1, n_levels + 1))
    
#     # Initialize per-level ARI lists for test
#     test_radj = {lvl: [] for lvl in level_list}

#     # Compute ARI for each model/fold on the test loader
#     for model in cv_models:
#         dset_radj = compute_radj(model, test_loader, n_levels)  # dict[level -> [ari,...]]
#         for lvl, aris in dset_radj.items():
#             test_radj[lvl].extend(aris)

#     if cv_val_loaders is not None:
#         # Per-level ARI lists for validation
#         val_radj = {lvl: [] for lvl in level_list}

#         for model, loader in zip(cv_models, cv_val_loaders):
#             dset_radj = compute_radj(model, loader, n_levels)
#             for lvl, aris in dset_radj.items():
#                 val_radj[lvl].extend(aris)

#         # Compute per-level stats
#         val_radj_stats = {}
#         overall_radj = {}
#         for lvl in level_list:
#             val_arr = np.asarray(val_radj[lvl], dtype=float)
#             test_arr = np.asarray(test_radj[lvl], dtype=float)
#             all_arr = np.concatenate([val_arr, test_arr])

#             val_radj_stats[lvl] = (val_arr.mean(), val_arr.std())
#             overall_radj[lvl] = (all_arr.mean(), all_arr.std())

#         return overall_radj, val_radj_stats, test_radj

#     return test_radj

def compute_CV_radj(cv_models: list,
                    test_loader: DataLoader,
                    cv_val_loaders: list | None = None):
    """
    Cross-validated ARI (Radj) per *dataset level* (1..K),
    where dataset level i maps to model level (N-K+i) if K < N.

    PBMC loader: batch is dict with keys X and y1..y4.
    Classic loader: batch is tuple where batch[0]=X and batch[1:]=labels.
    """
    first_batch = next(iter(test_loader))
    is_pbmc = _infer_pbmc_loader(first_batch)
    dataset_K = _count_dataset_levels(first_batch)

    level_list = list(range(1, dataset_K + 1))
    test_radj = {lvl: [] for lvl in level_list}

    for model in cv_models:
        model_n_levels = getattr(model, "n_levels", 1)
        if is_pbmc:
            dset_radj = compute_radj_pbmc(model, test_loader, dataset_K, model_n_levels)
        else:
            dset_radj = compute_radj_classic(model, test_loader, dataset_K, model_n_levels)

        for lvl, aris in dset_radj.items():
            test_radj[lvl].extend(aris)

    if cv_val_loaders is not None:
        val_radj = {lvl: [] for lvl in level_list}

        for model, loader in zip(cv_models, cv_val_loaders):
            first_val = next(iter(loader))
            is_pbmc_val = _infer_pbmc_loader(first_val)
            dataset_K_val = _count_dataset_levels(first_val)

            # We assume val/test have same label scheme; if not, handle gracefully by intersection.
            model_n_levels = getattr(model, "n_levels", 1)

            if is_pbmc_val:
                dset_radj = compute_radj_pbmc(model, loader, dataset_K_val, model_n_levels)
            else:
                dset_radj = compute_radj_classic(model, loader, dataset_K_val, model_n_levels)

            for lvl, aris in dset_radj.items():
                if lvl in val_radj:
                    val_radj[lvl].extend(aris)

        val_radj_stats = {}
        overall_radj = {}
        for lvl in level_list:
            val_arr = np.asarray(val_radj[lvl], dtype=float)
            test_arr = np.asarray(test_radj[lvl], dtype=float)

            # avoid nan if empty
            if val_arr.size == 0 and test_arr.size == 0:
                val_radj_stats[lvl] = (np.nan, np.nan)
                overall_radj[lvl] = (np.nan, np.nan)
                continue

            all_arr = np.concatenate([val_arr, test_arr]) if val_arr.size else test_arr
            val_radj_stats[lvl] = (val_arr.mean() if val_arr.size else np.nan,
                                   val_arr.std() if val_arr.size else np.nan)
            overall_radj[lvl] = (all_arr.mean(), all_arr.std())

        return overall_radj, val_radj_stats, test_radj

    return test_radj

def compute_ll(model, loader):
    ll_list = []
    model.eval()
    device = next(model.parameters()).device
    with torch.no_grad():
        for batch in tqdm(loader, total=len(loader)):
            try:
                x = batch["X"][:, 0, :]
            except Exception:
                x = batch[0]
            
            x = x.to(device)
            per_sample_ll = model.iwae(x)
            ll_list.append(per_sample_ll)
    ll_mean = torch.cat(ll_list, dim=0).mean(dim=0)
    return ll_mean.item()

def compute_CV_ll(cv_models: list, 
                  test_loader: DataLoader,
                  cv_val_loaders: list = None, 
                ):
    """
    Computes Cross-Validated Total marginal Log-likelihood
    using the IWAE estimator.
    """
    test_ll = {i: [] for i in range(len(cv_models))}

    for i,model in enumerate(cv_models):
        test_ll[i] = compute_ll(model, test_loader)

    test_ll = list(test_ll.values())
    
    if cv_val_loaders is not None:
        val_ll = {i:[] for i in range(1,5)}

        for model, loader in zip(cv_models, cv_val_loaders):
            val_ll = compute_ll(model, loader)

        overall_ll = {i:(np.mean(lls + test_ll[i]),np.std(lls+ test_ll[i])) 
                    for i,lls in val_ll.items()}

        val_ll = {i:(np.mean(lls),np.std(lls)) 
                    for i,lls in val_ll.items()}
        return overall_ll, val_ll, test_ll
    return test_ll

def initialize_gmm_params(loader, n_components, latent_dim, device):
    """
    Computes data-dependent initialization for priors using PCA + KMeans.
    """
    print(f"Initializing priors with PCA + KMeans ({n_components} components)...")
    
    # 1. Collect all data (careful with memory, subsample if >100k cells)
    all_x = []
    for batch in loader:
        # Handle dict or tuple batches
        x = batch["X"][:, 0, :] if isinstance(batch, dict) else batch[0]
        all_x.append(x)
    X_full = torch.cat(all_x, dim=0).cpu().numpy()

    # 2. PCA (Reduce noise)
    # Reducing to latent_dim is aggressive but aligns the initialization perfectly
    pca = PCA(n_components=latent_dim) 
    X_pca = pca.fit_transform(X_full)

    # 3. K-Means
    kmeans = KMeans(n_clusters=n_components, n_init=10, random_state=42)
    y_pred = kmeans.fit_predict(X_pca)
    
    # 4. Compute Statistics per Cluster
    cluster_means = []
    cluster_stds = []
    cluster_probs = []
    
    eps = 1e-6 # Stability
    
    for k in range(n_components):
        # Select data for cluster k
        X_k = X_pca[y_pred == k]
        
        # Proportions (with smoothing)
        n_k = len(X_k)
        prob = (n_k + 1) / (len(X_full) + n_components) # Add-1 smoothing
        cluster_probs.append(prob)
        
        # Mean & Std in the PCA-Latent space
        if n_k > 1:
            mean_k = X_k.mean(axis=0)
            std_k = X_k.std(axis=0) + eps # Avoid zero std
        else:
            # Fallback for empty clusters
            mean_k = np.zeros(latent_dim)
            std_k = np.ones(latent_dim)
            
        cluster_means.append(torch.tensor(mean_k))
        cluster_stds.append(torch.tensor(std_k))

    # Stack into tensors
    # Shape: (1, n_components * latent_dim) or (n_components, latent_dim) 
    
    # Categorical Params
    cat_params = {"probs": torch.tensor(cluster_probs, device=device).unsqueeze(1)}
    
    # Latent Params (Mean and Std)
    mu_tensor = torch.stack(cluster_means).to(device) # (K, D)
    std_tensor = torch.stack(cluster_stds).to(device) # (K, D)
    
    latent_params = {"mu": mu_tensor, "std": std_tensor}
    
    return cat_params, latent_params

def make_figure(config):
    # key = name of the model and posterior
    # value = list of cross validated model paths to load

    # dict of results used later to build a dataframe with columns:
    # Model, Posterior latent, IWAE, Radj
    # for each model 
    # compute LL across the cv (mean and std)
    # compute ARI (mean and std)
    # add it to the dictionary of results

    # return the dataframe of results
    pass
