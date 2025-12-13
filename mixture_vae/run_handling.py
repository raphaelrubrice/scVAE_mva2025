import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import os, sys
from torch.utils.data import DataLoader, TensorDataset, Subset
from sklearn.model_selection import KFold
from tqdm.auto import tqdm

# To ensure the custom package is found
path_to_repo = "/".join(os.path.abspath(__file__).split("/")[:-2])
if path_to_repo not in sys.path:
    sys.path.append(path_to_repo)

from mixture_vae.mvae import MixtureVAE, ind_MoMVAE, MoMixVAE
from mixture_vae.distributions import NormalDistribution, UniformDistribution, NegativeBinomial, Poisson, Student, CategoricalDistribution
from mixture_vae.training import training_mvae, training_momixvae
from mixture_vae.utils import *
from mixture_vae.viz import plot_loss_components, plot_latent

MODELS = {
    "MixtureVAE": MixtureVAE, 
    "ind_MoMVAE": ind_MoMVAE, 
    "MoMixVAE": MoMixVAE
}

DISTRIBUTIONS = {
    "Normal": NormalDistribution, 
    "Uniform": UniformDistribution, 
    "NegativeBinomial": NegativeBinomial,
    "Poisson": Poisson, 
    "Student": Student,
    "Categorical": CategoricalDistribution
}

def build_distribution(dist_name: str, params: dict, device='cpu'):
    """Factory to create distribution objects from config strings."""
    if dist_name not in DISTRIBUTIONS:
        raise ValueError(f"Distribution {dist_name} not supported.")
    
    formatted_params = {}
    for k, v in params.items():
        if isinstance(v, (list, float, int)):
            t = torch.tensor(v)
            if t.ndim == 0: t = t.unsqueeze(0)
            formatted_params[k] = t.to(device)
        else:
            formatted_params[k] = v.to(device)
            
    return DISTRIBUTIONS[dist_name](formatted_params)

# def compute_loader_mean(loader, device):
#     """Iterates over the loader to compute the mean of the features (dim 0)."""
#     sum_x = None
#     total_samples = 0
    
#     for batch in loader:
#         try:
#             x = batch["X"][:, 0, :]
#         except Exception:
#             x = batch[0]
        
#         x = x.to(device)
#         if sum_x is None:
#             sum_x = torch.zeros(x.shape[1], device=device)
            
#         sum_x += x.sum(dim=0)
#         total_samples += x.size(0)
        
#     return (sum_x / total_samples).reshape(1, -1)



def get_prior_parameters(dist_name, dim, device, loader=None):
    """
    Helper to generate initialization parameters based on distribution type.
    """
    params = {}
    
    if dist_name == "Normal":
        params["mu"] = torch.zeros((1, dim), device=device)
        params["std"] = torch.ones((1, dim), device=device)
        
    elif dist_name == "Student":
        params["df"] = torch.ones((1, dim), device=device) * 10.0 
        params["mu"] = torch.zeros((1, dim), device=device)
        params["scale"] = torch.ones((1, dim), device=device)
    
    elif dist_name == "NegativeBinomial":
        if loader is None:
            raise ValueError("Loader required to initialize NegativeBinomial")
        params = init_nb_params_mom(loader, device)
        
    elif dist_name == "Poisson":
        params["rate"] = torch.ones((1, dim), device=device)

    elif dist_name == "Uniform":
        # Used for the Categorical Prior (Classes)
        # Initializes a Uniform between 0 and 1 by default
        params["a"] = torch.zeros((1, dim), device=device)
        params["b"] = torch.ones((1, dim), device=device)
    else:
        print(f"Warning: No specific init logic for {dist_name}, using zeros/ones.")
        params["param1"] = torch.zeros((1, dim), device=device)
    
    return params

def build_categorical_prior(dist_name, n_components, device, cat_params=None):
    """Helper specifically for building the prior on Y (clusters)."""
    if dist_name == "Uniform":
        # 'dim' here implies the number of components/clusters
        params = get_prior_parameters(dist_name, n_components, device)
    else:
        assert cat_params is not None, f"When not using Uniform categorical, you must provide cat_params"
        params = cat_params
    return build_distribution(dist_name, params, device)

def get_continuous_priors(config, train_loader, input_dim, latent_dim, device, latent_prior_init=None):
    p_lat_name = config.get("prior_latent_dist", "Normal")
    p_in_name = config.get("prior_input_dist", "NegativeBinomial")
    post_lat_name = config.get("posterior_latent_dist", "Normal")

    # Latent Prior (allow data-dependent init)
    if latent_prior_init is not None:
        p_lat_params = latent_prior_init
    else:
        p_lat_params = get_prior_parameters(p_lat_name, latent_dim, device)
    prior_latent = build_distribution(p_lat_name, p_lat_params, device)

    # Input Prior (Decoder)
    p_in_params = get_prior_parameters(p_in_name, input_dim, device, loader=train_loader)
    prior_input = build_distribution(p_in_name, p_in_params, device)

    # Latent Posterior
    post_lat_params = get_prior_parameters(post_lat_name, latent_dim, device)
    posterior_latent = build_distribution(post_lat_name, post_lat_params, device)

    return prior_latent, prior_input, posterior_latent

def instantiate_model(config, train_loader, device):
    model_type  = config["model_type"]
    input_dim   = config["input_dim"]
    hidden_dim  = config["hidden_dim"]
    latent_dim  = config["latent_dim"]

    # Names of distributions
    p_lat_name     = config.get("prior_latent_dist", "Normal")
    p_in_name      = config.get("prior_input_dist", "NegativeBinomial")
    post_lat_name  = config.get("posterior_latent_dist", "Normal")

    # Categorical prior distribution name
    # Recommended: "Categorical" (your new class). If "Uniform", fallback to uniform.
    cat_dist_name  = config.get("prior_categorical_dist", "Categorical")

    # ------------------------------------------------------------
    # Helper: build input prior once (shared across models/levels)
    # ------------------------------------------------------------
    p_in_params = get_prior_parameters(p_in_name, input_dim, device, loader=train_loader)
    prior_input = build_distribution(p_in_name, p_in_params, device)

    # ------------------------------------------------------------
    # Helper: build a posterior latent distribution (fresh instance)
    # ------------------------------------------------------------
    def _make_posterior_latent():
        post_lat_params = get_prior_parameters(post_lat_name, latent_dim, device)
        return build_distribution(post_lat_name, post_lat_params, device)

    # ------------------------------------------------------------
    # Helper: per-(sub)mixture init for categorical + latent prior
    # ------------------------------------------------------------
    def _init_cat_and_latent_priors_for_K(K: int):
        """
        Returns:
          prior_cat: distribution instance for p(y)
          prior_lat: distribution instance for p(z|y) with (K,D) parameters
        """
        # Data-dependent init in latent space
        latent_params = initialize_gmm_params(
            train_loader, n_components=K, latent_dim=latent_dim, device=device
        )
        cat_params = init_categorical_uniform(K, device)

        # Categorical prior
        if cat_dist_name == "Uniform":
            prior_cat = build_categorical_prior("Uniform", K, device)
        else:
            # expects something like {"probs": (K,)}
            prior_cat = build_distribution(cat_dist_name, cat_params, device)

        if p_lat_name == "Student":
            latent_params = initialize_student_prior_params_from_gmm(latent_params, device, df0=10)
        # Latent prior (component-specific): expects {"mu": (K,D), "std": (K,D)} for Normal
        prior_lat = build_distribution(p_lat_name, latent_params, device)

        return prior_cat, prior_lat

    # ============================================================
    # 1) MixtureVAE
    # ============================================================
    if model_type == "MixtureVAE":
        n_components = config["n_components"]

        # Data-dependent init for p(y) and p(z|y)
        prior_cat, prior_latent = _init_cat_and_latent_priors_for_K(n_components)

        # Posterior q(z|x) (single)
        posterior_latent = _make_posterior_latent()

        model = MixtureVAE(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            n_components=n_components,
            n_layers=config.get("n_layers", 1),
            prior_latent=prior_latent,
            prior_input=prior_input,
            prior_categorical=prior_cat,
            posterior_latent=posterior_latent,
            act_func=config.get("act_func", nn.ReLU()),
            dropout=config.get("dropout", 0.0),
            norm_layer=config.get("norm_layer", None),
        )
        return model

    # ============================================================
    # 2) ind_MoMVAE (independent branches)
    # ============================================================
    if model_type == "ind_MoMVAE":
        # Accept either "branch_components" or "hierarchy_components"
        branches = config.get("branch_components", config.get("hierarchy_components"))
        if branches is None:
            raise ValueError("ind_MoMVAE requires 'branch_components' or 'hierarchy_components'.")

        PARAMS = []
        for n_c in branches:
            # Per-branch init
            prior_cat, prior_latent = _init_cat_and_latent_priors_for_K(n_c)
            posterior_latent = _make_posterior_latent()

            branch_config = {
                "input_dim": input_dim,
                "hidden_dim": hidden_dim,
                "n_components": n_c,
                "n_layers": config.get("n_layers", 1),
                "prior_latent": prior_latent,
                "prior_input": prior_input,
                "prior_categorical": prior_cat,
                "posterior_latent": posterior_latent,
                "act_func": config.get("act_func", nn.ReLU()),
                "dropout": config.get("dropout", 0.0),
                "norm_layer":config.get("norm_layer", None),
            }
            PARAMS.append(branch_config)

        model = ind_MoMVAE(PARAMS=PARAMS)
        return model

    # ============================================================
    # 3) MoMixVAE (hierarchy levels)
    #    Assumption: MoMixVAE accepts all_prior_latent=... (list)
    # ============================================================
    if model_type == "MoMixVAE":
        hierarchy = config.get("hierarchy_components", None)
        if hierarchy is None:
            raise ValueError("MoMixVAE requires 'hierarchy_components'.")

        all_prior_cat = []
        all_prior_lat = []
        all_post_lat  = []

        for n_c in hierarchy:
            prior_cat, prior_latent = _init_cat_and_latent_priors_for_K(n_c)
            all_prior_cat.append(prior_cat)
            all_prior_lat.append(prior_latent)
            all_post_lat.append(_make_posterior_latent())

        model = MoMixVAE(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            hierarchy_components=hierarchy,
            n_layers=config.get("n_layers", 2),
            prior_input=prior_input,
            all_prior_categorical=all_prior_cat,
            all_prior_latent=all_prior_lat,          # <--- you said assume this exists
            all_posterior_latent=all_post_lat,
            act_func=config.get("act_func", nn.ReLU()),
            dropout=config.get("dropout", 0.0),
            norm_layer=config.get("norm_layer", None),
        )
        return model

    raise ValueError(f"Unknown model_type: {model_type}")

def run_training(config: dict, 
                 train_loader: DataLoader, 
                 val_loader: DataLoader, 
                 plot_losses=True,
                 plot_latent_space=False,
                 **kwargs):
    """
    Orchestrates the training process based on a configuration dictionary.
    
    Args:
        config: Dict containing model hyperparams and training settings.
        train_loader: PyTorch DataLoader for training.
        val_loader: PyTorch DataLoader for validation.
    """
    run_tag = config['run_tag']
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running experiment: {run_tag} on {device}")
    
    
    # 1. Instantiate Model
    # We pass X_train_tensor to calculate data-dependent priors (like NB mean)
    model = instantiate_model(config, train_loader, device)
    model.to(device)

    # 2. Setup Optimizer
    lr = config.get("lr", 1e-3)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # 3. Setup Training Parameters
    epochs = config.get("epochs", 50)
    beta_kl = config.get("beta_kl", 1.0)
    reg_marg = config.get("reg_marg", 10)
    warmup = config.get("warmup", None)
    patience = config.get("patience", max(5,int(0.1*epochs)))
    tol = config.get("tol", 1e-3)
    save_path = config.get("save_path", f"./model_{config['model_type']}.ckpt")
    track_clusters = config.get("track_clusters", False)

    # 4. Select and Run Training Function
    model_type_str = config["model_type"]
    
    if model_type_str == "MixtureVAE":
        # model_type=0 corresponds to MixtureVAE in training_mvae
        model, losses, parts, clusters, betas = training_mvae(
            train_loader, val_loader, model, optimizer,
            epochs=epochs, beta_kl=beta_kl, reg_marg=reg_marg, warmup=warmup, 
            patience=patience, tol=tol, 
            model_type=0, # 0 for MixtureVAE
            track_clusters=track_clusters, save_path=save_path,
            **kwargs
        )
        
    elif model_type_str == "ind_MoMVAE":
        # model_type=1 corresponds to ind_MoMVAE in training_mvae
        model, losses, parts, clusters, betas = training_mvae(
            train_loader, val_loader, model, optimizer,
            epochs=epochs, beta_kl=beta_kl, reg_marg=reg_marg, warmup=warmup, 
            patience=patience, tol=tol, 
            model_type=1, # 1 for ind_MoMVAE
            track_clusters=track_clusters, save_path=save_path,
            **kwargs
        )
        
    elif model_type_str == "MoMixVAE":
        model, losses, parts, clusters, betas = training_momixvae(
            train_loader, val_loader, model, optimizer,
            epochs=epochs, beta_kl=beta_kl, reg_marg=reg_marg, warmup=warmup, 
            patience=patience, tol=tol, 
            track_clusters=track_clusters, save_path=save_path,
            **kwargs
        )
        
    else:
        raise ValueError(f"No training function mapped for {model_type_str}")

    print(f"Training complete. Model saved to {save_path}")
    
    results = {
        "model": model,
        "losses": losses,
        "parts": parts,
        "clusters": clusters,
        "betas": betas
    }
    
    model_parent_folder = "/".join(config["save_path"].split("/")[:-2])
    if plot_losses:
        plot_loss_components(results["parts"]["train"], 
                         results["parts"]["val"], 
                         results["betas"], 
                         title=f"Loss Breakdown - {run_tag}",
                         save_path=model_parent_folder + f"/Plots/losses_{run_tag}.pdf")

    if plot_latent_space:
        plot_latent(model, 
                val_loader,
                level=-1,
                true_labels=True,
                title="Latent Space",
                save_path=model_parent_folder + f"/Plots/true_latent_{run_tag}.pdf")
        plot_latent(model, 
                val_loader,
                level=-1,
                true_labels=False,
                title="Latent Space",
                save_path=model_parent_folder + f"/Plots/model_latent_{run_tag}.pdf")
    return results

def run_cv(config, folds, test_loader=None, 
           plot_losses=True, plot_latent_space=False, 
           in_folder=True, **kwargs):
    """
    Runs cross-validation training over folds, collects metrics,
    and returns trained models + performance DataFrame (if test set provided).
    """
    results_cv = []

    # ----- TRAINING LOOP OVER FOLDS -----
    for fold in tqdm(list(range(len(folds)))):
        train_loader, val_loader = folds[fold]

        # Copy config so each fold has its own run_tag and save_path
        config_copy = {key: val for key, val in config.items()}
        config_copy["run_tag"] = f'cv{fold}_' + config["run_tag"]

        # Adjust save path
        if isinstance(config["save_path"], str):
            save_path = config["save_path"]
            parts = save_path.split("/")
            if in_folder:
                save_path = "/".join(parts[:-1]) + f'/{config["run_tag"]}/Models/cv{fold}_' + parts[-1]
            else:
                save_path = "/".join(parts[:-1]) + f'/cv{fold}_' + parts[-1]
            print(save_path)
            config_copy["save_path"] = save_path

        # Train model for this fold
        results_cv.append(
            run_training(
                config_copy,
                train_loader,
                val_loader,
                plot_losses=plot_losses,
                plot_latent_space=plot_latent_space,
                **kwargs
            )
        )

    # ----- IF NO TEST LOADER, RETURN ONLY TRAINED MODELS -----
    if test_loader is None:
        return results_cv

    # ----- EVALUATION ON TEST CV MODELS -----
    cv_models = [results_cv[i]["model"] for i in range(len(folds))]

    # Log-likelihood (IWAE)
    cv_ll = compute_CV_ll(cv_models, test_loader)

    # ARI per hierarchical level
    cv_radj = compute_CV_radj(cv_models, test_loader)

    # ----- METRIC AGGREGATION -----
    metric_res = {
        "Model": [config["model_type"]],
        "Prior latent": [config["prior_latent_dist"]],
        "Mean IWAE": [np.mean(cv_ll)],
        "Std IWAE":  [np.std(cv_ll)],
    }

    # Add Radj metrics per hierarchical level
    for level in cv_radj.keys():
        metric_res[f"Mean Radj lvl.{level}"] = [np.mean(cv_radj[level])]
        metric_res[f"Std Radj lvl.{level}"]  = [np.std(cv_radj[level])]
    
    metric_df = pd.DataFrame(metric_res)
    print(metric_df)
    return results_cv, metric_df


def create_folds(dataset: TensorDataset, 
                 n_splits: int = 5, 
                 batch_size: int = 64, 
                 shuffle: bool = False):
    """
    Splits a TensorDataset into K folds and returns a list of (train_loader, val_loader) tuples.
    
    Args:
        dataset: The full dataset to split.
        n_splits: Number of folds (k).
        batch_size: Batch size for the loaders.
        shuffle: Whether to shuffle the data before splitting (default False for reproducibility).
        
    Returns:
        List of tuples: [(train_loader_1, val_loader_1), ..., (train_loader_k, val_loader_k)]
    """
    kf = KFold(n_splits=n_splits, shuffle=shuffle, random_state=42 if shuffle else None)
    
    # Extract indices (assuming dataset is indexable like a TensorDataset)
    indices = np.arange(len(dataset))
    
    folds_loaders = []
    
    print(f"Creating {n_splits}-fold cross-validation split...")
    
    for fold_idx, (train_indices, val_indices) in enumerate(kf.split(indices)):
        # Create Subsets for this fold
        train_subset = Subset(dataset, train_indices)
        val_subset = Subset(dataset, val_indices)
        
        # Create DataLoaders
        train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True) 
        val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False)
        
        folds_loaders.append((train_loader, val_loader))
        
    return folds_loaders

if __name__ == "__main__":
    file_parent = "/".join(os.path.abspath(__file__).split("/")[:-1])
    os.chdir(file_parent)
             
    n_genes = 5
    K = 4
    X, Y, _, _ = make_toy_nb_mixture(N=500, n_genes=n_genes, K=K, seed=0, r=10.0)
    X_val, Y_val, _, _ = make_toy_nb_mixture(N=50, n_genes=n_genes, K=K, seed=1, r=10.0)
    
    train_ds = torch.utils.data.TensorDataset(X, Y)
    val_ds = torch.utils.data.TensorDataset(X_val, Y_val)

    folds = create_folds(train_ds, 
                 n_splits=5, 
                 batch_size=128)

    
    # train_loader = DataLoader(train_ds, batch_size=64, shuffle=False)
    val_loader = DataLoader(val_ds, batch_size=64, shuffle=False)

    INPUT_DIM = 5
    HIDDEN = 32
    LATENT = 2
    EPOCHS = 2
    WARMUP = 2
    latent_dist = "Student"

    # --- Experiment 1: MixtureVAE Config ---
    config_mvae = {
        "run_tag": "Test_MixtureVAE",
        "model_type": "MixtureVAE",
        "prior_latent_dist": latent_dist,
        "prior_input_dist": "NegativeBinomial",
        "posterior_latent_dist": latent_dist,
        "prior_categorical_dist": "Categorical",
        "input_dim": INPUT_DIM,
        "hidden_dim": HIDDEN,
        "latent_dim": LATENT,
        "n_components": K,
        "n_layers": 2,
        "lr": 1e-3,
        "epochs": EPOCHS,
        "beta_kl": 0.5,
        "reg_marg": 10,
        "warmup": WARMUP,
        "save_path": "./mixture_vae.ckpt"
    }

    # --- Experiment 2: MoMixVAE Config ---
    config_momix = {
        "run_tag": "Test_MoMixVAE",
        "model_type": "MoMixVAE",
        "prior_latent_dist": latent_dist,
        "prior_input_dist": "NegativeBinomial",
        "posterior_latent_dist": latent_dist,
        "prior_categorical_dist": "Categorical",
        "input_dim": INPUT_DIM,
        "hidden_dim": HIDDEN,
        "latent_dim": LATENT,
        "hierarchy_components": [2, 3, K],
        "n_layers": 2,
        "lr": 1e-3,
        "epochs": EPOCHS,
        "beta_kl": 0.5,
        "reg_marg": 10,
        "warmup": WARMUP, # No warmup
        "save_path": "./momix_vae.ckpt"
    }

    # --- Experiment 3: ind_MoMVAE Config ---
    config_indmom = {
        "run_tag": "Test_ind_MoMVAE",
        "model_type": "ind_MoMVAE",
        "prior_latent_dist": latent_dist,
        "prior_input_dist": "NegativeBinomial",
        "posterior_latent_dist": latent_dist,
        "prior_categorical_dist": "Categorical",
        "input_dim": INPUT_DIM,
        "hidden_dim": HIDDEN,
        "latent_dim": LATENT,
        "hierarchy_components": [2, 3, K],
        "n_layers": 2,
        "lr": 1e-3,
        "epochs": EPOCHS,
        "beta_kl": 0.5,
        "reg_marg": 10,
        "warmup": WARMUP, # No warmup
        "save_path": "./ind_mom_vae.ckpt"
    }

    # CV
    latent = False
    cv_momix = run_cv(config_momix, folds, val_loader, plot_latent_space=latent, show_loss_every=1)
    cv_mvae = run_cv(config_mvae, folds, val_loader, plot_latent_space=latent, show_loss_every=1)
    cv_indmom = run_cv(config_indmom, folds, val_loader, plot_latent_space=latent, show_loss_every=1)

    # # --- Run ---
    # results_mvae = run_training(config_mvae, train_loader, val_loader)
    # results_momix = run_training(config_momix, train_loader, val_loader)
    # results_indmom = run_training(config_indmom, train_loader, val_loader)