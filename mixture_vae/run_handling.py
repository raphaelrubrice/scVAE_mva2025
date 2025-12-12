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
from mixture_vae.distributions import NormalDistribution, UniformDistribution, NegativeBinomial, Poisson, Student
from mixture_vae.training import training_mvae, training_momixvae
from mixture_vae.utils import compute_CV_ll, compute_CV_radj
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
    "Student": Student
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

def compute_loader_mean(loader, device):
    """Iterates over the loader to compute the mean of the features (dim 0)."""
    sum_x = None
    total_samples = 0
    
    for batch in loader:
        try:
            x = batch["X"][:, 0, :]
        except Exception:
            x = batch[0]
        
        x = x.to(device)
        if sum_x is None:
            sum_x = torch.zeros(x.shape[1], device=device)
            
        sum_x += x.sum(dim=0)
        total_samples += x.size(0)
        
    return (sum_x / total_samples).reshape(1, -1)

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
        params["p"] = 0.5 * torch.ones((1, dim), device=device)
        params["r"] = compute_loader_mean(loader, device)
        
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

def build_categorical_prior(dist_name, n_components, device):
    """Helper specifically for building the prior on Y (clusters)."""
    # 'dim' here implies the number of components/clusters
    params = get_prior_parameters(dist_name, n_components, device)
    return build_distribution(dist_name, params, device)

def get_continuous_priors(config, train_loader, input_dim, latent_dim, device):
    """
    Calculates the 3 continuous priors (Input, Latent Prior, Latent Posterior).
    """
    # 1. Get names
    p_lat_name = config.get("prior_latent_dist", "Normal")
    p_in_name = config.get("prior_input_dist", "NegativeBinomial")
    post_lat_name = config.get("posterior_latent_dist", "Normal")

    # 2. Build
    # Latent Prior
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
    model_type = config["model_type"]
    input_dim = config["input_dim"]
    hidden_dim = config["hidden_dim"]
    latent_dim = config["latent_dim"]
    
    # 1. Setup the 3 Continuous Distributions
    prior_latent, prior_input, posterior_latent = get_continuous_priors(
        config, train_loader, input_dim, latent_dim, device
    )

    # 2. Identify the Categorical Distribution Type (The 4th distribution)
    # Default to Uniform if not specified
    cat_dist_name = config.get("prior_categorical_dist", "Uniform")

    # 3. Model Specific Instantiation
    if model_type == "MixtureVAE":
        n_components = config["n_components"]
        
        # Build the 4th distribution: Prior on Y
        prior_cat = build_categorical_prior(cat_dist_name, n_components, device)

        model = MixtureVAE(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            n_components=n_components,
            n_layers=config.get("n_layers", 1),
            prior_latent=prior_latent,
            prior_input=prior_input,
            prior_categorical=prior_cat,       # <--- Injected here
            posterior_latent=posterior_latent,
            act_func=config.get("act_func", nn.ReLU()),
            dropout=config.get("dropout", 0.0)
        )

    elif model_type == "ind_MoMVAE":
        branches = config.get("branch_components", config.get("hierarchy_components"))
        
        PARAMS = []
        for n_c in branches:
            # Build the 4th distribution per branch
            prior_cat = build_categorical_prior(cat_dist_name, n_c, device)
            
            branch_config = {
                "input_dim": input_dim,
                "hidden_dim": hidden_dim, 
                "n_components": n_c,
                "n_layers": config.get("n_layers", 1),
                "prior_latent": prior_latent,
                "prior_input": prior_input,
                "prior_categorical": prior_cat, # <--- Injected here
                "posterior_latent": posterior_latent,
                "act_func": config.get("act_func", nn.ReLU()),
                "dropout": config.get("dropout", 0.0)
            }
            PARAMS.append(branch_config)
            
        model = ind_MoMVAE(PARAMS=PARAMS)

    elif model_type == "MoMixVAE":
        hierarchy = config["hierarchy_components"]
        
        all_prior_cat = []
        all_post_latent = []
        
        # For MoMix, we might want to vary posterior per level, 
        # but here we assume same config for all levels
        post_lat_name = config.get("posterior_latent_dist", "Normal")

        for n_c in hierarchy:
            # Build the 4th distribution per level
            all_prior_cat.append(build_categorical_prior(cat_dist_name, n_c, device))
            
            # Posterior Latent per level (Fresh instance per level)
            # (Re-using helper to ensure fresh parameters)
            post_lat_params = get_prior_parameters(post_lat_name, latent_dim, device)
            all_post_latent.append(build_distribution(post_lat_name, post_lat_params, device))

        model = MoMixVAE(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            hierarchy_components=hierarchy,
            n_layers=config.get("n_layers", 2),
            prior_latent=prior_latent,
            prior_input=prior_input,
            all_prior_categorical=all_prior_cat, # <--- Injected here (list)
            all_posterior_latent=all_post_latent,
            act_func=config.get("act_func", nn.ReLU()),
            dropout=config.get("dropout", 0.0)
        )
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    return model

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
    warmup = config.get("warmup", None)
    patience = config.get("patience", int(0.1*epochs))
    tol = config.get("tol", 1e-3)
    save_path = config.get("save_path", f"./model_{config['model_type']}.ckpt")
    track_clusters = config.get("track_clusters", False)

    # 4. Select and Run Training Function
    model_type_str = config["model_type"]
    
    if model_type_str == "MixtureVAE":
        # model_type=0 corresponds to MixtureVAE in training_mvae
        model, losses, parts, clusters, betas = training_mvae(
            train_loader, val_loader, model, optimizer,
            epochs=epochs, beta_kl=beta_kl, warmup=warmup, 
            patience=patience, tol=tol, 
            model_type=0, # 0 for MixtureVAE
            track_clusters=track_clusters, save_path=save_path,
            **kwargs
        )
        
    elif model_type_str == "ind_MoMVAE":
        # model_type=1 corresponds to ind_MoMVAE in training_mvae
        model, losses, parts, clusters, betas = training_mvae(
            train_loader, val_loader, model, optimizer,
            epochs=epochs, beta_kl=beta_kl, warmup=warmup, 
            patience=patience, tol=tol, 
            model_type=1, # 1 for ind_MoMVAE
            track_clusters=track_clusters, save_path=save_path,
            **kwargs
        )
        
    elif model_type_str == "MoMixVAE":
        model, losses, parts, clusters, betas = training_momixvae(
            train_loader, val_loader, model, optimizer,
            epochs=epochs, beta_kl=beta_kl, warmup=warmup, 
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
    print(metric_res)

    # Add Radj metrics per hierarchical level
    for level in cv_radj.keys():
        metric_res[f"Mean Radj lvl.{level}"] = [np.mean(cv_radj[level])]
        metric_res[f"Std Radj lvl.{level}"]  = [np.std(cv_radj[level])]

    return results_cv, pd.DataFrame(metric_res)


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
             
    X = torch.randint(0, 50, (5000, 5), dtype=torch.float)
    Y = torch.randint(0, 3, (5000,1), dtype=torch.float)
    X_val = torch.randint(0, 50, (500, 5), dtype=torch.float)
    Y_val = torch.randint(0, 3, (500,1), dtype=torch.float)
    
    train_ds = torch.utils.data.TensorDataset(X, Y)
    val_ds = torch.utils.data.TensorDataset(X_val, Y_val)

    folds = create_folds(train_ds, 
                 n_splits=5, 
                 batch_size=128)

    
    # train_loader = DataLoader(train_ds, batch_size=64, shuffle=False)
    val_loader = DataLoader(val_ds, batch_size=64, shuffle=False)

    # --- Experiment 1: MixtureVAE Config ---
    config_mvae = {
        "run_tag": "Test_MixtureVAE",
        "model_type": "MixtureVAE",
        "prior_latent_dist": "Normal",
        "prior_input_dist": "NegativeBinomial",
        "posterior_latent_dist": "Normal",
        "prior_categorical_dist": "Uniform",
        "input_dim": 5,
        "hidden_dim": 16,
        "latent_dim": 2,
        "n_components": 3,
        "n_layers": 1,
        "lr": 1e-3,
        "epochs": 5,
        "beta_kl": 0.5,
        "warmup": 10,
        "save_path": "./mixture_vae.ckpt"
    }

    # --- Experiment 2: MoMixVAE Config ---
    config_momix = {
        "run_tag": "Test_MoMixVAE",
        "model_type": "MoMixVAE",
        "prior_latent_dist": "Normal",
        "prior_input_dist": "NegativeBinomial",
        "posterior_latent_dist": "Normal",
        "prior_categorical_dist": "Uniform",
        "input_dim": 5,
        "hidden_dim": 16,
        "latent_dim": 2,
        "hierarchy_components": [2, 3, 5],
        "n_layers": 2,
        "lr": 1e-3,
        "epochs": 5,
        "beta_kl": 0.5,
        "warmup": None, # No warmup
        "save_path": "./momix_vae.ckpt"
    }

    # --- Experiment 3: ind_MoMVAE Config ---
    config_indmom = {
        "run_tag": "Test_ind_MoMVAE",
        "model_type": "ind_MoMVAE",
        "prior_latent_dist": "Normal",
        "prior_input_dist": "NegativeBinomial",
        "posterior_latent_dist": "Normal",
        "prior_categorical_dist": "Uniform",
        "input_dim": 5,
        "hidden_dim": 16,
        "latent_dim": 2,
        "hierarchy_components": [2, 3, 5],
        "n_layers": 2,
        "lr": 1e-3,
        "epochs": 5,
        "beta_kl": 0.5,
        "warmup": 10, # No warmup
        "save_path": "./momix_vae.ckpt"
    }

    # CV
    cv_mvae = run_cv(config_mvae, folds, val_loader)

    # # --- Run ---
    # results_mvae = run_training(config_mvae, train_loader, val_loader)
    # results_momix = run_training(config_momix, train_loader, val_loader)
    # results_indmom = run_training(config_indmom, train_loader, val_loader)