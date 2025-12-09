import torch
import torch.nn as nn
import os
import pickle

from mixture_vae.mvae import MixtureVAE, ind_MoMVAE, MoMixVAE

def extract_mixture_vae_config(model: MixtureVAE) -> dict:
    """Extracts initialization parameters from a MixtureVAE instance."""
    return {
        'input_dim': model.input_dim,
        'hidden_dim': model.hidden_dim,
        'n_components': model.n_components,
        'n_layers': model.n_layers,
        'prior_latent': model.prior_latent,
        'prior_input': model.prior_input,
        'prior_categorical': model.prior_categorical,
        'posterior_latent': model.posterior_latent,
        'act_func': model.act_func,
        'dropout': model.dropout,
        'norm_layer': model.norm_layer
    }

def extract_momix_vae_config(model: MoMixVAE) -> dict:
    """Extracts initialization parameters from a MoMixVAE instance."""
    return {
        'input_dim': model.input_dim,
        'hidden_dim': model.hidden_dim,
        'hierarchy_components': model.hierarchy_components,
        'n_layers': model.n_layers,
        'prior_latent': model.prior_latent,
        'prior_input': model.prior_input,
        'all_prior_categorical': model.all_prior_categorical,
        'all_posterior_latent': model.all_posterior_latent,
        'act_func': model.act_func,
        'dropout': model.dropout,
        'norm_layer': model.norm_layer
    }

def extract_ind_momvae_config(model: ind_MoMVAE) -> dict:
    """
    Reconstructs the PARAMS list of dicts required to initialize ind_MoMVAE.
    It does this by inspecting the internal branches.
    """
    params_list = []
    for branch in model.branches:
        # Extract config from each MixtureVAE branch
        params_list.append(extract_mixture_vae_config(branch))
    
    return {'PARAMS': params_list}

def save_model(model: nn.Module, path: str):
    """
    Saves the model state, configuration, and class name to a single file.
    
    Args:
        model: The model instance (MixtureVAE, MoMixVAE, or ind_MoMVAE).
        path: File path to save (e.g., 'checkpoint.pth').
    """
    if isinstance(model, MixtureVAE):
        config = extract_mixture_vae_config(model)
        model_type = 'MixtureVAE'
    elif isinstance(model, MoMixVAE):
        config = extract_momix_vae_config(model)
        model_type = 'MoMixVAE'
    elif isinstance(model, ind_MoMVAE):
        config = extract_ind_momvae_config(model)
        model_type = 'ind_MoMVAE'
    else:
        raise ValueError(f"Model type {type(model)} not supported by this save function.")

    checkpoint = {
        'model_type': model_type,
        'config': config,
        'state_dict': model.state_dict()
    }

    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)
    
    torch.save(checkpoint, path)
    print(f"Model saved to {path}")

def load_model(path: str, map_location=None):
    """
    Loads a model from a checkpoint file, re-instantiating the correct class
    with the saved configuration.

    Args:
        path: Path to the checkpoint file.
        map_location: device to load the model on (e.g., 'cuda' or 'cpu').

    Returns:
        model: The loaded model instance.
    """
    if map_location is None:
        map_location = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f"Loading checkpoint from {path}...")
    checkpoint = torch.load(path, map_location=map_location, weights_only=False)
    
    model_type = checkpoint['model_type']
    config = checkpoint['config']
    state_dict = checkpoint['state_dict']

    # Instantiate the correct class based on saved string
    if model_type == 'MixtureVAE':
        model = MixtureVAE(**config)
    elif model_type == 'MoMixVAE':
        model = MoMixVAE(**config)
    elif model_type == 'ind_MoMVAE':
        model = ind_MoMVAE(**config)
    else:
        raise ValueError(f"Unknown model type in checkpoint: {model_type}")

    # Load weights
    model.load_state_dict(state_dict)
    model.to(map_location)

    return model