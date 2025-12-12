import os, sys
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

# To ensure the custom package is found
path_to_repo = "/".join(os.path.abspath(__file__).split("/")[:-2])
if path_to_repo not in sys.path:
    sys.path.append(path_to_repo)

from mixture_vae.distributions import NormalDistribution, CategoricalDistribution, NegativeBinomial
from mixture_vae.mvae import MoMixVAE
from mixture_vae.training import training_momixvae
from mixture_vae.viz import plot_loss_components, plot_latent
from mixture_vae.saving import load_model
from mixture_vae.utils import compute_ll
from mixture_vae.utils import initialize_gmm_params

if __name__ == "__main__":
    # ensures the working dir is that of the file
    file_parent = "/".join(os.path.abspath(__file__).split("/")[:-1])
    os.chdir(file_parent)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # -------------------------
    # Train Toy data
    # -------------------------
    n_genes = 5
    X = torch.randint(0, 50, (5000, 5), dtype=torch.float)
    Y = nn.functional.one_hot(torch.randint(0, 3, (5000,1), dtype=torch.long))
    
    dataset = TensorDataset(X, Y)
    dataloader = DataLoader(dataset, batch_size=512, shuffle=False)

    # Val Toy data
    X_val = torch.randint(0, 50, (500, 5), dtype=torch.float)
    Y_val = nn.functional.one_hot(torch.randint(0, 3, (500,1), dtype=torch.long))
    val_dataset = TensorDataset(X_val, Y_val)
    val_dataloader = DataLoader(val_dataset, batch_size=256, shuffle=False)

    # -------------------------
    # Problem setup
    # -------------------------
    input_dim = n_genes
    hidden_dim = 16
    hierarchy_components = [2, 3]
    latent_dim = 2

    # -------------------------
    # Prior on input gene counts: NB for each gene
    # -------------------------
    p = 0.5 * torch.ones((1, input_dim), device=device)
    r = torch.mean(X.to(device), dim=0).reshape(1, -1)  # prior = average count in train data
    prior_input = NegativeBinomial({"p": p, "r": r})

    # -------------------------
    # NEW: data-dependent priors per level (PCA + KMeans)
    # -------------------------
    all_prior_categorical = []
    all_prior_latent = []

    for K in hierarchy_components:
        cat_params, latent_params = initialize_gmm_params(
            dataloader, n_components=K, latent_dim=latent_dim, device=device
        )

        # Categorical prior p(y_level): non-uniform init via probs
        prior_categorical = CategoricalDistribution(cat_params)
        all_prior_categorical.append(prior_categorical)

        # Latent prior p(z | y_level): component-specific (K, D) init
        # NOTE: This requires your NormalDistribution to accept (K,D) in ref_parameters
        prior_latent_level = NormalDistribution(latent_params)
        all_prior_latent.append(prior_latent_level)

    # -------------------------
    # Posterior on latent: Gaussian on R^D (one per level)
    # (kept standard; learned by encoder)
    # -------------------------
    all_posterior_latent = []
    for _K in hierarchy_components:
        mu0 = torch.zeros((1, latent_dim), device=device)
        std0 = torch.ones((1, latent_dim), device=device)
        posterior_latent = NormalDistribution({"mu": mu0, "std": std0})
        all_posterior_latent.append(posterior_latent)

    # -------------------------
    # Instantiate MoMixVAE (assume it has all_prior_latent)
    # -------------------------
    model = MoMixVAE(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        hierarchy_components=hierarchy_components,
        n_layers=2,
        all_prior_latent=all_prior_latent,          # <--- NEW
        prior_input=prior_input,
        all_prior_categorical=all_prior_categorical,
        all_posterior_latent=all_posterior_latent
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    EPOCHS = 50
    BETA_KL = 0.5
    WARMUP = None
    PATIENCE = 5
    TOL = 5e-3

    save_path = f"./model_{model.__class__.__name__}.ckpt"
    model, losses, parts, clusters, all_betas = training_momixvae(
        dataloader,
        val_dataloader,
        model,
        optimizer,
        epochs=EPOCHS,
        beta_kl=BETA_KL,
        warmup=WARMUP,
        patience=PATIENCE,
        tol=TOL,
        show_loss_every=1,
        track_clusters=True,
        save_path=save_path
    )

    print("Loading model..")
    model = load_model(save_path)
    model.to(device)

    print("Testing loaded model")
    compute_ll(model, val_dataloader)

    # plot training and validation losses
    plot_loss_components(
        parts["train"],
        parts["val"],
        all_betas,
        title="Loss Breakdown",
        save_path=f"./{model.__class__.__name__}_toy_losses.pdf"
    )

    plot_latent(
        model,
        val_dataloader,
        level=-1,
        true_labels=False,
        label_key=None,
        title="Latent Space",
        save_path=f"./{model.__class__.__name__}_model_latent.pdf"
    )

    plot_latent(
        model,
        val_dataloader,
        level=-1,
        true_labels=True,
        label_key=None,
        title="Latent Space",
        save_path=f"./{model.__class__.__name__}_true_latent.pdf"
    )