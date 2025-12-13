import os, sys
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

# To ensure the custom package is found
path_to_repo = "/".join(os.path.abspath(__file__).split("/")[:-2])
if path_to_repo not in sys.path:
    sys.path.append(path_to_repo)

from mixture_vae.distributions import NormalDistribution, CategoricalDistribution, NegativeBinomial, Student
from mixture_vae.mvae import MoMixVAE
from mixture_vae.training import training_momixvae
from mixture_vae.viz import plot_loss_components, plot_latent
from mixture_vae.saving import load_model
from mixture_vae.utils import *

if __name__ == "__main__":
    # ensures the working dir is that of the file
    file_parent = "/".join(os.path.abspath(__file__).split("/")[:-1])
    os.chdir(file_parent)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # -------------------------
    # Train Toy data
    # -------------------------
    n_genes = 5
    K = 4
    X, Y, _, _ = make_toy_nb_mixture(N=5000, n_genes=n_genes, K=K, seed=0, r=10.0)
    
    dataset = TensorDataset(X, Y)
    dataloader = DataLoader(dataset, batch_size=512, shuffle=False)

    # Val Toy data
    X_val, Y_val, _, _ = make_toy_nb_mixture(N=500, n_genes=n_genes, K=K, seed=1, r=10.0)
    val_dataset = TensorDataset(X_val, Y_val)
    val_dataloader = DataLoader(val_dataset, batch_size=256, shuffle=False)

    # -------------------------
    # Problem setup
    # -------------------------
    input_dim = n_genes
    hidden_dim = 64
    hierarchy_components = [2,3,K]
    latent_dim = 2

    # -------------------------
    # Prior on input gene counts: NB for each gene
    # -------------------------
    params = init_nb_params_mom(dataloader, "cpu")
    prior_input = NegativeBinomial(params)

    # -------------------------
    # NEW: data-dependent priors per level (PCA + KMeans)
    # -------------------------
    all_prior_categorical = []
    all_prior_latent = []

    for K in hierarchy_components:
        latent_params = initialize_gmm_params(
            dataloader, n_components=K, latent_dim=latent_dim, device=device
        )
    
        cat_params = init_categorical_uniform(K, device)

        # Categorical prior p(y_level): non-uniform init via probs
        prior_categorical = CategoricalDistribution(cat_params)
        all_prior_categorical.append(prior_categorical)

        # Latent prior p(z | y_level): component-specific (K, D) init
        # NOTE: This requires your NormalDistribution to accept (K,D) in ref_parameters
        latent_params = initialize_student_prior_params_from_gmm(latent_params, device, df0=10)
        prior_latent_level = Student(latent_params) #NormalDistribution(latent_params)
        all_prior_latent.append(prior_latent_level)

    # -------------------------
    # Posterior on latent: Gaussian on R^D (one per level)
    # (kept standard; learned by encoder)
    # -------------------------
    all_posterior_latent = []
    for _K in hierarchy_components:
        mu0 = torch.zeros((1, latent_dim), device=device)
        std0 = torch.ones((1, latent_dim), device=device)
        df0 = torch.ones((1, latent_dim), device=device) * 10.0
        
        posterior_latent = Student({"df": df0, "mu": mu0, "std": std0}) #NormalDistribution({"mu": mu0, "std": std0})
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

    EPOCHS = 1
    BETA_KL = 0.5
    REG_MARG = 10 # controls the "careful, use all components" message to the network
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
        reg_marg=REG_MARG,
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
    print(compute_ll(model, val_dataloader))

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

    radj = compute_radj_classic(
                        model,
                        val_dataloader,
                        1,
                        model.n_levels,
                        debug=True,
                    )
    print("Radj", radj)