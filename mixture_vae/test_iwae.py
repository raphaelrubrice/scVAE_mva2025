import os, sys
import torch
from torch.utils.data import TensorDataset, DataLoader

# To ensure the custom package is found
path_to_repo = "/".join(os.path.abspath(__file__).split("/")[:-2])
if path_to_repo not in sys.path:
    sys.path.append(path_to_repo)

from mixture_vae.distributions import NormalDistribution, UniformDistribution, NegativeBinomial
from mixture_vae.mvae import MoMixVAE
from mixture_vae.training import training_momixvae
from mixture_vae.viz import plot_loss_components
from mixture_vae.utils import compute_ll

if __name__ == "__main__":
    # ensures the working dir is that of the file
    file_parent = "/".join(os.path.abspath(__file__).split("/")[:-1])
    os.chdir(file_parent)

    # Train Toy data
    n_genes = 5
    X = torch.randint(0,50, (5000,n_genes), dtype=torch.float)  # count data, 100 samples, 5 features
    dataset = TensorDataset(X)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=False) 
    # shuffle = False because we need order to be safe for cluster trakcing

    # Val Toy data
    X_val = torch.randint(0,50, (500,n_genes), dtype=torch.float)  # count data, 100 samples, 5 features
    val_dataset = TensorDataset(X_val)
    val_dataloader = DataLoader(val_dataset, batch_size=64, shuffle=False)
    # shuffle = False because we need order to be safe for cluster trakcing
    
    # Problem setup
    input_dim = n_genes # 5 genes
    hidden_dim = 16 
    hierarchy_components = [2,3,5] # A hierarchy of 3 levels with 2, 3, 9 clusters respectively
    latent_dim = 2 # 2 dimension latent space
    
    # Prior on latent: Standard Gaussian in R2
    mu = torch.zeros((1,latent_dim))
    std = torch.ones((1,latent_dim))
    prior_latent = NormalDistribution({"mu":mu,
                                       "std":std})
    
    # Prior on input gene counts: NB for each gene 
    p = 0.5 * torch.ones((1,input_dim)) # 50/50 chance of expression
    r = torch.mean(X, dim=0).reshape(1,-1) # prior = average count in train data
    prior_input = NegativeBinomial({"p":p,
                                    "r":r})

    # Prior on cluster repartitions (mixture): Assume balanced 
    # cluster classes = Uniform on [0,1]
    # currently we assume the same for all levels
    all_prior_categorical = []
    for n_components in hierarchy_components:
        a = torch.zeros((1,n_components))
        b = torch.ones((1,n_components))
        prior_categorical = UniformDistribution({"a":a, 
                                                "b":b})
        all_prior_categorical.append(prior_categorical)

    # Posterior on latent: Gaussian on R2 
    # (here assumed posterior = assumed prior 
    # but it could have been differnet)
    # currently we assume the same for all levels
    all_posterior_latent = []
    for n_components in hierarchy_components:
        mu = torch.zeros((1,latent_dim))
        std = torch.ones((1,latent_dim))
        posterior_latent = NormalDistribution({"mu":mu,
                                            "std":std})
        all_posterior_latent.append(posterior_latent)
    
    # Instantiate MixtureVAE
    model = MoMixVAE(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        hierarchy_components=hierarchy_components,
        n_layers=2,
        prior_latent=prior_latent,
        prior_input=prior_input,
        all_prior_categorical=all_prior_categorical,
        all_posterior_latent=all_posterior_latent
    )

    print(compute_ll(model, val_dataloader))
    