"""Sub-Module to define training protocols"""
import torch
import numpy as np
import os, sys

# To ensure the custom package is found
path_to_repo = "/".join(os.path.abspath(__file__).split("/")[:-2])
if path_to_repo not in sys.path:
    sys.path.append(path_to_repo)

from mixture_vae.mvae import MixtureVAE, elbo_mixture_step

class EarlyStopping(object):
    """
    Implementation of early stopping with best epoch tracking
    """
    def __init__(self, patience):
        self.patience = patience
        self.losses = []
        self.loss_count = 0
        self.patience_count = 0
        self.best_loss = torch.inf
        self.best_model = None
        self.best_loss_idx = 0
    
    def register(self, model, loss):
        # Keep only patience losses
        if len(self.losses) == self.patience:
            self.losses = self.losses[1:]

        # register the loss
        self.losses.append(loss)
        self.loss_count += 1

        # check if this is the best loss
        if loss < self.best_loss:
            self.best_loss = loss
            self.best_loss_idx = self.loss_count
            self.best_model = model
            # reset the patience count
            self.patience_count = 0
        else:
            self.patience_count += 1
    
    def check_stop(self, model, loss):
        self.register(model, loss)
        if self.patience_count >= self.patience:
            return True
        return False
    
def training_mvae(dataloader: torch.utils.data.DataLoader,
                  val_dataloader:torch.utils.data.DataLoader,
                  model: MixtureVAE, 
                  optimizer: torch.optim.Optimizer, 
                  epochs: int = 50,
                  beta_kl : float = 1,
                  scheduler: torch.optim.lr_scheduler.LRScheduler = None,
                  patience: int | None = 5,
                  show_loss_every: int = 10):
    """
    Training protocol for MixtureVAE models.
    """
    assert isinstance(model, MixtureVAE), f"This training loop is tailored for MixtureVAE modules"
    # instantiate early stopper
    early_stopper = EarlyStopping(patience) if patience is not None else None

    losses = {"train":[], "val":[]}
    all_parts = {"train":{"recon":[],
                        "kl_latent":[],
                        "kl_cluster":[]},
                "val":{"recon":[],
                        "kl_latent":[],
                        "kl_cluster":[]}}
    for epoch in range(epochs):
        # TRAINING
        epoch_loss = 0
        epoch_parts = {"recon":[],
                       "kl_latent":[],
                       "kl_cluster":[]}
        for batch in dataloader:
            x = batch[0]
            optimizer.zero_grad()

            loss, parts = elbo_mixture_step(model, 
                                            x, 
                                            beta_kl=beta_kl)
            
            loss.backward()

            optimizer.step()

            if scheduler is not None:
                scheduler.step()
            epoch_loss += loss.item()
            for key in epoch_parts.keys():
                epoch_parts[key].append(parts[key])
        # register average epoch loss
        epoch_loss = epoch_loss / len(dataloader)
        # register average of epoch parts
        for key in epoch_parts.keys():
            epoch_parts[key] = np.mean(epoch_parts[key])
        
        # VALIDATION
        val_epoch_loss = 0
        val_epoch_parts = {"recon":[],
                            "kl_latent":[],
                            "kl_cluster":[]}
        with torch.no_grad():
            for batch in val_dataloader:
                x = batch[0]

                loss, parts = elbo_mixture_step(model, 
                                                x, 
                                                beta_kl=beta_kl)

                val_epoch_loss += loss.item()
                for key in epoch_parts.keys():
                    val_epoch_parts[key].append(parts[key])
            # register average epoch loss
            val_epoch_loss = val_epoch_loss / len(val_dataloader)
            # register average of epoch parts
            for key in val_epoch_parts.keys():
                val_epoch_parts[key] = np.mean(val_epoch_parts[key])
        
        # check early stoppage
        if early_stopper.check_stop(model, val_epoch_loss):
            print(f"\nEarly stoppage after {epoch} epochs with patience of {patience}.")
            final_loss_idx = early_stopper.best_loss_idx
            print(f"Best epoch: {final_loss_idx}")
            if final_loss_idx == 1:
                # 2 losses if the best epoch was the first
                # this avoids plotting a single point in other functions
                final_loss_idx = 2
            losses["train"] = losses["train"][:final_loss_idx]
            losses["val"] = losses["val"][:final_loss_idx]
            all_parts["train"] = {key:val[:final_loss_idx] for key, val in all_parts["train"].items()}
            all_parts["val"] = {key:val[:final_loss_idx] for key, val in all_parts["val"].items()}
            return early_stopper.best_model, losses, all_parts
        
        losses["train"].append(epoch_loss)
        losses["val"].append(epoch_loss)
        for key in all_parts["train"].keys():
            all_parts["train"][key].append(epoch_parts[key])
        for key in all_parts["val"].keys():
            all_parts["val"][key].append(val_epoch_parts[key])

        if epoch % show_loss_every == 0:
            print(f"epoch {epoch}: val = {losses["val"][-1]:.4f} | train = {losses["train"][-1]:.4f}")
    return model, losses, all_parts

if __name__ == "__main__":
    # ensures the working dir is that of the file
    file_parent = "/".join(os.path.abspath(__file__).split("/")[:-1])
    os.chdir(file_parent)

    from torch.utils.data import TensorDataset, DataLoader

    from mixture_vae.distributions import NormalDistribution, UniformDistribution, NegativeBinomial
    
    # Train Toy data
    X = torch.randint(0,50, (1000,5), dtype=torch.float)  # count data, 100 samples, 5 features
    dataset = TensorDataset(X)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

    # Val Toy data
    X_val = torch.randint(0,50, (300,5), dtype=torch.float)  # count data, 100 samples, 5 features
    val_dataset = TensorDataset(X)
    val_dataloader = DataLoader(val_dataset, batch_size=16, shuffle=True)
    
    # Problem setup
    input_dim = 5 # 5 genes
    hidden_dim = 16 # 8 hidden neurons per layer
    n_components = 3 # 3 clusters are assumed
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

    # Prior on cluster repartitions: Assume balanced 
    # cluster classes = Uniform on [0,1]
    a = torch.zeros((1,n_components))
    b = torch.ones((1,n_components))
    prior_categorical = UniformDistribution({"a":a, 
                                             "b":b})

    # Posterior on latent: Gaussian on R2 
    # (here assumed posterior = assumed prior 
    # but it could have been differnet)
    mu = torch.zeros((1,latent_dim))
    std = torch.ones((1,latent_dim))
    posterior_latent = NormalDistribution({"mu":mu,
                                           "std":std})
    
    # Instantiate MixtureVAE
    model = MixtureVAE(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        n_components=n_components,
        latent_dim=latent_dim,
        n_layers=1,
        prior_latent=prior_latent,
        prior_input=prior_input,
        prior_categorical=prior_categorical,
        posterior_latent=posterior_latent
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)

    trained_model, losses, parts = training_mvae(
        dataloader,
        val_dataloader,
        model,
        optimizer,
        epochs=5,
        beta_kl=0.1,
        patience=3,
        show_loss_every=1
    )

    print("Loss history:", losses)