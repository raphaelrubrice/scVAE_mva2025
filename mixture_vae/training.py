"""Sub-Module to define training protocols"""
import torch
import numpy as np
import os, sys
from tqdm import tqdm

# To ensure the custom package is found
path_to_repo = "/".join(os.path.abspath(__file__).split("/")[:-2])
if path_to_repo not in sys.path:
    sys.path.append(path_to_repo)

from mixture_vae.mvae import MixtureVAE, ind_MoMVAE, elbo_mixture_step, summed_elbo_mixture_step
from mixture_vae.viz import plot_loss_components

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

def format_loss(val_epoch_parts, beta_kl):
    """
    print the loss components in the form:
    -recon - beta_kl * (kl_latent + kl_cluster)
    while handling signs cleanly (no '--' or '+ -').
    """
    recon = val_epoch_parts["recon"][-1]
    kl_latent = val_epoch_parts["kl_latent"][-1]
    kl_cluster = val_epoch_parts["kl_cluster"][-1]

    # Recon is always shown as subtraction
    # the loss is - recon
    if recon >= 0:
        recon_str = f"- {recon:.4f}"
    else:
        recon_str = f"{-recon:.4f}"

    kl_latent_str = f"{kl_latent:.4f}"

    kl_cluster_str = f"{kl_cluster:.4f}"

    return f"{recon_str} - {beta_kl} * ({kl_latent_str} + ({kl_cluster_str}))"
   
def training_mvae(dataloader: torch.utils.data.DataLoader,
                  val_dataloader:torch.utils.data.DataLoader,
                  model: MixtureVAE, 
                  optimizer: torch.optim.Optimizer, 
                  epochs: int = 50,
                  beta_kl : float = 1,
                  scheduler: torch.optim.lr_scheduler.LRScheduler = None,
                  patience: int | None = 5,
                  show_loss_every: int = 10,
                  model_type: 0 | 1 | 2 = 0):
    """
    Training protocol for MixtureVAE models.
    """
    assert isinstance(model, MixtureVAE) or isinstance(model, ind_MoMVAE), f"This training loop is tailored for MixtureVAE or ind_MoMVAE modules"
    # instantiate early stopper
    early_stopper = EarlyStopping(patience) if patience is not None else None

    losses = {"train":[], "val":[]}
    all_parts = {"train":{"recon":[],
                        "kl_latent":[],
                        "kl_cluster":[]},
                "val":{"recon":[],
                        "kl_latent":[],
                        "kl_cluster":[]}}
    for epoch in tqdm(range(1,epochs+1), desc="TRAINING"):
        # TRAINING
        epoch_loss = 0
        epoch_parts = {"recon":[],
                       "kl_latent":[],
                       "kl_cluster":[]}
        for batch in dataloader:
            x = batch[0]
            optimizer.zero_grad()

            if model_type == 0:
                loss, parts = elbo_mixture_step(model, 
                                            x, 
                                            beta_kl=beta_kl)
            
            elif model_type == 1:
                loss, parts = summed_elbo_mixture_step(model, x)
            
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
            epoch_parts[key].append(np.mean(epoch_parts[key]))
        
        # VALIDATION
        val_epoch_loss = 0
        val_epoch_parts = {"recon":[],
                            "kl_latent":[],
                            "kl_cluster":[]}
        with torch.no_grad():
            for batch in val_dataloader:
                x = batch[0]
                if model_type == 0:
                    loss, parts = elbo_mixture_step(model, 
                                            x, 
                                            beta_kl=beta_kl)
                
                elif model_type == 1:
                    loss, parts = summed_elbo_mixture_step(model, x)

                val_epoch_loss += loss.item()
                for key in val_epoch_parts.keys():
                    val_epoch_parts[key].append(parts[key])
            # register average epoch loss
            val_epoch_loss = val_epoch_loss / len(val_dataloader)
            # register average of epoch parts
            for key in val_epoch_parts.keys():
                val_epoch_parts[key].append(np.mean(val_epoch_parts[key]))
        
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
        losses["val"].append(val_epoch_loss)
        for key in all_parts["train"].keys():
            all_parts["train"][key].append(np.mean(epoch_parts[key]))
        for key in all_parts["val"].keys():
            all_parts["val"][key].append(np.mean(val_epoch_parts[key]))

        if epoch == 1:
            print("Loss printing format:\nepoch x: val = loss (-recon - beta_kl * (kl_latent + kl_cluster)) | train = loss (-recon - beta_kl * (kl_latent + kl_cluster))\n")
        if epoch % show_loss_every == 0:
            print(f"epoch {epoch}: val = {losses["val"][-1]:.4f} ({format_loss(val_epoch_parts, beta_kl)}) | train = {losses["train"][-1]:.4f} ({format_loss(epoch_parts, beta_kl)})")
    return model, losses, all_parts

if __name__ == "__main__":
    # ensures the working dir is that of the file
    file_parent = "/".join(os.path.abspath(__file__).split("/")[:-1])
    os.chdir(file_parent)

    from torch.utils.data import TensorDataset, DataLoader

    from mixture_vae.distributions import NormalDistribution, UniformDistribution, NegativeBinomial

    model_type = 1
    
    # Train Toy data
    X = torch.randint(0,50, (5000,5), dtype=torch.float)  # count data, 100 samples, 5 features
    dataset = TensorDataset(X)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

    # Val Toy data
    X_val = torch.randint(0,50, (500,5), dtype=torch.float)  # count data, 100 samples, 5 features
    val_dataset = TensorDataset(X_val)
    val_dataloader = DataLoader(val_dataset, batch_size=64, shuffle=True)
    
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

    # Prior on cluster repartitions (mixture): Assume balanced 
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
    
    if model_type == 0:
        # Instantiate MixtureVAE
        model = MixtureVAE(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            n_components=n_components,
            n_layers=1,
            prior_latent=prior_latent,
            prior_input=prior_input,
            prior_categorical=prior_categorical,
            posterior_latent=posterior_latent
        )

    if model_type == 1:
        # Instatiate ind_MoMVAE
        model = ind_MoMVAE(
            PARAMS = [
            {"input_dim": input_dim,
            "hidden_dim": h,
            "n_components": n_components,
            "n_layers": 1,
            "prior_latent": prior_latent,
            "prior_input": prior_input,
            "prior_categorical": prior_categorical,
            "posterior_latent": posterior_latent} for h in [1, 2, 4, 8]]
        )

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    EPOCHS = 50
    BETA_KL = 0.5
    PATIENCE = 5

    trained_model, losses, parts = training_mvae(
        dataloader,
        val_dataloader,
        model,
        optimizer,
        epochs=EPOCHS,
        beta_kl=BETA_KL,
        patience=PATIENCE,
        show_loss_every=5,
        model_type=model_type
    )

    plot_loss_components(parts["train"], 
                         parts["val"], 
                         BETA_KL, 
                         title="Train Loss Breakdown",
                         save_path="./toy_losses.pdf")