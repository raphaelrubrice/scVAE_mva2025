"""Sub-Module to define training protocols"""
import torch
import numpy as np
import os, sys
from tqdm.auto import tqdm

# To ensure the custom package is found
path_to_repo = "/".join(os.path.abspath(__file__).split("/")[:-2])
if path_to_repo not in sys.path:
    sys.path.append(path_to_repo)

from mixture_vae.mvae import MixtureVAE, elbo_mixture_step, MoMixVAE, elbo_MoMix_step, ind_MoMVAE, summed_elbo_mixture_step
from mixture_vae.viz import plot_loss_components, plot_latent
from mixture_vae.saving import save_model, load_model
from mixture_vae.utils import compute_ll

class EarlyStopping(object):
    """
    Implementation of early stopping with best epoch tracking
    """
    def __init__(self, patience, tol=0.0):
        self.patience = patience
        self.tol = tol
        self.losses = []
        self.loss_count = 0
        self.patience_count = 0
        self.best_loss = torch.inf
        self.best_model = None
        self.best_loss_idx = 0
    
    def reset(self):
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
        if loss < self.best_loss - self.tol:
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

    return f"{recon_str} - {beta_kl:.4f} * ({kl_latent_str} + ({kl_cluster_str}))"
   
def training_mvae(
    dataloader: torch.utils.data.DataLoader,
    val_dataloader: torch.utils.data.DataLoader,
    model: MixtureVAE,
    optimizer: torch.optim.Optimizer,
    epochs: int = 50,
    beta_kl: float = 1,
    warmup: int | None = None,
    min_beta: float = 0.0,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
    track_clusters: bool = False,
    patience: int | None = 5,
    tol: float | None = 0.0,
    show_loss_every: int = 10,
    model_type: 0 | 1 | 2 = 0,
    progress_bar=False,
    save_path=None,
):
    """
    Training protocol for MixtureVAE models.
    """
    assert isinstance(model, MixtureVAE) or isinstance(
        model, ind_MoMVAE
    ), "This training loop is tailored for MixtureVAE or ind_MoMVAE modules"

    # ------------------------------------------------------------
    # Device selection: use CUDA if available, else CPU
    # ------------------------------------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model = model.to(device)

    if not progress_bar:
        def passthrough(obj, *args, **kwargs):
            return obj
        tqdm_func = passthrough
    else:
        tqdm_func = tqdm

    # ------------------------------------------------------------
    # Beta schedule
    # ------------------------------------------------------------
    if warmup is not None:
        beta_range = (min_beta, beta_kl)
        min_beta_range = min(beta_range)
        max_beta_range = max(beta_range)
        assert min_beta_range >= 0 and max_beta_range >= 0, (
            f"Invalid range: Must be positive. but got {beta_range}"
        )
        all_betas = (
            np.linspace(min_beta_range, max_beta_range, warmup).tolist()
            + [max_beta_range] * (epochs - warmup)
        )
        all_betas = all_betas[:epochs]
    else:
        all_betas = [beta_kl] * epochs

    # instantiate early stopper
    early_stopper = EarlyStopping(patience, tol) if patience is not None else None

    losses = {"train": [], "val": []}
    all_parts = {
        "train": {"recon": [], "kl_latent": [], "kl_cluster": []},
        "val": {"recon": [], "kl_latent": [], "kl_cluster": []},
    }
    clusters = {"train": [], "val": []}

    for epoch in tqdm(range(1, epochs + 1), desc="TRAINING"):
        beta_kl = all_betas[epoch - 1]

        # ==========================
        # TRAINING
        # ==========================
        model.train()
        epoch_loss = 0.0
        # Initialize lists to store detached tensors on GPU
        epoch_parts = {"recon": [], "kl_latent": [], "kl_cluster": []}
        epoch_clusters = []

        for batch in tqdm_func(dataloader, desc=f"Epoch {epoch}", total=len(dataloader)):
            try:
                x = batch["X"][:, 0, :]
            except Exception:
                x = batch[0]

            # move batch to device
            x = x.to(device)

            optimizer.zero_grad()

            if model_type == 0:
                loss, parts, batch_clusters = elbo_mixture_step(
                    model,
                    x,
                    beta_kl=beta_kl,
                    track_clusters=track_clusters,
                )

            elif model_type == 1:
                loss, parts, batch_clusters = summed_elbo_mixture_step(
                    model,
                    x,
                    beta_kl=beta_kl,
                    track_clusters=track_clusters,
                )

            # backward + step
            loss.backward()
            optimizer.step()

            if scheduler is not None:
                scheduler.step()

            # [PATCH APPLIED]
            # Accumulate loss as a tensor on GPU (detached), convert later
            epoch_loss += loss.detach()

            # [PATCH APPLIED]
            # Append detached tensors to list, don't move to CPU yet
            for key in epoch_parts.keys():
                val = parts[key]
                if torch.is_tensor(val):
                    epoch_parts[key].append(val.detach())
                else:
                    # Fallback if it's already a float (unlikely given mvae.py)
                    epoch_parts[key].append(torch.tensor(val, device=device))

            # keep clusters on GPU during accumulation; move to CPU later
            epoch_clusters.append(batch_clusters)

        # [PATCH APPLIED]
        # Process accumulations once per epoch (Sync Point)
        if isinstance(epoch_loss, torch.Tensor):
             epoch_loss = epoch_loss.item()
        epoch_loss = epoch_loss / len(dataloader)

        for key in epoch_parts.keys():
            if len(epoch_parts[key]) > 0:
                # Stack tensors on GPU, compute mean, then sync to CPU
                mean_val = torch.stack(epoch_parts[key]).mean().item()
            else:
                mean_val = 0.0
            # Replace list with single-element list containing mean 
            # (preserves format for format_loss accessing [-1])
            epoch_parts[key] = [mean_val]

        # register clusters for all inputs seen in epoch, then move to CPU
        if len(epoch_clusters) > 0:
            epoch_clusters = torch.cat(epoch_clusters, dim=0).detach().cpu()
        else:
            epoch_clusters = torch.empty(0)

        # ==========================
        # VALIDATION
        # ==========================
        model.eval()
        val_epoch_loss = 0.0
        val_epoch_parts = {"recon": [], "kl_latent": [], "kl_cluster": []}
        val_epoch_clusters = []

        with torch.no_grad():
            for batch in tqdm_func(val_dataloader, desc=f"Epoch {epoch}", total=len(val_dataloader)):
                try:
                    x = batch["X"][:, 0, :]
                except Exception:
                    x = batch[0]

                # move batch to device
                x = x.to(device)

                if model_type == 0:
                    loss, parts, batch_clusters = elbo_mixture_step(
                        model,
                        x,
                        beta_kl=beta_kl,
                        track_clusters=track_clusters,
                    )

                elif model_type == 1:
                    loss, parts, batch_clusters = summed_elbo_mixture_step(
                        model,
                        x,
                        beta_kl=beta_kl,
                        track_clusters=track_clusters,
                    )

                # [PATCH APPLIED]
                val_epoch_loss += loss.detach()

                # [PATCH APPLIED]
                for key in val_epoch_parts.keys():
                    val = parts[key]
                    if torch.is_tensor(val):
                        val_epoch_parts[key].append(val.detach())
                    else:
                        val_epoch_parts[key].append(torch.tensor(val, device=device))

                val_epoch_clusters.append(batch_clusters)

            # [PATCH APPLIED]
            if isinstance(val_epoch_loss, torch.Tensor):
                 val_epoch_loss = val_epoch_loss.item()
            val_epoch_loss = val_epoch_loss / len(val_dataloader)

            for key in val_epoch_parts.keys():
                if len(val_epoch_parts[key]) > 0:
                    mean_val = torch.stack(val_epoch_parts[key]).mean().item()
                else:
                    mean_val = 0.0
                val_epoch_parts[key] = [mean_val]

            # register clusters for all inputs seen in epoch (stored on CPU)
            if len(val_epoch_clusters) > 0:
                val_epoch_clusters = torch.cat(val_epoch_clusters, dim=0).detach().cpu()
            else:
                val_epoch_clusters = torch.empty(0)

        # ==========================
        # Early stopping / bookkeeping
        # ==========================
        # if we are out of the warmup zone, we reset the early stopper
        if warmup is not None and epoch <= warmup and early_stopper is not None:
            early_stopper.reset()

        # check early stoppage
        if early_stopper is not None and early_stopper.check_stop(model, val_epoch_loss):
            print(f"\nEarly stoppage after {epoch} epochs with patience of {patience}.")
            final_loss_idx = early_stopper.best_loss_idx
            print(f"Best epoch: {final_loss_idx}")
            if final_loss_idx == 1:
                # 2 losses if the best epoch was the first
                # this avoids plotting a single point in other functions
                final_loss_idx = 2
            losses["train"] = losses["train"][:final_loss_idx]
            losses["val"] = losses["val"][:final_loss_idx]
            all_parts["train"] = {
                key: val[:final_loss_idx] for key, val in all_parts["train"].items()
            }
            all_parts["val"] = {
                key: val[:final_loss_idx] for key, val in all_parts["val"].items()
            }
            clusters["train"] = clusters["train"][:final_loss_idx]
            clusters["val"] = clusters["val"][:final_loss_idx]

            # saving
            if save_path is not None:
                print(f"\nSaving model..")
                save_model(early_stopper.best_model, path=save_path)

            return (
                early_stopper.best_model,
                losses,
                all_parts,
                clusters,
                all_betas[:final_loss_idx],
            )

        # store losses/parts/clusters for this epoch
        losses["train"].append(epoch_loss)
        losses["val"].append(val_epoch_loss)

        for key in all_parts["train"].keys():
            all_parts["train"][key].append(np.mean(epoch_parts[key]))
        for key in all_parts["val"].keys():
            all_parts["val"][key].append(np.mean(val_epoch_parts[key]))

        clusters["train"].append(epoch_clusters)
        clusters["val"].append(val_epoch_clusters)

        if epoch == 1:
            print(
                "Loss printing format:\n"
                "epoch x: val = loss (-recon - beta_kl * (kl_latent + kl_cluster)) | "
                "train = loss (-recon - beta_kl * (kl_latent + kl_cluster))\n"
            )

        if epoch % show_loss_every == 0:
            print(
                f"epoch {epoch}: "
                f"val = {losses['val'][-1]:.4f} ({format_loss(val_epoch_parts, beta_kl)}) | "
                f"train = {losses['train'][-1]:.4f} ({format_loss(epoch_parts, beta_kl)})"
            )
    
    # saving
    if save_path is not None:
        print(f"\nSaving model..")
        save_model(model, path=save_path)

    return model, losses, all_parts, clusters, all_betas

def training_momixvae(
    dataloader: torch.utils.data.DataLoader,
    val_dataloader: torch.utils.data.DataLoader,
    model: MoMixVAE,
    optimizer: torch.optim.Optimizer,
    epochs: int = 50,
    beta_kl: float = 1,
    warmup: int | None = None,
    min_beta: float = 0.0,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
    track_clusters: bool = False,
    patience: int | None = 5,
    tol: float | None = 0.0,
    show_loss_every: int = 10,
    progress_bar=False,
    save_path=None,
):
    """
    Training protocol for MoMixVAE models.
    """
    assert isinstance(model, MoMixVAE), "This training loop is tailored for MoMixVAE modules"

    # ------------------------------------------------------------------
    # Device selection: use CUDA if available, else CPU
    # ------------------------------------------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model = model.to(device)

    if not progress_bar:
        def passthrough(obj, *args, **kwargs):
            return obj
        tqdm_func = passthrough
    else:
        tqdm_func = tqdm
    
    # ------------------------------------------------------------------
    # Beta schedule
    # ------------------------------------------------------------------
    if warmup is not None:
        beta_range = (min_beta, beta_kl)
        min_beta_range = min(beta_range)
        max_beta_range = max(beta_range)
        assert min_beta_range >= 0 and max_beta_range >= 0, (
            f"Invalid range: Must be positive, but got {beta_range}"
        )
        all_betas = (
            np.linspace(min_beta_range, max_beta_range, warmup).tolist()
            + [max_beta_range] * (epochs - warmup)
        )
        all_betas = all_betas[:epochs]
    else:
        all_betas = [beta_kl] * epochs

    # instantiate early stopper
    early_stopper = EarlyStopping(patience, tol) if patience is not None else None

    losses = {"train": [], "val": []}
    all_parts = {
        "train": {"recon": [], "kl_latent": [], "kl_cluster": []},
        "val": {"recon": [], "kl_latent": [], "kl_cluster": []},
    }
    clusters = {"train": [], "val": []}

    for epoch in tqdm(range(1, epochs + 1), desc="TRAINING"):
        beta_kl = all_betas[epoch - 1]

        # ==========================
        # TRAINING
        # ==========================
        model.train()
        epoch_loss = 0.0
        # Initialize lists to store detached tensors on GPU
        epoch_parts = {"recon": [], "kl_latent": [], "kl_cluster": []}
        epoch_clusters = []

        for batch in tqdm_func(dataloader, desc=f"Epoch {epoch}", total=len(dataloader)):
            try:
                x = batch["X"][:, 0, :]
            except Exception:
                x = batch[0]

            # move batch to device
            x = x.to(device)

            optimizer.zero_grad()

            loss, parts, batch_clusters = elbo_MoMix_step(
                model,
                x,
                beta_kl=beta_kl,
                track_clusters=track_clusters,
            )

            loss.backward()
            optimizer.step()

            if scheduler is not None:
                scheduler.step()

            # [PATCH APPLIED]
            epoch_loss += loss.detach()

            # [PATCH APPLIED]
            for key in epoch_parts.keys():
                val = parts[key]
                if torch.is_tensor(val):
                    epoch_parts[key].append(val.detach())
                else:
                    epoch_parts[key].append(torch.tensor(val, device=device))

            # keep clusters on GPU during accumulation, move to CPU later
            epoch_clusters.append(batch_clusters)

        # [PATCH APPLIED]
        if isinstance(epoch_loss, torch.Tensor):
             epoch_loss = epoch_loss.item()
        epoch_loss = epoch_loss / len(dataloader)

        for key in epoch_parts.keys():
            if len(epoch_parts[key]) > 0:
                mean_val = torch.stack(epoch_parts[key]).mean().item()
            else:
                mean_val = 0.0
            epoch_parts[key] = [mean_val]

        # register clusters for all inputs seen in epoch, then move to CPU for storage
        if len(epoch_clusters) > 0:
            epoch_clusters = torch.cat(epoch_clusters, dim=0).detach().cpu()
        else:
            epoch_clusters = torch.empty(0)

        # ==========================
        # VALIDATION
        # ==========================
        model.eval()
        val_epoch_loss = 0.0
        val_epoch_parts = {"recon": [], "kl_latent": [], "kl_cluster": []}
        val_epoch_clusters = []

        with torch.no_grad():
            for batch in tqdm_func(val_dataloader, desc=f"Epoch {epoch}", total=len(val_dataloader)):
                try:
                    x = batch["X"][:, 0, :]
                except Exception:
                    x = batch[0]

                # move batch to device
                x = x.to(device)

                loss, parts, batch_clusters = elbo_MoMix_step(
                    model,
                    x,
                    beta_kl=beta_kl,
                    track_clusters=track_clusters,
                )

                # [PATCH APPLIED]
                val_epoch_loss += loss.detach()

                # [PATCH APPLIED]
                for key in val_epoch_parts.keys():
                    val = parts[key]
                    if torch.is_tensor(val):
                        val_epoch_parts[key].append(val.detach())
                    else:
                        val_epoch_parts[key].append(torch.tensor(val, device=device))

                val_epoch_clusters.append(batch_clusters)

            # [PATCH APPLIED]
            if isinstance(val_epoch_loss, torch.Tensor):
                 val_epoch_loss = val_epoch_loss.item()
            val_epoch_loss = val_epoch_loss / len(val_dataloader)

            for key in val_epoch_parts.keys():
                if len(val_epoch_parts[key]) > 0:
                    mean_val = torch.stack(val_epoch_parts[key]).mean().item()
                else:
                    mean_val = 0.0
                val_epoch_parts[key] = [mean_val]

            # register clusters for all inputs seen in epoch (stored on CPU)
            if len(val_epoch_clusters) > 0:
                val_epoch_clusters = torch.cat(val_epoch_clusters, dim=0).detach().cpu()
            else:
                val_epoch_clusters = torch.empty(0)

        # ==========================
        # Early stopping / bookkeeping
        # ==========================
        # if we are out of the warmup zone, we reset the early stopper
        if warmup is not None and epoch <= warmup and early_stopper is not None:
            early_stopper.reset()

        # check early stoppage
        if early_stopper is not None and early_stopper.check_stop(model, val_epoch_loss):
            print(f"\nEarly stoppage after {epoch} epochs with patience of {patience}.")
            final_loss_idx = early_stopper.best_loss_idx
            print(f"Best epoch: {final_loss_idx}")
            if final_loss_idx == 1:
                # 2 losses if the best epoch was the first
                # this avoids plotting a single point in other functions
                final_loss_idx = 2
            losses["train"] = losses["train"][:final_loss_idx]
            losses["val"] = losses["val"][:final_loss_idx]
            all_parts["train"] = {
                key: val[:final_loss_idx] for key, val in all_parts["train"].items()
            }
            all_parts["val"] = {
                key: val[:final_loss_idx] for key, val in all_parts["val"].items()
            }
            clusters["train"] = clusters["train"][:final_loss_idx]
            clusters["val"] = clusters["val"][:final_loss_idx]

            # saving
            if save_path is not None:
                print(f"\nSaving model..")
                save_model(early_stopper.best_model, path=save_path)

            return early_stopper.best_model, losses, all_parts, clusters, all_betas[:final_loss_idx]

        # store losses/parts/clusters for this epoch
        losses["train"].append(epoch_loss)
        losses["val"].append(val_epoch_loss)

        for key in all_parts["train"].keys():
            all_parts["train"][key].append(np.mean(epoch_parts[key]))
        for key in all_parts["val"].keys():
            all_parts["val"][key].append(np.mean(val_epoch_parts[key]))

        clusters["train"].append(epoch_clusters)
        clusters["val"].append(val_epoch_clusters)

        if epoch == 1:
            print(
                "Loss printing format:\n"
                "epoch x: val = loss (-recon - beta_kl * (kl_latent + kl_cluster)) | "
                "train = loss (-recon - beta_kl * (kl_latent + kl_cluster))\n"
            )

        if epoch % show_loss_every == 0:
            print(
                f"epoch {epoch}: "
                f"val = {losses['val'][-1]:.4f} ({format_loss(val_epoch_parts, beta_kl)}) | "
                f"train = {losses['train'][-1]:.4f} ({format_loss(epoch_parts, beta_kl)})"
            )

    # saving
    if save_path is not None:
        print(f"\nSaving model..")
        save_model(model, path=save_path)
        
    return model, losses, all_parts, clusters, all_betas


def retrieve_latent(model, dataloader):
    latent_mix_params = []
    latent_mix_samples = []
    latent_cluster_params = []
    latent_cluster_samples = []

    model.eval()
    with torch.no_grad():
        for batch in dataloader:
                try:
                    x = batch["X"][:, 0, :]
                except:
                    x = batch[0]
                (input_params, 
                z_mixture, 
                latent_params_mixture, 
                cluster_probas, 
                all_z, 
                all_latent
                ) = model(x)
                clusters = model.cluster_input(cluster_probas=cluster_probas)

                latent_mix_params.append(latent_params_mixture)
                latent_mix_samples.append(z_mixture)

                # Fetch the inferred z conditioned on the cluster of the sample
                latent_cluster_samples = [all_z[clusters[i]][i:i+1,:] for i in range(x.size(0))]
                latent_cluster_samples = torch.cat(latent_cluster_samples, dim=0)
                latent_cluster_params = [all_latent[clusters[i]][i:i+1,:] for i in range(x.size(0))]
                latent_cluster_params = torch.cat(latent_cluster_params, dim=0)
    return (latent_mix_params,
            latent_mix_samples,
            latent_cluster_params,
            latent_cluster_samples)

if __name__ == "__main__":
    # ensures the working dir is that of the file
    file_parent = "/".join(os.path.abspath(__file__).split("/")[:-1])
    os.chdir(file_parent)

    from torch.utils.data import TensorDataset, DataLoader
    from argparse import ArgumentParser
    from mixture_vae.distributions import NormalDistribution, UniformDistribution, NegativeBinomial

    parser = ArgumentParser()
    parser.add_argument("--model", default=0, type=int, help="Model type")

    args = parser.parse_args()
    model_type = args.model
    
    # Train Toy data
    X = torch.randint(0,50, (5000,5), dtype=torch.float)  # count data, 100 samples, 5 features
    dataset = TensorDataset(X)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=False) 
    # shuffle = False because we need order to be safe for cluster trakcing

    # Val Toy data
    X_val = torch.randint(0,50, (500,5), dtype=torch.float)  # count data, 100 samples, 5 features
    val_dataset = TensorDataset(X_val)
    val_dataloader = DataLoader(val_dataset, batch_size=64, shuffle=False)
    # shuffle = False because we need order to be safe for cluster trakcing
    
    # Problem setup
    input_dim = 5 # 5 genes
    hidden_dim = 16 # 8 hidden neurons per layer
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

    # Posterior on latent: Gaussian on R2 
    # (here assumed posterior = assumed prior 
    # but it could have been differnet)
    mu = torch.zeros((1,latent_dim))
    std = torch.ones((1,latent_dim))
    posterior_latent = NormalDistribution({"mu":mu,
                                           "std":std})
    
    if model_type == 0:
        n_components = 3 # 3 clusters are assumed
        # Prior on cluster repartitions (mixture): Assume balanced 
        # cluster classes = Uniform on [0,1]
        a = torch.zeros((1,n_components))
        b = torch.ones((1,n_components))
        prior_categorical = UniformDistribution({"a":a, 
                                                "b":b})
    
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
        # Prior on cluster repartitions (mixture): Assume balanced 
        # cluster classes = Uniform on [0,1]
        cat_priors = {}
        for n_components in [2, 4, 8]:
            a = torch.zeros((1,n_components))
            b = torch.ones((1,n_components))
            prior_categorical = UniformDistribution({"a":a, 
                                                    "b":b})
            cat_priors[n_components] = prior_categorical
        
        # Instatiate ind_MoMVAE
        model = ind_MoMVAE(
            PARAMS = [
            {"input_dim": input_dim,
            "hidden_dim": 128,
            "n_components": n_components,
            "n_layers": 1,
            "prior_latent": prior_latent,
            "prior_input": prior_input,
            "prior_categorical": cat_priors[n_components],
            "posterior_latent": posterior_latent} for n_components in [2, 4, 8]]
        )
        
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    EPOCHS = 5
    BETA_KL = 0.5
    WARMUP_BETA = int(0.2*EPOCHS)
    PATIENCE = 5
    TOL = 5e-3

    save_path = f"./model_{model.__class__.__name__}.ckpt"

    model, losses, parts, clusters, all_betas = training_mvae(
        dataloader,
        val_dataloader,
        model,
        optimizer,
        epochs=EPOCHS,
        beta_kl=BETA_KL,
        warmup=WARMUP_BETA,
        patience=PATIENCE,
        tol=TOL,
        show_loss_every=1,
        model_type=model_type,
        track_clusters=True,
        save_path=save_path
    )

    print("Loading model..")
    model = load_model(save_path)
    print("Testing loaded model")
    compute_ll(model, val_dataloader)

    # plot training and validation losses
    plot_loss_components(parts["train"], 
                         parts["val"], 
                         all_betas, 
                         title="Loss Breakdown",
                         save_path=f"./{model.__class__.__name__}toy_losses.pdf")
    
    plot_latent(model, 
                val_dataloader,
                level=-1,
                true_labels=False,
                label_key=None,
                title="Latent Space",
                save_path=f"./{model.__class__.__name__}_latent.pdf")
    