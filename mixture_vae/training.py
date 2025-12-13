"""Sub-Module to define training protocols"""
import torch
import math
from torch.optim.lr_scheduler import LambdaLR, CosineAnnealingLR, SequentialLR
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
from mixture_vae.utils import *

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

def format_loss(val_epoch_parts, beta_kl, reg_marg=0.0):
    """
    print the loss components in the form:
    -recon + beta_kl * (kl_latent + kl_cluster)
    while handling signs cleanly (no '--' or '+ -').
    """
    recon = val_epoch_parts["recon"][-1]
    kl_latent = val_epoch_parts["kl_latent"][-1]
    kl_cluster = val_epoch_parts["kl_cluster"][-1]

    # Recon is always shown as subtraction
    # the loss is - recon
    if recon >= 0:
        recon_str = f"- {recon:.3f}"
    else:
        recon_str = f"{-recon:.3f}"

    kl_latent_str = f"{kl_latent:.3f}"

    kl_cluster_str = f"{kl_cluster:.3f}"

    if "kl_marginal" in val_epoch_parts.keys():
        kl_marg = val_epoch_parts["kl_marginal"][-1]
        kl_marg_str = f"{kl_marg:.3f}"
        return f"{recon_str} + {beta_kl:.3f} * ({kl_latent_str} + ({kl_cluster_str})) + {reg_marg:.3f} * {kl_marg_str}"

    return f"{recon_str} + {beta_kl:.3f} * ({kl_latent_str} + ({kl_cluster_str}))"


def _build_default_cosine_scheduler(optimizer: torch.optim.Optimizer, epochs: int):
    """
    Default: 5% epoch warmup from 0 -> base_lr, then cosine anneal to base_lr/10 by final epoch.
    Stepped once per epoch.
    """
    warmup_epochs = max(1, int(0.05 * epochs))
    cosine_epochs = max(1, epochs - warmup_epochs)

    # --- warmup: scale LR linearly from 0 to 1 over warmup_epochs
    def warmup_lambda(e):
        # e is epoch index starting at 0
        return float(e + 1) / float(warmup_epochs)

    warmup = LambdaLR(optimizer, lr_lambda=warmup_lambda)

    # --- cosine: from base_lr -> base_lr/10
    # CosineAnnealingLR goes to eta_min at T_max.
    # Set eta_min per param group to base_lr/10.
    base_lrs = [pg["lr"] for pg in optimizer.param_groups]
    eta_min = min(base_lrs) / 10.0  # conservative choice; or set per group if you prefer

    cosine = CosineAnnealingLR(optimizer, T_max=cosine_epochs, eta_min=eta_min)

    # Chain them
    sched = SequentialLR(optimizer, schedulers=[warmup, cosine], milestones=[warmup_epochs])
    return sched, warmup_epochs


def training_mvae(
    dataloader: torch.utils.data.DataLoader,
    val_dataloader: torch.utils.data.DataLoader,
    model: MixtureVAE,
    optimizer: torch.optim.Optimizer,
    epochs: int = 50,
    beta_kl: float = 1,
    reg_marg: float = 2,
    warmup: int | None = None,
    min_beta: float = 0.0,
    scheduler: torch.optim.lr_scheduler.LRScheduler | bool | None = None,
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
    ), f"This training loop is tailored for MixtureVAE or ind_MoMVAE modules but got {type(model)}"

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
    if reg_marg > 0:
        all_parts["train"]["kl_marginal"] = []
        all_parts["val"]["kl_marginal"] = []
        
    clusters = {"train": [], "val": []}

    if scheduler is None:
        scheduler, _lr_warmup_epochs = _build_default_cosine_scheduler(optimizer, epochs)
    elif scheduler == False:
        scheduler = None

    for epoch in tqdm(range(1, epochs + 1), desc="TRAINING"):
        beta_kl = all_betas[epoch - 1]

        # ==========================
        # TRAINING
        # ==========================
        model.train()
        epoch_loss = 0.0
        # Initialize lists to store detached tensors on GPU
        epoch_parts = {key:[] for key in all_parts["train"].keys()}
        epoch_clusters = []

        for batch in tqdm_func(dataloader, desc=f"Epoch {epoch} (train)", total=len(dataloader)):
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
                    reg_marginal=reg_marg,
                    track_clusters=track_clusters,
                )

            elif model_type == 1:
                loss, parts, batch_clusters = summed_elbo_mixture_step(
                    model,
                    x,
                    beta_kl=beta_kl,
                    reg_marginal=reg_marg,
                    track_clusters=track_clusters,
                )

            # backward + step
            loss.backward()
            optimizer.step()

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
        if len(epoch_clusters) > 0 and None not in epoch_clusters:
            epoch_clusters = torch.cat(epoch_clusters, dim=0).detach().cpu()
        else:
            epoch_clusters = torch.empty(0)

        # ==========================
        # VALIDATION
        # ==========================
        model.eval()
        val_epoch_loss = 0.0
        val_epoch_parts = {key:[] for key in all_parts["val"].keys()}
        val_epoch_clusters = []

        with torch.no_grad():
            for batch in tqdm_func(val_dataloader, desc=f"Epoch {epoch} (val)", total=len(val_dataloader)):
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
                        reg_marginal=reg_marg,
                        track_clusters=track_clusters,
                    )

                elif model_type == 1:
                    loss, parts, batch_clusters = summed_elbo_mixture_step(
                        model,
                        x,
                        beta_kl=beta_kl,
                        reg_marginal=reg_marg,
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
            if len(val_epoch_clusters) > 0 and None not in val_epoch_clusters:
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
            if reg_marg > 0:
                addon = " + reg_marg * kl_marg"
            else:
                addon = ""
            print(
                "Loss printing format:\n"
                f"epoch x: val = loss (-recon + beta_kl * (kl_latent + kl_cluster){addon}) | "
                f"train = loss (-recon + beta_kl * (kl_latent + kl_cluster){addon})"
                " | lr=current_lr\n"
            )
        
        if scheduler is not None and scheduler != False:
            scheduler.step()

        if epoch == 1 or epoch % show_loss_every == 0:
            current_lr = optimizer.param_groups[0]["lr"]
            print(
                f"epoch {epoch}: "
                f"val = {losses['val'][-1]:.3f} ({format_loss(val_epoch_parts, beta_kl, reg_marg)}) | "
                f"train = {losses['train'][-1]:.3f} ({format_loss(val_epoch_parts, beta_kl, reg_marg)})"
                f" | lr={current_lr:.3e} "
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
    reg_marg: float = 2,
    warmup: int | None = None,
    min_beta: float = 0.0,
    scheduler: torch.optim.lr_scheduler.LRScheduler | bool | None = None,
    track_clusters: bool = False,
    patience: int | None = 5,
    tol: float | None = 0.0,
    show_loss_every: int = 1,
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

    if reg_marg > 0:
        all_parts["train"]["kl_marginal"] = []
        all_parts["val"]["kl_marginal"] = []

    clusters = {"train": [], "val": []}

    if scheduler is None:
        scheduler, _lr_warmup_epochs = _build_default_cosine_scheduler(optimizer, epochs)
    elif scheduler == False:
        scheduler = None

    for epoch in tqdm(range(1, epochs + 1), desc="TRAINING"):
        beta_kl = all_betas[epoch - 1]

        # ==========================
        # TRAINING
        # ==========================
        model.train()
        epoch_loss = 0.0
        # Initialize lists to store detached tensors on GPU
        epoch_parts = {key:[] for key in all_parts["train"].keys()}
        epoch_clusters = []

        for batch in tqdm_func(dataloader, desc=f"Epoch {epoch} (train)", total=len(dataloader)):
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
                reg_marginal=reg_marg,
                track_clusters=track_clusters,
            )

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

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
        if len(epoch_clusters) > 0 and None not in epoch_clusters:
            epoch_clusters = torch.cat(epoch_clusters, dim=0).detach().cpu()
        else:
            epoch_clusters = torch.empty(0)

        # ==========================
        # VALIDATION
        # ==========================
        model.eval()
        val_epoch_loss = 0.0
        val_epoch_parts = {key:[] for key in all_parts["val"].keys()}
        val_epoch_clusters = []

        with torch.no_grad():
            for batch in tqdm_func(val_dataloader, desc=f"Epoch {epoch} (val)", total=len(val_dataloader)):
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
                    reg_marginal=reg_marg,
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
            if len(val_epoch_clusters) > 0 and None not in val_epoch_clusters:
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
            if reg_marg > 0:
                addon = " + reg_marg * kl_marg"
            else:
                addon = ""
            print(
                "Loss printing format:\n"
                f"epoch x: val = loss (-recon + beta_kl * (kl_latent + kl_cluster){addon}) | "
                f"train = loss (-recon + beta_kl * (kl_latent + kl_cluster){addon})"
                " | lr=current_lr\n"
            )

        # step LR scheduler once per epoch (after epoch work)
        if scheduler is not None and scheduler != False:
            scheduler.step()
        
        if epoch == 1 or epoch % show_loss_every == 0:
            current_lr = optimizer.param_groups[0]["lr"]
            print(
                f"epoch {epoch}: "
                f"val = {losses['val'][-1]:.3f} ({format_loss(val_epoch_parts, beta_kl, reg_marg)}) | "
                f"train = {losses['train'][-1]:.3f} ({format_loss(epoch_parts, beta_kl, reg_marg)})"
                f" | lr={current_lr:.3e} "
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

    from mixture_vae.distributions import (
        NormalDistribution,
        Student,
        NegativeBinomial,
        CategoricalDistribution,   # <-- NEW
    )

    parser = ArgumentParser()
    parser.add_argument("--model", default=0, type=int, help="0: MixtureVAE, 1: ind_MoMVAE")
    args = parser.parse_args()
    model_type = args.model

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Train toy data
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
    latent_dim = 2

    # -------------------------
    # Prior on input gene counts: NB for each gene
    # -------------------------
    params = init_nb_params_mom(dataloader, "cpu")
    prior_input = NegativeBinomial(params)

    # -------------------------
    # Posterior on latent: Gaussian on R^D (shared across all models/branches)
    # Learned params are still (B, 2D) from the encoder.
    # -------------------------
    mu0 = torch.zeros((1, latent_dim), device=device)
    std0 = torch.ones((1, latent_dim), device=device)
    df0 = torch.ones((1, latent_dim), device=device) * 10.0
    
    posterior_latent = Student({"df": df0, "mu": mu0, "std": std0}) #NormalDistribution({"mu": mu0, "std": std0})

    # ============================================================
    # Model 0: MixtureVAE (single mixture)
    # ============================================================
    if model_type == 0:
        n_components = K

        # NEW: PCA+KMeans init for this K
        latent_params = initialize_gmm_params(
            dataloader, n_components=n_components, latent_dim=latent_dim, device=device
        )
        cat_params = init_categorical_uniform(n_components, device)

        # NEW: non-uniform categorical prior
        prior_categorical = CategoricalDistribution(cat_params)

        # NEW: component-specific latent prior (mu/std are (K,D))
        latent_params = initialize_student_prior_params_from_gmm(latent_params, device, df0=10)
        prior_latent = Student(latent_params) #NormalDistribution(latent_params)

        model = MixtureVAE(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            n_components=n_components,
            n_layers=1,
            prior_latent=prior_latent,
            prior_input=prior_input,
            prior_categorical=prior_categorical,
            posterior_latent=posterior_latent
        ).to(device)

    # ============================================================
    # Model 1: ind_MoMVAE (multiple independent branches)
    # ============================================================
    elif model_type == 1:
        branch_components = [2,3,K]

        PARAMS = []
        for K in branch_components:
            # NEW: init per-branch K
            latent_params = initialize_gmm_params(
                dataloader, n_components=K, latent_dim=latent_dim, device=device
            )
            cat_params = init_categorical_uniform(K, device)

            prior_categorical = CategoricalDistribution(cat_params)
            latent_params = initialize_student_prior_params_from_gmm(latent_params, device, df0=10)
            prior_latent = Student(latent_params) #NormalDistribution(latent_params)

            PARAMS.append({
                "input_dim": input_dim,
                "hidden_dim": hidden_dim,
                "n_components": K,
                "n_layers": 1,
                "prior_latent": prior_latent,
                "prior_input": prior_input,
                "prior_categorical": prior_categorical,
                "posterior_latent": posterior_latent,
            })

        model = ind_MoMVAE(PARAMS=PARAMS).to(device)

    else:
        raise ValueError(f"Unsupported model_type={model_type}. Use 0 (MixtureVAE) or 1 (ind_MoMVAE).")

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    EPOCHS = 1
    BETA_KL = 0.5
    REG_MARG = 10 # controls the "careful, use all components" message to the network
    WARMUP_BETA = int(0.2 * EPOCHS)
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
        reg_marg=REG_MARG,
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
    model.to(device)

    print("Testing loaded model")
    print(compute_ll(model, val_dataloader))

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