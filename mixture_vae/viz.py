import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

def save_plot(save_path):
    if "/" in save_path:
        parent_path = "/".join(save_path.split("/")[:-1])
        os.makedirs(parent_path, exist_ok=True)
    plot_ext = os.path.splitext(save_path)[-1][1:]
    plt.savefig(save_path, 
                bbox_inches='tight', 
                format=plot_ext)
    print(f"\nPlot saved at: {save_path}.")

def plot_loss_components(train_parts,
                         val_parts,
                         beta_kl_list,
                         title="Loss Breakdown",
                         save_path=None):
    """
    Plot the total loss and its components for train and val

    Args:
        parts (dict): Dictionary with keys "recon", "kl_latent", "kl_cluster".
                      Each value is a list/array of floats (per epoch).
        beta_kl_list (iterable): Weight for KL term.
        title (str): Title for the plot.
        save_path (str): Path to save the plot.
    """
    beta_kl = np.array(beta_kl_list) if not isinstance(beta_kl_list, np.ndarray) else beta_kl_list
    def extract(parts):
        recon = np.array(parts["recon"]).ravel()
        kl_latent = np.array(parts["kl_latent"]).ravel()
        kl_cluster = np.array(parts["kl_cluster"]).ravel()
        kl_total = kl_latent + kl_cluster
        weighted_kl = beta_kl * kl_total
        total_loss = -(recon - weighted_kl)
        return {
            "Reconstruction": recon,
            "KL Latent": kl_latent,
            "KL Cluster": kl_cluster,
            f"βKL Total": weighted_kl,
            "Total Loss": total_loss,
        }

    tr = extract(train_parts)
    va = extract(val_parts)

    epochs = np.arange(1, len(tr["Total Loss"]) + 1)

    plt.figure(figsize=(11, 9))
    sns.set_theme(style="whitegrid")

    # ----- Row 1: total loss
    ax = plt.subplot(3, 1, 1)
    sns.lineplot(x=epochs, y=tr["Total Loss"], label="train", ax=ax)
    sns.lineplot(x=epochs, y=va["Total Loss"], label="val", ax=ax)
    ax.set_title(f"{title} — total loss")
    ax.set_ylabel("Loss")

    # ----- Row 2: train components
    ax = plt.subplot(3, 1, 2)
    for key in ["Reconstruction", "KL Latent", "KL Cluster", "βKL Total"]:
        sns.lineplot(x=epochs, y=tr[key], label=key, ax=ax)
    ax.set_title("train components")
    ax.set_ylabel("value")

    # ----- Row 3: val components
    ax = plt.subplot(3, 1, 3)
    for key in ["Reconstruction", "KL Latent", "KL Cluster", "βKL Total"]:
        sns.lineplot(x=epochs, y=va[key], label=key, ax=ax)
    ax.set_title("val components")
    ax.set_xlabel("epoch")
    ax.set_ylabel("value")

    plt.tight_layout()

    if save_path:
        save_plot(save_path)
    else:
        plt.show()
