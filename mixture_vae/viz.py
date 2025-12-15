import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme("paper")
import numpy as np
import os
import torch

from sklearn.manifold import TSNE
import umap
from collections import Counter

from mixture_vae.mvae import MixtureVAE
from mixture_vae.utils import match_labels

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


def plot_latent(model, 
                loader,
                level=-1,
                true_labels=False,
                label_key=4,
                title="Latent Space",
                save_path=None):
    """
    Visualizes the latent space of the model. 
    If dimension > 2, it projects using UMAP and t-SNE.
    """
    
    # 1. Validation & Setup
    device = next(model.parameters()).device
    model.eval()
    
    
    # Warning for MixtureVAE ignoring level
    if isinstance(model, MixtureVAE) and level != -1:
        print(f"Warning: Model is 'MixtureVAE' which does not support other levels than -1. Ignoring level={level} and using standard encoding.")
    
    if isinstance(model, MixtureVAE):
      level = label_key if label_key is not None else level

    if level == -1:
        level = model.n_levels

    all_latent = []
    all_labels = []
    
    # 2. Extract Latents and Labels
    print("Extracting latent representations...")
    with torch.no_grad():
        for batch in loader:
            try:
                x = batch["X"][:, 0, :]
                y_true = batch[f"y{level}"] if len(batch) > 1 else None
                y_true = torch.argmax(y_true.squeeze(), dim=1)
            except:
                x = batch[0]
                if len(batch) > 1:
                    if level < len(batch):
                        y_true = batch[level]
                    else:
                        y_true = batch[1]
                    y_true = torch.argmax(y_true.squeeze(), dim=1)
                else:
                    y_true = None

            x = x.to(device)

            # --- Encode ---
            # Handle different method signatures
            if isinstance(model, MixtureVAE):
                # MixtureVAE.encode returns: z, latent_params, cluster_probas, ...
                enc_out = model.encode(x)
                latent = enc_out[1] # z is the first return
                
                # Get model inferred clusters
                if not true_labels:
                    # mixture_vae cluster_input logic might differ slightly, usually argmax probas
                    # enc_out[2] are cluster_probas
                    batch_labels = torch.argmax(enc_out[2], dim=1).cpu().numpy()
                    
            else: 
                # MoMixVAE / ind_MoMVAE
                # They support at_level
                enc_out = model.encode(x, at_level=level)
                latent = enc_out[1]
                
                # Get model inferred clusters
                if not true_labels:
                    # model.cluster_input needs x or probas. 
                    # We can use the probas returned in enc_out to save computation
                    # For MoMix: enc_out[2] is cluster_probas for the *specific level* requested
                    probas = enc_out[2] 
                    batch_labels = torch.argmax(probas, dim=1).cpu().numpy()

            all_latent.append(latent.cpu().numpy())

            # --- Handle Labels ---
            if true_labels:
                if y_true is None:
                    raise ValueError("true_labels=True but DataLoader did not yield labels.")
                
                # If label_key is complex (e.g., dict), handle extraction here if needed
                # For now assuming y_true is the tensor of labels
                if isinstance(y_true, dict) and label_key:
                     batch_labels = y_true[label_key].cpu().numpy()
                else:
                     batch_labels = y_true.cpu().numpy()
                
                all_labels.append(batch_labels)
            else:
                # We calculated model-inferred labels above
                all_labels.append(batch_labels)

    # Concatenate all batches
    LATENT = np.concatenate(all_latent, axis=0)
    labels = np.concatenate(all_labels, axis=0).flatten()
    
    # 3. Dimensionality Reduction Logic
    latent_dim = LATENT.shape[1]
    
    projections = {}
    
    if latent_dim > 2:
        print(f"Latent dim is {latent_dim} (>2). Computing UMAP and t-SNE projections...")
        
        # UMAP
        print("Running UMAP...")
        reducer = umap.UMAP()
        latent_umap = reducer.fit_transform(LATENT)
        projections["UMAP"] = latent_umap
        
        # t-SNE
        print("Running t-SNE...")
        tsne = TSNE(n_components=2, n_jobs=-1, init='pca', learning_rate='auto')
        latent_tsne = tsne.fit_transform(LATENT)
        projections["t-SNE"] = latent_tsne
    else:
        # If 2D (or 1D), just plot directly
        projections["Latent"] = LATENT

    # 4. Plotting
    num_plots = len(projections)
    fig, axes = plt.subplots(1, num_plots, figsize=(6 * num_plots, 5), squeeze=False)
    axes = axes.flatten()

    # Determine unique labels for legend
    unique_labels = np.unique(labels)
    # Use a categorical palette
    palette = sns.color_palette("tab10", len(unique_labels))

    for ax, (proj_name, data) in zip(axes, projections.items()):
        
        # Determine axes to plot
        x_axis = data[:, 0].flatten()
        y_axis = data[:, 1].flatten() if data.shape[1] > 1 else np.zeros_like(x_axis) # Handle 1D
        
        sns.scatterplot(
            x=x_axis, 
            y=y_axis, 
            hue=labels, 
            palette=palette, 
            alpha=0.7, 
            s=40, 
            ax=ax,
            legend="full"
        )
        
        ax.set_title(f"{title} ({'Model' if not true_labels else 'True'} labels) - {proj_name}")
        ax.set_xlabel(f"{proj_name} 1")
        ax.set_ylabel(f"{proj_name} 2")
        
        # Improve legend placement
        if len(unique_labels) > 10: # If many clusters, hide legend or put outside
            ax.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
        else:
            ax.legend(loc='best')

    plt.tight_layout()

    # 5. Save or Show
    if save_path:
        # ensure dir exists
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
        plt.close()
    else:
        plt.show()

def plot_latent_comparison(
    model,
    loader,
    level=-1,
    label_key=None,
    title="Latent Space Comparison",
    save_path=None,
    apply_hungarian=True,
):
    """
    Plot latent space with TRUE vs MODEL-PREDICTED labels in a single figure (2xN layout).

    Row 1: True labels (named + colored)
    Row 2: Model labels (Hungarian-matched to True numeric convention, then named + colored)
    Columns: Latent / UMAP | t-SNE (if applicable)
    """
    # ------------------------------------------------------------------
    # Label associations + colors (MUST live inside the function)
    # ------------------------------------------------------------------
    associations = [
        {0: "Stem_cell", 1: "Non_stem"},
        {0: "CD34_HSC", 1: "B_cell", 2: "NK_cell", 3: "T_cell"},
        {0: "CD34_HSC", 1: "B_cell", 2: "NK_cell", 3: "CD4_T", 4: "CD8_T"},
        {
            0: "CD34_HSC",
            1: "CD19_B_cell",
            2: "CD56_NK_cell",
            3: "CD4_CD3_T_helper",
            4: "CD4_CD25_Treg",
            5: "CD4_CD45RA_CD25neg_naive_T",
            6: "CD4_CD45RO_memory_T",
            7: "CD8_cytotoxic_T",
            8: "CD8_CD45RA_naive_T",
        },
    ]

    colors = [
        {0: "lime", 1: "salmon"},
        {0: "lime", 1: "crimson", 2: "darkslategray", 3: "mediumorchid"},
        {0: "lime", 1: "crimson", 2: "darkslategray", 3: "royalblue", 4: "peru"},
        {
            0: "lime",
            1: "crimson",
            2: "darkslategray",
            3: "slateblue",
            4: "rebeccapurple",
            5: "thistle",
            6: "fuchsia",
            7: "darkorange",
            8: "tan",
        },
    ]

    sns.set_theme("paper")
    device = next(model.parameters()).device
    model.eval()

    # ------------------------------------------------------------------
    # Resolve hierarchy level
    # ------------------------------------------------------------------
    if isinstance(model, MixtureVAE) and level != -1:
        print("Warning: MixtureVAE does not support level != -1. Using standard encoding.")

    if isinstance(model, MixtureVAE):
        level = label_key if label_key is not None else level

    if level == -1:
        level = model.n_levels

    assoc_idx = level - 1
    assoc = associations[assoc_idx] if 0 <= assoc_idx < len(associations) else {}
    colmap = colors[assoc_idx] if 0 <= assoc_idx < len(colors) else {}

    # ------------------------------------------------------------------
    # DEBUG: numeric → textual mapping for this level
    # ------------------------------------------------------------------
    print("\n" + "=" * 80)
    print(f"[DEBUG] Active hierarchy level: y{level}")
    print("[DEBUG] Numeric → textual label mapping:")
    for k in sorted(assoc.keys()):
        print(f"  {k:>2} → {assoc[k]}")
    print("=" * 80 + "\n")

    def to_name(lbl: int) -> str:
        return assoc.get(int(lbl), str(int(lbl)))

    def to_color(lbl: int):
        return colmap.get(int(lbl), None)

    all_latent, all_true_labels, all_model_labels = [], [], []

    print("Extracting latent representations...")
    with torch.no_grad():
        for batch in loader:
            try:
                x = batch["X"][:, 0, :]
                y_true = batch[f"y{level}"] if len(batch) > 1 else None
                size_true = y_true.squeeze().size(1)
                y_true = torch.argmax(y_true.squeeze(), dim=1)
            except Exception:
                x = batch[0]
                if len(batch) > 1:
                    if level < len(batch):
                        y_true = batch[level]
                    else:
                        y_true = batch[1]
                        level = model.n_levels
                    size_true = y_true.squeeze().size(1)
                    y_true = torch.argmax(y_true.squeeze(), dim=1)
                else:
                    y_true = None

            if y_true is None:
                raise ValueError("True labels required for comparison plot.")

            x = x.to(device)

            if isinstance(model, MixtureVAE):
                enc_out = model.encode(x)
                latent = enc_out[1]
                size_model = enc_out[2].size(1)
                model_labels = torch.argmax(enc_out[2], dim=1)
            else:
                enc_out = model.encode(x, at_level=level - 1)
                latent = enc_out[1]
                size_model = enc_out[2].size(1)
                model_labels = torch.argmax(enc_out[2], dim=1)

            all_latent.append(latent.cpu().numpy())
            all_true_labels.append(y_true.cpu().numpy())
            all_model_labels.append(model_labels.cpu().numpy())

    LATENT = np.concatenate(all_latent, axis=0)
    true_labels = np.concatenate(all_true_labels).astype(int)
    model_labels_raw = np.concatenate(all_model_labels).astype(int)
    
    if size_true < size_model:
        print(f"\nSKIPPING LEVEL {level}")
        return None
    # ------------------------------------------------------------------
    # DEBUG: cluster counts BEFORE alignment
    # ------------------------------------------------------------------
    print("[DEBUG] Cluster counts BEFORE Hungarian alignment")
    print(f"  True labels    : {len(np.unique(true_labels))} clusters")
    print(f"  Model labels   : {len(np.unique(model_labels_raw))} clusters")

    print("  True label distribution:")
    print("   ", Counter(true_labels))
    print("  Model label distribution (raw):")
    print("   ", Counter(model_labels_raw))
    print()

    # ------------------------------------------------------------------
    # Hungarian alignment
    # ------------------------------------------------------------------
    if apply_hungarian:
        model_labels_aligned = match_labels(true_labels, model_labels_raw)
    else:
        model_labels_aligned = model_labels_raw

    # ------------------------------------------------------------------
    # DEBUG: cluster counts AFTER alignment
    # ------------------------------------------------------------------
    print("[DEBUG] Cluster counts AFTER Hungarian alignment")
    print(f"  Model labels (aligned): {len(np.unique(model_labels_aligned))} clusters")
    print("  Model label distribution (aligned):")
    print("   ", Counter(model_labels_aligned))
    print()

    # ------------------------------------------------------------------
    # DEBUG: explicit numeric remapping (raw → aligned)
    # ------------------------------------------------------------------
    print("[DEBUG] Hungarian remapping (model raw → model aligned → textual)")
    raw_ids = sorted(np.unique(model_labels_raw).tolist())
    for rid in raw_ids:
        aligned_ids = np.unique(model_labels_aligned[model_labels_raw == rid])
        aligned_ids = aligned_ids.tolist()
        aligned_txt = [to_name(i) for i in aligned_ids]
        print(f"  raw {rid:>2} → aligned {aligned_ids} → {aligned_txt}")
    print()

    model_labels = model_labels_aligned.astype(int)

    # ------------------------------------------------------------------
    # Projections
    # ------------------------------------------------------------------
    projections = {}
    latent_dim = LATENT.shape[1]

    if latent_dim > 2:
        print("Latent dim > 2 → computing UMAP and t-SNE")
        reducer = umap.UMAP()
        projections["UMAP"] = reducer.fit_transform(LATENT)

        tsne = TSNE(n_components=2, n_jobs=-1, init="pca", learning_rate="auto")
        projections["t-SNE"] = tsne.fit_transform(LATENT)
    else:
        projections["Latent"] = LATENT

    # ------------------------------------------------------------------
    # Build palette + named labels
    # ------------------------------------------------------------------
    label_universe = np.unique(np.concatenate([true_labels, model_labels])).astype(int)
    ordered_ids = sorted(label_universe.tolist())
    ordered_names = [to_name(i) for i in ordered_ids]

    palette = {to_name(i): to_color(i) for i in ordered_ids if to_color(i) is not None}

    true_names = np.array([to_name(i) for i in true_labels], dtype=object)
    model_names = np.array([to_name(i) for i in model_labels], dtype=object)

    # ------------------------------------------------------------------
    # Plotting
    # ------------------------------------------------------------------
    n_cols = len(projections)
    fig, axes = plt.subplots(2, n_cols, figsize=(6 * n_cols, 10), sharex="col", sharey="col")
    if n_cols == 1:
        axes = np.expand_dims(axes, axis=1)

    for col_idx, (proj_name, data) in enumerate(projections.items()):
        x_axis = data[:, 0]
        y_axis = data[:, 1] if data.shape[1] > 1 else np.zeros_like(x_axis)

        sns.scatterplot(
            x=x_axis,
            y=y_axis,
            hue=true_names,
            hue_order=ordered_names,
            palette=palette if palette else None,
            s=40,
            alpha=0.7,
            ax=axes[0, col_idx],
            legend="full",
        )
        axes[0, col_idx].set_title(f"TRUE labels – {proj_name}")

        sns.scatterplot(
            x=x_axis,
            y=y_axis,
            hue=model_names,
            hue_order=ordered_names,
            palette=palette if palette else None,
            s=40,
            alpha=0.7,
            ax=axes[1, col_idx],
            legend="full",
        )
        axes[1, col_idx].set_title(f"MODEL labels – {proj_name}")

        axes[0, col_idx].set_xlabel("")
        axes[1, col_idx].set_xlabel(f"{proj_name} 1")
        axes[0, col_idx].set_ylabel(f"{proj_name} 2")
        axes[1, col_idx].set_ylabel(f"{proj_name} 2")

    fig.suptitle(title, fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    if save_path:
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else ".", exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"Plot saved to {save_path}")
    else:
        plt.show()