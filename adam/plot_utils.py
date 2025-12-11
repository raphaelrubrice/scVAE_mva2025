import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from string import ascii_uppercase


def tab_clustering_plot(true_labels: list[np.ndarray[int]], latents: list[np.ndarray[float]], labels: list[np.ndarray[int]], sizes: list[int], associations: list[dict[int: str]], tsne: bool = True) -> None:
    """
    """

    match tsne:

        case True:

            f = lambda x: TSNE().fit_transform(PCA().fit_transform(x))

        case False:
            
            f = lambda x: x

        case _:

            raise TypeError(":param tsne: must be a bool.")
        
    assert len(latents) == len(labels) == len(sizes) == len(associations)
        
    match tsne:
        case True:
            f = lambda x: TSNE().fit_transform(PCA().fit_transform(x))
        case False:
            f = lambda x: x
        case _:
            raise TypeError(":param tsne: must be a bool.")
        
    assert len(latents) == len(labels) == len(sizes)
        
    fig = plt.figure(figsize=(20, 10))
    fig.suptitle("COMPARISON BETWEEN LABELS AND LATENTS", fontsize=24)

    alphabet = ascii_uppercase
    rows = []
    for i in range(0, len(latents) * 3, 3):
        rows.append(alphabet[i] + alphabet[i+1] + alphabet[i+2])

    ax = fig.subplot_mosaic("\n".join(rows))

    ax["A"].set_title("TRUE LABELS / LATENT")
    ax["B"].set_title("PREDICTED LABELS / LATENT")
    ax["C"].set_title("LABELS")

    for lvl, (latent, true_label, pred_label, size) in enumerate(zip(latents, true_labels, labels, sizes), start=1):

        base = (lvl - 1) * 3
        p_true = alphabet[base]
        p_pred = alphabet[base + 1]
        p_text = alphabet[base + 2]

        latent = f(latent)

        ax[p_true].set_ylabel(f"{size=}", fontsize=16, rotation=90)
        ax[p_pred].scatter(latent[:, 0], latent[:, 1], c=pred_label, cmap="tab10", alpha=0.7)

        sc_true = ax[p_true].scatter(latent[:, 0], latent[:, 1], c=true_label, cmap="tab10", alpha=0.7)

        cmap = sc_true.cmap 
        norm = sc_true.norm

        unique_ids = sorted(associations[lvl - 1].keys())
        for idx, lab_id in enumerate(unique_ids):
            color = cmap(norm(lab_id))
            ax[p_text].text(0.1, 1 - idx * 0.1, f"{associations[lvl - 1][lab_id]}", color=color, fontsize=12)
        ax[p_text].set_axis_off()

    plt.show()
    plt.close()

    return None


def matrix_clustering_plot(latents: list[np.ndarray[float]], labels: list[np.ndarray[int]], sizes: list[int], associations: list[dict[int: str]], tsne: bool = True) -> None:
    """
    """
    match tsne:

        case True:

            f = lambda x: TSNE().fit_transform(PCA().fit_transform(x))

        case False:
            
            f = lambda x: x

        case _:

            raise TypeError(":param tsne: must be a bool.")
        
    assert len(latents) == len(labels) == len(sizes) == len(associations)

    n = len(labels)
        
    fig = plt.figure(figsize=(4*n, 4*(n+1)))
    gs = fig.add_gridspec(n+1, n)

    fig.supxlabel("LABELS", fontsize=22)
    fig.supylabel("LATENTS", fontsize=22)
    fig.suptitle("COMPARISON BETWEEN LABELS AND LATENTS", fontsize=24)

    col_scatter = [None] * n
    col_assocs  = associations

    for j, (latent, size_lat) in enumerate(zip(latents, sizes)):
        latent2d = f(latent)

        for i, (label, size_lab) in enumerate(zip(labels[:j+1], sizes[:j+1])):
            ax = fig.add_subplot(gs[j, i])

            sc = ax.scatter(latent2d[:, 0], latent2d[:, 1], c=label, cmap="tab10", alpha=0.7)

            col_scatter[i] = sc

            if i == 0:
                ax.set_ylabel(f"latent({size_lat})")
            if j == n - 1:
                ax.set_xlabel(f"label({size_lab})")

    for col in range(n):

        ax_leg = fig.add_subplot(gs[n, col])
        ax_leg.set_axis_off()

        sc = col_scatter[col]
        if sc is None:
            continue

        cmap = sc.cmap
        norm = sc.norm

        assoc = col_assocs[col]
        unique_ids = sorted(assoc.keys())

        for k, lab_id in enumerate(unique_ids):
            color = cmap(norm(lab_id))
            ax_leg.text(0.1, 1 - k*0.15, assoc[lab_id], color=color, fontsize=12)

    plt.tight_layout()
    plt.show()

    return None