import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from string import ascii_uppercase


sns.set_theme("paper")


def tab_clustering_plot(true_labels: list[np.ndarray[int]], latents: list[np.ndarray[float]], labels: list[np.ndarray[int]], sizes: list[int], colors: list[dict[int: str]], tsne: bool = True) -> None:
    """
    """
    match tsne:

        case True:

            f = lambda x: TSNE().fit_transform(PCA().fit_transform(x))

        case False:
            
            f = lambda x: x

        case _:

            raise TypeError(":param tsne: must be a bool.")
        
    assert len(latents) == len(labels) == len(sizes) == len(colors)
        
    match tsne:
        case True:
            f = lambda x: TSNE().fit_transform(PCA().fit_transform(x))
        case False:
            f = lambda x: x
        case _:
            raise TypeError(":param tsne: must be a bool.")
        
    fig = plt.figure(figsize=(18, 10))
    fig.suptitle("COMPARISON BETWEEN LABELS AND LATENTS", fontsize=24)

    alphabet = ascii_uppercase

    rows = [[alphabet[i], alphabet[i+1]] for i in range(0, len(latents) * 2, 2)]

    fig = plt.figure(figsize=(18, 10))
    fig.suptitle("COMPARISON BETWEEN LABELS AND LATENTS", fontsize=24)

    ax = fig.subplot_mosaic(rows)

    ax["A"].set_title("TRUE LABELS / LATENT")
    ax["B"].set_title("PREDICTED LABELS / LATENT")

    for lvl, (latent, true_label, pred_label, size) in enumerate(zip(latents, true_labels, labels, sizes), start=1):

        base = (lvl - 1) * 2
        p_true = alphabet[base]
        p_pred = alphabet[base + 1]

        latent = f(latent)

        true_colors = [colors[lvl-1][lab] for lab in true_label]
        pred_colors = [colors[lvl-1][lab] for lab in pred_label]

        ax[p_true].set_ylabel(f"{size=}", fontsize=14)
        ax[p_true].scatter(latent[:, 0], latent[:, 1], c=true_colors, alpha=0.3)
        ax[p_pred].scatter(latent[:, 0], latent[:, 1], c=pred_colors, alpha=0.3)

    plt.tight_layout()
    plt.savefig("tab.svg")
    plt.show()
    plt.close()

    return None


def matrix_clustering_plot(latents: list[np.ndarray[float]], labels: list[np.ndarray[int]], sizes: list[int], colors: list[dict[int: str]], tsne: bool = True) -> None:
    """
    """
    match tsne:

        case True:

            f = lambda x: TSNE().fit_transform(PCA().fit_transform(x))

        case False:
            
            f = lambda x: x

        case _:

            raise TypeError(":param tsne: must be a bool.")
        
    assert len(latents) == len(labels) == len(sizes)
    n = len(labels)

    fig = plt.figure(figsize=(4*n, 4*(n+1)))
    gs = fig.add_gridspec(n, n)

    fig.supxlabel("LABELS", fontsize=22)
    fig.supylabel("LATENTS", fontsize=22)
    fig.suptitle("COMPARISON BETWEEN LABELS AND LATENTS", fontsize=24)

    for j, (latent, size_lat) in enumerate(zip(latents, sizes)):

        latent = f(latent)

        for i, (label, size_lab) in enumerate(zip(labels[:j+1], sizes[:j+1])):

            ax = fig.add_subplot(gs[j, i])

            mapped_colors = [colors[i][lab] for lab in label]

            ax.scatter(latent[:, 0], latent[:, 1], c=mapped_colors, alpha=0.3)

            if i == 0:
                ax.set_ylabel(f"latent({size_lat})")
            if j == n - 1:
                ax.set_xlabel(f"label({size_lab})")

    plt.tight_layout()
    plt.savefig("matrix.svg")
    plt.show()
    plt.close()

    return None