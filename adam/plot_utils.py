import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


def tab_clustering_plot(true_labels: np.ndarray[float], latents: list[np.ndarray[float]], labels: list[np.ndarray[int]], sizes: list[int], tsne: bool = True) -> None:
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
        
    fig = plt.figure(figsize=(20, 10))
    fig.suptitle("COMPARISON BETWEEN LABELS AND LATENTS", fontsize=24)

    ax = fig.subplot_mosaic("\n".join([f"{i}{i+1}" for i in range(1, len(latents)*2+1, 2)]))
    ax["1"].set_title(f"TRUE LABELS / LATENT")
    ax["2"].set_title(f"PREDICTED LABELS / LATENT")


    for lvl, (latent, label, size) in enumerate(zip(latents, labels, sizes), 1):

        latent = f(latent)

        ax[str(lvl*2 - 1)].scatter(latent[:, 0], latent[:, 1], c=true_labels, cmap="tab10", alpha=0.7)

        ax[str(lvl*2 - 1)].set_ylabel(f"{size=}")

        ax[str(lvl*2)].scatter(latent[:, 0], latent[:, 1], c=label, cmap="tab10", alpha=0.7)

    plt.show()
    plt.close()

    return None


def matrix_clustering_plot(latents: list[np.ndarray[float]], labels: list[np.ndarray[int]], sizes: list[int], tsne: bool = True) -> None:
    """
    """
    match tsne:

        case True:

            f = lambda x: TSNE().fit_transform(PCA().fit_transform(x))

        case False:
            
            f = lambda x: x

        case _:

            raise TypeError(":param tsne: must be a bool.")
        
    assert len(latents) == len(labels)
        
    fig = plt.figure(figsize=(4*len(labels), 4*len(labels)))

    fig.supxlabel("LABELS", fontsize=22)
    fig.supylabel("LATENTS", fontsize=22)
    fig.suptitle("COMPARISON BETWEEN LABELS AND LATENTS", fontsize=24)

    gs = fig.add_gridspec(len(labels), len(labels))
    
    for j, (latent, size1) in enumerate(zip(latents, sizes)):

        latent: np.ndarray[float] = f(latent)
        
        for i, (label, size2) in enumerate(zip(labels[:j+1], sizes[:j+1])):

            ax = fig.add_subplot(gs[j, i])
            ax.scatter(latent[:, 0], latent[:, 1], c=label, cmap="tab10", alpha=0.7)

            if i == 0:
                ax.set_ylabel(f"latent({size1})")

            if j == len(labels) - 1:
                ax.set_xlabel(f"label({size2})")

        
    plt.show()
    plt.close()

    return None