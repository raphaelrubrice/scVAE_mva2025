from mixture_vae.utils import compute_CV_ll, compute_CV_radj
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import numpy as np
import pandas as pd


def plot_figure1(df_list,
                 radj_level='lvl.4',
                 save_path=None):
    """
    Plots the generative and clustering (Total marginal LL and Radj)
    for several models with improved aesthetics:
      - capped, fine error bars
      - scatter points overlayed on top
      - legend placed outside the plot
    """

    # Concatenate and reset index to avoid duplicate index issues
    figure1_plot_data = pd.concat(df_list, axis=0, ignore_index=True)

    # Identify Radj columns for the requested level
    radj_cols = [col for col in figure1_plot_data.columns if radj_level in col]
    assert len(radj_cols) == 2, f"Error, retrieved more than 2 candidate Radj columns: {radj_cols}"
    mean_radj = [col for col in radj_cols if 'Mean' in col][0]
    std_radj = [col for col in radj_cols if col != mean_radj][0]

    # Keep only relevant columns
    cols = [col for col in figure1_plot_data.columns if 'Radj' not in col] + radj_cols
    figure1_plot_data = figure1_plot_data[cols]

    plt.figure(figsize=(7, 5))

    # -----------------------------
    # Error bars
    # -----------------------------
    plt.errorbar(
        figure1_plot_data['Mean IWAE'],
        figure1_plot_data[mean_radj],
        yerr=figure1_plot_data[std_radj],
        xerr=figure1_plot_data['Std IWAE'],
        fmt='none',
        ecolor='gray',
        elinewidth=1.0,
        capsize=4,
        zorder=1       # draw underneath scatter points
    )

    # -----------------------------
    # Scatter Points (on top)
    # -----------------------------
    sns.scatterplot(
        data=figure1_plot_data,
        x="Mean IWAE",
        y=mean_radj,
        hue="Prior latent",
        style="Model",
        palette='viridis',
        s=80,
        zorder=3      # draw above error bars
    )

    plt.title("Generative and Clustering Performances")

    # -----------------------------
    # Legend outside the plot
    # -----------------------------
    plt.legend(
        bbox_to_anchor=(1.05, 1),
        loc='upper left',
        borderaxespad=0.
    )

    plt.tight_layout()

    if save_path is None:
        plt.show()
    else:
        plt.savefig(save_path, bbox_inches='tight')



# def get_metrics(cv_models, loader):
#     return compute_CV_ll(cv_models,loader), compute_CV_radj(cv_models,loader)

# def get_figure1_data(prior_latent_tag,
#                  cv_scVAE,
#                  cv_IndMoM,
#                  cv_MoMix,
#                  test_loader):

#     scvae_lls, scvae_radjs = get_metrics(cv_scVAE, test_loader)
#     indmom_lls, indmom_radjs = get_metrics(cv_IndMoM, test_loader)
#     momix_lls, momix_radjs = get_metrics(cv_MoMix, test_loader)

#     plot_data = pd.DataFrame({"Latent Prior":[prior_latent_tag]*3,
#                             "Model":["scVAE", "IndMoM", "MoMix"],
#                             "Radj":[np.mean(cv_radjs)
#                                 for cv_radjs in [scvae_radjs, indmom_radjs, momix_radjs]],
#                             "Std Radj": [np.std(cv_radjs)
#                                 for cv_radjs in [scvae_radjs, indmom_radjs, momix_radjs]],
#                             "$\mathcal{L}$":[np.mean(cv_lls)
#                                 for cv_lls in [scvae_lls, indmom_lls, momix_lls]],
#                             "Std $\mathcal{L}$": [np.std(cv_lls)
#                                 for cv_lls in [scvae_lls, indmom_lls, momix_lls]]})
#     return plot_data

# def plot_figure1(cv_dico,
#                 test_loader,
#                 save_path=None):
#     """
#     Plots the generative and clustering (Total marginal LL and Radj)
#     for several models.
#     cv_dico should be:
#         key = latent prior name
#         values = [cv_scVAE, cv_IndMoM, cv_MoMix]
#         where 1 cv_model = [list of models from the CV for this model type]
#     """
#     df_list = []
#     for latent_tag, CVs in cv_dico.items():
#         cv_scVAE, cv_IndMoM, cv_MoMix = CVs[0], CVs[1], CVs[2]

#         df_data = get_figure1_data(latent_tag,
#                                     cv_scVAE,
#                                     cv_IndMoM,
#                                     cv_MoMix,
#                                     test_loader)
#         df_list.append(df_data)

#     figure1_plot_data = pd.concat(df_list, axis=0)

#     plt.errorbar(figure1_plot_data['$\mathcal{L}$'],
#                 figure1_plot_data['Radj'],
#                 yerr=figure1_plot_data['Std Radj'],
#                 xerr=figure1_plot_data['Std $\mathcal{L}$']
#                 )
#     sns.scatter(figure1_plot_data,
#                 x="$\mathcal{L}$", y="Radj",
#                 hue="Latent Prior", shape="Model")
#     plt.title("Generative and Clustering performances")

#     if save_path is None:
#         plt.tight_layout()
#         plt.show()
#     else:
#         plt.savefig(save_path, bbox_inches='tight')