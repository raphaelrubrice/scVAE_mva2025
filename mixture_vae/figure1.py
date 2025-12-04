from mixture_vae.utils import compute_CV_ll, compute_CV_radj
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import numpy as np
import pandas as pd

def get_metrics(cv_models, loader):
    return compute_CV_ll(cv_models,loader), compute_CV_radj(cv_models,loader)

def get_figure1_data(prior_latent_tag, 
                 cv_scVAE, 
                 cv_IndMoM, 
                 cv_MoMix, 
                 test_loader):

    scvae_lls, scvae_radjs = get_metrics(cv_scVAE, test_loader)
    indmom_lls, indmom_radjs = get_metrics(cv_IndMoM, test_loader)
    momix_lls, momix_radjs = get_metrics(cv_MoMix, test_loader)

    plot_data = pd.DataFrame({"Latent Prior":[prior_latent_tag]*3,
                            "Model":["scVAE", "IndMoM", "MoMix"],
                            "Radj":[np.mean(cv_radjs) 
                                for cv_radjs in [scvae_radjs, indmom_radjs, momix_radjs]],
                            "Std Radj": [np.std(cv_radjs) 
                                for cv_radjs in [scvae_radjs, indmom_radjs, momix_radjs]],
                            "$\mathcal{L}$":[np.mean(cv_lls) 
                                for cv_lls in [scvae_lls, indmom_lls, momix_lls]],
                            "Std $\mathcal{L}$": [np.std(cv_lls) 
                                for cv_lls in [scvae_lls, indmom_lls, momix_lls]]})
    return plot_data

def plot_figure1(cv_dico,
                test_loader,
                save_path=None):
    """
    Plots the generative and clustering (Total marginal LL and Radj)
    for several models.
    cv_dico should be:
        key = latent prior name
        values = [cv_scVAE, cv_IndMoM, cv_MoMix]
        where 1 cv_model = [list of models from the CV for this model type]
    """
    df_list = []
    for latent_tag, CVs in cv_dico.items():
        cv_scVAE, cv_IndMoM, cv_MoMix = CVs[0], CVs[1], CVs[2]

        df_data = get_figure1_data(latent_tag, 
                                    cv_scVAE, 
                                    cv_IndMoM, 
                                    cv_MoMix, 
                                    test_loader)
        df_list.append(df_data)
    
    figure1_plot_data = pd.concat(df_list, axis=0)

    plt.errorbar(figure1_plot_data['$\mathcal{L}$'], 
                figure1_plot_data['Radj'], 
                yerr=figure1_plot_data['Std Radj'],
                xerr=figure1_plot_data['Std $\mathcal{L}$']
                )
    sns.scatter(figure1_plot_data, 
                x="$\mathcal{L}$", y="Radj", 
                hue="Latent Prior", shape="Model")
    plt.title("Generative and Clustering performances")
    
    if save_path is None:
        plt.tight_layout()
        plt.show()
    else:
        plt.savefig(save_path, bbox_inches='tight')