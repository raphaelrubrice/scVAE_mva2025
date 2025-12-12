# Variational Auto-Encoders for single cell data
Github for the project on VAEs for single cell data for the "Introduction to Probabilistic Graphical Models and Deep Generative Models" course of the MVA.
We primarily focused on the scVAE paper.

## **Installation**
Let's use the modern package management system for python called [uv](https://github.com/astral-sh/uv)

1) To install uv on your device follow instructions [here](https://github.com/astral-sh/uv)

2) Clone the repo
```bash
git clone https://github.com/raphaelrubrice/scVAE_mva2025.git
```

3) Go to the current repo (scVAE_mva2025) and create a virtual environment
```bash
cd HVAE/
uv venv
```

4) Activtae the virtual environment and add the repo's library 
```bash
source .venv/bin/activate
uv sync
```

# **Training on PBMC**
- 3 model-types in one notebook: <a target="_blank" href="https://colab.research.google.com/github/raphaelrubrice/scVAE_mva2025/blob/main/PBMC_experiments.ipynb"> <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/> </a>  
- Experiments on `MixtureVAE`: <a target="_blank" href="https://colab.research.google.com/github/raphaelrubrice/scVAE_mva2025/blob/main/PBMC_experiments_MixtureVAE.ipynb"> <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/> </a>   
- Experiments on `ind_MoMVAE`: <a target="_blank" href="https://colab.research.google.com/github/raphaelrubrice/scVAE_mva2025/blob/main/PBMC_experiments_Ind_MoMVAE.ipynb"> <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/> </a>  
- Experiments on `MoMixVAE`: <a target="_blank" href="https://colab.research.google.com/github/raphaelrubrice/scVAE_mva2025/blob/main/PBMC_experiments_MoMixVAE.ipynb"> <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/> </a>  
