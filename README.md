# Variational Auto-Encoders for Single-Cell RNA-seq Data

<div align="center">

**Reimplementation and Hierarchical Extensions of scVAE (Grønbech et al., 2020)**

[![Paper](https://img.shields.io/badge/DOI-10.1093/bioinformatics/btaa293-b31b1b.svg)](https://academic.oup.com/bioinformatics/article/36/16/4415/5838187?login=false)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![MVA: Introduction to Probabilistic Graphical Models](https://img.shields.io/badge/MVA-Probabilistic%20Graphical%20Models-darkgreen.svg)](https://www.master-mva.com/cours/probabilistic-graphical-models/)


[Paper](https://academic.oup.com/bioinformatics/article/36/16/4415/5838187?login=false) • [Original Code](https://github.com/scvae/scvae) • [Report](./RaphaelRubriceTiffneyAinaAdamKeddis_scVAE_report.pdf) • [Poster](./PGM_Poster.pdf)

*Course project – Introduction to Probabilistic Graphical Models  
Master MVA (ENS Paris-Saclay)*

**Authors:** Raphaël Rubrice · Adam Keddis · Tiffney Aina

</div>

---

## Overview

This repository provides a **from-scratch reimplementation of scVAE** and extends it with **hierarchical mixture models** for single-cell RNA-seq data.

scRNA-seq data are:
- high-dimensional,
- sparse,
- overdispersed,
- and biologically hierarchical (lineages → subtypes).

While **scVAE** models counts using a **Gaussian Mixture VAE** with a Negative Binomial likelihood, it relies on a **flat mixture prior**, which cannot explicitly encode biological hierarchies.

We address this limitation by introducing two extensions:

- **IndMoMVAE** – Independent Mixture-of-Mixtures VAE  
- **MoMixVAE** – Hierarchical Mixture-of-Mixtures VAE  

All models are evaluated on **PBMC datasets** with a curated **4-level cell-type hierarchy**.

---

## Implemented Models

### 1. MixtureVAE (scVAE)

A flexible generalization of scVAE:
- Arbitrary latent priors (Normal, Student-t)
- Explicit categorical prior for clustering
- Modular distributions and training loops

---

### 2. IndMoMVAE (Independent Mixture-of-Mixtures)

- Multiple **independent mixture branches**
- Each branch learns a different partition of the data
- No hierarchical dependency between levels

This model acts as an **ablation baseline** to isolate the effect of hierarchy.

---

### 3. MoMixVAE (Hierarchical Mixture-of-Mixtures)

- Explicit **hierarchical dependencies** between clustering levels
- Coarse-to-fine latent organization
- Structured variational posterior
- Hierarchical ELBO with:
  - β-scaled KL terms
  - marginal-usage regularization to prevent component collapse

This model best reflects biological lineage structure.

---

## PBMC Hierarchy

We constructed a **four-level hierarchy**:

| Level | Description |
|------|------------|
| 1 | Stem vs Non-stem |
| 2 | Major lineage (B / NK / T) |
| 3 | CD4 vs CD8 (T-cells) |
| 4 | Terminal subtypes (naive, memory, regulatory, etc.) |

Nine PBMC datasets from 10x Genomics are downloaded, annotated, harmonized, and merged using a fully reproducible pipeline.

---

## Installation

We use **uv**, a modern and fast Python package manager.

```bash
# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone the repository
git clone https://github.com/raphaelrubrice/scVAE_mva2025.git
cd scVAE_mva2025

# Create and activate environment
uv venv
source .venv/bin/activate
uv sync
```

## Data Pipeline (PBMC)

The full PBMC processing pipeline is implemented in `data_pipeline/` and is fully reproducible.

### Pipeline stages

1. **Download** raw PBMC datasets from 10x Genomics  
2. **Load** raw 10x matrices into `AnnData` objects  
3. **Annotate** cells with a curated 4-level hierarchical cell-type taxonomy  
4. **Freeze** label vocabularies across datasets  
5. **Shard** each dataset as `.h5ad`  
6. **Combine** shards into a unified `AnnCollection`  
7. **Build** stratified train / validation / test PyTorch `DataLoader`s  

The pipeline is compatible with **local execution and Google Colab**.

### Hierarchical labels

Each cell is annotated with four hierarchical levels:

| Level | Meaning |
|------:|---------|
| 1 | Stem vs Non-stem |
| 2 | Major lineage (B / NK / T) |
| 3 | Intermediate lineage (CD4 / CD8) |
| 4 | Terminal subtype |

Label metadata and dataset URLs are defined in `data_pipeline/src/config.py`.

## Implemented Models

### scVAE (Baseline)

Reimplementation of scVAE (Grønbech et al., 2020):

- Continuous latent variable `z`
- Discrete cluster variable `y`
- Gaussian mixture prior
- Negative Binomial likelihood
- ELBO objective with KL warm-up

This model serves as the reference baseline.

---

### MixtureVAE

A modular generalization of scVAE:

- Flexible latent priors (Normal, Student-t)
- Explicit categorical cluster prior
- Modular distribution and training framework
- Supports clustering and generative modeling

---

### IndMoMVAE (Independent Mixture-of-Mixtures)

- Multiple independent mixture branches
- Each branch learns a distinct clustering
- No dependencies between hierarchy levels

This model acts as an **ablation baseline** to isolate the effect of hierarchical coupling.

---

### MoMixVAE (Hierarchical Mixture-of-Mixtures)

- Explicit hierarchical dependencies between clustering levels
- Coarse-to-fine latent organization
- Structured variational posterior
- Hierarchical ELBO with:
  - β-scaled KL terms
  - Marginal-usage regularization to prevent component collapse

This model best captures biological lineage structure.

## Experiments

All experiments are conducted on PBMC data using the curated hierarchical labels.

### Training notebooks

- **All models (single notebook)**  
  PBMC_experiments.ipynb

- **MixtureVAE experiments**  
  PBMC_experiments_MixtureVAE.ipynb

- **IndMoMVAE experiments**  
  PBMC_experiments_IndMoMVAE.ipynb

- **MoMixVAE experiments**  
  PBMC_experiments_MoMixVAE.ipynb

Each notebook supports execution on **Google Colab** and locally.

## Evaluation Metrics

We evaluate models using complementary metrics:

- **IWAE log-likelihood**  
  Measures generative modeling quality

- **Weighted F1-score**  
  Measures clustering quality after Hungarian label alignment

- **Adjusted Rand Index (ARI)**  
  Measures agreement between predicted and true clusters

### Observations

- Student-t latent priors improve generative quality
- MoMixVAE provides the best balance between reconstruction and clustering
- F1-score is more stable than ARI on nonlinear latent manifolds

## Project Structure
```text
scVAE_mva2025/
├── mixture_vae/
│   ├── mvae.py              # MixtureVAE, IndMoMVAE, MoMixVAE
│   ├── distributions.py    # Latent & likelihood distributions
│   ├── training.py         # Training protocols
│   ├── utils.py            # Metrics & evaluation
│   └── viz.py              # Visualization utilities
│
├── data_pipeline/
│   ├── src/
│   │   ├── downloader.py
│   │   ├── load_anndata.py
│   │   ├── combine.py
│   │   ├── dataloader.py
│   │   └── config.py
│
├── PBMC_experiments*.ipynb
├── pyproject.toml
├── uv.lock
└── README.md
```

## Key Contributions

- Full reimplementation of scVAE
- Hierarchical Mixture-of-Mixtures formulation
- Robust and reproducible PBMC data pipeline
- Stable training via KL warm-up and marginal regularization
- Extensive quantitative and qualitative evaluation

## References

- Grønbech et al., *scVAE: Variational Auto-Encoders for Single-Cell Gene Expression Data*, Bioinformatics, 2020  
- Kingma & Welling, *Auto-Encoding Variational Bayes*, ICLR 2014
