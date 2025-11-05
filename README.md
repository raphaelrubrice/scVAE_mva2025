# VAE for singlee cell data
Github for the project on VAEs for single cell data for the "Introduction to Probabilistic Graphical Models and Deep Generative Models" course of the MVA.
We primarily focused on the scVAE paper.

## **Installation**
Let's use the modern package management system for python called [uv](https://github.com/astral-sh/uv)

1) To install uv on your device follow instructions [here](https://github.com/astral-sh/uv)

2) Clone the repo
```bash
git clone https://github.com/blackswan-advitamaeternam/HVAE.git
```

3) Go to the current repo (HVAE) and create a virtual environment
```bash
cd HVAE/
uv venv
```

4) Activtae the virtual environment and add the repo's library 
```bash
source .venv/bin/activate
uv sync
```
