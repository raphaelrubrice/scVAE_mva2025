# data_pipeline/converters.py
import torch
import numpy as np


def adata_to_tensors(adata):
    X = adata.X.toarray() if hasattr(adata.X, "toarray") else adata.X
    X = torch.tensor(X, dtype=torch.float32)
    y = torch.zeros(X.shape[0], dtype=torch.long)  # placeholder labels
    return X, y