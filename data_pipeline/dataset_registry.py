# data_pipeline/dataset_registry.py
import torch
from torch.utils.data import DataLoader, TensorDataset, random_split
from .preprocess import preprocess_pbmc3k
from .converters import adata_to_tensors

def get_dataloaders(dataset='pmbc-3k', batch_size=128, val_split=0.1):
    if dataset == 'pmbc-3k':
        adata = preprocess_pbmc3k("data/raw/filtered_gene_bc_matrices/hg19")
    elif dataset == 'tcga':
         adata = preprocess_pbmc3k("data/raw/filtered_gene_bc_matrices/hg19")
    elif dataset == 'pmbc':
         adata = preprocess_pbmc3k("data/raw/filtered_gene_bc_matrices/hg19")
    else:
        print("Dataset not found. Check spelling")
   
    X, y = adata_to_tensors(adata)

    dataset = TensorDataset(X, y)
    n_val = int(len(dataset) * val_split) # split into train and test sets 
    n_train = len(dataset) - n_val
    train_ds, val_ds = random_split(dataset, [n_train, n_val])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader