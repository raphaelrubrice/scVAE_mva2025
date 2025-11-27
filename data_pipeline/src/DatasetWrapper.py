from torch.utils.data import Dataset


class AnnDatasetWrapper(Dataset):
    """Thin wrapper; applies a converter that returns X + labels."""
    def __init__(self, ann_coll, transform):
        self.ann_coll = ann_coll
        self.transform = transform

    def __len__(self):
        return self.ann_coll.n_obs

    def __getitem__(self, idx):
        batch = self.ann_coll[idx:idx + 1]
        return self.transform(batch)