import torch
from torchvision import datasets
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np

class AlbumentationsDataset(Dataset):
    def __init__(self, subset, transform=None):
        self.subset=subset
        self.transform=transform

    def __len__(self):
        return len(self.subset)
    
    def __getitem__(self, idx):
        img, label = self.subset[idx]
        img = np.array(img)

        if img.ndim == 2:
            img = np.stack([img] * 3, axis=-1)

        if self.transform:
            img = self.transform(image=img)["image"]

        return img, label

def get_subsets(data_dir: str, test_split: float=0.2):
    ds = datasets.ImageFolder(root=data_dir)

    test_size = int(len(ds) * test_split)
    train_size = len(ds) - test_size
    train_subset, test_subset = random_split(
        ds, 
        [train_size, test_size], 
        generator=torch.Generator().manual_seed(42)
    )

    return train_subset, test_subset, ds.classes

def get_data_loader(dataset, transforms, batch_size: int=32, num_workers: int=2, shuffle: bool=False):

    ds = AlbumentationsDataset(dataset, transform=transforms)

    dataloader = DataLoader(dataset=ds, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)

    return dataloader