import json
import torch
from torch.utils.data import DataLoader, random_split, Subset
import torchvision.datasets as datasets
import torchvision.transforms as T
import numpy as np
from torch.utils.data import Dataset


def get_loaders(batch_size, val_size=10000, split_seed=0, train_subset=None):
    transform = T.ToTensor()
    full_train = datasets.MNIST(root="data", train=True, download=True, transform=transform)
    test_data = datasets.MNIST(root="data", train=False, download=True, transform=transform)

    split_gen = torch.Generator().manual_seed(split_seed)
    train_data, val_data = random_split(
        full_train, [len(full_train) - val_size, val_size], generator=split_gen
    )

    if train_subset is not None:           # label-efficiency runs only
        positions = load_subset(train_subset["n"], train_subset["seed"], train_subset["path"])
        train_data = Subset(train_data, positions)

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True,
                              generator=torch.Generator().manual_seed(split_seed))
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, test_loader

def balanced_subset(labels, n, seed):
    """n // 10 random indices per class, reproducible via the seed."""
    gen = torch.Generator().manual_seed(seed)
    per_class = [(labels == c).nonzero().squeeze(1) for c in range(10)]
    picks = [idx[torch.randperm(len(idx), generator=gen)[: n // 10]] for idx in per_class]
    return torch.cat(picks).sort().values.tolist()


def load_subset(n, seed, path="subsets.json"):
    """Indices into the training split for one labelled subset."""
    with open(path) as f:
        return json.load(f)[str(n)][str(seed)]


class DSprites(Dataset):
    """dSprites images as (1, 64, 64) float tensors, with one ground-truth factor as the label."""

    FACTORS = ["color", "shape", "scale", "orientation", "pos_x", "pos_y"]

    def __init__(self, path="data/dsprites.npz", target="shape"):
        data = np.load(path, allow_pickle=True, encoding="latin1")
        self.imgs = np.packbits(data["imgs"], axis=-1)            # (N, 64, 8): 8 pixels per byte
        self.factors = torch.from_numpy(data["latents_classes"])  # (N, 6) integer factor indices
        self.target = None if target is None else self.FACTORS.index(target)

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, i):
        img = np.unpackbits(self.imgs[i], axis=-1)                # back to (64, 64) of 0/1
        x = torch.from_numpy(img).float().unsqueeze(0)            # (1, 64, 64), like MNIST batches
        y = self.factors[i] if self.target is None else self.factors[i, self.target]
        return x, y