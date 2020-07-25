from torchvision.datasets import CIFAR100
import numpy as np


class CIFAR100_Full(CIFAR100):
    train_list = CIFAR100.train_list + CIFAR100.test_list
    test_list = []

    def __init__(self, root, transform=None, target_transform=None,
                 download=False):
        super().__init__(root, train=True, transform=transform, target_transform=target_transform,
                         download=download)


def prepare_cifar(ds, threshold, first):
    if first:
        mask = np.array(ds.targets) < threshold
    else:
        mask = np.array(ds.targets) >= threshold
    ds.targets = (np.array(ds.targets)[mask]).tolist()
    ds.data = ds.data[mask]


