import os
from torch.utils.data import Dataset
import torchvision

class CIFAR10Dataset(Dataset):
    def __init__(self, root_path, train=True, transform=None, download=True):
        self.root_path = root_path
        self.transform = transform
        
        self.cifar10 = torchvision.datasets.CIFAR10(
            root=self.root_path,
            train=train,
            download=download,
            transform=self.transform
        )
        
        self.labels = self.cifar10.classes
        
        self.label_to_idx = self.cifar10.class_to_idx

    def __len__(self):
        return len(self.cifar10)

    def __getitem__(self, idx):
        image, label_idx = self.cifar10[idx]
        return image, label_idx
