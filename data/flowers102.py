import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision.datasets import Flowers102
from torch.utils.data import ConcatDataset

class Flowers102Dataset(Dataset):
    def __init__(self, root_path, split='train', transform=None, download=True):
        """
        Args:
            root_path (str): Directory where the dataset will be downloaded/loaded.
            split (str): 'train', 'val', or 'test'.
            transform (callable, optional): Optional transform to be applied on a sample.
            download (bool): If True, downloads the dataset from the internet.
        """
        self.root_path = root_path
        self.split = split
        self.transform = transform
        
        # We use the built-in torchvision class as the engine.
        # This safely handles parsing the 'imagelabels.mat' and 'setid.mat' files,
        # and automatically shifts the 1-102 labels to standard 0-101 PyTorch indices.
        test_base = Flowers102(
            root=self.root_path,
            split="test",
            transform=self.transform,
            download=download
        )

        val_base = Flowers102(
            root=self.root_path,
            split="val",
            transform=self.transform,
            download=download
        )

        train_base = Flowers102(
            root=self.root_path,
            split="train",
            transform=self.transform,
            download=download
        )
        
        self.dataset = ConcatDataset([train_base, val_base, test_base])
        # Flowers102 does not inherently map to string names in the base dataset,
        # so we generate generic class strings (0 to 101) to keep the attribute intact.
        self.classes = [str(i) for i in range(102)]
        self.label_to_idx = {str(i): i for i in range(102)}

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        # The torchvision base class automatically handles PIL Image opening,
        # RGB conversion, and applying the transform pipeline.
        image, label_idx = self.dataset[idx]
            
        return image, label_idx