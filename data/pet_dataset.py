import os
from torch.utils.data import Dataset
import torchvision
from sklearn.preprocessing import LabelEncoder
from torchvision.io import ImageReadMode
from PIL import Image


class PetDataset(Dataset):
    def __init__(self, root_path, transform=None):
        self.root_path = root_path
        self.transform = transform
        self.image_paths = [
            os.path.join(root_path, f)
            for f in os.listdir(root_path)
            if f.endswith(".jpg")
        ]

        self.labels = list(
            set([self._get_label(image_path) for image_path in self.image_paths])
        )
        self.le = LabelEncoder()
        self.le.fit(self.labels)

        self.label_to_idx = {label: idx for idx, label in enumerate(self.le.classes_)}

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert("RGB")
        if self.transform:
            image = self.transform(image)

        label = self._get_label(image_path)
        label_idx = self.label_to_idx[label]

        return image, label_idx

    def _get_label(self, filepath):

        id_end = -1
        for index, char in enumerate(filepath):
            if char.isdigit():
                id_end = index
                break
        label = filepath[
            : id_end - 1
        ]  # image_path is in the form of label_imagenumber.jpg

        return label
