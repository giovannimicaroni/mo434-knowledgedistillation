from data.pet_dataset import PetDataset
from data.cifar_10_dataset import CIFAR10Dataset

_DATASETS = {
    "oxford_pet": PetDataset,
    "cifar_10": CIFAR10Dataset,
}


def build_dataset(cfg, transform=None):
    cls = _DATASETS[cfg.dataset.name]
    return cls(cfg.dataset.path, transform=transform)
