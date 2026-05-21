from data.pet_dataset import PetDataset

_DATASETS = {
    "oxford_pet": PetDataset,
}


def build_dataset(cfg, transform=None):
    cls = _DATASETS[cfg.dataset.name]
    return cls(cfg.dataset.path, transform=transform)
