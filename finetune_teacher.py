import torch
from torchvision import transforms

from config.config_loader import load_config
from data.pet_dataset import PetDataset
from models.vgg_teacher import VGGTeacher
from trainers.model_trainer import ModelTrainer


def finetune_teacher(config_path=None):
    cfg = load_config(config_path)

    transform = transforms.Compose([
        transforms.Resize((cfg.dataset.image_size, cfg.dataset.image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ConvertImageDtype(torch.float),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    dataset = PetDataset(cfg.dataset.path, transform=transform)

    teacher = VGGTeacher(
        pretrained=cfg.teacher.pretrained,
        num_classes=cfg.teacher.num_classes,
        weights_path=cfg.teacher.weights_path,
    )
    teacher.freeze_feature_extractor()

    trainer = ModelTrainer(
        model=teacher,
        lr=cfg.training.fine_tune_lr,
        epochs=cfg.training.fine_tune_epochs,
        batch_size=cfg.training.batch_size,
    )
    trainer.fit(dataset)
    trainer.evaluate(dataset)

    return teacher


if __name__ == "__main__":
    finetune_teacher()
