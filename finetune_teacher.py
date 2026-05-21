import torch
from pathlib import Path
from torchvision import transforms

from config.config_loader import load_config
from data.factory import build_dataset
from models.factory import build_teacher
from trainers.model_trainer import ModelTrainer


def finetune_teacher(config_path=None):
    cfg = load_config(config_path)

    save_path = Path(cfg.teacher.save_path) if cfg.teacher.save_path else None

    if save_path and save_path.exists() and not cfg.teacher.force_retrain:
        print(f"Loading finetuned teacher from {save_path}")
        teacher = build_teacher(cfg)
        teacher.model.load_state_dict(torch.load(save_path, map_location=teacher.device))
        return teacher

    transform = transforms.Compose([
        transforms.Resize((cfg.dataset.image_size, cfg.dataset.image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ConvertImageDtype(torch.float),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    dataset = build_dataset(cfg, transform=transform)
    teacher = build_teacher(cfg)
    teacher.freeze_feature_extractor()

    trainer = ModelTrainer(
        model=teacher,
        lr=cfg.training.fine_tune_lr,
        epochs=cfg.training.fine_tune_epochs,
        batch_size=cfg.training.batch_size,
    )
    trainer.fit(dataset)
    trainer.evaluate(dataset)

    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(teacher.model.state_dict(), save_path)
        print(f"Teacher saved to {save_path}")

    return teacher


if __name__ == "__main__":
    finetune_teacher()
