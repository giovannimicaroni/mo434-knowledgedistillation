import torch
from pathlib import Path
from torchvision import transforms
from torch.utils.data import random_split

from config.config_loader import load_config
from data.factory import build_dataset
from models.factory import build_teacher
from trainers.model_trainer import ModelTrainer
from models.factory import build_student

import torch.nn as nn

class CompleteStudent(nn.Module):
    def __init__(self, student_backbone: nn.Module, num_classes: int, feature_dim: int = 512):
        super().__init__()
        self.backbone = student_backbone
        
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        
        self.classifier = nn.Linear(feature_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        
        if features.dim() == 4:
            features = self.pool(features)
            features = torch.flatten(features, 1)  # Shape becomes [Batch, feature_dim]
        elif features.dim() == 2:
            pass
            
        logits = self.classifier(features)
        return logits
        

def train_from_zero(config_path=None):
    cfg = load_config(config_path)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.Resize((cfg.dataset.image_size, cfg.dataset.image_size)),
        transforms.ConvertImageDtype(torch.float),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    dataset = build_dataset(cfg, transform=transform)

    # Have to keep in mind seed for both teacher and student train/val split
    seed = 42
    gen = torch.Generator().manual_seed(seed)

    train_dataset, test_dataset = random_split(
        dataset, 
        lengths=[0.8, 0.2],
        generator=gen
    )
    student = build_student(cfg)
    complete_student = CompleteStudent(student, 102, 768)

    trainer = ModelTrainer(
        model=complete_student,
        lr=cfg.training.fine_tune_lr,
        epochs=cfg.training.fine_tune_epochs,
        batch_size=cfg.training.batch_size,
    )
    trainer.fit(train_dataset)
    trainer.evaluate(test_dataset)

    return student


if __name__ == "__main__":
    train_from_zero()
