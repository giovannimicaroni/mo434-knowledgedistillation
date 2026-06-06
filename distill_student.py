import torch
import torch.nn as nn
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torch.utils.data import random_split
from tqdm import tqdm

from config.config_loader import load_config
from data.factory import build_dataset
from models.factory import build_student
from trainers.model_trainer import ModelTrainer, ModelTrainerCombined
from finetune_teacher import finetune_teacher

import torch.nn.functional as F

import matplotlib.pyplot as plt


class DistillationDataset(Dataset):
    """Pairs each image with the pre-extracted teacher feature vector for that image."""

    def __init__(self, image_dataset, teacher_features):
        self.image_dataset = image_dataset
        self.teacher_features = teacher_features

    def __len__(self):
        return len(self.image_dataset)

    def __getitem__(self, idx):
        image, _ = self.image_dataset[idx]
        return image, self.teacher_features[idx]


def _extract_teacher_features(teacher, dataset, batch_size, cache_path):
    if cache_path.exists():
        print(f"Loading cached features from {cache_path}")
        return torch.load(cache_path)

    device = teacher.device
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    all_features = []

    teacher.eval()
    with torch.no_grad():
        for images, _ in tqdm(loader, desc="Extracting teacher features"):
            x = teacher.extract_features(images.to(device))  # (B, 512, 7, 7)
            all_features.append(x.cpu())

    features = torch.cat(all_features)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(features, cache_path)
    print(f"Features cached to {cache_path}")
    return features


def _evaluate_student(student, teacher, dataset, batch_size):
    device = teacher.device
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    student.eval()
    student.to(device)

    correct = total = 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            features = student(images)
            outputs = teacher.forward_classifier(features)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f"Student accuracy: {accuracy:.2f}%")
    return accuracy


def distill_student(config_path=None):
    cfg = load_config(config_path)

    teacher = finetune_teacher(config_path)

    # Deterministic transforms only — no augmentation, since features are extracted once
    transform = transforms.Compose([
        transforms.Resize((cfg.dataset.image_size, cfg.dataset.image_size)),
        transforms.ConvertImageDtype(torch.float),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    dataset = build_dataset(cfg, transform=transform)

    cache_path = Path(cfg.cache.dir) / f"features_{cfg.teacher.architecture}_{cfg.dataset.name}.pt"
    if cfg.student.force_reextract and cache_path.exists():
        cache_path.unlink()

    teacher_features = _extract_teacher_features(teacher, dataset, cfg.training.batch_size, cache_path)
    distillation_dataset = DistillationDataset(dataset, teacher_features)

    student = build_student(cfg)
    trainer = ModelTrainer(
        model=student,
        lr=cfg.training.distillation_lr,
        epochs=cfg.training.distillation_epochs,
        batch_size=cfg.training.batch_size,
        criterion=nn.MSELoss(),
    )
    trainer.fit(distillation_dataset)

    print("\nEvaluating student with teacher classifier...")
    _evaluate_student(student, teacher, dataset, cfg.training.batch_size)

class DistillationDatasetCombined(Dataset):
    def __init__(self, image_dataset, teacher_features):
        self.image_dataset = image_dataset
        self.teacher_features = teacher_features

    def __len__(self):
        return len(self.image_dataset)

    def __getitem__(self, idx):
        # Extract both the image and the true label
        image, label = self.image_dataset[idx]
        
        # Return all three components
        return image, self.teacher_features[idx], label
    
class CombinedDistillationLoss(nn.Module):
    def __init__(self, teacher_model, alpha=0.5):
        """
        alpha: Weight balancing the two losses. 
               alpha=1.0 is pure MSE, alpha=0.0 is pure Cross Entropy.
        """
        super().__init__()
        self.teacher = teacher_model
        self.alpha = alpha
        self.mse_loss = nn.MSELoss()
        self.entropy_loss = nn.CrossEntropyLoss()

    def forward(self, student_features, teacher_features, labels):


        student_pooled = torch.mean(student_features, dim=[2, 3])
        teacher_pooled = torch.mean(teacher_features, dim=[2, 3])
        
        # Calculate MSE on the pooled vectors instead of the raw spatial maps
        loss_mse = self.mse_loss(student_pooled, teacher_pooled)
        student_logits = self.teacher.forward_classifier(student_features)
        loss_ce = self.entropy_loss(student_logits, labels)
        
        # 3. Combine them
        total_loss = (self.alpha * loss_mse) + ((1 - self.alpha) * loss_ce)
        
        return total_loss, loss_mse, loss_ce

def distill_student_combined(config_path=None, alpha=0.5):
    cfg = load_config(config_path)

    teacher = finetune_teacher(config_path)

    # Deterministic transforms only — no augmentation, since features are extracted once
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

    cache_path = Path(cfg.cache.dir) / f"features_{cfg.teacher.architecture}_{cfg.dataset.name}.pt"
    if cfg.student.force_reextract and cache_path.exists():
        cache_path.unlink()

    teacher_features = _extract_teacher_features(teacher, train_dataset, cfg.training.batch_size, cache_path)
    distillation_dataset = DistillationDatasetCombined(train_dataset, teacher_features)

    combined_criterion = CombinedDistillationLoss(teacher, alpha=alpha)

    student = build_student(cfg)

    teacher.eval()
    # Freeze the entire teacher model to save memory and prevent any updates
    for param in teacher.parameters():
        param.requires_grad = False
        
    print("Teacher model successfully frozen!")

    trainer = ModelTrainerCombined(
        model=student,
        lr=cfg.training.distillation_lr,
        epochs=cfg.training.distillation_epochs,
        batch_size=cfg.training.batch_size,
        criterion=combined_criterion,
    )
    trainer.fit(distillation_dataset)

    print("\nEvaluating student with teacher classifier...")
    acc = _evaluate_student(student, teacher, test_dataset, cfg.training.batch_size)
    print("The ratio of the size of the student and the teacher is: ", (student.get_student_parameters()["total"] + teacher.get_teacher_parameters()["classifier_module"]) / teacher.get_teacher_parameters()["total"])
    return acc, student, teacher

if __name__ == "__main__":
    accuracies = []
    for i in range(0, 11):
        accuracies.append(distill_student_combined(alpha=0.1 * i)[0])
    plt.plot([0.1 * i for i in range(0, 11)], accuracies, color='green', marker='o')
    
    plt.xlabel("Alpha")
    plt.ylabel("Accuracy")
    plt.show()

# if __name__ == '__main__':
#      distill_student_combined(alpha=0.8)