import sys
from contextlib import contextmanager

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
from trainers.model_trainer import ModelTrainer, ModelTrainerCombined, ReletionalModelTrainer
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


def _evaluate_teacher(teacher, dataset, batch_size):
    """Evaluate the finetuned teacher on its own (full forward pass) for comparison."""
    device = teacher.device
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    teacher.eval()

    correct = total = 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = teacher(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f"Teacher accuracy: {accuracy:.2f}%")
    return accuracy


def _save_loss_curves(history, run_dir, run_name):
    """Plot every loss series in `history` against epoch and save to run_dir/loss_curve.png."""
    plt.figure()
    for label, values in history.items():
        if label.endswith("_acc"):
            continue  # accuracy series are plotted separately by _save_accuracy_curves
        epochs = range(1, len(values) + 1)
        plt.plot(epochs, values, marker="o", label=label)

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(run_name)
    plt.legend()
    plt.grid(True, alpha=0.3)

    plot_path = Path(run_dir) / "loss_curve.png"
    plt.savefig(plot_path, bbox_inches="tight")
    plt.close()
    print(f"Loss curve saved to {plot_path}")


def _save_accuracy_curves(history, run_dir, run_name):
    """Plot every *_acc series in `history` against epoch and save to run_dir/accuracy_curve.png."""
    acc_series = {k: v for k, v in history.items() if k.endswith("_acc")}
    if not acc_series:
        return

    plt.figure()
    for label, values in acc_series.items():
        epochs = range(1, len(values) + 1)
        plt.plot(epochs, values, marker="o", label=label)

    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.title(run_name)
    plt.legend()
    plt.grid(True, alpha=0.3)

    plot_path = Path(run_dir) / "accuracy_curve.png"
    plt.savefig(plot_path, bbox_inches="tight")
    plt.close()
    print(f"Accuracy curve saved to {plot_path}")


class _Tee:
    """Duplicates writes to multiple streams (e.g. terminal + log file)."""

    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for stream in self.streams:
            stream.write(data)

    def flush(self):
        for stream in self.streams:
            stream.flush()


@contextmanager
def _capture_logs(log_path):
    """Tee everything printed to stdout into log_path while still showing it on the terminal."""
    Path(log_path).parent.mkdir(parents=True, exist_ok=True)
    original_stdout = sys.stdout
    with open(log_path, "w") as log_file:
        sys.stdout = _Tee(original_stdout, log_file)
        try:
            yield
        finally:
            sys.stdout = original_stdout
    print(f"Training log saved to {log_path}")


def distill_student(config_path=None):
    cfg = load_config(config_path)

    run_name = f"distill_{cfg.teacher.architecture}_{cfg.dataset.name}"
    run_dir = Path("training_results") / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    with _capture_logs(run_dir / "training_log.txt"):
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
        history = trainer.fit(distillation_dataset)
        _save_loss_curves(history, run_dir, run_name)

        print("\nEvaluating teacher and student...")
        _evaluate_teacher(teacher, dataset, cfg.training.batch_size)
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

    run_name = f"combined_{cfg.teacher.architecture}_{cfg.dataset.name}_alpha{alpha}"
    run_dir = Path("training_results") / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    with _capture_logs(run_dir / "training_log.txt"):
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
        history = trainer.fit(
            distillation_dataset,
            train_eval_dataset=train_dataset,
            val_dataset=test_dataset,
        )
        _save_loss_curves(history, run_dir, run_name)
        _save_accuracy_curves(history, run_dir, run_name)

        print("\nEvaluating teacher and student...")
        _evaluate_teacher(teacher, test_dataset, cfg.training.batch_size)
        acc = _evaluate_student(student, teacher, test_dataset, cfg.training.batch_size)
        print("Number of parameters in the student: ", student.get_student_parameters()["total"] + teacher.get_teacher_parameters()["classifier_module"])
        print("Number of parameters in the teacher: ", teacher.get_teacher_parameters()["total"])
        print("The ratio of the size of the student and the teacher is: ", (student.get_student_parameters()["total"] + teacher.get_teacher_parameters()["classifier_module"]) / teacher.get_teacher_parameters()["total"])
    return acc, student, teacher

def distill_student_reletional(config_path=None, alpha=0.5):
    cfg = load_config(config_path)

    run_name = f"relational_{cfg.teacher.architecture}_{cfg.dataset.name}_alpha{alpha}"
    run_dir = Path("training_results") / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    with _capture_logs(run_dir / "training_log.txt"):
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

        trainer = ReletionalModelTrainer(
            model=student,
            lr=cfg.training.distillation_lr,
            epochs=cfg.training.distillation_epochs,
            batch_size=cfg.training.batch_size,
            criterion=combined_criterion,
        )
        history = trainer.fit(
            distillation_dataset,
            train_eval_dataset=train_dataset,
            val_dataset=test_dataset,
        )
        _save_loss_curves(history, run_dir, run_name)
        _save_accuracy_curves(history, run_dir, run_name)

        print("\nEvaluating teacher and student...")
        _evaluate_teacher(teacher, test_dataset, cfg.training.batch_size)
        acc = _evaluate_student(student, teacher, test_dataset, cfg.training.batch_size)
        print("Number of parameters in the student: ", student.get_student_parameters()["total"] + teacher.get_teacher_parameters()["classifier_module"])
        print("Number of parameters in the teacher: ", teacher.get_teacher_parameters()["total"])
        print("The ratio of the size of the student and the teacher is: ", (student.get_student_parameters()["total"] + teacher.get_teacher_parameters()["classifier_module"]) / teacher.get_teacher_parameters()["total"])
    return acc, student, teacher

# if __name__ == "__main__":
#     accuracies = []
#     for i in range(0, 11):
#         accuracies.append(distill_student_combined(alpha=0.1 * i)[0])
#     plt.plot([0.1 * i for i in range(0, 11)], accuracies, color='green', marker='o')
    
#     plt.xlabel("Alpha")
#     plt.ylabel("Accuracy")
#     plt.show()

if __name__ == '__main__':
    distill_student_combined(alpha=0.8)
    # distill_student_reletional(alpha=0.8)