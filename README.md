# mo434-knowledgedistillation

Knowledge distillation pipeline for image classification. A large pretrained teacher model is finetuned on a target dataset, and a smaller student model is trained to replicate the teacher's internal feature representations. The student is then evaluated by plugging its features into the teacher's classifier head.

## Project structure

```
├── config/
│   ├── config.yaml            # Central configuration for the full pipeline
│   └── config_loader.py       # Parses config.yaml into typed dataclasses
├── data/
│   ├── pet_dataset.py         # Dataset class for the Oxford-IIIT Pet dataset
│   └── factory.py             # Maps dataset names to dataset classes
├── models/
│   ├── base_teacher.py        # Abstract base class for teacher models
│   ├── vgg_teacher.py         # VGG11 teacher implementation
│   ├── student.py             # Lightweight student CNN architecture
│   └── factory.py             # Maps architecture names to model classes
├── trainers/
│   └── model_trainer.py       # General-purpose training and evaluation class
├── finetune_teacher.py        # Entry point: finetunes the teacher on the target dataset
└── distill_student.py         # Entry point: trains the student via feature distillation
```

## Components

### `config/config.yaml`
Single source of truth for the entire pipeline. Controls the teacher architecture, student architecture, dataset, training hyperparameters, checkpoint paths, and cache behaviour. Changing the teacher or dataset only requires updating this file and registering the new class in the respective factory.

### `config/config_loader.py`
Loads and parses `config.yaml` into typed Python dataclasses (`TeacherConfig`, `StudentConfig`, `TrainingConfig`, `DatasetConfig`, `CacheConfig`), grouped under a top-level `PipelineConfig`.

### `data/pet_dataset.py`
PyTorch `Dataset` for the Oxford-IIIT Pet dataset. Reads images from a flat directory, extracts breed labels from filenames (e.g. `Abyssinian_001.jpg` → `Abyssinian`), and encodes them as integer class indices. Supports transforms applied lazily per sample.

### `data/factory.py`
Registry that maps dataset name strings to dataset classes. To add a new dataset, implement a `Dataset` subclass and register it here.

### `models/base_teacher.py`
Abstract base class for all teacher models. Provides shared logic: weight loading, split forward pass (`extract_features` / `forward_classifier`), gradient-free `predict`, and `freeze_feature_extractor` / `freeze_classifier` / `unfreeze_classifier` for controlled finetuning.

### `models/vgg_teacher.py`
Concrete teacher built on VGG11 with ImageNet pretrained weights. Replaces the final classifier layer to output the configured number of classes.

### `models/student.py`
Lightweight CNN with 4 convolutional layers and BatchNorm, producing a 25088-dim flat feature vector — the same shape as VGG11's backbone output — so it can be evaluated directly with the teacher's classifier head.

### `models/factory.py`
Registry that maps architecture name strings to model classes for both teachers and students. To add a new architecture, implement the class and register it here.

### `trainers/model_trainer.py`
General-purpose `ModelTrainer` class that handles training and evaluation for any `nn.Module`. Accepts a configurable `criterion` (defaults to `CrossEntropyLoss`), so it works for both classifier finetuning and MSE-based feature distillation without modification.

### `finetune_teacher.py`
Finetunes the teacher's classifier head on the target dataset while keeping the backbone frozen. Saves the resulting weights to `teacher.save_path` and skips retraining on subsequent runs unless `teacher.force_retrain: true` is set.

### `distill_student.py`
Trains the student to replicate the teacher's feature representations using MSE loss. Pre-extracts teacher features once and caches them to disk — keyed by teacher architecture and dataset name — to avoid recomputing them across runs. After training, evaluates the student by passing its features through the teacher's classifier head.

## Pipeline workflow

### Setup

```bash
uv sync
```

Installs all dependencies from `uv.lock` into the project's virtual environment.

### Step 1 — Finetune the teacher

```bash
uv run finetune_teacher.py
```

- Loads VGG11 with ImageNet weights and freezes the backbone
- Trains only the classifier head on the target dataset (with augmentation)
- Saves finetuned weights to `checkpoints/teacher_vgg11.pth`

On subsequent runs this step is skipped and the saved weights are loaded automatically. To force retraining, set `teacher.force_retrain: true` in `config.yaml`.

### Step 2 — Distill the student

```bash
uv run distill_student.py
```

- Loads the finetuned teacher (runs step 1 first if weights are not saved yet)
- Passes every image through the frozen backbone and caches the feature vectors to `cache/features_vgg11_oxford_pet.pt`
- Trains the student CNN to reproduce those features using MSE loss
- Evaluates accuracy by passing student features through the teacher's classifier head

On subsequent runs the feature cache is reloaded automatically. To force re-extraction, set `student.force_reextract: true` in `config.yaml`.

## Extending the pipeline

### Swapping the teacher
1. Implement a new class extending `BaseTeacherModel`
2. Register it in `models/factory.py`
3. Set `teacher.architecture` and `teacher.save_path` in `config.yaml`

### Swapping the dataset
1. Implement a `torch.utils.data.Dataset` subclass
2. Register it in `data/factory.py`
3. Set `dataset.name` and `dataset.path` in `config.yaml`
