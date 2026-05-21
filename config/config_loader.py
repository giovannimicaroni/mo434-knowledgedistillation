from dataclasses import dataclass
from pathlib import Path
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = REPO_ROOT / "config" / "config.yaml"


@dataclass
class TeacherConfig:
    architecture: str
    weights: str | None = None
    weights_path: str | None = None
    save_path: str | None = None
    force_retrain: bool = False
    pretrained: bool = True
    num_classes: int = 2
    freeze_feature_extractor: bool = True
    train_classifier_only: bool = True


@dataclass
class StudentConfig:
    architecture: str
    force_reextract: bool = False


@dataclass
class TrainingConfig:
    batch_size: int
    fine_tune_epochs: int
    fine_tune_lr: float
    distillation_epochs: int
    distillation_lr: float
    optimizer: str


@dataclass
class DatasetConfig:
    name: str
    path: str
    image_size: int


@dataclass
class CacheConfig:
    dir: str = "cache"


@dataclass
class PipelineConfig:
    teacher: TeacherConfig
    student: StudentConfig
    training: TrainingConfig
    dataset: DatasetConfig
    cache: CacheConfig


def _resolve_config_path(config_path: str | Path | None = None) -> Path:
    if config_path is None:
        return DEFAULT_CONFIG_PATH

    path = Path(config_path)
    return path if path.is_absolute() else REPO_ROOT / path


def load_config(config_path: str | Path | None = None) -> PipelineConfig:
    path = _resolve_config_path(config_path)
    with open(path, "r") as f:
        data = yaml.safe_load(f)

    if "teacher" in data:
        teacher_cfg = TeacherConfig(**data["teacher"])
    else:
        teacher_cfg = TeacherConfig(
            architecture=data["model"]["architecture"],
            weights=data["model"].get("weights"),
            num_classes=data["model"].get("num_classes", 2),
        )

    student_cfg = StudentConfig(**data["student"])
    training_cfg = TrainingConfig(**data["training"])
    dataset_cfg = DatasetConfig(**data["dataset"])
    cache_cfg = CacheConfig(**data.get("cache", {}))

    return PipelineConfig(
        teacher=teacher_cfg,
        student=student_cfg,
        training=training_cfg,
        dataset=dataset_cfg,
        cache=cache_cfg,
    )
