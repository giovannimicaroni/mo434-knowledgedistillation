"""Pre-stage everything the comparison run needs from the network.

Run this ON A LOGIN NODE (which has internet) before submitting cluster jobs whose
compute nodes are offline. It reads config/comparison_matrix.yaml and:

  * downloads the torchvision-backed datasets (cifar_10, flowers) into the exact
    `path` values from the matrix, and
  * warms the pretrained teacher weights (ResNet34/VGG11/ConvNeXt) into the torch
    hub cache (TORCH_HOME, default ~/.cache/torch/hub/checkpoints).

Afterwards the compute-node run finds everything locally and never touches the
network.

IMPORTANT: the dataset `path` values in the matrix must live on a SHARED filesystem
($HOME / project space) that compute nodes can read — not node-local /tmp or scratch.

Usage:
    uv run predownload.py                         # datasets + teacher weights
    uv run predownload.py --skip-weights          # datasets only
    uv run predownload.py --skip-datasets         # teacher weights only
    uv run predownload.py path/to/matrix.yaml     # use a different matrix
"""

import argparse
import sys
import traceback

import yaml

from config.config_loader import load_config
from data.factory import build_dataset
from models.factory import build_teacher

MATRIX_DEFAULT = "config/comparison_matrix.yaml"

# Datasets that download themselves via torchvision (root=path, download=True).
# Everything else (oxford_pet, tiny_imagenet) reads pre-extracted local files and
# must be staged onto shared storage by hand.
TORCHVISION_DATASETS = {"cifar_10", "flowers"}


def _load_matrix(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def download_datasets(matrix):
    template_config = matrix["template_config"]
    failures = []
    for d in matrix["datasets"]:
        name = d["name"]
        if name not in TORCHVISION_DATASETS:
            print(f"[skip] {name}: not torchvision-backed — stage '{d['path']}' manually.")
            continue
        cfg = load_config(template_config)
        cfg.dataset.name = name
        cfg.dataset.path = d["path"]
        cfg.dataset.image_size = d.get("image_size", 224)
        print(f"[dataset] downloading {name} -> {d['path']}")
        try:
            build_dataset(cfg)  # constructs with download=True
            print(f"[dataset] {name} ready.")
        except Exception:
            print(f"[dataset] FAILED: {name}")
            traceback.print_exc()
            failures.append(("dataset", name))
    return failures


def download_teacher_weights(matrix):
    template_config = matrix["template_config"]
    failures = []
    for teacher_arch in matrix["teachers"]:
        cfg = load_config(template_config)
        cfg.teacher.architecture = teacher_arch
        cfg.teacher.pretrained = True
        cfg.teacher.weights_path = None  # force torchvision pretrained download, not a checkpoint
        print(f"[weights] warming pretrained weights for {teacher_arch}")
        try:
            build_teacher(cfg)  # downloads + caches under TORCH_HOME
            print(f"[weights] {teacher_arch} ready.")
        except Exception:
            print(f"[weights] FAILED: {teacher_arch}")
            traceback.print_exc()
            failures.append(("weights", teacher_arch))
    return failures


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("matrix", nargs="?", default=MATRIX_DEFAULT, help="path to comparison matrix YAML")
    parser.add_argument("--skip-datasets", action="store_true", help="don't download datasets")
    parser.add_argument("--skip-weights", action="store_true", help="don't warm pretrained teacher weights")
    args = parser.parse_args()

    matrix = _load_matrix(args.matrix)

    failures = []
    if not args.skip_datasets:
        failures += download_datasets(matrix)
    if not args.skip_weights:
        failures += download_teacher_weights(matrix)

    print("\n" + "=" * 60)
    if failures:
        print("Completed WITH FAILURES:")
        for kind, name in failures:
            print(f"  - {kind}: {name}")
        sys.exit(1)
    print("All requested items pre-downloaded successfully.")


if __name__ == "__main__":
    main()
