"""Driver: benchmark the 5 comparison students for every teacher x dataset combo.

Reads config/comparison_matrix.yaml, runs distill_student_combined() for each
(teacher, dataset, student) triple, and writes a per-combo comparison table
(markdown + CSV) under training_results/comparison_<teacher>_<dataset>/.

Per-student loss/accuracy curves are saved by distill_student_combined() itself,
in training_results/combined_<teacher>_<dataset>_<student>_alpha<alpha>/.

Usage:
    uv run run_comparison.py
    uv run run_comparison.py config/comparison_matrix.yaml
"""

import csv
import sys
import traceback
from pathlib import Path

import yaml

from config.config_loader import load_config
from finetune_teacher import finetune_teacher
from distill_student import distill_student_combined

MATRIX_DEFAULT = "config/comparison_matrix.yaml"


def _load_matrix(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def _build_cfg(template_config, teacher_arch, dataset):
    """Load the template config and override teacher/dataset fields for this combo."""
    cfg = load_config(template_config)
    cfg.teacher.architecture = teacher_arch
    cfg.teacher.num_classes = dataset["num_classes"]
    # Unique checkpoint per (teacher, dataset) so finetuned heads don't collide.
    cfg.teacher.save_path = f"checkpoints/teacher_{teacher_arch}_{dataset['name']}.pth"
    cfg.dataset.name = dataset["name"]
    cfg.dataset.path = dataset["path"]
    cfg.dataset.image_size = dataset.get("image_size", 224)
    return cfg


def _write_table(rows, out_dir, teacher_arch, dataset_name):
    out_dir.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "student",
        "num_blocks",
        "student_params",
        "student_plus_classifier",
        "teacher_params",
        "ratio_vs_teacher",
        "accuracy_pct",
    ]

    csv_path = out_dir / "comparison_table.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    md_path = out_dir / "comparison_table.md"
    header = f"# Student comparison — teacher: {teacher_arch}, dataset: {dataset_name}\n\n"
    col_titles = [
        "Student", "Blocks", "Student params", "Student+classifier",
        "Teacher params", "Ratio vs teacher", "Accuracy (%)",
    ]
    lines = [header]
    lines.append("| " + " | ".join(col_titles) + " |")
    lines.append("|" + "|".join(["---"] * len(col_titles)) + "|")
    for r in rows:
        if r.get("accuracy_pct") is None:
            acc = "FAILED"
            ratio = "-"
        else:
            acc = f"{r['accuracy_pct']:.2f}"
            ratio = f"{r['ratio_vs_teacher']:.4f}"
        lines.append(
            "| " + " | ".join([
                str(r["student"]),
                str(r["num_blocks"]),
                f"{r['student_params']:,}" if r["student_params"] is not None else "-",
                f"{r['student_plus_classifier']:,}" if r["student_plus_classifier"] is not None else "-",
                f"{r['teacher_params']:,}" if r["teacher_params"] is not None else "-",
                ratio,
                acc,
            ]) + " |"
        )
    md_path.write_text("\n".join(lines) + "\n")
    print(f"\nComparison table written to {md_path} and {csv_path}")


def run_matrix(matrix_path=MATRIX_DEFAULT):
    matrix = _load_matrix(matrix_path)
    template_config = matrix["template_config"]
    alpha = matrix.get("alpha", 0.5)
    teachers = matrix["teachers"]
    students = matrix["students"]
    datasets = matrix["datasets"]

    for teacher_arch in teachers:
        for dataset in datasets:
            combo = f"{teacher_arch} x {dataset['name']}"
            print(f"\n{'=' * 70}\nCOMBO: {combo}\n{'=' * 70}")

            cfg = _build_cfg(template_config, teacher_arch, dataset)

            # Finetune the teacher once for this combo, then reuse the checkpoint
            # for all 5 student runs (and the feature cache, which is keyed by
            # teacher+dataset only, is extracted on the first student run).
            cfg.teacher.force_retrain = True
            try:
                finetune_teacher(cfg=cfg)
            except Exception:
                print(f"!! Teacher finetune failed for {combo}; skipping combo.")
                traceback.print_exc()
                continue
            cfg.teacher.force_retrain = False

            rows = []
            for student_arch in students:
                cfg.student.architecture = student_arch
                print(f"\n--- {combo} | student: {student_arch} ---")
                row = {
                    "student": student_arch,
                    "num_blocks": None,
                    "student_params": None,
                    "student_plus_classifier": None,
                    "teacher_params": None,
                    "ratio_vs_teacher": None,
                    "accuracy_pct": None,
                }
                try:
                    acc, student, teacher = distill_student_combined(cfg=cfg, alpha=alpha)
                    s_params = student.get_student_parameters()["total"]
                    t_params = teacher.get_teacher_parameters()
                    s_plus_clf = s_params + t_params["classifier_module"]
                    row.update({
                        "num_blocks": student.num_blocks(),
                        "student_params": s_params,
                        "student_plus_classifier": s_plus_clf,
                        "teacher_params": t_params["total"],
                        "ratio_vs_teacher": s_plus_clf / t_params["total"],
                        "accuracy_pct": acc,
                    })
                except Exception:
                    print(f"!! Run failed: {combo} | {student_arch}")
                    traceback.print_exc()
                rows.append(row)

            out_dir = Path("training_results") / f"comparison_{teacher_arch}_{dataset['name']}"
            _write_table(rows, out_dir, teacher_arch, dataset["name"])


if __name__ == "__main__":
    matrix_path = sys.argv[1] if len(sys.argv) > 1 else MATRIX_DEFAULT
    run_matrix(matrix_path)
