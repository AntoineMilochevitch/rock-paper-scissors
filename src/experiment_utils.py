from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable

import matplotlib
import numpy as np

matplotlib.use("Agg")
from matplotlib import pyplot as plt


@dataclass
class RunSummary:
    model_name: str
    run_name: str
    optimizer: str
    epochs: int
    best_epoch: int
    best_val_loss: float
    best_val_accuracy: float
    test_loss: float
    test_accuracy: float
    precision_macro: float
    recall_macro: float
    f1_macro: float
    timestamp: str
    data_root: str
    batch_size: int
    learning_rate: float
    weight_decay: float
    dropout: float
    use_batchnorm: bool
    kernel_size: int
    channels: str
    target_size: int


def ensure_directory(directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)


def current_timestamp() -> str:
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


def save_json(data: dict, output_path: Path) -> None:
    ensure_directory(output_path.parent)
    output_path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def append_run_summary(csv_path: Path, summary: RunSummary) -> None:
    ensure_directory(csv_path.parent)
    row = asdict(summary)
    fieldnames = list(row.keys())

    file_exists = csv_path.exists()
    with csv_path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def save_training_curves(history: list[dict[str, float]], output_path: Path) -> None:
    if not history:
        return

    ensure_directory(output_path.parent)
    epochs = [item["epoch"] for item in history]
    train_loss = [item["train_loss"] for item in history]
    val_loss = [item["val_loss"] for item in history]
    train_accuracy = [item["train_accuracy"] for item in history]
    val_accuracy = [item["val_accuracy"] for item in history]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    axes[0].plot(epochs, train_loss, marker="o", label="train")
    axes[0].plot(epochs, val_loss, marker="o", label="validation")
    axes[0].set_title("Loss curve")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].grid(alpha=0.25)
    axes[0].legend()

    axes[1].plot(epochs, train_accuracy, marker="o", label="train")
    axes[1].plot(epochs, val_accuracy, marker="o", label="validation")
    axes[1].set_title("Accuracy curve")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].grid(alpha=0.25)
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def save_confusion_matrix(confusion: np.ndarray, class_names: list[str], output_path: Path) -> None:
    ensure_directory(output_path.parent)
    fig, axis = plt.subplots(figsize=(7, 6))
    image = axis.imshow(confusion, interpolation="nearest", cmap="Blues")
    axis.figure.colorbar(image, ax=axis)
    axis.set_title("Confusion matrix")
    axis.set_xlabel("Predicted label")
    axis.set_ylabel("True label")
    ticks = np.arange(len(class_names))
    axis.set_xticks(ticks)
    axis.set_yticks(ticks)
    axis.set_xticklabels(class_names)
    axis.set_yticklabels(class_names)

    threshold = confusion.max() / 2 if confusion.size else 0
    for row_index in range(confusion.shape[0]):
        for col_index in range(confusion.shape[1]):
            axis.text(
                col_index,
                row_index,
                int(confusion[row_index, col_index]),
                ha="center",
                va="center",
                color="white" if confusion[row_index, col_index] > threshold else "black",
            )

    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def save_classification_report_bars(report: dict[str, dict[str, float]], class_names: list[str], output_path: Path) -> None:
    ensure_directory(output_path.parent)
    precision = [report[class_name]["precision"] for class_name in class_names]
    recall = [report[class_name]["recall"] for class_name in class_names]
    f1_score = [report[class_name]["f1-score"] for class_name in class_names]

    fig, axis = plt.subplots(figsize=(9, 5))
    x_positions = np.arange(len(class_names))
    width = 0.25

    axis.bar(x_positions - width, precision, width=width, label="precision")
    axis.bar(x_positions, recall, width=width, label="recall")
    axis.bar(x_positions + width, f1_score, width=width, label="f1-score")
    axis.set_xticks(x_positions)
    axis.set_xticklabels(class_names)
    axis.set_ylim(0, 1.05)
    axis.set_title("Per-class metrics")
    axis.set_ylabel("Score")
    axis.grid(axis="y", alpha=0.25)
    axis.legend()

    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
