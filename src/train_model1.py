from __future__ import annotations

import copy
import json
import random
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import classification_report, confusion_matrix
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from experiment_utils import (
    RunSummary,
    append_run_summary,
    current_timestamp,
    save_classification_report_bars,
    save_confusion_matrix,
    save_json,
    save_training_curves,
)
from model_baseline import BaselineCNN


DATA_ROOT = Path("data/rock-paper-scissors-prepared")
OUTPUT_ROOT = Path("reports/experiments/model1_baseline")
RESULTS_CSV = Path("reports/experiments/results.csv")
TARGET_SIZE = 300
EPOCHS = 100
BATCH_SIZE = 32
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
DROPOUT = 0.3
PATIENCE = 67
SEED = 42
NUM_WORKERS = 2
CHANNELS = (32, 64, 128)
KERNEL_SIZE = 3
USE_BATCHNORM = True
OPTIMIZER_NAME = "adam"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train_model() -> None:
    set_seed(SEED)
    print(f"Using device: {DEVICE}")
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    RESULTS_CSV.parent.mkdir(parents=True, exist_ok=True)

    stats_transform = transforms.Compose([
        transforms.Resize((TARGET_SIZE, TARGET_SIZE)),
        transforms.ToTensor(),
    ])
    raw_train = datasets.ImageFolder(DATA_ROOT / "train", transform=stats_transform)
    stats_loader = DataLoader(raw_train, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    channel_sum = torch.zeros(3)
    channel_squared_sum = torch.zeros(3)
    pixel_count = 0
    for images, _ in stats_loader:
        pixel_count += images.size(0) * images.size(2) * images.size(3)
        channel_sum += images.sum(dim=(0, 2, 3))
        channel_squared_sum += (images ** 2).sum(dim=(0, 2, 3))

    mean = channel_sum / pixel_count
    std = torch.sqrt(channel_squared_sum / pixel_count - mean ** 2).clamp_min(1e-6)

    train_transform = transforms.Compose([
        transforms.Resize((TARGET_SIZE, TARGET_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean.tolist(), std=std.tolist()),
    ])

    train_dataset = datasets.ImageFolder(DATA_ROOT / "train", transform=train_transform)
    validation_dataset = datasets.ImageFolder(DATA_ROOT / "validation", transform=train_transform)
    test_dataset = datasets.ImageFolder(DATA_ROOT / "test", transform=train_transform)
    class_names = train_dataset.classes

    pin_memory = DEVICE.type == "cuda"
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=pin_memory)
    validation_loader = DataLoader(validation_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=pin_memory)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=pin_memory)

    model = BaselineCNN(
        num_classes=len(class_names),
        channels=CHANNELS,
        kernel_size=KERNEL_SIZE,
        use_batchnorm=USE_BATCHNORM,
        dropout=DROPOUT,
    ).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    history: list[dict[str, float]] = []
    best_state = None
    best_val_loss = float("inf")
    best_val_accuracy = 0.0
    best_epoch = 0
    patience_counter = 0

    for epoch in range(1, EPOCHS + 1):
        model.train()
        train_running_loss = 0.0
        train_correct = 0
        train_total = 0

        for inputs, targets in train_loader:
            inputs = inputs.to(DEVICE)
            targets = targets.to(DEVICE)

            optimizer.zero_grad(set_to_none=True)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            train_running_loss += loss.item() * inputs.size(0)
            train_correct += (outputs.argmax(dim=1) == targets).sum().item()
            train_total += targets.size(0)

        train_loss = train_running_loss / train_total
        train_accuracy = train_correct / train_total

        model.eval()
        val_running_loss = 0.0
        val_correct = 0
        val_total = 0
        val_targets: list[int] = []
        val_predictions: list[int] = []

        with torch.no_grad():
            for inputs, targets in validation_loader:
                inputs = inputs.to(DEVICE)
                targets = targets.to(DEVICE)
                outputs = model(inputs)
                loss = criterion(outputs, targets)

                val_running_loss += loss.item() * inputs.size(0)
                predictions = outputs.argmax(dim=1)
                val_correct += (predictions == targets).sum().item()
                val_total += targets.size(0)
                val_targets.extend(targets.cpu().tolist())
                val_predictions.extend(predictions.cpu().tolist())

        val_loss = val_running_loss / val_total
        val_accuracy = val_correct / val_total

        history.append(
            {
                "epoch": float(epoch),
                "train_loss": train_loss,
                "train_accuracy": train_accuracy,
                "val_loss": val_loss,
                "val_accuracy": val_accuracy,
            }
        )

        improved = val_loss < best_val_loss
        if improved:
            best_val_loss = val_loss
            best_val_accuracy = val_accuracy
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1

        print(
            f"Epoch {epoch:02d}/{EPOCHS} - "
            f"train_loss={train_loss:.4f} train_acc={train_accuracy:.4f} "
            f"val_loss={val_loss:.4f} val_acc={val_accuracy:.4f}"
        )

        if patience_counter >= PATIENCE:
            print(f"Early stopping triggered at epoch {epoch}.")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    test_running_loss = 0.0
    test_correct = 0
    test_total = 0
    targets: list[int] = []
    predictions: list[int] = []

    with torch.no_grad():
        for inputs, target_batch in test_loader:
            inputs = inputs.to(DEVICE)
            target_batch = target_batch.to(DEVICE)
            outputs = model(inputs)
            loss = criterion(outputs, target_batch)

            test_running_loss += loss.item() * inputs.size(0)
            batch_predictions = outputs.argmax(dim=1)
            test_correct += (batch_predictions == target_batch).sum().item()
            test_total += target_batch.size(0)
            targets.extend(target_batch.cpu().tolist())
            predictions.extend(batch_predictions.cpu().tolist())

    test_loss = test_running_loss / test_total
    test_accuracy = test_correct / test_total
    report = classification_report(targets, predictions, target_names=class_names, output_dict=True, zero_division=0)
    confusion = confusion_matrix(targets, predictions)

    run_timestamp = current_timestamp()
    run_name = f"model1_{run_timestamp}"
    run_dir = OUTPUT_ROOT / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    summary = RunSummary(
        model_name="baseline_cnn",
        run_name=run_name,
        optimizer=OPTIMIZER_NAME,
        epochs=len(history),
        best_epoch=best_epoch,
        best_val_loss=best_val_loss,
        best_val_accuracy=best_val_accuracy,
        test_loss=test_loss,
        test_accuracy=test_accuracy,
        precision_macro=report["macro avg"]["precision"],
        recall_macro=report["macro avg"]["recall"],
        f1_macro=report["macro avg"]["f1-score"],
        timestamp=run_timestamp,
        data_root=str(DATA_ROOT),
        batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        dropout=DROPOUT,
        use_batchnorm=USE_BATCHNORM,
        kernel_size=KERNEL_SIZE,
        channels="-".join(str(channel) for channel in CHANNELS),
        target_size=TARGET_SIZE,
    )

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "class_names": class_names,
            "mean": mean.tolist(),
            "std": std.tolist(),
            "config": {
                "data_root": str(DATA_ROOT),
                "output_root": str(OUTPUT_ROOT),
                "results_csv": str(RESULTS_CSV),
                "target_size": TARGET_SIZE,
                "epochs": EPOCHS,
                "batch_size": BATCH_SIZE,
                "learning_rate": LEARNING_RATE,
                "weight_decay": WEIGHT_DECAY,
                "dropout": DROPOUT,
                "patience": PATIENCE,
                "seed": SEED,
                "num_workers": NUM_WORKERS,
                "channels": CHANNELS,
                "kernel_size": KERNEL_SIZE,
                "use_batchnorm": USE_BATCHNORM,
                "device": str(DEVICE),
            },
            "best_epoch": best_epoch,
        },
        run_dir / "best_model.pt",
    )
    save_json({"summary": summary.__dict__, "history": history, "classification_report": report}, run_dir / "metrics.json")
    save_training_curves(history, run_dir / "training_curves.png")
    save_confusion_matrix(confusion, class_names, run_dir / "confusion_matrix.png")
    save_classification_report_bars(report, class_names, run_dir / "per_class_metrics.png")
    append_run_summary(RESULTS_CSV, summary)

    print("\nTraining finished.")
    print(f"Best epoch: {best_epoch}")
    print(f"Best validation accuracy: {best_val_accuracy:.4f}")
    print(f"Test accuracy: {test_accuracy:.4f}")
    print(f"Artifacts saved in: {run_dir}")


def main() -> None:
    train_model()


if __name__ == "__main__":
    main()
