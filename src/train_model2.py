from __future__ import annotations

import copy
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
from model_baseline import ConvBlock


DATA_ROOT = Path("data/rock-paper-scissors-prepared")
OUTPUT_ROOT = Path("reports/experiments/model2_deeper")
RESULTS_CSV = Path("reports/experiments/results.csv")
TARGET_SIZE = 300
EPOCHS = 100
BATCH_SIZE = 32
LEARNING_RATE = 3e-4
WEIGHT_DECAY = 5e-4
DROPOUT = 0.4
PATIENCE = 67
SEED = 42
NUM_WORKERS = 2
CHANNELS = (64, 128, 256, 256)
KERNEL_SIZE = 3
USE_BATCHNORM = True
OPTIMIZER_NAME = "adamw"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class DeeperCNN(nn.Module):
    """Deeper CNN used as the second from-scratch experiment.

    Compared to the baseline model, this version increases representational
    capacity by stacking an additional convolutional block and using wider
    feature maps.
    """

    def __init__(
        self,
        num_classes: int = 3,
        channels: tuple[int, ...] = CHANNELS,
        kernel_size: int = KERNEL_SIZE,
        use_batchnorm: bool = USE_BATCHNORM,
        dropout: float = DROPOUT,
    ) -> None:
        super().__init__()

        feature_layers: list[nn.Module] = []
        in_channels = 3
        for index, out_channels in enumerate(channels):
            # Increase the regularization slightly as the network grows deeper.
            block_dropout = dropout * 0.25 if index == 0 else dropout * 0.5 if index < len(channels) - 1 else dropout
            feature_layers.append(ConvBlock(in_channels, out_channels, kernel_size, use_batchnorm, block_dropout))
            in_channels = out_channels

        self.features = nn.Sequential(*feature_layers)
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_channels, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout * 0.5),
            nn.Linear(128, num_classes),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        features = self.features(inputs)
        pooled = self.global_pool(features)
        return self.classifier(pooled)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def build_train_transform(image_size: int, mean: list[float], std: list[float]) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=15, fill=(255, 255, 255)),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
            transforms.RandomErasing(p=0.2, scale=(0.02, 0.15), value="random"),
        ]
    )


def build_eval_transform(image_size: int, mean: list[float], std: list[float]) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )


def compute_dataset_mean_std(dataset: datasets.ImageFolder, batch_size: int, num_workers: int) -> tuple[list[float], list[float]]:
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    channel_sum = torch.zeros(3)
    channel_squared_sum = torch.zeros(3)
    pixel_count = 0

    for images, _ in loader:
        # Images are already converted to tensors in [0, 1].
        pixel_count += images.size(0) * images.size(2) * images.size(3)
        channel_sum += images.sum(dim=(0, 2, 3))
        channel_squared_sum += (images ** 2).sum(dim=(0, 2, 3))

    mean = channel_sum / pixel_count
    std = torch.sqrt(channel_squared_sum / pixel_count - mean ** 2).clamp_min(1e-6)
    return mean.tolist(), std.tolist()


def build_datasets(data_root: Path, image_size: int, batch_size: int, num_workers: int) -> tuple[datasets.ImageFolder, datasets.ImageFolder, datasets.ImageFolder, list[str], list[float], list[float]]:
    stats_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
    ])
    raw_train = datasets.ImageFolder(data_root / "train", transform=stats_transform)
    mean, std = compute_dataset_mean_std(raw_train, batch_size=batch_size, num_workers=num_workers)

    train_dataset = datasets.ImageFolder(data_root / "train", transform=build_train_transform(image_size, mean, std))
    validation_dataset = datasets.ImageFolder(data_root / "validation", transform=build_eval_transform(image_size, mean, std))
    test_dataset = datasets.ImageFolder(data_root / "test", transform=build_eval_transform(image_size, mean, std))
    return train_dataset, validation_dataset, test_dataset, train_dataset.classes, mean, std


def build_dataloaders(train_dataset, validation_dataset, test_dataset, batch_size: int, num_workers: int) -> tuple[DataLoader, DataLoader, DataLoader]:
    pin_memory = DEVICE.type == "cuda"
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory)
    validation_loader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)
    return train_loader, validation_loader, test_loader


def build_optimizer(model: nn.Module) -> torch.optim.Optimizer:
    # AdamW usually behaves better than Adam when the model is deeper and more expressive.
    return torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)


def build_scheduler(optimizer: torch.optim.Optimizer) -> torch.optim.lr_scheduler.ReduceLROnPlateau:
    # Reduce the learning rate when validation loss stops improving.
    return torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=4)


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, criterion: nn.Module, device: torch.device) -> tuple[float, float, list[int], list[int]]:
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_targets: list[int] = []
    all_predictions: list[int] = []

    for inputs, targets in loader:
        inputs = inputs.to(device)
        targets = targets.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        running_loss += loss.item() * inputs.size(0)
        predictions = outputs.argmax(dim=1)
        correct += (predictions == targets).sum().item()
        total += targets.size(0)
        all_targets.extend(targets.cpu().tolist())
        all_predictions.extend(predictions.cpu().tolist())

    return running_loss / total, correct / total, all_targets, all_predictions


def train_one_epoch(model: nn.Module, loader: DataLoader, criterion: nn.Module, optimizer: torch.optim.Optimizer, device: torch.device) -> tuple[float, float]:
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, targets in loader:
        inputs = inputs.to(device)
        targets = targets.to(device)

        optimizer.zero_grad(set_to_none=True)
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        correct += (outputs.argmax(dim=1) == targets).sum().item()
        total += targets.size(0)

    return running_loss / total, correct / total


def train_model() -> None:
    set_seed(SEED)
    print(f"Using device: {DEVICE}")
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    RESULTS_CSV.parent.mkdir(parents=True, exist_ok=True)

    train_dataset, validation_dataset, test_dataset, class_names, mean, std = build_datasets(
        DATA_ROOT,
        image_size=TARGET_SIZE,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
    )
    train_loader, validation_loader, test_loader = build_dataloaders(
        train_dataset,
        validation_dataset,
        test_dataset,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
    )

    model = DeeperCNN(
        num_classes=len(class_names),
        channels=CHANNELS,
        kernel_size=KERNEL_SIZE,
        use_batchnorm=USE_BATCHNORM,
        dropout=DROPOUT,
    ).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = build_optimizer(model)
    scheduler = build_scheduler(optimizer)

    history: list[dict[str, float]] = []
    best_state = None
    best_val_loss = float("inf")
    best_val_accuracy = 0.0
    best_epoch = 0
    patience_counter = 0

    for epoch in range(1, EPOCHS + 1):
        train_loss, train_accuracy = train_one_epoch(model, train_loader, criterion, optimizer, DEVICE)
        val_loss, val_accuracy, _, _ = evaluate(model, validation_loader, criterion, DEVICE)
        scheduler.step(val_loss)

        history.append(
            {
                "epoch": float(epoch),
                "train_loss": train_loss,
                "train_accuracy": train_accuracy,
                "val_loss": val_loss,
                "val_accuracy": val_accuracy,
            }
        )

        if val_loss < best_val_loss:
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

    test_loss, test_accuracy, targets, predictions = evaluate(model, test_loader, criterion, DEVICE)
    report = classification_report(targets, predictions, target_names=class_names, output_dict=True, zero_division=0)
    confusion = confusion_matrix(targets, predictions)

    run_timestamp = current_timestamp()
    run_name = f"model2_{run_timestamp}"
    run_dir = OUTPUT_ROOT / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    summary = RunSummary(
        model_name="deep_cnn",
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

    # Save the best checkpoint and every artifact needed for the report.
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "class_names": class_names,
            "mean": mean,
            "std": std,
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
                "optimizer": OPTIMIZER_NAME,
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
