"""Run inference on real test images with all three trained models and save a report figure."""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
import torch
from PIL import Image
from torchvision import transforms
from torchvision.models import resnet18

matplotlib.use("Agg")
from matplotlib import pyplot as plt
from matplotlib.patches import FancyBboxPatch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from torch import nn
from model_baseline import BaselineCNN, ConvBlock


class DeeperCNN(nn.Module):
    def __init__(self, num_classes: int = 3, channels: tuple = (64, 128, 256, 256),
                 kernel_size: int = 3, use_batchnorm: bool = True, dropout: float = 0.4) -> None:
        super().__init__()
        feature_layers: list[nn.Module] = []
        in_channels = 3
        for index, out_channels in enumerate(channels):
            block_dropout = dropout * 0.25 if index == 0 else dropout * 0.5 if index < len(channels) - 1 else dropout
            feature_layers.append(ConvBlock(in_channels, out_channels, kernel_size, use_batchnorm, block_dropout))
            in_channels = out_channels
        self.features = nn.Sequential(*feature_layers)
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_channels, 256), nn.ReLU(inplace=True), nn.Dropout(dropout),
            nn.Linear(256, 128), nn.ReLU(inplace=True), nn.Dropout(dropout * 0.5),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.global_pool(self.features(x)))

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TEST_DIR = ROOT / "data" / "test_real_prepared"
EXPERIMENTS_DIR = ROOT / "reports" / "experiments"
OUTPUT_PATH = ROOT / "reports" / "real_world_predictions.png"

CLASS_COLORS = {"paper": "#3b82f6", "rock": "#f59e0b", "scissors": "#10b981"}


def find_latest_checkpoint(experiment_dir: Path) -> Path | None:
    checkpoints = sorted(experiment_dir.rglob("best_model.pt"))
    return checkpoints[-1] if checkpoints else None


def load_checkpoint(path: Path) -> dict:
    return torch.load(path, map_location=DEVICE, weights_only=False)


def build_eval_transform(target_size: int, mean: list, std: list) -> transforms.Compose:
    # Use Resize (shorter edge) + CenterCrop to match ImageOps.fit used during preprocessing,
    # rather than squashing a portrait image into a square (which distorts hand shape).
    return transforms.Compose([
        transforms.Resize(target_size),
        transforms.CenterCrop(target_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])


def load_model1(ckpt_path: Path):
    ckpt = load_checkpoint(ckpt_path)
    config = ckpt["config"]
    channels = tuple(config["channels"])
    model = BaselineCNN(
        num_classes=len(ckpt["class_names"]),
        channels=channels,
        kernel_size=config["kernel_size"],
        use_batchnorm=config["use_batchnorm"],
        dropout=config["dropout"],
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval().to(DEVICE)
    transform = build_eval_transform(config["target_size"], ckpt["mean"], ckpt["std"])
    return model, ckpt["class_names"], transform


def load_model2(ckpt_path: Path):
    ckpt = load_checkpoint(ckpt_path)
    config = ckpt["config"]
    channels_raw = config["channels"]
    channels = tuple(channels_raw) if isinstance(channels_raw, (list, tuple)) else tuple(int(c) for c in str(channels_raw).split("-"))
    model = DeeperCNN(num_classes=len(ckpt["class_names"]), channels=channels)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval().to(DEVICE)
    transform = build_eval_transform(config["target_size"], ckpt["mean"], ckpt["std"])
    return model, ckpt["class_names"], transform


def load_model3(ckpt_path: Path):
    ckpt = load_checkpoint(ckpt_path)
    config = ckpt["config"]
    model = resnet18(weights=None)
    model.fc = torch.nn.Linear(model.fc.in_features, len(ckpt["class_names"]))
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval().to(DEVICE)
    transform = build_eval_transform(config["target_size"], ckpt["mean"], ckpt["std"])
    return model, ckpt["class_names"], transform


@torch.no_grad()
def predict(model: torch.nn.Module, transform: transforms.Compose, image_path: Path, class_names: list[str]) -> tuple[str, dict[str, float]]:
    image = Image.open(image_path).convert("RGB")
    tensor = transform(image).unsqueeze(0).to(DEVICE)
    probs = torch.softmax(model(tensor), dim=1).squeeze().cpu()
    pred_idx = probs.argmax().item()
    return class_names[pred_idx], {class_names[i]: float(probs[i]) for i in range(len(class_names))}


def save_figure(image_paths: list[Path], results: list[dict], output_path: Path) -> None:
    """Save a grid figure: one column per image, one row per model, with confidence bars."""
    n_images = len(image_paths)
    n_models = len(results)
    fig, axes = plt.subplots(n_models + 1, n_images, figsize=(5 * n_images, 4 * (n_models + 1)))

    if n_images == 1:
        axes = [[ax] for ax in axes]
    if n_models + 1 == 1:
        axes = [axes]

    # Row 0: original images
    for col, image_path in enumerate(image_paths):
        ax = axes[0][col]
        image = Image.open(image_path).convert("RGB")
        ax.imshow(image)
        ax.set_title(image_path.stem, fontsize=13, fontweight="bold")
        ax.axis("off")

    # Rows 1..n_models: confidence bars for each model
    for row, entry in enumerate(results, start=1):
        model_name = entry["model_name"]
        for col, image_path in enumerate(image_paths):
            ax = axes[row][col]
            pred, probs = entry["predictions"][image_path.name]
            classes = sorted(probs.keys())
            values = [probs[c] for c in classes]
            colors = [CLASS_COLORS.get(c, "#6b7280") for c in classes]
            bars = ax.barh(classes, values, color=colors, height=0.5)
            ax.set_xlim(0, 1)
            ax.axvline(x=0, color="gray", linewidth=0.5)
            ax.set_xlabel("Confidence")
            for bar, val in zip(bars, values):
                ax.text(min(val + 0.02, 0.98), bar.get_y() + bar.get_height() / 2,
                        f"{val:.1%}", va="center", fontsize=10,
                        fontweight="bold" if classes[values.index(val)] == pred else "normal")
            title_color = CLASS_COLORS.get(pred, "#374151")
            ax.set_title(f"{model_name}\n→ {pred.upper()}", fontsize=10, color=title_color, fontweight="bold")
            ax.grid(axis="x", alpha=0.3)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

    fig.suptitle("Real-world predictions — all three models", fontsize=15, fontweight="bold", y=1.01)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"\nFigure saved to: {output_path}")


def main() -> None:
    image_paths = sorted(p for p in TEST_DIR.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"})
    if not image_paths:
        print(f"No images found in {TEST_DIR}")
        return

    print(f"Using device: {DEVICE}")
    print(f"Images found: {[p.name for p in image_paths]}\n")

    loaders = [
        ("Model 1 (Baseline CNN)", EXPERIMENTS_DIR / "model1_baseline", load_model1),
        ("Model 2 (Deeper CNN)",   EXPERIMENTS_DIR / "model2_deeper",   load_model2),
        ("Model 3 (ResNet18)",     EXPERIMENTS_DIR / "model3_resnet",   load_model3),
    ]

    loaded_models: list[tuple[str, torch.nn.Module, list[str], transforms.Compose]] = []
    target_sizes: list[int] = []
    for name, exp_dir, loader in loaders:
        ckpt_path = find_latest_checkpoint(exp_dir)
        if ckpt_path is None:
            print(f"  [SKIP] {name} — no checkpoint found in {exp_dir}")
            print(f"         Run the corresponding train script to regenerate it.")
        else:
            model, class_names, transform = loader(ckpt_path)
            # CenterCrop is transforms[1]; its size attribute gives the target
            size = transform.transforms[1].size
            size = size[0] if isinstance(size, (list, tuple)) else size
            loaded_models.append((name, model, class_names, transform))
            target_sizes.append(size)
            print(f"  [OK]   {name} — {ckpt_path.parent.name} (input {size}×{size})")

    if not loaded_models:
        print("\nNo models loaded.")
        return

    # Collect all predictions
    results: list[dict] = []
    for model_name, model, class_names, transform in loaded_models:
        entry: dict = {"model_name": model_name, "predictions": {}}
        for image_path in image_paths:
            pred, probs = predict(model, transform, image_path, class_names)
            entry["predictions"][image_path.name] = (pred, probs)
        results.append(entry)

    # Print ASCII table
    col_w = 22
    separator = "-" * (16 + col_w * len(loaded_models))
    print("\n" + "=" * len(separator))
    header = f"{'Image':<16}" + "".join(f"{name:<{col_w}}" for name, *_ in loaded_models)
    print(header)
    print(separator)
    for image_path in image_paths:
        row = f"{image_path.name:<16}"
        for entry in results:
            pred, probs = entry["predictions"][image_path.name]
            cell = f"{pred} ({probs[pred]:.0%})"
            row += f"{cell:<{col_w}}"
        print(row)

    # Print detailed probabilities
    print("\nDetailed probabilities:")
    for image_path in image_paths:
        print(f"\n  {image_path.name}")
        for entry in results:
            pred, probs = entry["predictions"][image_path.name]
            prob_str = "   ".join(f"{c}: {p:.1%}" for c, p in sorted(probs.items()))
            marker = "✓" if True else " "
            print(f"    [{entry['model_name']}]  → {pred:<10}  |  {prob_str}")

    save_figure(image_paths, results, OUTPUT_PATH)
    save_model_view(image_paths, loaded_models, target_sizes)


def save_model_view(image_paths: list[Path], loaded_models: list, target_sizes: list[int]) -> None:
    """Save a grid showing what each model actually receives after preprocessing (no normalization)."""
    if not loaded_models:
        return
    out_path = ROOT / "reports" / "real_world_model_view.png"
    n_images = len(image_paths)
    n_models = len(loaded_models)
    fig, axes = plt.subplots(n_images, n_models + 1, figsize=(4 * (n_models + 1), 4 * n_images))
    if n_images == 1:
        axes = [axes]

    for row, image_path in enumerate(image_paths):
        axes[row][0].imshow(Image.open(image_path).convert("RGB"))
        axes[row][0].set_title(f"Original\n{Image.open(image_path).size}", fontsize=9)
        axes[row][0].axis("off")

        for col, ((model_name, *_), size) in enumerate(zip(loaded_models, target_sizes), start=1):
            crop_transform = transforms.Compose([transforms.Resize(size), transforms.CenterCrop(size)])
            cropped = crop_transform(Image.open(image_path).convert("RGB"))
            axes[row][col].imshow(cropped)
            short_name = model_name.split("(")[1].rstrip(")")
            axes[row][col].set_title(f"{short_name}\n{size}×{size}", fontsize=9)
            axes[row][col].axis("off")

    fig.suptitle("Images seen by each model (center-crop + resize, before normalization)", fontsize=11, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"Model-view figure saved to: {out_path}")


if __name__ == "__main__":
    main()
