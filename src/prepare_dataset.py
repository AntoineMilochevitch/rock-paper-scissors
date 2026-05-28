from __future__ import annotations

import argparse
import shutil
from collections import Counter
from pathlib import Path
from random import Random

import matplotlib
from PIL import Image, ImageEnhance, ImageOps

matplotlib.use("Agg")
from matplotlib import pyplot as plt

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp"}
EXPECTED_SPLITS = ("train", "test", "validation")
EXPECTED_CLASSES = ("paper", "rock", "scissors")
DEFAULT_VALIDATION_RATIO = 0.10
DEFAULT_AUGMENTATION_RATIO = 0.20
DEFAULT_TARGET_SIZE = 300
DEFAULT_RANDOM_SEED = 42


def is_image_file(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS


def infer_class_from_filename(image_path: Path) -> str | None:
    stem = image_path.stem.lower()
    for class_name in EXPECTED_CLASSES:
        if stem.startswith(class_name):
            return class_name
    return None


def iter_split_images(split_dir: Path):
    class_dirs = [split_dir / class_name for class_name in EXPECTED_CLASSES if (split_dir / class_name).is_dir()]

    if class_dirs:
        for class_dir in class_dirs:
            class_name = class_dir.name
            for image_path in sorted(path for path in class_dir.iterdir() if is_image_file(path)):
                yield class_name, image_path
        return

    for image_path in sorted(path for path in split_dir.iterdir() if is_image_file(path)):
        class_name = infer_class_from_filename(image_path)
        if class_name is not None:
            yield class_name, image_path


def count_images(split_dir: Path) -> dict[str, int]:
    counts = {class_name: 0 for class_name in EXPECTED_CLASSES}
    for class_name, _ in iter_split_images(split_dir):
        counts[class_name] += 1
    return counts


def ensure_resolution_directory(directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)


def load_normalized_image(image_path: Path, target_size: int) -> Image.Image:
    with Image.open(image_path) as image:
        rgb_image = image.convert("RGB")
        if target_size > 0:
            return ImageOps.fit(rgb_image, (target_size, target_size), method=Image.Resampling.LANCZOS)
        return rgb_image.copy()


def zoom_image(image: Image.Image, zoom_factor: float, target_size: int) -> Image.Image:
    if abs(zoom_factor - 1.0) < 1e-3:
        return image

    width, height = image.size
    if zoom_factor > 1.0:
        crop_width = max(1, int(round(width / zoom_factor)))
        crop_height = max(1, int(round(height / zoom_factor)))
        left = (width - crop_width) // 2
        top = (height - crop_height) // 2
        cropped = image.crop((left, top, left + crop_width, top + crop_height))
        return cropped.resize((width, height), Image.Resampling.LANCZOS)

    scaled_width = max(1, int(round(width * zoom_factor)))
    scaled_height = max(1, int(round(height * zoom_factor)))
    scaled = image.resize((scaled_width, scaled_height), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (width, height), (255, 255, 255))
    paste_x = (width - scaled_width) // 2
    paste_y = (height - scaled_height) // 2
    canvas.paste(scaled, (paste_x, paste_y))
    return canvas


def augment_image(image: Image.Image, rng: Random) -> tuple[Image.Image, dict[str, float]]:
    augmented = image.copy()
    augmentation_params = {
        "rotation": rng.uniform(-18.0, 18.0),
        "zoom": rng.uniform(0.85, 1.15),
        "brightness": rng.uniform(0.80, 1.20),
        "contrast": rng.uniform(0.80, 1.20),
    }

    augmented = augmented.rotate(
        augmentation_params["rotation"],
        resample=Image.Resampling.BICUBIC,
        expand=False,
        fillcolor=(255, 255, 255),
    )
    augmented = zoom_image(augmented, augmentation_params["zoom"], augmented.size[0])
    augmented = ImageEnhance.Brightness(augmented).enhance(augmentation_params["brightness"])
    augmented = ImageEnhance.Contrast(augmented).enhance(augmentation_params["contrast"])
    return augmented, augmentation_params


def collect_stats(dataset_root: Path, target_size: int) -> tuple[dict[str, Counter[str]], Counter[tuple[int, int]]]:
    split_class_counts: dict[str, Counter[str]] = {split_name: Counter() for split_name in EXPECTED_SPLITS}
    resolution_counts: Counter[tuple[int, int]] = Counter()

    for split_name in EXPECTED_SPLITS:
        split_dir = dataset_root / split_name
        if not split_dir.exists():
            continue

        for class_name, image_path in iter_split_images(split_dir):
            split_class_counts[split_name][class_name] += 1
            with Image.open(image_path) as image:
                width, height = image.convert("RGB").size
                if target_size > 0:
                    width = target_size
                    height = target_size
            resolution_counts[(width, height)] += 1

    return split_class_counts, resolution_counts


def save_image(image: Image.Image, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)


def write_normalized_images(class_name: str, image_paths: list[Path], target_split_dir: Path, target_size: int, prefix: str | None = None) -> None:
    for image_path in image_paths:
        target_name = f"{image_path.stem}.png" if prefix is None else f"{prefix}{image_path.stem}.png"
        output_path = target_split_dir / class_name / target_name
        save_image(load_normalized_image(image_path, target_size), output_path)


def plot_grouped_split_counts(report_dir: Path, before_counts: dict[str, Counter[str]], after_counts: dict[str, Counter[str]]) -> None:
    report_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)

    for axis, counts, title in [
        (axes[0], before_counts, "Before preprocessing"),
        (axes[1], after_counts, "After preprocessing"),
    ]:
        x_positions = range(len(EXPECTED_CLASSES))
        split_names = [split_name for split_name in EXPECTED_SPLITS if sum(counts[split_name].values()) > 0]
        width = 0.25

        for index, split_name in enumerate(split_names):
            offsets = [position + (index - (len(split_names) - 1) / 2) * width for position in x_positions]
            values = [counts[split_name].get(class_name, 0) for class_name in EXPECTED_CLASSES]
            axis.bar(offsets, values, width=width, label=split_name)

        axis.set_title(title)
        axis.set_xlabel("Class")
        axis.set_xticks(list(x_positions))
        axis.set_xticklabels(EXPECTED_CLASSES)
        axis.grid(axis="y", alpha=0.25)

    axes[0].set_ylabel("Number of images")
    axes[1].legend()
    fig.suptitle("Class distribution before and after preprocessing")
    fig.tight_layout()
    fig.savefig(report_dir / "class_distribution_before_after.png", dpi=160)
    plt.close(fig)


def plot_validation_growth(report_dir: Path, before_counts: dict[str, Counter[str]], after_counts: dict[str, Counter[str]]) -> None:
    fig, axis = plt.subplots(figsize=(8, 5))
    x_positions = range(len(EXPECTED_CLASSES))
    before_values = [before_counts["validation"].get(class_name, 0) for class_name in EXPECTED_CLASSES]
    after_values = [after_counts["validation"].get(class_name, 0) for class_name in EXPECTED_CLASSES]

    axis.bar([position - 0.18 for position in x_positions], before_values, width=0.36, label="before")
    axis.bar([position + 0.18 for position in x_positions], after_values, width=0.36, label="after")
    axis.set_title("Validation class counts before and after preprocessing")
    axis.set_xlabel("Class")
    axis.set_ylabel("Number of images")
    axis.set_xticks(list(x_positions))
    axis.set_xticklabels(EXPECTED_CLASSES)
    axis.legend()
    axis.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(report_dir / "validation_before_after.png", dpi=160)
    plt.close(fig)


def plot_resolution_comparison(report_dir: Path, before_resolutions: Counter[tuple[int, int]], after_resolutions: Counter[tuple[int, int]]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)

    for axis, resolutions, title in [
        (axes[0], before_resolutions, "Before preprocessing"),
        (axes[1], after_resolutions, "After preprocessing"),
    ]:
        if resolutions:
            ordered = sorted(resolutions.items(), key=lambda item: (item[0][0] * item[0][1], item[0][0], item[0][1]))
            labels = [f"{width}x{height}" for (width, height), _ in ordered]
            values = [count for _, count in ordered]
            axis.bar(labels, values, color="#2563eb")
            for index, count in enumerate(values):
                axis.text(index, count, str(count), ha="center", va="bottom", fontsize=9)
        axis.set_title(title)
        axis.set_xlabel("Resolution")
        axis.grid(axis="y", alpha=0.25)

    axes[0].set_ylabel("Number of images")
    fig.suptitle("Exact image resolutions before and after preprocessing")
    fig.tight_layout()
    fig.savefig(report_dir / "resolution_before_after.png", dpi=160)
    plt.close(fig)


def plot_augmentation_parameters(report_dir: Path, augmentation_records: list[dict[str, float]]) -> None:
    if not augmentation_records:
        return

    rotations = [record["rotation"] for record in augmentation_records]
    zooms = [record["zoom"] for record in augmentation_records]
    brightness = [record["brightness"] for record in augmentation_records]
    contrast = [record["contrast"] for record in augmentation_records]

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    hist_specs = [
        (axes[0, 0], rotations, "Rotation angles", "Angle (degrees)"),
        (axes[0, 1], zooms, "Zoom factors", "Zoom factor"),
        (axes[1, 0], brightness, "Brightness factors", "Brightness factor"),
        (axes[1, 1], contrast, "Contrast factors", "Contrast factor"),
    ]

    for axis, values, title, xlabel in hist_specs:
        axis.hist(values, bins=min(20, max(5, len(set(round(value, 2) for value in values)))))
        axis.set_title(title)
        axis.set_xlabel(xlabel)
        axis.set_ylabel("Count")
        axis.grid(axis="y", alpha=0.25)

    fig.suptitle("Distribution of augmentation parameters")
    fig.tight_layout()
    fig.savefig(report_dir / "augmentation_parameters.png", dpi=160)
    plt.close(fig)


def plot_sample_preprocessing_comparison(report_dir: Path, sample_pairs: dict[str, tuple[Path, Path, dict[str, float]]]) -> None:
    available_classes = [class_name for class_name in EXPECTED_CLASSES if class_name in sample_pairs]
    if not available_classes:
        return

    fig, axes = plt.subplots(len(available_classes), 2, figsize=(8, 4 * len(available_classes)))

    if len(available_classes) == 1:
        axes = [axes]

    for row_index, class_name in enumerate(available_classes):
        original_path, augmented_path, params = sample_pairs[class_name]

        with Image.open(original_path) as original_image:
            axes[row_index][0].imshow(original_image)
            axes[row_index][0].set_title(f"{class_name} original")
            axes[row_index][0].axis("off")

        with Image.open(augmented_path) as augmented_image:
            axes[row_index][1].imshow(augmented_image)
            axes[row_index][1].set_title(
                f"{class_name} augmented\nrot={params['rotation']:.1f}, zoom={params['zoom']:.2f}, br={params['brightness']:.2f}, ct={params['contrast']:.2f}"
            )
            axes[row_index][1].axis("off")

    fig.suptitle("Before and after preprocessing samples")
    fig.tight_layout()
    fig.savefig(report_dir / "sample_preprocessing_before_after.png", dpi=160)
    plt.close(fig)


def ensure_clean_directory(directory: Path) -> None:
    if directory.exists():
        shutil.rmtree(directory)
    directory.mkdir(parents=True, exist_ok=True)


def copy_classified_images(class_name: str, image_paths: list[Path], target_split_dir: Path, prefix: str | None = None) -> None:
    for image_path in image_paths:
        target_dir = target_split_dir / class_name
        target_dir.mkdir(parents=True, exist_ok=True)

        target_name = image_path.name if prefix is None else f"{prefix}{image_path.name}"
        shutil.copy2(image_path, target_dir / target_name)


def build_rebalanced_dataset(source_root: Path, output_root: Path, report_root: Path, validation_ratio: float, augmentation_ratio: float, target_size: int, seed: int) -> None:
    train_dir = source_root / "train"
    test_dir = source_root / "test"
    validation_dir = source_root / "validation"

    if not train_dir.exists() or not test_dir.exists() or not validation_dir.exists():
        raise FileNotFoundError("Source dataset must contain train, test and validation folders")

    ensure_clean_directory(output_root)
    ensure_clean_directory(report_root)

    before_counts, before_resolutions = collect_stats(source_root, target_size=0)

    random = Random(seed)
    augmentation_records: list[dict[str, float]] = []
    sample_pairs: dict[str, tuple[Path, Path, dict[str, float]]] = {}

    # Copy and normalize test as-is.
    for class_name, image_path in iter_split_images(test_dir):
        write_normalized_images(class_name, [image_path], output_root / "test", target_size)

    # Read all train images by class.
    train_images_by_class: dict[str, list[Path]] = {class_name: [] for class_name in EXPECTED_CLASSES}
    for class_name, image_path in iter_split_images(train_dir):
        train_images_by_class[class_name].append(image_path)

    # Read existing validation images and normalize them into class folders.
    validation_images_by_class: dict[str, list[Path]] = {class_name: [] for class_name in EXPECTED_CLASSES}
    for class_name, image_path in iter_split_images(validation_dir):
        validation_images_by_class[class_name].append(image_path)

    selected_from_train: dict[str, list[Path]] = {class_name: [] for class_name in EXPECTED_CLASSES}

    for class_name in EXPECTED_CLASSES:
        train_images = train_images_by_class[class_name]
        target_validation_count = max(len(validation_images_by_class[class_name]), round(len(train_images) * validation_ratio))
        additional_needed = max(0, target_validation_count - len(validation_images_by_class[class_name]))

        if additional_needed > 0:
            selected_from_train[class_name] = random.sample(train_images, additional_needed)
        else:
            selected_from_train[class_name] = []

    # Copy the new train split, excluding the sampled validation images.
    for class_name, train_images in train_images_by_class.items():
        excluded_paths = set(selected_from_train[class_name])
        kept_images = [image_path for image_path in train_images if image_path not in excluded_paths]

        write_normalized_images(class_name, kept_images, output_root / "train", target_size)

        additional_augmented = round(len(kept_images) * augmentation_ratio)
        if additional_augmented > 0 and kept_images:
            chosen_sources = random.choices(kept_images, k=additional_augmented)
            for index, source_path in enumerate(chosen_sources):
                base_image = load_normalized_image(source_path, target_size)
                augmented_image, params = augment_image(base_image, random)
                augmentation_records.append(params)

                augmented_name = f"aug_{index:04d}_{source_path.stem}.png"
                augmented_path = output_root / "train" / class_name / augmented_name
                save_image(augmented_image, augmented_path)

                if class_name not in sample_pairs:
                    sample_pairs[class_name] = (source_path, augmented_path, params)

    # Copy existing validation images plus the sampled train images.
    for class_name, validation_images in validation_images_by_class.items():
        write_normalized_images(class_name, validation_images, output_root / "validation", target_size)
        write_normalized_images(class_name, selected_from_train[class_name], output_root / "validation", target_size, prefix="train_")

    after_counts, after_resolutions = collect_stats(output_root, target_size)

    plot_grouped_split_counts(report_root, before_counts, after_counts)
    plot_validation_growth(report_root, before_counts, after_counts)
    plot_resolution_comparison(report_root, before_resolutions, after_resolutions)
    plot_augmentation_parameters(report_root, augmentation_records)
    plot_sample_preprocessing_comparison(report_root, sample_pairs)

    print(f"Prepared dataset written to: {output_root}")
    print(f"Preprocessing report written to: {report_root}")
    print("Validation was rebuilt from the original validation split plus a stratified sample from train.")
    print("Training images selected for validation were removed from the new train split to avoid data leakage.")
    print(f"All images were resized to {target_size}x{target_size} before saving.")
    print(f"Additional train augmentation ratio: {augmentation_ratio:.2f}")
    print()

    for split_name, counts in after_counts.items():
        total = sum(counts.values())
        print(f"[{split_name}]")
        for class_name in EXPECTED_CLASSES:
            print(f"  {class_name:<10} {counts[class_name]:>5} images")
        print(f"  {'total':<10} {total:>5} images")
        print()

    print("Generated plots:")
    for plot_name in [
        "class_distribution_before_after.png",
        "validation_before_after.png",
        "resolution_before_after.png",
        "augmentation_parameters.png",
        "sample_preprocessing_before_after.png",
    ]:
        plot_path = report_root / plot_name
        if plot_path.exists():
            print(f"  {plot_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare a balanced Rock Paper Scissors dataset with preprocessing for CNN training.")
    parser.add_argument(
        "--source-root",
        type=Path,
        default=Path("data/rock-paper-scissors-dataset"),
        help="Path to the original dataset root.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("data/rock-paper-scissors-prepared"),
        help="Path where the prepared dataset will be written.",
    )
    parser.add_argument(
        "--validation-ratio",
        type=float,
        default=DEFAULT_VALIDATION_RATIO,
        help="Target validation ratio computed from the train split, per class.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_RANDOM_SEED,
        help="Random seed used to select train images for validation.",
    )
    parser.add_argument(
        "--augment-ratio",
        type=float,
        default=DEFAULT_AUGMENTATION_RATIO,
        help="Number of augmented train images to generate per class, as a ratio of the kept train split.",
    )
    parser.add_argument(
        "--target-size",
        type=int,
        default=DEFAULT_TARGET_SIZE,
        help="Square size used to normalize all images before saving.",
    )
    parser.add_argument(
        "--report-root",
        type=Path,
        default=Path("reports/dataset_preparation"),
        help="Directory where before-after preprocessing graphs will be written.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    build_rebalanced_dataset(
        args.source_root,
        args.output_root,
        args.report_root,
        args.validation_ratio,
        args.augment_ratio,
        args.target_size,
        args.seed,
    )


if __name__ == "__main__":
    main()
