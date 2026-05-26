from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
from random import Random
from statistics import mean, median

import matplotlib
from PIL import Image, ImageStat

matplotlib.use("Agg")
from matplotlib import pyplot as plt

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp"}
EXPECTED_SPLITS = ("train", "test", "validation")
EXPECTED_CLASSES = ("paper", "rock", "scissors")
RANDOM_SEED = 42
SAMPLE_IMAGES_PER_CLASS = 3


def is_image_file(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS


def has_expected_structure(directory: Path) -> bool:
    return all((directory / split_name).is_dir() for split_name in EXPECTED_SPLITS)


def resolve_dataset_root(dataset_dir: Path) -> Path:
    if has_expected_structure(dataset_dir):
        return dataset_dir

    candidates = [directory for directory in [dataset_dir, *dataset_dir.rglob("*")] if directory.is_dir()]
    valid_candidates = [directory for directory in candidates if has_expected_structure(directory)]

    if not valid_candidates:
        raise FileNotFoundError(f"Could not find train/test/validation folders inside: {dataset_dir}")

    return min(valid_candidates, key=lambda directory: len(directory.relative_to(dataset_dir).parts))


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


def iter_class_images(dataset_root: Path):
    for split_name in EXPECTED_SPLITS:
        split_dir = dataset_root / split_name
        if not split_dir.exists():
            continue

        for class_name, image_path in iter_split_images(split_dir):
            yield split_name, class_name, image_path


def collect_dataset_stats(dataset_root: Path):
    split_class_counts: dict[str, Counter[str]] = {split_name: Counter() for split_name in EXPECTED_SPLITS}
    split_totals: Counter[str] = Counter()
    class_totals: Counter[str] = Counter()
    resolution_counts: Counter[tuple[int, int]] = Counter()
    image_dimensions: list[tuple[int, int]] = []
    image_areas: list[int] = []
    aspect_ratios: list[float] = []
    luminance_values: list[float] = []
    channel_means: dict[str, list[float]] = {channel: [] for channel in ("red", "green", "blue")}
    split_luminance: dict[str, list[float]] = {split_name: [] for split_name in EXPECTED_SPLITS}
    split_image_areas: dict[str, list[int]] = {split_name: [] for split_name in EXPECTED_SPLITS}
    class_luminance: dict[str, list[float]] = defaultdict(list)
    class_channel_means: dict[str, dict[str, list[float]]] = defaultdict(lambda: {channel: [] for channel in ("red", "green", "blue")})
    sample_candidates: dict[str, list[Path]] = defaultdict(list)

    for split_name, class_name, image_path in iter_class_images(dataset_root):
        split_class_counts[split_name][class_name] += 1
        split_totals[split_name] += 1
        class_totals[class_name] += 1

        with Image.open(image_path) as image:
            rgb_image = image.convert("RGB")
            width, height = rgb_image.size
            stat = ImageStat.Stat(rgb_image)
            red_mean, green_mean, blue_mean = stat.mean
            luminance = ImageStat.Stat(rgb_image.convert("L")).mean[0]

        image_dimensions.append((width, height))
        image_areas.append(width * height)
        aspect_ratios.append(width / height if height else 0.0)
        resolution_counts[(width, height)] += 1
        luminance_values.append(luminance)
        split_luminance[split_name].append(luminance)
        split_image_areas[split_name].append(width * height)
        channel_means["red"].append(red_mean)
        channel_means["green"].append(green_mean)
        channel_means["blue"].append(blue_mean)
        class_luminance[class_name].append(luminance)
        class_channel_means[class_name]["red"].append(red_mean)
        class_channel_means[class_name]["green"].append(green_mean)
        class_channel_means[class_name]["blue"].append(blue_mean)

        if split_name == "train":
            sample_candidates[class_name].append(image_path)

    random = Random(RANDOM_SEED)
    sample_images: dict[str, list[Path]] = {}
    for class_name, candidates in sample_candidates.items():
        if len(candidates) <= SAMPLE_IMAGES_PER_CLASS:
            sample_images[class_name] = candidates
        else:
            sample_images[class_name] = random.sample(candidates, SAMPLE_IMAGES_PER_CLASS)

    return (
        split_class_counts,
        split_totals,
        class_totals,
        resolution_counts,
        image_dimensions,
        image_areas,
        aspect_ratios,
        luminance_values,
        channel_means,
        split_luminance,
        split_image_areas,
        class_luminance,
        class_channel_means,
        sample_images,
    )


def plot_class_distribution(output_dir: Path, split_class_counts: dict[str, Counter[str]]) -> None:
    output_path = output_dir / "class_distribution_by_split.png"
    split_names = [split_name for split_name in EXPECTED_SPLITS if sum(split_class_counts[split_name].values()) > 0]
    class_names = list(EXPECTED_CLASSES)

    if not split_names:
        return

    fig, ax = plt.subplots(figsize=(10, 6))
    x_positions = range(len(class_names))
    width = 0.25

    for index, split_name in enumerate(split_names):
        offsets = [position + (index - (len(split_names) - 1) / 2) * width for position in x_positions]
        counts = [split_class_counts[split_name].get(class_name, 0) for class_name in class_names]
        ax.bar(offsets, counts, width=width, label=split_name)

    ax.set_title("Image count by class and split")
    ax.set_xlabel("Class")
    ax.set_ylabel("Number of images")
    ax.set_xticks(list(x_positions))
    ax.set_xticklabels(class_names)
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_resolution_counts(output_dir: Path, resolution_counts: Counter[tuple[int, int]]) -> None:
    if not resolution_counts:
        return

    sizes = sorted(resolution_counts.items(), key=lambda item: (item[0][0] * item[0][1], item[0][0], item[0][1]))
    labels = [f"{width}x{height}" for (width, height), _ in sizes]
    counts = [count for _, count in sizes]

    fig, axis = plt.subplots(figsize=(9, 5))
    axis.bar(labels, counts, color="#3b82f6")
    axis.set_title("Exact image resolutions")
    axis.set_xlabel("Resolution")
    axis.set_ylabel("Number of images")
    axis.grid(axis="y", alpha=0.25)
    for index, count in enumerate(counts):
        axis.text(index, count, str(count), ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    fig.savefig(output_dir / "image_resolution_counts.png", dpi=160)
    plt.close(fig)

def plot_pixel_counts_by_split(output_dir: Path, split_image_areas: dict[str, list[int]]) -> None:
    available_splits = [split_name for split_name in EXPECTED_SPLITS if split_image_areas.get(split_name)]
    if not available_splits:
        return

    split_values = [split_image_areas[split_name] for split_name in available_splits]

    fig, axis = plt.subplots(figsize=(8, 4.8))
    axis.boxplot(split_values, tick_labels=available_splits, showfliers=False)
    axis.set_title("Pixels per image by split")
    axis.set_xlabel("Split")
    axis.set_ylabel("Width × height (pixels)")
    axis.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / "pixel_count_by_split.png", dpi=160)
    plt.close(fig)


def plot_aspect_ratios(output_dir: Path, aspect_ratios: list[float]) -> None:
    if not aspect_ratios:
        return

    fig, axis = plt.subplots(figsize=(8, 4.5))
    axis.hist(aspect_ratios, bins=min(20, max(5, len(set(round(value, 3) for value in aspect_ratios)))))
    axis.set_title("Aspect ratio distribution")
    axis.set_xlabel("Width / height")
    axis.set_ylabel("Count")
    axis.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / "aspect_ratio_distribution.png", dpi=160)
    plt.close(fig)


def plot_brightness_distribution(output_dir: Path, split_class_counts: dict[str, Counter[str]], split_luminance: dict[str, list[float]]) -> None:
    if not any(split_luminance.values()):
        return

    split_values = []
    labels = []
    for split_name in EXPECTED_SPLITS:
        values = split_luminance.get(split_name, [])
        if values:
            split_values.append(values)
            labels.append(split_name)

    if split_values:
        fig, axis = plt.subplots(figsize=(8, 4.5))
        axis.boxplot(split_values, tick_labels=labels, showfliers=False)
        axis.set_title("Brightness distribution by split")
        axis.set_xlabel("Split")
        axis.set_ylabel("Mean luminance (0-255)")
        axis.grid(axis="y", alpha=0.25)
        fig.tight_layout()
        fig.savefig(output_dir / "brightness_distribution_by_split.png", dpi=160)
        plt.close(fig)


def plot_channel_means(output_dir: Path, class_channel_means: dict[str, dict[str, list[float]]]) -> None:
    available_classes = [class_name for class_name in EXPECTED_CLASSES if class_channel_means.get(class_name)]
    if not available_classes:
        return

    channel_names = ("red", "green", "blue")
    fig, axis = plt.subplots(figsize=(10, 6))
    x_positions = range(len(available_classes))
    width = 0.22

    for index, channel_name in enumerate(channel_names):
        offsets = [position + (index - 1) * width for position in x_positions]
        values = [mean(class_channel_means[class_name][channel_name]) if class_channel_means[class_name][channel_name] else 0.0 for class_name in available_classes]
        axis.bar(offsets, values, width=width, label=channel_name.capitalize())

    axis.set_title("Average RGB channel intensity by class")
    axis.set_xlabel("Class")
    axis.set_ylabel("Mean channel value (0-255)")
    axis.set_xticks(list(x_positions))
    axis.set_xticklabels(available_classes)
    axis.legend()
    axis.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / "rgb_channel_means_by_class.png", dpi=160)
    plt.close(fig)


def plot_sample_grid(output_dir: Path, sample_images: dict[str, list[Path]]) -> None:
    available_classes = [class_name for class_name in EXPECTED_CLASSES if sample_images.get(class_name)]
    if not available_classes:
        return

    rows = len(available_classes)
    cols = max(len(sample_images[class_name]) for class_name in available_classes)
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))

    if rows == 1 and cols == 1:
        axes = [[axes]]
    elif rows == 1:
        axes = [axes]
    elif cols == 1:
        axes = [[axis] for axis in axes]

    for row_index, class_name in enumerate(available_classes):
        image_paths = sample_images[class_name]
        for col_index in range(cols):
            axis = axes[row_index][col_index]
            axis.axis("off")

            if col_index >= len(image_paths):
                continue

            image_path = image_paths[col_index]
            with Image.open(image_path) as image:
                axis.imshow(image)
                axis.set_title(f"{class_name} #{col_index + 1}\n{image.width}x{image.height}", fontsize=10)

    fig.suptitle("Sample training images")
    fig.tight_layout()
    fig.savefig(output_dir / "sample_training_images.png", dpi=160)
    plt.close(fig)


def summarize_dataset(dataset_dir: Path, output_dir: Path) -> None:
    if not dataset_dir.exists():
        raise FileNotFoundError(f"Dataset folder not found: {dataset_dir}")

    dataset_dir = resolve_dataset_root(dataset_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    (
        split_class_counts,
        split_totals,
        class_totals,
        resolution_counts,
        image_dimensions,
        image_areas,
        aspect_ratios,
        luminance_values,
        channel_means,
        split_luminance,
        split_image_areas,
        class_luminance,
        class_channel_means,
        sample_images,
    ) = collect_dataset_stats(dataset_dir)

    plot_class_distribution(output_dir, split_class_counts)
    plot_resolution_counts(output_dir, resolution_counts)
    plot_pixel_counts_by_split(output_dir, split_image_areas)
    plot_aspect_ratios(output_dir, aspect_ratios)
    plot_brightness_distribution(output_dir, split_class_counts, split_luminance)
    plot_channel_means(output_dir, class_channel_means)
    plot_sample_grid(output_dir, sample_images)

    total_images = sum(split_totals.values())
    widths = [width for width, _ in image_dimensions]
    heights = [height for _, height in image_dimensions]

    print(f"Dataset directory: {dataset_dir}")
    print(f"Reports directory: {output_dir}")
    print()

    for split_name in EXPECTED_SPLITS:
        split_total = split_totals.get(split_name, 0)
        if split_total == 0 and not any(split_class_counts[split_name].values()):
            print(f"[{split_name}] missing")
            continue

        print(f"[{split_name}]")
        for class_name in EXPECTED_CLASSES:
            print(f"  {class_name:<10} {split_class_counts[split_name].get(class_name, 0):>5} images")
        print(f"  {'total':<10} {split_total:>5} images")
        print()

    if image_dimensions:
        print("Geometry")
        print(f"  mean width:  {mean(widths):.1f}")
        print(f"  median width: {median(widths):.1f}")
        print(f"  mean height: {mean(heights):.1f}")
        print(f"  median height: {median(heights):.1f}")
        print(f"  mean area:   {mean(image_areas):.1f}")
        print(f"  median area: {median(image_areas):.1f}")
        print(f"  mean aspect: {mean(aspect_ratios):.3f}")
        print()

    if luminance_values:
        print("Color / brightness")
        print(f"  mean luminance: {mean(luminance_values):.1f}")
        print(f"  median luminance: {median(luminance_values):.1f}")
        print(f"  mean red:   {mean(channel_means['red']):.1f}")
        print(f"  mean green: {mean(channel_means['green']):.1f}")
        print(f"  mean blue:  {mean(channel_means['blue']):.1f}")
        print()

    print("Summary")
    print(f"  total images: {total_images}")
    for split_name in EXPECTED_SPLITS:
        print(f"  {split_name:<10} {split_totals.get(split_name, 0):>5}")
    for class_name in EXPECTED_CLASSES:
        print(f"  {class_name:<10} {class_totals.get(class_name, 0):>5}")
    print()
    print("Generated plots:")
    for plot_name in [
        "class_distribution_by_split.png",
        "image_resolution_counts.png",
        "pixel_count_by_split.png",
        "aspect_ratio_distribution.png",
        "brightness_distribution_by_split.png",
        "rgb_channel_means_by_class.png",
        "sample_training_images.png",
    ]:
        plot_path = output_dir / plot_name
        if plot_path.exists():
            print(f"  {plot_path}")


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    dataset_dir = project_root / "data" / "rock-paper-scissors-dataset"
    output_dir = project_root / "reports" / "dataset_analysis"
    summarize_dataset(dataset_dir, output_dir)


if __name__ == "__main__":
    main()
