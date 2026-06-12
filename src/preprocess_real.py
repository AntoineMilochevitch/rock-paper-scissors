"""Preprocess real test images to match the prepared dataset format (300×300 RGB PNG)."""
from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parents[1]
INPUT_DIR = ROOT / "data" / "test_real"
OUTPUT_DIR = ROOT / "data" / "test_real_prepared"
TARGET_SIZE = 300


def preprocess(image_path: Path, target_size: int) -> Image.Image:
    with Image.open(image_path) as img:
        rgb = img.convert("RGB")
        return ImageOps.fit(rgb, (target_size, target_size), method=Image.Resampling.LANCZOS)


def main() -> None:
    image_paths = sorted(
        p for p in INPUT_DIR.iterdir()
        if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
    )
    if not image_paths:
        print(f"No images found in {INPUT_DIR}")
        return

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Preprocessing {len(image_paths)} image(s) → {TARGET_SIZE}×{TARGET_SIZE} PNG")
    print(f"Output directory: {OUTPUT_DIR}\n")

    for image_path in image_paths:
        processed = preprocess(image_path, TARGET_SIZE)
        output_path = OUTPUT_DIR / (image_path.stem + ".png")
        processed.save(output_path, format="PNG")
        print(f"  {image_path.name} ({Image.open(image_path).size}) → {output_path.name} ({processed.size})")

    print(f"\nDone. {len(image_paths)} images saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
