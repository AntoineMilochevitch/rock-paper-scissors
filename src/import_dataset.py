from __future__ import annotations

import shutil
from pathlib import Path

import kagglehub


DATASET_SLUG = "sanikamal/rock-paper-scissors-dataset"
EXPECTED_SPLITS = ("train", "test", "validation")


def has_expected_structure(directory: Path) -> bool:
	return all((directory / split_name).is_dir() for split_name in EXPECTED_SPLITS)


def find_dataset_root(downloaded_path: Path) -> Path:
	candidates = [directory for directory in [downloaded_path, *downloaded_path.rglob("*")] if directory.is_dir()]
	valid_candidates = [directory for directory in candidates if has_expected_structure(directory)]

	if not valid_candidates:
		raise FileNotFoundError(f"Could not find a dataset root inside: {downloaded_path}")

	return min(valid_candidates, key=lambda directory: len(directory.relative_to(downloaded_path).parts))


def main() -> None:
	project_root = Path(__file__).resolve().parents[1]
	data_dir = project_root / "data"
	dataset_dir = data_dir / "rock-paper-scissors-dataset"

	data_dir.mkdir(exist_ok=True)

	if dataset_dir.exists() and has_expected_structure(dataset_dir):
		print(f"Dataset already present at: {dataset_dir}")
		return

	if dataset_dir.exists():
		shutil.rmtree(dataset_dir)

	downloaded_path = Path(kagglehub.dataset_download(DATASET_SLUG))
	dataset_root = find_dataset_root(downloaded_path)
	shutil.copytree(dataset_root, dataset_dir)

	print(f"Dataset copied to: {dataset_dir}")


if __name__ == "__main__":
	main()