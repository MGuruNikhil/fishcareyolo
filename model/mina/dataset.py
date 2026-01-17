"""
Dataset download and organization logic.
"""

import os
import shutil
from pathlib import Path

from roboflow import Roboflow

from mina.core.constants import DISEASE_CLASSES, DATA_DIR, TEST_DATA_DIR, MODEL_DIR
from mina.core.dataset import create_data_yaml


def download_dataset(
    api_key: str,
    workspace: str,
    project_name: str,
    version_number: int,
    format: str = "yolov8",
):
    """
    Download dataset from Roboflow.

    Args:
        api_key: Roboflow API key
        workspace: Roboflow workspace name
        project_name: Roboflow project name
        version_number: Dataset version number
        format: Export format

    Returns:
        Downloaded dataset object
    """
    rf = Roboflow(api_key=api_key)
    project = rf.workspace(workspace).project(project_name)
    version = project.version(version_number)
    dataset = version.download(format)
    return dataset


def organize_dataset(
    source_dir: Path,
    target_dir: Path,
    test_dir: Path,
) -> None:
    """
    Organize downloaded dataset into the expected structure.

    Source structure (Roboflow):
        source_dir/
            train/images/, train/labels/
            valid/images/, valid/labels/
            test/images/, test/labels/
            data.yaml

    Target structure:
        target_dir/
            images/train/, images/val/
            labels/train/, labels/val/
            data.yaml

        test_dir/ (separate, for final evaluation only)
            images/
            labels/

    Args:
        source_dir: Downloaded dataset directory
        target_dir: Target directory for train/val data
        test_dir: Target directory for test data
    """
    target_dir.mkdir(parents=True, exist_ok=True)

    # Create target directories
    (target_dir / "images" / "train").mkdir(parents=True, exist_ok=True)
    (target_dir / "images" / "val").mkdir(parents=True, exist_ok=True)
    (target_dir / "labels" / "train").mkdir(parents=True, exist_ok=True)
    (target_dir / "labels" / "val").mkdir(parents=True, exist_ok=True)

    # Copy training data
    _copy_split(source_dir / "train", target_dir, "train")

    # Copy validation data (Roboflow uses "valid")
    _copy_split(source_dir / "valid", target_dir, "val")

    # Copy test data to separate directory
    test_dir.mkdir(parents=True, exist_ok=True)
    (test_dir / "images").mkdir(parents=True, exist_ok=True)
    (test_dir / "labels").mkdir(parents=True, exist_ok=True)

    src_test_images = source_dir / "test" / "images"
    src_test_labels = source_dir / "test" / "labels"

    if src_test_images.exists():
        for img in src_test_images.glob("*"):
            shutil.copy2(img, test_dir / "images" / img.name)

    if src_test_labels.exists():
        for lbl in src_test_labels.glob("*"):
            shutil.copy2(lbl, test_dir / "labels" / lbl.name)

    # Create data.yaml
    create_data_yaml(target_dir)

    # Print summary
    train_count = len(list((target_dir / "images" / "train").glob("*")))
    val_count = len(list((target_dir / "images" / "val").glob("*")))
    test_count = len(list((test_dir / "images").glob("*")))

    print(f"\nDataset organized successfully!")
    print(f"  Training images: {train_count}")
    print(f"  Validation images: {val_count}")
    print(f"  Test images: {test_count} (in {test_dir})")
    print(f"  Classes: {DISEASE_CLASSES}")


def _copy_split(src_split_dir: Path, target_dir: Path, split_name: str) -> None:
    """Copy images and labels for a single split."""
    src_images = src_split_dir / "images"
    src_labels = src_split_dir / "labels"

    if src_images.exists():
        for img in src_images.glob("*"):
            shutil.copy2(img, target_dir / "images" / split_name / img.name)

    if src_labels.exists():
        for lbl in src_labels.glob("*"):
            shutil.copy2(lbl, target_dir / "labels" / split_name / lbl.name)


def download_and_organize(
    api_key: str | None = None,
    workspace: str = "mina-orfdd",
    project: str = "mina-u7bag",
    version: int = 2,
    target_dir: Path | None = None,
    test_dir: Path | None = None,
) -> None:
    """
    Download and organize the fish disease dataset.

    Args:
        api_key: Roboflow API key. If None, reads from ROBOFLOW_API_KEY env var.
        workspace: Roboflow workspace name
        project: Roboflow project name
        version: Dataset version number
        target_dir: Target directory for train/val data
        test_dir: Target directory for test data
    """
    if api_key is None:
        api_key = os.getenv("ROBOFLOW_API_KEY")
        if not api_key:
            raise RuntimeError(
                "ROBOFLOW_API_KEY is not set.\n"
                "Set it via environment variable or pass api_key parameter."
            )

    if target_dir is None:
        target_dir = DATA_DIR
    if test_dir is None:
        test_dir = TEST_DATA_DIR

    print("Downloading dataset from Roboflow...")
    dataset = download_dataset(
        api_key=api_key,
        workspace=workspace,
        project_name=project,
        version_number=version,
    )

    source_dir = Path(dataset.location)
    print(f"Downloaded to: {source_dir}")

    print(f"\nOrganizing dataset to: {target_dir}")
    print(f"Test data will be saved to: {test_dir}")
    organize_dataset(source_dir, target_dir, test_dir)

    # Clean up original download
    shutil.rmtree(source_dir)
    print(f"\nCleaned up temporary download directory")
