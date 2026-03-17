"""
data/datasets/fundus_dataset.py
================================
Retina-GPT: Base Fundus Dataset Class

Provides a production-grade PyTorch Dataset base class for retinal fundus imaging datasets.
Handles multi-task labels (classification, segmentation masks, lesion bounding boxes),
integrates with the preprocessing pipeline, and supports augmentation.

Subclass this for specific datasets (EyePACS, DRIVE, etc.).

Author: Retina-GPT Engineering Team
Date: 2026
"""

from __future__ import annotations

import json
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import albumentations as A
import cv2
import numpy as np
import pandas as pd
import torch
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from utils.preprocessing import PreprocessingConfig, RetinaPreprocessor

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────
# Label Definitions
# ─────────────────────────────────────────────────────────────

# Diabetic Retinopathy grading scale (International Clinical DR Scale)
DR_GRADE_LABELS = {
    0: "No DR",
    1: "Mild NPDR",
    2: "Moderate NPDR",
    3: "Severe NPDR",
    4: "Proliferative DR",
}

# Image quality labels
QUALITY_LABELS = {
    0: "Ungradable",
    1: "Gradable",
}


# ─────────────────────────────────────────────────────────────
# Sample Dataclass
# ─────────────────────────────────────────────────────────────

@dataclass
class RetinaySample:
    """
    A single multi-task retinal sample.

    Attributes:
        image_id: Unique identifier for this image
        image_path: Path to raw fundus image
        image_tensor: Preprocessed tensor (C, H, W)
        dr_grade: Diabetic retinopathy grade [0-4], -1 if unavailable
        quality_label: Image quality label [0=ungradable, 1=gradable], -1 if unavailable
        vessel_mask: Binary vessel segmentation mask (H, W), None if unavailable
        lesion_boxes: List of [x1,y1,x2,y2,class_id] lesion bounding boxes, None if unavailable
        metadata: Additional metadata dict (e.g., patient ID, camera type, eye side)
    """
    image_id: str
    image_path: Path
    image_tensor: torch.Tensor
    dr_grade: int = -1
    quality_label: int = -1
    vessel_mask: Optional[torch.Tensor] = None
    lesion_boxes: Optional[List[List[float]]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


# ─────────────────────────────────────────────────────────────
# Augmentation Pipelines
# ─────────────────────────────────────────────────────────────

def get_train_augmentation(target_size: Tuple[int, int] = (512, 512)) -> A.Compose:
    """
    Training augmentation pipeline for fundus images.
    
    Augmentations are carefully chosen to be clinically realistic:
    - Flips: Fundus images can be mirrored (left/right eye symmetry)
    - Rotations: Camera positioning variation
    - Color jitter: Variation in camera settings / patient eye pigmentation
    - Brightness/contrast: Illumination variation
    
    EXCLUDED (clinically unrealistic for fundus):
    - Large crops that remove optic disc
    - Heavy distortions that destroy vessel topology
    """
    return A.Compose(
        [
            A.Resize(height=target_size[0], width=target_size[1]),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.2),
            A.Rotate(limit=15, p=0.5, border_mode=cv2.BORDER_CONSTANT, value=0),
            A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.05, p=0.5),
            A.GaussianBlur(blur_limit=(3, 5), p=0.2),
            A.GaussNoise(var_limit=(5.0, 25.0), p=0.2),
            A.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.15, p=0.4),
            A.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
            ),
            ToTensorV2(),
        ],
        # Support for paired image + mask augmentation
        additional_targets={"mask": "mask"},
    )


def get_val_augmentation(target_size: Tuple[int, int] = (512, 512)) -> A.Compose:
    """
    Validation/test augmentation (only resize + normalize; no random transforms).
    """
    return A.Compose(
        [
            A.Resize(height=target_size[0], width=target_size[1]),
            A.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
            ),
            ToTensorV2(),
        ],
        additional_targets={"mask": "mask"},
    )


# ─────────────────────────────────────────────────────────────
# Abstract Base Dataset
# ─────────────────────────────────────────────────────────────

class BaseFundusDataset(Dataset, ABC):
    """
    Abstract base class for all retinal fundus datasets in Retina-GPT.

    Provides:
    - Standardized interface for multi-task label loading
    - Integration with RetinaPreprocessor
    - Augmentation support
    - Utilities for class balancing

    Subclasses must implement:
    - _load_metadata(): Load image paths and labels into self._samples list
    """

    def __init__(
        self,
        root_dir: Union[str, Path],
        split: str = "train",
        config: Optional[PreprocessingConfig] = None,
        augmentation: Optional[A.Compose] = None,
        use_augmentation: bool = True,
        load_masks: bool = False,
        cache_preprocessed: bool = False,
    ):
        """
        Args:
            root_dir: Root directory of the dataset
            split: Data split — 'train', 'val', or 'test'
            config: Preprocessing configuration
            augmentation: Custom albumentations pipeline (overrides default)
            use_augmentation: Whether to apply training augmentation
            load_masks: Whether to load segmentation masks if available
            cache_preprocessed: Cache preprocessed tensors in memory (for small datasets)
        """
        super().__init__()

        self.root_dir = Path(root_dir)
        self.split = split
        self.load_masks = load_masks
        self.cache_preprocessed = cache_preprocessed
        self._cache: Dict[int, dict] = {}

        # Preprocessing pipeline
        self.config = config or PreprocessingConfig()
        self.preprocessor = RetinaPreprocessor(self.config)

        # Augmentation
        if augmentation is not None:
            self.transform = augmentation
        elif use_augmentation and split == "train":
            self.transform = get_train_augmentation(self.config.target_size)
        else:
            self.transform = get_val_augmentation(self.config.target_size)

        # Load dataset metadata (must be implemented by subclass)
        self._samples: List[Dict] = []
        self._load_metadata()

        logger.info(
            f"{self.__class__.__name__} | split={split} | "
            f"n_samples={len(self._samples)} | load_masks={load_masks}"
        )

    @abstractmethod
    def _load_metadata(self) -> None:
        """
        Load image paths and labels into self._samples.

        Each item in self._samples should be a dict with at minimum:
            {
                'image_id': str,
                'image_path': Path,
                'dr_grade': int,        # -1 if not available
                'quality_label': int,   # -1 if not available
                'mask_path': Path,      # None if not available
                'lesion_boxes': list,   # None if not available
            }
        """
        raise NotImplementedError

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Load, preprocess, and return a single sample.

        Returns dict with keys:
            - image (Tensor): preprocessed image (C, H, W)
            - dr_grade (int): DR classification label
            - quality_label (int): image quality label
            - vessel_mask (Tensor | None): binary segmentation mask (1, H, W)
            - lesion_boxes (Tensor | None): bounding boxes (N, 5) [x1,y1,x2,y2,cls]
            - image_id (str): unique sample identifier
        """
        if self.cache_preprocessed and idx in self._cache:
            return self._cache[idx]

        meta = self._samples[idx]
        image_path = Path(meta["image_path"])

        # ── Load image ──
        bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if bgr is None:
            raise RuntimeError(f"Cannot read image: {image_path}")
        image = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

        # ── Load optional segmentation mask ──
        mask = None
        if self.load_masks and meta.get("mask_path") is not None:
            mask_path = Path(meta["mask_path"])
            if mask_path.exists():
                mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
                mask = (mask > 127).astype(np.uint8)  # binarize

        # ── Apply augmentation / normalization ──
        if mask is not None:
            augmented = self.transform(image=image, mask=mask)
            image_tensor = augmented["image"]   # (C, H, W) float tensor
            mask_tensor = augmented["mask"].unsqueeze(0).float()  # (1, H, W)
        else:
            augmented = self.transform(image=image)
            image_tensor = augmented["image"]
            mask_tensor = None

        # ── Build output dict ──
        sample = {
            "image": image_tensor,
            "dr_grade": torch.tensor(meta.get("dr_grade", -1), dtype=torch.long),
            "quality_label": torch.tensor(meta.get("quality_label", -1), dtype=torch.long),
            "vessel_mask": mask_tensor,
            "lesion_boxes": meta.get("lesion_boxes", None),
            "image_id": meta["image_id"],
        }

        if self.cache_preprocessed:
            self._cache[idx] = sample

        return sample

    # ── Utility: class weights for imbalanced datasets ──

    def get_class_weights(self, label_key: str = "dr_grade") -> torch.Tensor:
        """
        Compute inverse-frequency class weights for weighted sampling.

        Args:
            label_key: Which label to compute weights for ('dr_grade' or 'quality_label')

        Returns:
            1D tensor of weights for each class
        """
        labels = [s.get(label_key, -1) for s in self._samples]
        labels = [l for l in labels if l >= 0]

        unique, counts = np.unique(labels, return_counts=True)
        weights = 1.0 / counts.astype(np.float32)
        weight_map = dict(zip(unique, weights))

        sample_weights = torch.tensor(
            [weight_map.get(l, 0.0) for l in labels], dtype=torch.float32
        )
        return sample_weights

    def get_weighted_sampler(self, label_key: str = "dr_grade") -> WeightedRandomSampler:
        """
        Create a WeightedRandomSampler for class-balanced training.

        Returns:
            PyTorch WeightedRandomSampler instance
        """
        sample_weights = self.get_class_weights(label_key)
        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True,
        )
        return sampler


# ─────────────────────────────────────────────────────────────
# EyePACS Dataset Implementation
# ─────────────────────────────────────────────────────────────

class EyePACSDataset(BaseFundusDataset):
    """
    EyePACS Diabetic Retinopathy dataset loader.
    
    Dataset structure expected:
        root_dir/
            train/          ← JPEG fundus images
            test/
            trainLabels.csv ← image,level columns
            testLabels.csv

    Reference: Kaggle Diabetic Retinopathy Detection 2015
    URL: https://www.kaggle.com/c/diabetic-retinopathy-detection
    """

    def _load_metadata(self) -> None:
        label_file_map = {
            "train": "trainLabels.csv",
            "val":   "trainLabels.csv",  # use train CSV with split
            "test":  "testLabels.csv",
        }

        csv_file = self.root_dir / label_file_map.get(self.split, "trainLabels.csv")
        image_dir = self.root_dir / ("train" if self.split != "test" else "test")

        if not csv_file.exists():
            logger.warning(f"Label CSV not found: {csv_file}. Using dummy data for testing.")
            self._load_dummy_samples()
            return

        df = pd.read_csv(csv_file)

        # Optional: split train into train/val (80/20)
        if self.split in ("train", "val"):
            split_idx = int(len(df) * 0.8)
            df = df.iloc[:split_idx] if self.split == "train" else df.iloc[split_idx:]

        for _, row in df.iterrows():
            image_name = str(row["image"])
            # EyePACS images can be .jpeg or .jpg
            for ext in [".jpeg", ".jpg", ".png"]:
                image_path = image_dir / f"{image_name}{ext}"
                if image_path.exists():
                    break

            self._samples.append({
                "image_id": image_name,
                "image_path": str(image_path),
                "dr_grade": int(row.get("level", -1)),
                "quality_label": -1,  # EyePACS doesn't provide quality labels
                "mask_path": None,
                "lesion_boxes": None,
            })

    def _load_dummy_samples(self) -> None:
        """Load synthetic sample stubs for unit testing without real data."""
        for i in range(10):
            self._samples.append({
                "image_id": f"dummy_{i:04d}",
                "image_path": "/dev/null",  # won't exist, handled in __getitem__
                "dr_grade": i % 5,
                "quality_label": 1,
                "mask_path": None,
                "lesion_boxes": None,
            })


# ─────────────────────────────────────────────────────────────
# DataLoader Factory
# ─────────────────────────────────────────────────────────────

def create_dataloader(
    dataset: BaseFundusDataset,
    batch_size: int = 16,
    num_workers: int = 4,
    use_weighted_sampler: bool = False,
    pin_memory: bool = True,
    prefetch_factor: int = 2,
) -> DataLoader:
    """
    Factory function to create optimized DataLoaders for fundus datasets.

    Args:
        dataset: A BaseFundusDataset instance
        batch_size: Number of samples per batch
        num_workers: Parallel data loading processes
        use_weighted_sampler: Balance classes via WeightedRandomSampler
        pin_memory: Enable pinned memory for faster GPU transfer
        prefetch_factor: Batches pre-loaded per worker

    Returns:
        Configured PyTorch DataLoader
    """
    sampler = None
    shuffle = dataset.split == "train"

    if use_weighted_sampler and dataset.split == "train":
        sampler = dataset.get_weighted_sampler()
        shuffle = False  # sampler and shuffle are mutually exclusive

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
        persistent_workers=num_workers > 0,
        drop_last=dataset.split == "train",  # avoid incomplete batch issues
    )
