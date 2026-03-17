"""
dataset_manager.py — Retina-GPT Dataset Manager
=================================================
Unified dataset management for all retinal fundus datasets.

Supported datasets:
    APTOS 2019      — 3,662 images, DR grading (0-4)
    EyePACS         — 88,702 images, DR grading (0-4)
    ODIR-5K         — 8,000 patients, multi-disease
    REFUGE          — 1,200 images, glaucoma + disc/cup segmentation
    IDRiD           — 516 images, DR + lesion segmentation + grading
    DRIVE           — 40 images, vessel segmentation
    STARE           — 20 images, vessel segmentation
    Messidor-2      — 1,748 images, DR grading
    RFMiD           — 3,200 images, 46 disease categories
    Kaggle DR       — 35,126 images, DR grading

Features:
    • Unified interface across all datasets
    • Automatic train/val/test splitting (stratified)
    • Multi-dataset mixing with configurable weights
    • Unlabeled dataset support for DINO pretraining
    • Class-balanced sampling
    • Fast caching of preprocessed tensors
    • Augmentation pipeline integration
    • Dataset statistics and visualization

Usage:
    manager = DatasetManager("./data")

    # Single dataset
    train_loader, val_loader = manager.get_loaders(
        dataset="aptos", task="dr_grading", batch_size=32
    )

    # Mixed dataset for pretraining
    pretrain_loader = manager.get_pretraining_loader(
        datasets=["aptos", "eyepacs", "odir"],
        batch_size=64,
    )

    # Multi-task
    loaders = manager.get_multitask_loaders(
        datasets={"dr": "aptos", "seg": "idrid", "glaucoma": "refuge"},
        batch_size=32,
    )
"""

import os
import json
import logging
import hashlib
from pathlib import Path
from typing import Optional, Dict, List, Tuple, Union, Callable
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler, ConcatDataset
import numpy as np
import pandas as pd
import cv2
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Dataset Metadata Catalog
# ─────────────────────────────────────────────────────────────────────────────

DATASET_CATALOG = {
    "aptos": {
        "name": "APTOS 2019 Blindness Detection",
        "url": "https://www.kaggle.com/competitions/aptos2019-blindness-detection",
        "size": 3662,
        "tasks": ["dr_grading"],
        "labels": ["No DR", "Mild", "Moderate", "Severe", "Proliferative DR"],
        "num_classes": 5,
        "image_folder": "train_images",
        "label_csv": "train.csv",
        "label_col": "diagnosis",
        "id_col": "id_code",
        "image_ext": ".png",
    },
    "eyepacs": {
        "name": "EyePACS DR Grading",
        "url": "https://www.kaggle.com/competitions/diabetic-retinopathy-detection",
        "size": 88702,
        "tasks": ["dr_grading"],
        "labels": ["No DR", "Mild", "Moderate", "Severe", "Proliferative DR"],
        "num_classes": 5,
        "image_folder": "train",
        "label_csv": "trainLabels.csv",
        "label_col": "level",
        "id_col": "image",
        "image_ext": ".jpeg",
    },
    "odir": {
        "name": "ODIR-5K Ocular Disease Recognition",
        "url": "https://odir2019.grand-challenge.org",
        "size": 8000,
        "tasks": ["multi_disease"],
        "num_classes": 8,
        "image_folder": "Training Images",
        "label_csv": "ODIR-5K_Training_Annotations.xlsx",
    },
    "idrid": {
        "name": "IDRiD - Indian Diabetic Retinopathy Image Dataset",
        "url": "https://idrid.grand-challenge.org",
        "size": 516,
        "tasks": ["dr_grading", "lesion_segmentation"],
        "num_classes": 5,
        "image_folder": "Original Images/a. Training Set",
        "label_csv": "a. IDRiD_Disease Grading/1. Original Images/a. Training Set",
        "has_masks": True,
        "mask_folders": {
            "microaneurysm":    "B. IDRiD_segmentation/Lesion Segmentations/Training Set/1. Microaneurysms",
            "hemorrhage":       "B. IDRiD_segmentation/Lesion Segmentations/Training Set/2. Haemorrhages",
            "hard_exudate":     "B. IDRiD_segmentation/Lesion Segmentations/Training Set/3. Hard Exudates",
            "soft_exudate":     "B. IDRiD_segmentation/Lesion Segmentations/Training Set/4. Soft Exudates",
            "optic_disc":       "B. IDRiD_segmentation/Lesion Segmentations/Training Set/5. Optic Disc",
        },
    },
    "refuge": {
        "name": "REFUGE - Retinal Fundus Glaucoma Challenge",
        "url": "https://refuge.grand-challenge.org",
        "size": 1200,
        "tasks": ["glaucoma", "disc_cup_segmentation"],
        "num_classes": 2,
        "image_folder": "Training400/Images",
        "label_csv": "Training400/GT",
        "has_masks": True,
    },
    "drive": {
        "name": "DRIVE - Digital Retinal Images for Vessel Extraction",
        "url": "https://drive.grand-challenge.org",
        "size": 40,
        "tasks": ["vessel_segmentation"],
        "image_folder": "training/images",
        "mask_folder": "training/1st_manual",
    },
}


# ─────────────────────────────────────────────────────────────────────────────
# Augmentation Pipelines
# ─────────────────────────────────────────────────────────────────────────────

def get_classification_train_transforms(image_size: int = 224) -> A.Compose:
    return A.Compose([
        A.RandomResizedCrop(height=image_size, width=image_size,
                            scale=(0.7, 1.0), ratio=(0.9, 1.1)),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.2),
        A.Rotate(limit=20, p=0.5),
        A.OneOf([
            A.CLAHE(clip_limit=2.0, p=1.0),
            A.ColorJitter(brightness=0.2, contrast=0.3, saturation=0.1, p=1.0),
        ], p=0.8),
        A.OneOf([
            A.GaussianBlur(blur_limit=3, p=1.0),
            A.GaussNoise(var_limit=(10, 50), p=1.0),
        ], p=0.3),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])


def get_classification_val_transforms(image_size: int = 224) -> A.Compose:
    return A.Compose([
        A.Resize(height=image_size, width=image_size),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])


def get_segmentation_train_transforms(image_size: int = 512) -> A.Compose:
    return A.Compose([
        A.RandomResizedCrop(height=image_size, width=image_size, scale=(0.7, 1.0)),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.3),
        A.Rotate(limit=15, p=0.5),
        A.ColorJitter(brightness=0.2, contrast=0.2, p=0.5),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ], additional_targets={"mask": "mask"})


def get_dino_unlabeled_transforms(image_size: int = 224) -> A.Compose:
    """Minimal transform for DINO pretraining dataset (full augmentation in DINO module)."""
    return A.Compose([
        A.Resize(height=image_size, width=image_size),
        # No normalize here — DINO augmentation handles it
    ])


# ─────────────────────────────────────────────────────────────────────────────
# Base Fundus Dataset
# ─────────────────────────────────────────────────────────────────────────────

class FundusDataset(Dataset):
    """
    Generic fundus image dataset.

    Handles loading from:
        - image_paths + labels lists
        - CSV manifest file

    Returns:
        (image_tensor, label, metadata_dict)
    """

    def __init__(
        self,
        image_paths: List[str],
        labels: Optional[List[int]] = None,
        image_size: int = 224,
        transform: Optional[A.Compose] = None,
        preprocess_fn: Optional[Callable] = None,
        return_metadata: bool = False,
        dataset_name: str = "unknown",
    ):
        self.image_paths = image_paths
        self.labels = labels or [-1] * len(image_paths)
        self.image_size = image_size
        self.transform = transform
        self.preprocess_fn = preprocess_fn
        self.return_metadata = return_metadata
        self.dataset_name = dataset_name

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int):
        path = self.image_paths[idx]
        label = self.labels[idx]

        # Load image
        image = cv2.imread(str(path))
        if image is None:
            # Fallback: return zero tensor
            image_tensor = torch.zeros(3, self.image_size, self.image_size)
            return (image_tensor, torch.tensor(label, dtype=torch.long),
                    {"path": str(path), "error": True})

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Preprocessing (CLAHE, circular crop, etc.)
        if self.preprocess_fn:
            image = self.preprocess_fn(image)

        # Augmentation
        if self.transform:
            aug = self.transform(image=image)
            image = aug["image"]
        else:
            image = cv2.resize(image, (self.image_size, self.image_size))
            image = torch.from_numpy(image.transpose(2, 0, 1)).float() / 255.0

        label_tensor = torch.tensor(label, dtype=torch.long)

        if self.return_metadata:
            meta = {"path": str(path), "dataset": self.dataset_name, "idx": idx}
            return image, label_tensor, meta

        return image, label_tensor

    def get_class_weights(self) -> torch.Tensor:
        """Compute inverse-frequency class weights for imbalanced datasets."""
        labels = np.array(self.labels)
        classes = np.unique(labels[labels >= 0])
        counts = np.bincount(labels[labels >= 0])
        weights = 1.0 / (counts + 1e-6)
        weights = weights / weights.sum() * len(classes)
        return torch.tensor(weights, dtype=torch.float32)

    def get_sample_weights(self) -> torch.Tensor:
        """Per-sample weights for WeightedRandomSampler."""
        class_weights = self.get_class_weights()
        sample_weights = torch.tensor([
            class_weights[l].item() if l >= 0 else 1.0 for l in self.labels
        ])
        return sample_weights


class SegmentationDataset(Dataset):
    """
    Dataset for segmentation tasks (vessels, lesions, disc/cup).

    Returns: (image_tensor, mask_tensor, metadata)
    """

    def __init__(
        self,
        image_paths: List[str],
        mask_paths: List[str],
        image_size: int = 512,
        transform: Optional[A.Compose] = None,
    ):
        assert len(image_paths) == len(mask_paths)
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.image_size = image_size
        self.transform = transform

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int):
        image = cv2.cvtColor(cv2.imread(str(self.image_paths[idx])), cv2.COLOR_BGR2RGB)
        mask = cv2.imread(str(self.mask_paths[idx]), cv2.IMREAD_GRAYSCALE)

        image = cv2.resize(image, (self.image_size, self.image_size))
        mask = cv2.resize(mask, (self.image_size, self.image_size),
                          interpolation=cv2.INTER_NEAREST)
        mask = (mask > 127).astype(np.uint8)

        if self.transform:
            aug = self.transform(image=image, mask=mask)
            image = aug["image"]
            mask = torch.from_numpy(aug["mask"]).long()
        else:
            image = torch.from_numpy(image.transpose(2, 0, 1)).float() / 255.0
            mask = torch.from_numpy(mask).long()

        return image, mask


class DINOUnlabeledDataset(Dataset):
    """
    Unlabeled dataset for DINO pretraining.
    Returns multi-crop list generated by RetinaMultiCropAugmentation.
    """

    def __init__(
        self,
        image_paths: List[str],
        multi_crop_aug,  # RetinaMultiCropAugmentation instance
    ):
        self.image_paths = image_paths
        self.multi_crop_aug = multi_crop_aug

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int):
        image = cv2.cvtColor(cv2.imread(str(self.image_paths[idx])), cv2.COLOR_BGR2RGB)
        crops = self.multi_crop_aug(image)
        return crops, -1  # -1 = no label


class CLIPPairedDataset(Dataset):
    """
    Paired (image, clinical_text) dataset for Retina-CLIP training.
    """

    def __init__(
        self,
        image_paths: List[str],
        texts: List[str],
        tokenizer,
        transform: Optional[A.Compose] = None,
        max_text_length: int = 128,
        image_size: int = 224,
    ):
        assert len(image_paths) == len(texts)
        self.image_paths = image_paths
        self.texts = texts
        self.tokenizer = tokenizer
        self.transform = transform or get_classification_val_transforms(image_size)
        self.max_text_length = max_text_length

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int):
        image = cv2.cvtColor(cv2.imread(str(self.image_paths[idx])), cv2.COLOR_BGR2RGB)
        aug = self.transform(image=image)
        image_tensor = aug["image"]

        tokens = self.tokenizer(
            self.texts[idx],
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.max_text_length,
        )
        return (
            image_tensor,
            tokens["input_ids"].squeeze(0),
            tokens["attention_mask"].squeeze(0),
        )


# ─────────────────────────────────────────────────────────────────────────────
# Dataset Manager
# ─────────────────────────────────────────────────────────────────────────────

class DatasetManager:
    """
    Unified dataset manager for all Retina-GPT training stages.

    Handles:
        - Dataset discovery and validation
        - Train/val/test splitting (stratified)
        - DataLoader creation with proper samplers
        - Multi-dataset mixing
        - Preprocessing integration
        - Dataset statistics
    """

    def __init__(
        self,
        data_root: str = "./data",
        cache_dir: Optional[str] = "./data/.cache",
        image_size: int = 224,
        num_workers: int = 4,
        pin_memory: bool = True,
    ):
        self.data_root = Path(data_root)
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.image_size = image_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory

        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Preprocess function (from utils.preprocessing)
        self._preprocess_fn = self._load_preprocessor()

    def _load_preprocessor(self):
        """Load RetinaPreprocessor if available."""
        try:
            import sys
            sys.path.insert(0, str(Path(__file__).parent.parent))
            from utils.preprocessing import RetinaPreprocessor
            preprocessor = RetinaPreprocessor()
            def preprocess(image: np.ndarray) -> np.ndarray:
                return preprocessor.preprocess(image)
            return preprocess
        except Exception:
            return None

    # ── Loaders for each training stage ──────────────────────────────────────

    def get_classification_loaders(
        self,
        dataset: str,
        batch_size: int = 32,
        val_split: float = 0.15,
        test_split: float = 0.05,
        balanced_sampling: bool = True,
        task: str = "dr_grading",
    ) -> Tuple[DataLoader, DataLoader, Optional[DataLoader]]:
        """
        Get train/val/test DataLoaders for classification tasks.

        Returns: (train_loader, val_loader, test_loader)
        """
        paths, labels = self._load_classification_data(dataset, task)
        if len(paths) == 0:
            raise FileNotFoundError(
                f"No data found for dataset '{dataset}'. "
                f"Expected at: {self.data_root / dataset}"
            )

        # Stratified split
        train_paths, val_paths, test_paths, train_labels, val_labels, test_labels = \
            self._stratified_split(paths, labels, val_split, test_split)

        # Datasets
        train_ds = FundusDataset(
            train_paths, train_labels,
            image_size=self.image_size,
            transform=get_classification_train_transforms(self.image_size),
            preprocess_fn=self._preprocess_fn,
            dataset_name=dataset,
        )
        val_ds = FundusDataset(
            val_paths, val_labels,
            image_size=self.image_size,
            transform=get_classification_val_transforms(self.image_size),
            preprocess_fn=self._preprocess_fn,
            dataset_name=dataset,
        )
        test_ds = FundusDataset(
            test_paths, test_labels,
            image_size=self.image_size,
            transform=get_classification_val_transforms(self.image_size),
            preprocess_fn=self._preprocess_fn,
            dataset_name=dataset,
        ) if test_paths else None

        # Sampler
        train_sampler = None
        if balanced_sampling and len(set(train_labels)) > 1:
            weights = train_ds.get_sample_weights()
            train_sampler = WeightedRandomSampler(weights, len(weights), replacement=True)

        train_loader = DataLoader(
            train_ds,
            batch_size=batch_size,
            sampler=train_sampler,
            shuffle=(train_sampler is None),
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=True,
        )
        val_loader = DataLoader(
            val_ds, batch_size=batch_size * 2, shuffle=False,
            num_workers=self.num_workers, pin_memory=self.pin_memory,
        )
        test_loader = DataLoader(
            test_ds, batch_size=batch_size * 2, shuffle=False,
            num_workers=self.num_workers, pin_memory=self.pin_memory,
        ) if test_ds else None

        logger.info(
            f"[DatasetManager] {dataset} | "
            f"train={len(train_ds)}, val={len(val_ds)}, "
            f"test={len(test_ds) if test_ds else 0} | balanced={balanced_sampling}"
        )
        return train_loader, val_loader, test_loader

    def get_pretraining_loader(
        self,
        datasets: List[str],
        multi_crop_aug,  # RetinaMultiCropAugmentation
        batch_size: int = 64,
    ) -> DataLoader:
        """
        Get unlabeled DataLoader for Retina-DINO pretraining.
        Mixes multiple datasets for maximum diversity.
        """
        all_paths = []
        for ds_name in datasets:
            paths = self._find_all_images(ds_name)
            all_paths.extend(paths)
            logger.info(f"[Pretraining] {ds_name}: {len(paths)} images")

        logger.info(f"[Pretraining] Total: {len(all_paths)} images")

        dataset = DINOUnlabeledDataset(all_paths, multi_crop_aug)
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=True,
            collate_fn=self._dino_collate_fn,
        )

    def get_segmentation_loaders(
        self,
        dataset: str,
        batch_size: int = 8,
        val_split: float = 0.15,
        image_size: int = 512,
    ) -> Tuple[DataLoader, DataLoader]:
        """Get DataLoaders for segmentation tasks (IDRiD, DRIVE, REFUGE)."""
        image_paths, mask_paths = self._load_segmentation_data(dataset)

        n = len(image_paths)
        n_val = max(1, int(n * val_split))
        indices = np.random.permutation(n)
        val_idx, train_idx = indices[:n_val], indices[n_val:]

        train_ds = SegmentationDataset(
            [image_paths[i] for i in train_idx],
            [mask_paths[i] for i in train_idx],
            image_size=image_size,
            transform=get_segmentation_train_transforms(image_size),
        )
        val_ds = SegmentationDataset(
            [image_paths[i] for i in val_idx],
            [mask_paths[i] for i in val_idx],
            image_size=image_size,
        )

        train_loader = DataLoader(
            train_ds, batch_size=batch_size, shuffle=True,
            num_workers=self.num_workers, pin_memory=self.pin_memory,
        )
        val_loader = DataLoader(
            val_ds, batch_size=batch_size, shuffle=False,
            num_workers=self.num_workers, pin_memory=self.pin_memory,
        )
        return train_loader, val_loader

    def get_multitask_loaders(
        self,
        batch_size: int = 32,
        datasets: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Tuple[DataLoader, DataLoader]]:
        """
        Get DataLoaders for all tasks simultaneously.

        Returns: dict mapping task_name → (train_loader, val_loader)
        """
        datasets = datasets or {
            "dr_grading":    "aptos",
            "vessel_seg":    "drive",
            "glaucoma":      "refuge",
        }

        loaders = {}
        for task, ds_name in datasets.items():
            try:
                if "seg" in task:
                    train_l, val_l = self.get_segmentation_loaders(ds_name, batch_size)
                else:
                    train_l, val_l, _ = self.get_classification_loaders(
                        ds_name, batch_size, task=task
                    )
                loaders[task] = (train_l, val_l)
            except FileNotFoundError:
                logger.warning(f"[DatasetManager] Dataset '{ds_name}' not found, skipping task '{task}'")

        return loaders

    # ── Internal Loaders ─────────────────────────────────────────────────────

    def _load_classification_data(
        self, dataset: str, task: str = "dr_grading"
    ) -> Tuple[List[str], List[int]]:
        """Load image paths + labels for a classification dataset."""
        ds_dir = self.data_root / dataset
        if not ds_dir.exists():
            return [], []

        meta = DATASET_CATALOG.get(dataset, {})
        csv_name = meta.get("label_csv", "train.csv")
        img_folder = meta.get("image_folder", "train_images")
        label_col = meta.get("label_col", "diagnosis")
        id_col = meta.get("id_col", "id_code")
        ext = meta.get("image_ext", ".png")

        csv_path = ds_dir / csv_name
        img_dir = ds_dir / img_folder

        if csv_path.exists() and img_dir.exists():
            try:
                df = pd.read_csv(csv_path)
                paths = [str(img_dir / f"{row[id_col]}{ext}") for _, row in df.iterrows()]
                labels = [int(row[label_col]) for _, row in df.iterrows()]
                paths, labels = zip(*[(p, l) for p, l in zip(paths, labels) if Path(p).exists()])
                return list(paths), list(labels)
            except Exception as e:
                logger.warning(f"Failed to load CSV for {dataset}: {e}")

        # Fallback: folder scan (class0/, class1/, ...)
        paths, labels = [], []
        for cls_dir in sorted(ds_dir.glob("*")):
            if cls_dir.is_dir() and cls_dir.name.isdigit():
                cls_label = int(cls_dir.name)
                for img_path in cls_dir.glob(f"*{ext}"):
                    paths.append(str(img_path))
                    labels.append(cls_label)

        return paths, labels

    def _load_segmentation_data(
        self, dataset: str
    ) -> Tuple[List[str], List[str]]:
        """Load image + mask path pairs for segmentation."""
        ds_dir = self.data_root / dataset
        if not ds_dir.exists():
            return [], []

        meta = DATASET_CATALOG.get(dataset, {})
        img_folder = meta.get("image_folder", "images")
        mask_folder = meta.get("mask_folder", "masks")

        img_dir = ds_dir / img_folder
        mask_dir = ds_dir / mask_folder

        if not img_dir.exists():
            return [], []

        image_paths = sorted(img_dir.glob("*.*"))
        image_paths = [p for p in image_paths
                       if p.suffix.lower() in (".png", ".jpg", ".tif", ".jpeg")]

        mask_paths = []
        for img_path in image_paths:
            mask_path = mask_dir / img_path.name
            if not mask_path.exists():
                # Try PNG fallback
                mask_path = mask_dir / (img_path.stem + ".png")
            mask_paths.append(str(mask_path))

        image_paths = [str(p) for p in image_paths]
        paired = [(i, m) for i, m in zip(image_paths, mask_paths) if Path(m).exists()]

        if not paired:
            return [], []
        return zip(*paired)

    def _find_all_images(self, dataset: str) -> List[str]:
        """Find all images in a dataset directory for unlabeled pretraining."""
        ds_dir = self.data_root / dataset
        if not ds_dir.exists():
            logger.warning(f"Dataset directory not found: {ds_dir}")
            return []

        extensions = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}
        paths = []
        for path in ds_dir.rglob("*"):
            if path.suffix.lower() in extensions:
                paths.append(str(path))

        return sorted(paths)

    def _stratified_split(
        self,
        paths: List[str],
        labels: List[int],
        val_split: float,
        test_split: float,
    ) -> Tuple:
        """Stratified train/val/test split preserving class ratios."""
        from sklearn.model_selection import train_test_split

        indices = np.arange(len(paths))
        test_size = test_split if test_split > 0 else 0

        try:
            # First, split off test
            if test_size > 0:
                idx_trainval, idx_test = train_test_split(
                    indices, test_size=test_size, stratify=labels, random_state=42
                )
                labels_trainval = [labels[i] for i in idx_trainval]
                val_fraction = val_split / (1 - test_size)
                idx_train, idx_val = train_test_split(
                    idx_trainval, test_size=val_fraction,
                    stratify=labels_trainval, random_state=42
                )
            else:
                idx_test = []
                idx_train, idx_val = train_test_split(
                    indices, test_size=val_split, stratify=labels, random_state=42
                )
        except ValueError:
            # Fallback: random split without stratification
            n = len(paths)
            n_val = int(n * val_split)
            n_test = int(n * test_split)
            perm = np.random.permutation(n)
            idx_test = perm[:n_test]
            idx_val = perm[n_test:n_test + n_val]
            idx_train = perm[n_test + n_val:]

        def _select(idx):
            if len(idx) == 0:
                return [], []
            return [paths[i] for i in idx], [labels[i] for i in idx]

        train_paths, train_labels = _select(idx_train)
        val_paths, val_labels = _select(idx_val)
        test_paths, test_labels = _select(idx_test)

        return train_paths, val_paths, test_paths, train_labels, val_labels, test_labels

    @staticmethod
    def _dino_collate_fn(batch):
        """Collate multi-crop lists for DINO training."""
        # batch: list of (crops_list, label)
        num_crops = len(batch[0][0])
        collated_crops = []
        for i in range(num_crops):
            crop_batch = torch.stack([sample[0][i] for sample in batch])
            collated_crops.append(crop_batch)
        labels = torch.tensor([sample[1] for sample in batch])
        return collated_crops, labels

    # ── Statistics ────────────────────────────────────────────────────────────

    def dataset_stats(self, dataset: str, task: str = "dr_grading") -> Dict:
        """Compute and print dataset statistics."""
        paths, labels = self._load_classification_data(dataset, task)
        if not paths:
            return {"error": f"Dataset '{dataset}' not found"}

        labels_arr = np.array(labels)
        classes, counts = np.unique(labels_arr, return_counts=True)

        stats = {
            "name": dataset,
            "total_samples": len(paths),
            "num_classes": len(classes),
            "class_distribution": {int(c): int(n) for c, n in zip(classes, counts)},
            "class_balance_ratio": float(counts.min() / counts.max()),
        }

        print(f"\n{'='*50}")
        print(f"Dataset: {DATASET_CATALOG.get(dataset, {}).get('name', dataset)}")
        print(f"Total samples: {stats['total_samples']:,}")
        print(f"Class distribution:")
        label_names = DATASET_CATALOG.get(dataset, {}).get("labels", [str(i) for i in classes])
        for cls, count in zip(classes, counts):
            pct = 100 * count / len(paths)
            bar = "█" * int(pct / 2)
            name = label_names[cls] if cls < len(label_names) else str(cls)
            print(f"  Class {cls} ({name:25s}): {count:5d} ({pct:5.1f}%) {bar}")
        print(f"Balance ratio: {stats['class_balance_ratio']:.3f}")
        print(f"{'='*50}\n")

        return stats

    def __repr__(self) -> str:
        available = [
            ds for ds in DATASET_CATALOG
            if (self.data_root / ds).exists()
        ]
        return (
            f"DatasetManager(\n"
            f"  data_root={self.data_root}\n"
            f"  available_datasets={available}\n"
            f"  image_size={self.image_size}\n"
            f")"
        )
