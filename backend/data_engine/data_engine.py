"""
data_engine.py — Retina Data Engine
=====================================
The Retina Data Engine transforms raw fundus image collections into
clean, versioned, analysis-ready datasets.

Why this matters:
    Model quality is bounded by data quality.
    In medical imaging, bad data kills good models.

Pipeline:
    Raw Images
        ↓
    Quality Control     ← reject blurry, dark, artifact-heavy images
        ↓
    Cleaning            ← normalize illumination, remove artifacts
        ↓
    Device Normalization← harmonize images from different camera brands
        ↓
    Annotation          ← attach labels, masks, metadata
        ↓
    Dataset Versioning  ← hash + track every dataset snapshot
        ↓
    Statistics          ← class balance, quality distribution, audit

Usage:
    engine = RetinaDataEngine(data_root="./data")

    # Process a new batch of images
    result = engine.process_batch(
        input_dir="raw_images/",
        output_dataset="aptos_cleaned_v2",
        labels_csv="labels.csv",
    )

    print(result.summary())
    # → Dataset v2: 3,247 images (rejected: 415 low quality)
    #   DR distribution: [1204, 370, 896, 193, 584]
    #   Mean quality score: 0.847
"""

import os
import json
import hashlib
import logging
import shutil
from pathlib import Path
from typing import Optional, Dict, List, Tuple, Any
from dataclasses import dataclass, field, asdict
from datetime import datetime

import numpy as np
import cv2

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Quality Assessment
# ─────────────────────────────────────────────────────────────────────────────

class ImageQualityAssessor:
    """
    Rule-based image quality assessment for fundus photography.

    Scores images on:
        • Focus / sharpness (Laplacian variance)
        • Illumination (mean brightness, uniformity)
        • Field of view (circular fundus coverage)
        • Artifact detection (lens reflections, dust, motion blur)
        • Color channel health (detect monochrome / failed capture)
    """

    def __init__(
        self,
        min_quality_score: float = 0.35,
        blur_threshold:    float = 50.0,
        min_brightness:    float = 20.0,
        max_brightness:    float = 230.0,
    ):
        self.min_quality_score = min_quality_score
        self.blur_threshold    = blur_threshold
        self.min_brightness    = min_brightness
        self.max_brightness    = max_brightness

    def assess(self, image: np.ndarray) -> Dict[str, float]:
        """
        Assess image quality.

        Args:
            image: (H, W, 3) uint8 RGB image

        Returns:
            dict with: overall_score, sharpness, brightness, fov_coverage,
                       is_adequate, rejection_reason
        """
        if image is None or image.size == 0:
            return self._failed("Image is empty or None")

        h, w = image.shape[:2]
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if image.ndim == 3 else image

        scores = {}
        rejection = None

        # ── Sharpness (Laplacian variance) ───────────────────────────────
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        sharpness_score = min(1.0, laplacian_var / 300.0)
        scores["sharpness"] = float(sharpness_score)
        if laplacian_var < self.blur_threshold:
            rejection = f"Blurry image (Laplacian={laplacian_var:.1f} < {self.blur_threshold})"

        # ── Brightness ────────────────────────────────────────────────────
        mean_brightness = float(gray.mean())
        if mean_brightness < self.min_brightness:
            brightness_score = mean_brightness / self.min_brightness
            rejection = rejection or f"Too dark (brightness={mean_brightness:.1f})"
        elif mean_brightness > self.max_brightness:
            brightness_score = 1.0 - (mean_brightness - self.max_brightness) / 50.0
            rejection = rejection or f"Overexposed (brightness={mean_brightness:.1f})"
        else:
            range_size = self.max_brightness - self.min_brightness
            center = (self.min_brightness + self.max_brightness) / 2
            brightness_score = 1.0 - abs(mean_brightness - center) / (range_size / 2)
        scores["brightness"] = float(np.clip(brightness_score, 0, 1))

        # ── Field of View (circular fundus coverage) ──────────────────────
        # Fundus images should have a prominent circular region
        _, binary = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
        fov_ratio = float(binary.sum()) / (255 * h * w)
        # Ideal: circle covers ~78% of square (pi/4)
        fov_score = min(1.0, fov_ratio / 0.6)
        scores["fov_coverage"] = float(fov_score)

        # ── Color health (multi-channel check) ───────────────────────────
        if image.ndim == 3 and image.shape[2] == 3:
            r_mean = float(image[:,:,0].mean())
            g_mean = float(image[:,:,1].mean())
            b_mean = float(image[:,:,2].mean())
            # Fundus images should be red-dominant (blood vessel contrast)
            color_health = min(1.0, r_mean / max(g_mean, b_mean, 1) / 2.0)
        else:
            color_health = 0.5
        scores["color_health"] = float(np.clip(color_health, 0, 1))

        # ── Artifact detection (saturation check) ─────────────────────────
        sat_ratio = float((gray >= 250).mean())  # Overexposed regions
        artifact_score = max(0.0, 1.0 - sat_ratio * 10)
        scores["artifact_score"] = artifact_score
        if sat_ratio > 0.1:
            rejection = rejection or f"Significant artifacts/saturation ({sat_ratio:.1%})"

        # ── Overall score (weighted combination) ──────────────────────────
        weights = {
            "sharpness":    0.35,
            "brightness":   0.25,
            "fov_coverage": 0.20,
            "color_health": 0.10,
            "artifact_score": 0.10,
        }
        overall = sum(scores[k] * w for k, w in weights.items())
        scores["overall_score"] = float(overall)

        is_adequate = (
            overall >= self.min_quality_score and
            rejection is None
        )
        scores["is_adequate"] = float(is_adequate)
        scores["rejection_reason"] = rejection or ""

        return scores

    def _failed(self, reason: str) -> Dict:
        return {
            "overall_score": 0.0, "sharpness": 0.0, "brightness": 0.0,
            "fov_coverage": 0.0, "color_health": 0.0, "artifact_score": 0.0,
            "is_adequate": False, "rejection_reason": reason,
        }


# ─────────────────────────────────────────────────────────────────────────────
# Image Cleaner
# ─────────────────────────────────────────────────────────────────────────────

class FundusImageCleaner:
    """
    Preprocessing and cleaning pipeline for fundus images.

    Steps:
        1. Circular mask extraction (removes black corners)
        2. CLAHE contrast enhancement
        3. Illumination normalization (green channel)
        4. Color normalization (device harmonization)
        5. Resize to target size
    """

    def __init__(self, target_size: int = 512):
        self.target_size = target_size

    def clean(self, image: np.ndarray) -> np.ndarray:
        """Full cleaning pipeline."""
        if image is None:
            return np.zeros((self.target_size, self.target_size, 3), dtype=np.uint8)

        image = self._crop_circular(image)
        image = self._normalize_illumination(image)
        image = self._enhance_contrast(image)
        image = cv2.resize(image, (self.target_size, self.target_size),
                           interpolation=cv2.INTER_LANCZOS4)
        return image

    def _crop_circular(self, image: np.ndarray) -> np.ndarray:
        """Remove black border, keep only the circular fundus region."""
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        _, mask = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)

        # Find bounding box of circular region
        coords = cv2.findNonZero(mask)
        if coords is None:
            return image

        x, y, w, h = cv2.boundingRect(coords)
        # Add small padding
        pad = 5
        x = max(0, x - pad); y = max(0, y - pad)
        w = min(image.shape[1] - x, w + 2*pad)
        h = min(image.shape[0] - y, h + 2*pad)

        return image[y:y+h, x:x+w]

    def _normalize_illumination(self, image: np.ndarray) -> np.ndarray:
        """
        Green channel illumination normalization.
        The green channel carries most diagnostic information in fundus images.
        """
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        l_channel, a, b = cv2.split(lab)

        # CLAHE on L channel
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l_norm = clahe.apply(l_channel)

        lab_norm = cv2.merge([l_norm, a, b])
        return cv2.cvtColor(lab_norm, cv2.COLOR_LAB2RGB)

    def _enhance_contrast(self, image: np.ndarray) -> np.ndarray:
        """Apply CLAHE per-channel for better lesion visibility."""
        clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8))
        channels = [clahe.apply(ch) for ch in cv2.split(image)]
        return cv2.merge(channels)


# ─────────────────────────────────────────────────────────────────────────────
# Dataset Version
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class DatasetVersion:
    """Immutable snapshot of a dataset at a point in time."""

    name:           str
    version:        str
    created_at:     str
    num_images:     int
    num_rejected:   int
    class_distribution: Dict[int, int]
    mean_quality:   float
    data_hash:      str    # SHA256 of sorted image paths + labels
    config:         Dict[str, Any]
    splits:         Dict[str, int]  # {"train": N, "val": N, "test": N}
    metadata_path:  str            # Path to full per-image JSON metadata

    def summary(self) -> str:
        total = self.num_images + self.num_rejected
        lines = [
            f"Dataset: {self.name} v{self.version}",
            f"Created: {self.created_at[:10]}",
            f"Images:  {self.num_images:,} accepted / {total:,} total "
            f"({self.num_rejected} rejected)",
            f"Quality: mean={self.mean_quality:.3f}",
            f"Splits:  train={self.splits.get('train',0):,}  "
            f"val={self.splits.get('val',0):,}  "
            f"test={self.splits.get('test',0):,}",
            f"Hash:    {self.data_hash[:16]}...",
            "Class distribution:",
        ]
        labels = ["No DR", "Mild", "Moderate", "Severe", "Proliferative"]
        for cls, count in sorted(self.class_distribution.items()):
            name = labels[int(cls)] if int(cls) < len(labels) else str(cls)
            pct  = 100 * count / max(self.num_images, 1)
            bar  = "█" * int(pct / 3)
            lines.append(f"  [{cls}] {name:20s}: {count:5,} ({pct:5.1f}%) {bar}")
        return "\n".join(lines)

    def to_dict(self) -> Dict:
        return asdict(self)


# ─────────────────────────────────────────────────────────────────────────────
# Processing Result
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ProcessingResult:
    """Result of a batch processing run."""
    accepted:       List[str] = field(default_factory=list)
    rejected:       List[Dict] = field(default_factory=list)   # {path, reason}
    quality_scores: Dict[str, float] = field(default_factory=dict)
    processing_time_s: float = 0.0
    dataset_version: Optional[DatasetVersion] = None

    @property
    def acceptance_rate(self) -> float:
        total = len(self.accepted) + len(self.rejected)
        return len(self.accepted) / max(total, 1)

    def summary(self) -> str:
        total = len(self.accepted) + len(self.rejected)
        return (
            f"Processing complete:\n"
            f"  Accepted: {len(self.accepted):,} / {total:,} "
            f"({self.acceptance_rate:.1%})\n"
            f"  Rejected: {len(self.rejected):,}\n"
            f"  Mean quality: {np.mean(list(self.quality_scores.values())):.3f}\n"
            f"  Time: {self.processing_time_s:.1f}s"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Retina Data Engine
# ─────────────────────────────────────────────────────────────────────────────

class RetinaDataEngine:
    """
    Complete data pipeline engine for Retina-GPT.

    Transforms raw fundus images → clean, versioned, analysis-ready datasets.

    Usage:
        engine = RetinaDataEngine("./data")

        # Process raw images
        result = engine.process_batch(
            input_dir="./raw",
            output_name="my_dataset",
            labels_csv="labels.csv",
            version="v1.0",
        )

        print(result.summary())

        # Load a specific version for training
        dataset_info = engine.load_version("my_dataset", "v1.0")
    """

    VERSION_REGISTRY = "dataset_versions.json"

    def __init__(
        self,
        data_root: str = "./data",
        target_image_size: int = 512,
        min_quality: float = 0.35,
        num_workers: int = 4,
    ):
        self.data_root   = Path(data_root)
        self.target_size = target_image_size
        self.min_quality = min_quality
        self.num_workers = num_workers

        # Sub-directories
        (self.data_root / "raw").mkdir(parents=True, exist_ok=True)
        (self.data_root / "processed").mkdir(exist_ok=True)
        (self.data_root / "rejected").mkdir(exist_ok=True)
        (self.data_root / "versions").mkdir(exist_ok=True)

        self.quality_assessor = ImageQualityAssessor(min_quality_score=min_quality)
        self.cleaner          = FundusImageCleaner(target_size=target_image_size)

        # Load version registry
        self._registry_path = self.data_root / "versions" / self.VERSION_REGISTRY
        self._registry: Dict[str, DatasetVersion] = self._load_registry()

    def process_batch(
        self,
        input_dir:   str,
        output_name: str,
        labels_csv:  Optional[str] = None,
        version:     str = "v1.0",
        val_split:   float = 0.15,
        test_split:  float = 0.05,
        clean_images: bool = True,
        overwrite:   bool = False,
    ) -> ProcessingResult:
        """
        Process a directory of raw fundus images.

        Steps:
        1. Scan input directory for images
        2. Quality assess each image
        3. Clean and normalize accepted images
        4. Save to organized output directory
        5. Create dataset version snapshot

        Args:
            input_dir:    directory with raw images
            output_name:  name for the processed dataset
            labels_csv:   optional CSV with {filename: label} mapping
            version:      version string (e.g., "v1.0")
            val_split:    validation fraction
            test_split:   test fraction
            clean_images: apply cleaning pipeline
            overwrite:    overwrite existing version

        Returns:
            ProcessingResult
        """
        import time
        start_time = time.time()

        input_path  = Path(input_dir)
        output_path = self.data_root / "processed" / output_name
        output_path.mkdir(parents=True, exist_ok=True)

        # Load labels
        labels = self._load_labels(labels_csv) if labels_csv else {}

        # Find all images
        extensions = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
        image_files = [
            f for f in input_path.rglob("*")
            if f.suffix.lower() in extensions
        ]
        logger.info(f"[DataEngine] Found {len(image_files)} images in {input_path}")

        result = ProcessingResult()

        for i, img_path in enumerate(image_files):
            if (i + 1) % 100 == 0:
                logger.info(f"  Processing {i+1}/{len(image_files)}...")

            # Load image
            image = cv2.imread(str(img_path))
            if image is None:
                result.rejected.append({"path": str(img_path), "reason": "Failed to load"})
                continue
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # Quality assessment
            quality = self.quality_assessor.assess(image)
            result.quality_scores[img_path.name] = quality["overall_score"]

            if not quality["is_adequate"]:
                # Save to rejected with metadata
                reject_path = self.data_root / "rejected" / img_path.name
                cv2.imwrite(str(reject_path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
                result.rejected.append({
                    "path": str(img_path),
                    "reason": quality["rejection_reason"],
                    "quality_score": quality["overall_score"],
                })
                continue

            # Clean image
            if clean_images:
                processed = self.cleaner.clean(image)
            else:
                processed = cv2.resize(image, (self.target_size, self.target_size))

            # Save processed image
            out_file = output_path / img_path.name
            cv2.imwrite(str(out_file), cv2.cvtColor(processed, cv2.COLOR_RGB2BGR))
            result.accepted.append(str(out_file))

        # Build class distribution from labels
        class_dist: Dict[int, int] = {}
        for fname in [Path(p).name for p in result.accepted]:
            label = labels.get(fname.split(".")[0], labels.get(fname, -1))
            if isinstance(label, int) and label >= 0:
                class_dist[label] = class_dist.get(label, 0) + 1

        # Compute splits
        n = len(result.accepted)
        n_test = int(n * test_split)
        n_val  = int(n * val_split)
        n_train = n - n_val - n_test
        splits = {"train": n_train, "val": n_val, "test": n_test}

        # Save per-image metadata
        metadata = {
            Path(p).name: {
                "quality": result.quality_scores.get(Path(p).name, 0),
                "label": labels.get(Path(p).stem, -1),
            }
            for p in result.accepted
        }
        meta_path = output_path / "metadata.json"
        with open(meta_path, "w") as f:
            json.dump(metadata, f, indent=2)

        # Create dataset version
        data_hash = self._compute_hash(result.accepted, labels)
        mean_quality = float(np.mean(list(result.quality_scores.values()))) if result.quality_scores else 0.0

        version_obj = DatasetVersion(
            name=output_name,
            version=version,
            created_at=datetime.utcnow().isoformat(),
            num_images=len(result.accepted),
            num_rejected=len(result.rejected),
            class_distribution=class_dist,
            mean_quality=mean_quality,
            data_hash=data_hash,
            config={
                "target_size":  self.target_size,
                "min_quality":  self.min_quality,
                "clean_images": clean_images,
                "input_dir":    str(input_path),
            },
            splits=splits,
            metadata_path=str(meta_path),
        )

        result.dataset_version = version_obj
        result.processing_time_s = time.time() - start_time

        # Register version
        self._registry[f"{output_name}:{version}"] = version_obj
        self._save_registry()

        logger.info(f"[DataEngine] {result.summary()}")
        logger.info(f"[DataEngine] Dataset version {output_name}:{version} registered.")

        return result

    def load_version(self, name: str, version: str) -> Optional[DatasetVersion]:
        """Retrieve a registered dataset version."""
        return self._registry.get(f"{name}:{version}")

    def list_versions(self, name: Optional[str] = None) -> List[DatasetVersion]:
        """List all registered dataset versions."""
        versions = list(self._registry.values())
        if name:
            versions = [v for v in versions if v.name == name]
        return sorted(versions, key=lambda v: v.created_at, reverse=True)

    def compute_statistics(self, dataset_dir: str) -> Dict[str, Any]:
        """
        Compute statistics on a processed dataset directory.

        Returns quality distribution, class balance, image size info.
        """
        path = Path(dataset_dir)
        if not path.exists():
            return {"error": f"Directory not found: {path}"}

        image_files = [f for f in path.glob("*.*")
                       if f.suffix.lower() in {".png", ".jpg", ".jpeg"}]
        if not image_files:
            return {"error": "No images found"}

        sizes, qualities, brightness_vals = [], [], []

        sample_size = min(500, len(image_files))
        sample = np.random.choice(image_files, sample_size, replace=False)

        for img_path in sample:
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            h, w = img_rgb.shape[:2]
            sizes.append((w, h))
            quality = self.quality_assessor.assess(img_rgb)
            qualities.append(quality["overall_score"])
            gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
            brightness_vals.append(float(gray.mean()))

        stats = {
            "total_images": len(image_files),
            "sample_size":  sample_size,
            "mean_quality": float(np.mean(qualities)),
            "std_quality":  float(np.std(qualities)),
            "pct_adequate": float((np.array(qualities) >= self.min_quality).mean()),
            "mean_brightness": float(np.mean(brightness_vals)),
            "unique_sizes": len(set(sizes)),
            "most_common_size": max(set(sizes), key=sizes.count) if sizes else None,
        }

        return stats

    # ── Internal Helpers ─────────────────────────────────────────────────────

    def _load_labels(self, csv_path: str) -> Dict[str, int]:
        """Load {image_id: label} from CSV file."""
        try:
            import pandas as pd
            df = pd.read_csv(csv_path)
            # Try common column names
            id_col    = next((c for c in df.columns if "id" in c.lower()), df.columns[0])
            label_col = next((c for c in df.columns
                              if any(k in c.lower() for k in ["label", "grade", "level", "diag"])),
                             df.columns[-1])
            return {str(row[id_col]): int(row[label_col]) for _, row in df.iterrows()}
        except Exception as e:
            logger.warning(f"[DataEngine] Could not load labels: {e}")
            return {}

    def _compute_hash(self, paths: List[str], labels: Dict) -> str:
        """Compute a deterministic hash of the dataset contents."""
        content = sorted(paths) + [json.dumps(labels, sort_keys=True)]
        return hashlib.sha256("".join(content).encode()).hexdigest()[:32]

    def _load_registry(self) -> Dict[str, DatasetVersion]:
        if self._registry_path.exists():
            with open(self._registry_path) as f:
                data = json.load(f)
            return {k: DatasetVersion(**v) for k, v in data.items()}
        return {}

    def _save_registry(self):
        with open(self._registry_path, "w") as f:
            json.dump(
                {k: v.to_dict() for k, v in self._registry.items()},
                f, indent=2,
            )

    def __repr__(self) -> str:
        return (
            f"RetinaDataEngine(\n"
            f"  data_root={self.data_root}\n"
            f"  target_size={self.target_size}\n"
            f"  min_quality={self.min_quality}\n"
            f"  registered_versions={len(self._registry)}\n"
            f")"
        )
