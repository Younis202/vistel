"""
utils/preprocessing.py
=======================
Retina-GPT: Retinal Fundus Image Preprocessing Pipeline

This module implements the complete preprocessing pipeline for retinal fundus images,
including CLAHE enhancement, green channel extraction, circular masking, and normalization.
Designed for clinical-grade image preparation compatible with all major fundus datasets
(EyePACS, DRIVE, STARE, CHASE_DB1, DIARETDB).

Author: Retina-GPT Engineering Team
Date: 2026
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Tuple, Union

import cv2
import numpy as np
import torch
from PIL import Image

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────

@dataclass
class PreprocessingConfig:
    """
    Configuration dataclass for the retinal preprocessing pipeline.
    
    All parameters are tuned for standard fundus camera images (e.g., Topcon, Zeiss, Canon).
    Modify these based on your specific dataset characteristics.
    """
    # Target image dimensions (H, W) — must be divisible by patch_size in ViT
    target_size: Tuple[int, int] = (224, 224)

    # CLAHE (Contrast Limited Adaptive Histogram Equalization) parameters
    # Applied to green channel which has highest signal-to-noise for retinal structures
    clahe_clip_limit: float = 2.0
    clahe_tile_grid_size: Tuple[int, int] = (8, 8)

    # Whether to extract and use the green channel for preprocessing
    # Green channel has highest contrast for vessels and lesions
    use_green_channel: bool = True

    # Circular mask: remove black border artifacts common in fundus cameras
    apply_circular_mask: bool = True
    mask_scale: float = 0.95  # fraction of image radius to keep

    # Normalization: ImageNet stats (for pretrained encoders) or retina-specific
    normalize: bool = True
    use_imagenet_stats: bool = True  # set False to compute dataset-specific stats

    # Custom normalization stats (used if use_imagenet_stats=False)
    custom_mean: Tuple[float, float, float] = (0.4254, 0.2244, 0.0718)
    custom_std: Tuple[float, float, float] = (0.2966, 0.1621, 0.0793)

    # Output format
    output_dtype: str = "float32"


# ─────────────────────────────────────────────────────────────
# Core Preprocessing Functions
# ─────────────────────────────────────────────────────────────

def load_fundus_image(
    image_path: Union[str, Path],
    return_original: bool = False,
) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """
    Load a fundus image from disk, handling multiple formats robustly.

    Args:
        image_path: Path to image file (.jpg, .png, .tif, .bmp supported)
        return_original: If True, also returns the unmodified original image

    Returns:
        RGB numpy array (H, W, 3), uint8 [0, 255]
        If return_original=True: tuple of (rgb_image, original_image)

    Raises:
        FileNotFoundError: if image path does not exist
        ValueError: if image cannot be decoded
    """
    image_path = Path(image_path)
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    # OpenCV reads BGR by default — convert to RGB
    bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError(f"cv2.imread failed to decode: {image_path}")

    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    logger.debug(f"Loaded image: {image_path.name} | shape={rgb.shape} | dtype={rgb.dtype}")

    if return_original:
        return rgb, rgb.copy()
    return rgb


def resize_image(
    image: np.ndarray,
    target_size: Tuple[int, int] = (224, 224),
    interpolation: int = cv2.INTER_LANCZOS4,
) -> np.ndarray:
    """
    Resize fundus image with high-quality Lanczos interpolation.
    
    Lanczos4 is preferred over bilinear/bicubic for medical images as it
    preserves fine vessel structures better.

    Args:
        image: Input image (H, W, C) or (H, W)
        target_size: (height, width) target dimensions
        interpolation: OpenCV interpolation flag

    Returns:
        Resized image of shape (target_size[0], target_size[1], C)
    """
    h, w = target_size
    resized = cv2.resize(image, (w, h), interpolation=interpolation)
    return resized


def create_circular_mask(
    height: int,
    width: int,
    scale: float = 0.95,
) -> np.ndarray:
    """
    Create a circular binary mask to remove fundus camera border artifacts.
    
    Fundus cameras produce circular field-of-view images with black corners.
    This mask zeros out those corners, preventing the model from learning
    irrelevant border artifacts.

    Args:
        height: Image height in pixels
        width: Image width in pixels  
        scale: Fraction of min(H,W)/2 to use as radius (0.95 recommended)

    Returns:
        Boolean mask of shape (H, W), True inside circle
    """
    center_y, center_x = height // 2, width // 2
    radius = int(min(height, width) // 2 * scale)

    Y, X = np.ogrid[:height, :width]
    dist_from_center = np.sqrt((X - center_x) ** 2 + (Y - center_y) ** 2)
    mask = dist_from_center <= radius
    return mask


def apply_circular_mask(
    image: np.ndarray,
    scale: float = 0.95,
    fill_value: int = 0,
) -> np.ndarray:
    """
    Apply circular mask to fundus image, zeroing out border areas.

    Args:
        image: Input RGB image (H, W, 3)
        scale: Fraction of radius to preserve
        fill_value: Pixel value for masked-out regions (0 = black)

    Returns:
        Masked image with same shape as input
    """
    h, w = image.shape[:2]
    mask = create_circular_mask(h, w, scale)

    result = image.copy()
    if image.ndim == 3:
        result[~mask, :] = fill_value
    else:
        result[~mask] = fill_value

    return result


def apply_clahe(
    image: np.ndarray,
    clip_limit: float = 2.0,
    tile_grid_size: Tuple[int, int] = (8, 8),
    use_green_channel: bool = True,
) -> np.ndarray:
    """
    Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) to enhance
    retinal structures including vessels, disc, and lesions.

    For RGB fundus images, enhancement is applied to the green channel (highest
    contrast for retinal structures) and then recombined. Alternatively, enhancement
    can be applied in LAB color space for perceptual quality.

    Args:
        image: Input RGB image (H, W, 3), uint8
        clip_limit: CLAHE contrast clipping limit (higher = more enhancement)
        tile_grid_size: Grid size for local histogram computation
        use_green_channel: If True, enhance green channel separately

    Returns:
        CLAHE-enhanced RGB image (H, W, 3), uint8
    """
    clahe = cv2.createCLAHE(
        clipLimit=clip_limit,
        tileGridSize=tile_grid_size,
    )

    if use_green_channel:
        # Split into channels, enhance green (index 1), recombine
        r, g, b = cv2.split(image)
        g_enhanced = clahe.apply(g)
        enhanced = cv2.merge([r, g_enhanced, b])
    else:
        # Apply CLAHE in LAB color space (perceptually uniform)
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        l_enhanced = clahe.apply(l)
        lab_enhanced = cv2.merge([l_enhanced, a, b])
        enhanced = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2RGB)

    return enhanced


def normalize_image(
    image: np.ndarray,
    use_imagenet_stats: bool = True,
    mean: Optional[Tuple[float, float, float]] = None,
    std: Optional[Tuple[float, float, float]] = None,
) -> np.ndarray:
    """
    Normalize image to float32 with mean/std normalization.
    
    Converts uint8 [0, 255] → float32 [0, 1] → normalized (z-score per channel).
    ImageNet stats are used for pretrained ViT encoders; custom stats for
    domain-specific training from scratch.

    Args:
        image: Input image (H, W, 3), uint8 or float
        use_imagenet_stats: Use standard ImageNet mean/std
        mean: Custom per-channel mean (overrides imagenet if provided)
        std: Custom per-channel std (overrides imagenet if provided)

    Returns:
        Normalized float32 image (H, W, 3)
    """
    IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    # Convert to float [0, 1]
    if image.dtype == np.uint8:
        img_float = image.astype(np.float32) / 255.0
    else:
        img_float = image.astype(np.float32)
        if img_float.max() > 1.0:
            img_float /= 255.0

    # Select normalization stats
    if mean is not None and std is not None:
        norm_mean = np.array(mean, dtype=np.float32)
        norm_std  = np.array(std, dtype=np.float32)
    elif use_imagenet_stats:
        norm_mean = IMAGENET_MEAN
        norm_std  = IMAGENET_STD
    else:
        # Compute on-the-fly (per-image; dataset-level is preferred)
        norm_mean = img_float.mean(axis=(0, 1))
        norm_std  = img_float.std(axis=(0, 1)) + 1e-7

    normalized = (img_float - norm_mean) / norm_std
    return normalized.astype(np.float32)


def to_tensor(image: np.ndarray) -> torch.Tensor:
    """
    Convert numpy image (H, W, C) → PyTorch tensor (C, H, W).
    
    Args:
        image: Float32 numpy array (H, W, C)

    Returns:
        PyTorch float32 tensor (C, H, W)
    """
    # HWC → CHW
    tensor = torch.from_numpy(image.transpose(2, 0, 1)).float()
    return tensor


# ─────────────────────────────────────────────────────────────
# Main Preprocessing Pipeline
# ─────────────────────────────────────────────────────────────

class RetinaPreprocessor:
    """
    Complete retinal fundus image preprocessing pipeline.

    Chains all preprocessing steps in the correct clinical order:
        1. Load & validate image
        2. Resize to target dimensions
        3. Apply circular mask (remove border artifacts)
        4. CLAHE enhancement (boost vessel/lesion contrast)
        5. Normalize (float conversion + mean/std normalization)
        6. Convert to PyTorch tensor

    Example:
        >>> preprocessor = RetinaPreprocessor(PreprocessingConfig())
        >>> tensor = preprocessor("path/to/fundus.jpg")
        >>> print(tensor.shape)  # torch.Size([3, 512, 512])
    """

    def __init__(self, config: Optional[PreprocessingConfig] = None):
        self.config = config or PreprocessingConfig()
        logger.info(f"RetinaPreprocessor initialized | target_size={self.config.target_size}")

    def __call__(
        self,
        image_input: Union[str, Path, np.ndarray],
        return_intermediates: bool = False,
    ) -> Union[torch.Tensor, dict]:
        """
        Run full preprocessing pipeline on a single fundus image.

        Args:
            image_input: Path to image file OR numpy RGB array (H, W, 3)
            return_intermediates: If True, return dict with all intermediate outputs

        Returns:
            Preprocessed tensor (C, H, W) or dict of intermediate stages
        """
        cfg = self.config
        intermediates = {}

        # Step 1: Load
        if isinstance(image_input, (str, Path)):
            image = load_fundus_image(image_input)
        else:
            image = image_input.copy()
        intermediates["original"] = image

        # Step 2: Resize
        image = resize_image(image, target_size=cfg.target_size)
        intermediates["resized"] = image

        # Step 3: Circular mask
        if cfg.apply_circular_mask:
            image = apply_circular_mask(image, scale=cfg.mask_scale)
            intermediates["masked"] = image

        # Step 4: CLAHE enhancement
        image = apply_clahe(
            image,
            clip_limit=cfg.clahe_clip_limit,
            tile_grid_size=cfg.clahe_tile_grid_size,
            use_green_channel=cfg.use_green_channel,
        )
        intermediates["clahe"] = image

        # Step 5: Normalize
        if cfg.normalize:
            mean = cfg.custom_mean if not cfg.use_imagenet_stats else None
            std  = cfg.custom_std  if not cfg.use_imagenet_stats else None
            image = normalize_image(
                image,
                use_imagenet_stats=cfg.use_imagenet_stats,
                mean=mean,
                std=std,
            )
        else:
            image = image.astype(np.float32) / 255.0
        intermediates["normalized"] = image

        # Step 6: To tensor (CHW)
        tensor = to_tensor(image)
        intermediates["tensor"] = tensor

        logger.debug(f"Preprocessing complete | tensor shape: {tensor.shape}")

        if return_intermediates:
            return intermediates
        return tensor

    def preprocess_batch(
        self,
        image_paths: list,
        device: str = "cpu",
    ) -> torch.Tensor:
        """
        Preprocess a list of image paths into a batched tensor.

        Args:
            image_paths: List of image file paths
            device: Target device ('cpu', 'cuda', 'mps')

        Returns:
            Batched tensor of shape (B, C, H, W) on specified device
        """
        tensors = []
        for path in image_paths:
            try:
                t = self(path)
                tensors.append(t)
            except Exception as e:
                logger.warning(f"Failed to preprocess {path}: {e}")
                continue

        if not tensors:
            raise RuntimeError("No images successfully preprocessed.")

        batch = torch.stack(tensors, dim=0).to(device)
        logger.info(f"Batch preprocessed: {batch.shape} on {device}")
        return batch


# ─────────────────────────────────────────────────────────────
# Utility: Denormalize for visualization
# ─────────────────────────────────────────────────────────────

def denormalize_tensor(
    tensor: torch.Tensor,
    use_imagenet_stats: bool = True,
) -> np.ndarray:
    """
    Reverse normalization for visualization purposes.

    Args:
        tensor: Normalized tensor (C, H, W) or (B, C, H, W)
        use_imagenet_stats: Whether ImageNet stats were used

    Returns:
        uint8 numpy image (H, W, C) or (B, H, W, C) in range [0, 255]
    """
    IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406])
    IMAGENET_STD  = torch.tensor([0.229, 0.224, 0.225])

    t = tensor.clone().cpu()

    if t.ndim == 3:
        mean = IMAGENET_MEAN[:, None, None]
        std  = IMAGENET_STD[:, None, None]
        t = t * std + mean
        t = t.clamp(0, 1)
        img = (t.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    elif t.ndim == 4:
        mean = IMAGENET_MEAN[None, :, None, None]
        std  = IMAGENET_STD[None, :, None, None]
        t = t * std + mean
        t = t.clamp(0, 1)
        img = (t.permute(0, 2, 3, 1).numpy() * 255).astype(np.uint8)
    else:
        raise ValueError(f"Expected 3D or 4D tensor, got {t.ndim}D")

    return img
