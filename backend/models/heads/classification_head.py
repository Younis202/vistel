"""
models/heads/classification_head.py
=====================================
Retina-GPT: Multi-Task Prediction Heads

This module implements all task-specific prediction heads that attach to
the RetinaViT backbone's output embeddings:

  1. QualityHead          — image quality assessment (binary)
  2. ClassificationHead   — DR grading (5-class ordinal)
  3. SegmentationHead     — retinal vessel segmentation (U-Net decoder)
  4. DetectionHead        — lesion detection (anchor-free FCOS-style)

All heads accept the standardized output dict from RetinaViT.

Author: Retina-GPT Engineering Team
Date: 2026
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────
# Shared Utilities
# ─────────────────────────────────────────────────────────────

class MLP(nn.Module):
    """Simple multi-layer perceptron with optional dropout."""

    def __init__(
        self,
        in_dim: int,
        hidden_dims: List[int],
        out_dim: int,
        drop: float = 0.0,
        act: nn.Module = nn.GELU(),
    ):
        super().__init__()
        dims = [in_dim] + hidden_dims + [out_dim]
        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                layers.append(act)
                if drop > 0:
                    layers.append(nn.Dropout(drop))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ConvBNReLU(nn.Module):
    """Convolution → BatchNorm → ReLU block."""

    def __init__(self, in_c: int, out_c: int, kernel: int = 3, padding: int = 1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel, padding=padding, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


# ─────────────────────────────────────────────────────────────
# 1. Image Quality Assessment Head
# ─────────────────────────────────────────────────────────────

class QualityHead(nn.Module):
    """
    Image Quality Assessment Head.
    
    Predicts whether a fundus image is gradable (sufficient quality for
    clinical diagnosis) or ungradable (artifact-ridden, blurry, overexposed).
    
    Input:  cls_token (B, D) — global image embedding from ViT
    Output: logits (B, 2) — [ungradable, gradable]
    
    Loss: Binary cross-entropy (optionally with label smoothing)
    Metric: Accuracy, AUC
    """

    def __init__(
        self,
        embed_dim: int = 768,
        hidden_dim: int = 256,
        num_classes: int = 2,
        drop: float = 0.3,
    ):
        super().__init__()
        self.head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(hidden_dim, num_classes),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, backbone_output: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Args:
            backbone_output: Output dict from RetinaViT containing 'cls_token'
        Returns:
            dict with 'logits' (B, 2) and 'probabilities' (B, 2)
        """
        cls = backbone_output["cls_token"]          # (B, D)
        logits = self.head(cls)                     # (B, 2)
        probs = F.softmax(logits, dim=-1)

        return {"logits": logits, "probabilities": probs}


# ─────────────────────────────────────────────────────────────
# 2. Diabetic Retinopathy Classification Head
# ─────────────────────────────────────────────────────────────

class ClassificationHead(nn.Module):
    """
    Diabetic Retinopathy Grading Head (5-class ordinal classification).
    
    Grades: 0=No DR, 1=Mild, 2=Moderate, 3=Severe, 4=Proliferative
    
    Implements ordinal regression via cumulative logits, which respects
    the natural ordering of DR grades (clinically appropriate).
    
    Input:  cls_token (B, D)
    Output: 
        class_logits (B, 5)     — standard cross-entropy target
        ordinal_logits (B, 4)   — for ordinal loss (cumulative link)
    
    Reference: Frank & Hall (2001), "A Simple Approach to Ordinal Classification"
    """

    def __init__(
        self,
        embed_dim: int = 768,
        hidden_dim: int = 512,
        num_classes: int = 5,
        drop: float = 0.3,
        use_ordinal: bool = True,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.use_ordinal = use_ordinal

        # Shared feature extractor
        self.feature_extractor = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(drop),
        )

        # Standard classification head
        self.cls_head = nn.Linear(hidden_dim // 2, num_classes)

        # Ordinal regression head: num_classes - 1 binary decisions
        # P(y > k) for k in {0, 1, 2, 3}
        if use_ordinal:
            self.ordinal_head = nn.Linear(hidden_dim // 2, num_classes - 1)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, backbone_output: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Args:
            backbone_output: Output dict from RetinaViT containing 'cls_token'
        Returns:
            dict with:
                'class_logits'   (B, 5)  — multinomial class scores
                'ordinal_logits' (B, 4)  — ordinal binary logits (if use_ordinal)
                'probabilities'  (B, 5)  — softmax class probabilities
                'predicted_grade' (B,)   — argmax predicted grade [0-4]
        """
        cls = backbone_output["cls_token"]
        features = self.feature_extractor(cls)      # (B, hidden//2)

        class_logits = self.cls_head(features)      # (B, 5)
        probs = F.softmax(class_logits, dim=-1)
        predicted_grade = torch.argmax(probs, dim=-1)

        out = {
            "class_logits": class_logits,
            "probabilities": probs,
            "predicted_grade": predicted_grade,
        }

        if self.use_ordinal:
            ordinal_logits = self.ordinal_head(features)  # (B, 4)
            # Convert to probabilities: P(y > k) = sigmoid(logit_k)
            ordinal_probs = torch.sigmoid(ordinal_logits)
            out["ordinal_logits"] = ordinal_logits
            out["ordinal_probs"] = ordinal_probs

        return out


# ─────────────────────────────────────────────────────────────
# 3. Vessel Segmentation Head
# ─────────────────────────────────────────────────────────────

class SegmentationHead(nn.Module):
    """
    Retinal Vessel Segmentation Head — U-Net style decoder.
    
    Reconstructs full-resolution binary segmentation mask from multi-scale
    ViT intermediate features. Uses hierarchical upsampling with skip connections
    (ViT-Adapter / Segmenter pattern).
    
    Input:  
        patch_tokens (B, N, D)           — final patch features from ViT
        intermediate (List[(B, N, D)])   — multi-scale features from ViT blocks
    
    Output:
        logits (B, 1, H, W)  — raw binary vessel segmentation logits
        mask   (B, 1, H, W)  — thresholded binary mask (sigmoid > 0.5)
    
    Architecture:
        Patch Tokens (32x32) → Upsample → Fuse with Intermediate → ...
        → 64x64 → 128x128 → 256x256 → 512x512 → 1-channel sigmoid output
    """

    def __init__(
        self,
        embed_dim: int = 768,
        decoder_channels: List[int] = [512, 256, 128, 64],
        num_classes: int = 1,          # 1 = binary vessel/background
        image_size: int = 512,
        patch_size: int = 16,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.grid_size = image_size // patch_size  # e.g., 32 for 512/16

        # Project patch tokens from embed_dim to decoder_channels[0]
        self.input_proj = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, decoder_channels[0]),
        )

        # Decoder: series of upsample + ConvBNReLU stages
        self.decoder_blocks = nn.ModuleList()
        in_ch = decoder_channels[0]
        for out_ch in decoder_channels[1:]:
            self.decoder_blocks.append(
                nn.Sequential(
                    nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
                    ConvBNReLU(in_ch, out_ch, kernel=3, padding=1),
                    ConvBNReLU(out_ch, out_ch, kernel=3, padding=1),
                )
            )
            in_ch = out_ch

        # Final upsample to patch_size (16x) and produce mask
        num_final_upsamples = int(torch.log2(torch.tensor(patch_size)).item()) - len(decoder_channels) + 1
        final_up = []
        for _ in range(num_final_upsamples):
            final_up.append(nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False))
            final_up.append(ConvBNReLU(in_ch, in_ch // 2, kernel=3, padding=1))
            in_ch = in_ch // 2

        final_up.append(nn.Conv2d(in_ch, num_classes, kernel_size=1))
        self.final_up = nn.Sequential(*final_up)

    def forward(self, backbone_output: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Args:
            backbone_output: Output dict from RetinaViT
        Returns:
            dict with 'logits' (B, 1, H, W) and 'mask' (B, 1, H, W)
        """
        patch_tokens = backbone_output["patch_tokens"]   # (B, N, D)
        B, N, D = patch_tokens.shape
        g = self.grid_size

        # Project and reshape to spatial feature map
        x = self.input_proj(patch_tokens)                # (B, N, C)
        x = rearrange(x, "b (h w) c -> b c h w", h=g, w=g)  # (B, C, g, g)

        # Decode: progressively upsample
        for block in self.decoder_blocks:
            x = block(x)

        # Final upsampling to target resolution
        logits = self.final_up(x)                        # (B, 1, H, W)
        mask = (torch.sigmoid(logits) > 0.5).float()

        return {"logits": logits, "mask": mask}


# ─────────────────────────────────────────────────────────────
# 4. Lesion Detection Head
# ─────────────────────────────────────────────────────────────

class DetectionHead(nn.Module):
    """
    Retinal Lesion Detection Head — anchor-free FCOS-style detector.
    
    Detects diabetic retinopathy lesions:
        Class 0: Microaneurysms (MA)
        Class 1: Hemorrhages (HE)
        Class 2: Hard Exudates (EX)
        Class 3: Soft Exudates / Cotton Wool Spots (SE)
    
    For each patch location, predicts:
        - Classification score (num_lesion_classes)
        - Bounding box regression (4 values: l, t, r, b distances to box edges)
        - Centerness score (how centered the location is in an object)
    
    Input:  patch_tokens (B, N, D) — spatial patch features from ViT
    Output: per-location class scores, box regression, centerness
    
    Reference: Tian et al. "FCOS: Fully Convolutional One-Stage Object Detection" (2019)
    """

    def __init__(
        self,
        embed_dim: int = 768,
        num_classes: int = 4,      # MA, HE, EX, SE
        hidden_dim: int = 256,
        grid_size: int = 32,       # = image_size // patch_size
    ):
        super().__init__()
        self.num_classes = num_classes
        self.grid_size = grid_size

        # Shared feature projection
        self.proj = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
        )

        # Classification branch: N locations × num_classes
        self.cls_branch = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes),
        )

        # Box regression branch: N locations × 4 (l, t, r, b)
        self.box_branch = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 4),
        )

        # Centerness branch: N locations × 1
        self.centerness_branch = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        # Initialize cls bias for focal loss stability (π = 0.01)
        nn.init.constant_(self.cls_branch[-1].bias, -torch.log(torch.tensor((1 - 0.01) / 0.01)).item())

    def forward(self, backbone_output: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Args:
            backbone_output: Output dict from RetinaViT
        Returns:
            dict with:
                'cls_logits'   (B, N, num_classes) — class scores per location
                'box_pred'     (B, N, 4)           — LTRB box predictions
                'centerness'   (B, N, 1)            — centerness scores
                'cls_probs'    (B, N, num_classes) — sigmoid class probabilities
        """
        patch_tokens = backbone_output["patch_tokens"]   # (B, N, D)
        features = self.proj(patch_tokens)               # (B, N, hidden)

        cls_logits   = self.cls_branch(features)         # (B, N, num_classes)
        box_pred     = F.relu(self.box_branch(features)) # (B, N, 4) — must be positive
        centerness   = self.centerness_branch(features)  # (B, N, 1)

        return {
            "cls_logits":  cls_logits,
            "box_pred":    box_pred,
            "centerness":  centerness,
            "cls_probs":   torch.sigmoid(cls_logits),
        }

    def decode_predictions(
        self,
        preds: Dict[str, torch.Tensor],
        score_threshold: float = 0.3,
        nms_iou_threshold: float = 0.4,
        image_size: int = 512,
    ) -> List[Dict[str, torch.Tensor]]:
        """
        Decode raw predictions into bounding boxes for a batch.
        
        Args:
            preds: Output dict from forward()
            score_threshold: Minimum confidence to keep a detection
            nms_iou_threshold: IoU threshold for NMS
            image_size: Original image size for coordinate conversion

        Returns:
            List of per-image detection dicts with 'boxes', 'scores', 'labels'
        """
        from torchvision.ops import nms

        B = preds["cls_logits"].shape[0]
        g = self.grid_size
        patch_stride = image_size // g

        # Generate patch center coordinates
        ys = (torch.arange(g, device=preds["cls_logits"].device) + 0.5) * patch_stride
        xs = (torch.arange(g, device=preds["cls_logits"].device) + 0.5) * patch_stride
        cy, cx = torch.meshgrid(ys, xs, indexing="ij")
        centers = torch.stack([cx.flatten(), cy.flatten()], dim=-1)  # (N, 2)

        results = []
        for b in range(B):
            cls_p  = preds["cls_probs"][b]      # (N, C)
            box_p  = preds["box_pred"][b]        # (N, 4) — l, t, r, b
            ctrn   = torch.sigmoid(preds["centerness"][b]).squeeze(-1)  # (N,)

            # Convert LTRB offsets to absolute boxes
            x1 = centers[:, 0] - box_p[:, 0] * patch_stride
            y1 = centers[:, 1] - box_p[:, 1] * patch_stride
            x2 = centers[:, 0] + box_p[:, 2] * patch_stride
            y2 = centers[:, 1] + box_p[:, 3] * patch_stride
            boxes = torch.stack([x1, y1, x2, y2], dim=-1)   # (N, 4)

            # Score = sqrt(cls_prob × centerness)
            scores, labels = cls_p.max(dim=-1)
            scores = torch.sqrt(scores * ctrn)

            # Filter by threshold
            keep_mask = scores > score_threshold
            if keep_mask.sum() == 0:
                results.append({"boxes": torch.empty(0, 4), "scores": torch.empty(0), "labels": torch.empty(0, dtype=torch.long)})
                continue

            boxes  = boxes[keep_mask]
            scores = scores[keep_mask]
            labels = labels[keep_mask]

            # NMS per class
            keep_ids = nms(boxes, scores, nms_iou_threshold)

            results.append({
                "boxes":  boxes[keep_ids].clamp(0, image_size),
                "scores": scores[keep_ids],
                "labels": labels[keep_ids],
            })

        return results
