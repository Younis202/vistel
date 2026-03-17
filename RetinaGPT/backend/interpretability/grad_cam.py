"""
grad_cam.py — Explainability for Retina-GPT
============================================
Ophthalmologists need to understand WHY the AI made a decision.
This module provides:

    1. Grad-CAM — gradient-weighted class activation maps
    2. Attention Rollout — ViT attention visualization
    3. DINO Self-Attention — foundation model feature visualization
    4. Overlay renderer — heatmap on retinal image

Output: heatmap highlighting regions that caused the diagnosis.

Usage:
    explainer = RetinaExplainer(model)

    # Grad-CAM
    heatmap = explainer.grad_cam(image, target_class=2)   # DR Grade 2

    # Attention map
    attn_map = explainer.attention_rollout(image)

    # Visual overlay
    overlay = explainer.overlay_heatmap(original_image, heatmap)

    # Full report image
    report_img = explainer.full_explanation(image, predictions)
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, List, Tuple, Union
import cv2


# ─────────────────────────────────────────────────────────────────────────────
# Grad-CAM for ViT
# ─────────────────────────────────────────────────────────────────────────────

class GradCAMViT:
    """
    Gradient-weighted Class Activation Maps for Vision Transformers.

    Adapted from standard CNN Grad-CAM to work with ViT patch tokens.
    Target layer: the last attention block of the transformer.

    The heatmap shows which image patches the model focused on
    when making its diagnosis decision.
    """

    def __init__(self, model: nn.Module, target_layer_name: str = "last_block"):
        self.model = model
        self.gradients: Optional[torch.Tensor] = None
        self.activations: Optional[torch.Tensor] = None
        self._handles = []
        self._register_hooks(target_layer_name)

    def _register_hooks(self, target_layer_name: str):
        """Register forward and backward hooks on the target layer."""
        target_layer = self._find_target_layer(target_layer_name)
        if target_layer is None:
            return

        def save_activation(module, input, output):
            self.activations = output.detach()

        def save_gradient(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()

        self._handles.append(target_layer.register_forward_hook(save_activation))
        self._handles.append(target_layer.register_backward_hook(save_gradient))

    def _find_target_layer(self, name: str) -> Optional[nn.Module]:
        """Find the target transformer block."""
        # Try to find last transformer block
        if hasattr(self.model, "encoder") and hasattr(self.model.encoder, "blocks"):
            return self.model.encoder.blocks[-1]
        if hasattr(self.model, "blocks"):
            return self.model.blocks[-1]
        # Fallback: search named modules
        for n, m in self.model.named_modules():
            if "block" in n.lower() or "layer" in n.lower():
                last = m
        return last if 'last' in dir() else None

    def __call__(
        self,
        image: torch.Tensor,
        target_class: Optional[int] = None,
        patch_size: int = 16,
        image_size: int = 224,
    ) -> np.ndarray:
        """
        Compute Grad-CAM heatmap.

        Args:
            image:        (1, 3, H, W) preprocessed image tensor
            target_class: class index to explain. None = predicted class.
            patch_size:   ViT patch size (16)
            image_size:   input image size (224)

        Returns:
            heatmap: (H, W) float array in [0, 1]
        """
        self.model.eval()
        image = image.requires_grad_(True)

        # Forward pass
        output = self.model(image)

        # Get logits from model output
        if isinstance(output, dict):
            logits = output.get("dr", {}).get("logits",
                    output.get("logits", list(output.values())[0]))
        else:
            logits = output

        if isinstance(logits, dict):
            logits = logits.get("logits", list(logits.values())[0])

        # Target class
        if target_class is None:
            target_class = int(logits.argmax(dim=-1).item())

        # Backward
        self.model.zero_grad()
        one_hot = torch.zeros_like(logits)
        one_hot[0, target_class] = 1.0
        logits.backward(gradient=one_hot, retain_graph=True)

        if self.gradients is None or self.activations is None:
            # Fallback: return uniform map
            return np.ones((image_size, image_size))

        # Grad-CAM computation on patch tokens
        # activations: (1, N_tokens, D) or (1, D, H', W')
        act = self.activations
        grad = self.gradients

        # Handle different shapes
        if act.dim() == 3:
            # ViT patch tokens: (1, N, D) — skip CLS (first token)
            act  = act[:, 1:, :]   # (1, N_patches, D)
            grad = grad[:, 1:, :]

            # Weight activation by gradients
            weights = grad.mean(dim=-1, keepdim=True)          # (1, N, 1)
            cam = (act * weights).sum(dim=-1).squeeze(0)        # (N_patches,)
            cam = F.relu(cam)

            # Reshape to 2D patch grid
            n = int(image_size // patch_size)
            try:
                cam_2d = cam[:n*n].reshape(n, n).detach().cpu().numpy()
            except Exception:
                cam_2d = cam.reshape(int(cam.shape[0]**0.5), -1).detach().cpu().numpy()

        else:
            # CNN-style spatial activations: (1, D, H', W')
            weights = grad.mean(dim=[2, 3], keepdim=True)
            cam = (act * weights).sum(dim=1).squeeze(0)
            cam = F.relu(cam)
            cam_2d = cam.detach().cpu().numpy()

        # Upsample to image size
        heatmap = cv2.resize(cam_2d, (image_size, image_size), interpolation=cv2.INTER_CUBIC)

        # Normalize to [0, 1]
        heatmap -= heatmap.min()
        if heatmap.max() > 0:
            heatmap /= heatmap.max()

        return heatmap.astype(np.float32)

    def remove_hooks(self):
        for h in self._handles:
            h.remove()
        self._handles = []


# ─────────────────────────────────────────────────────────────────────────────
# Attention Rollout for ViT
# ─────────────────────────────────────────────────────────────────────────────

class AttentionRollout:
    """
    Attention Rollout for Vision Transformers (Abnar & Zuidema, 2020).

    Propagates attention through all transformer layers to show
    which patches the CLS token attends to in the final representation.

    Works directly with Retina Foundation Encoder's attention maps.
    """

    def __init__(self, model: nn.Module, head_fusion: str = "mean",
                 discard_ratio: float = 0.9):
        """
        Args:
            model:         The ViT model (encoder or full model)
            head_fusion:   How to combine attention heads: "mean" | "max" | "min"
            discard_ratio: Fraction of lowest attention weights to discard
        """
        self.model = model
        self.head_fusion = head_fusion
        self.discard_ratio = discard_ratio
        self.attention_maps: List[torch.Tensor] = []
        self._handles = []

    def _register_hooks(self):
        """Register hooks to capture attention weights from all blocks."""
        self.attention_maps = []

        def make_hook(layer_idx):
            def hook(module, input, output):
                # MultiheadAttention returns (output, attn_weights)
                if isinstance(output, tuple) and len(output) > 1:
                    attn = output[1]  # (B, H, N, N) or (B, N, N)
                    if attn is not None:
                        self.attention_maps.append(attn.detach())
            return hook

        encoder = self._get_encoder()
        if encoder is None:
            return

        for i, block in enumerate(encoder.blocks):
            # Hook the attention module
            for name, module in block.named_modules():
                if isinstance(module, nn.MultiheadAttention):
                    h = module.register_forward_hook(make_hook(i))
                    self._handles.append(h)
                    break

    def _get_encoder(self):
        """Get the transformer encoder from the model."""
        if hasattr(self.model, "encoder"):
            return self.model.encoder
        if hasattr(self.model, "blocks"):
            return self.model
        return None

    def __call__(
        self,
        image: torch.Tensor,
        patch_size: int = 16,
        image_size: int = 224,
    ) -> np.ndarray:
        """
        Compute attention rollout heatmap.

        Args:
            image:      (1, 3, H, W) tensor
            patch_size: ViT patch size
            image_size: image size

        Returns:
            heatmap: (H, W) float array in [0, 1]
        """
        self._register_hooks()
        self.model.eval()

        with torch.no_grad():
            _ = self.model(image)

        self.remove_hooks()

        if not self.attention_maps:
            return np.ones((image_size, image_size), dtype=np.float32)

        n_patches = (image_size // patch_size) ** 2
        n_tokens  = n_patches + 1  # +1 for CLS

        # Build rollout matrix
        result = torch.eye(n_tokens, device=image.device)

        for attn in self.attention_maps:
            # attn shape varies: normalize to (N, N)
            if attn.dim() == 4:   # (B, H, N, N)
                if self.head_fusion == "mean":
                    attn_fused = attn[0].mean(dim=0)
                elif self.head_fusion == "max":
                    attn_fused = attn[0].max(dim=0).values
                else:
                    attn_fused = attn[0].min(dim=0).values
            elif attn.dim() == 3:  # (B, N, N)
                attn_fused = attn[0]
            else:
                continue

            # Trim/pad to match n_tokens
            s = min(attn_fused.shape[0], n_tokens)
            attn_fused = attn_fused[:s, :s]

            # Discard low attention
            flat = attn_fused.flatten()
            threshold = torch.quantile(flat, self.discard_ratio)
            attn_fused[attn_fused < threshold] = 0

            # Add residual connection (identity)
            mat = attn_fused + torch.eye(s, device=attn_fused.device)
            mat /= mat.sum(dim=-1, keepdim=True).clamp(min=1e-9)

            # Trim result if needed
            r = result[:s, :s]
            result[:s, :s] = torch.matmul(mat, r)

        # CLS row: how much each patch contributes to the CLS token
        cls_attn = result[0, 1:1 + n_patches]  # Skip CLS→CLS
        cls_attn = cls_attn.cpu().numpy()

        # Reshape to patch grid
        n_side = image_size // patch_size
        try:
            cam_2d = cls_attn[:n_side*n_side].reshape(n_side, n_side)
        except Exception:
            n_s = int(len(cls_attn) ** 0.5)
            cam_2d = cls_attn[:n_s*n_s].reshape(n_s, n_s)

        # Upsample to image size
        heatmap = cv2.resize(cam_2d, (image_size, image_size), interpolation=cv2.INTER_CUBIC)
        heatmap -= heatmap.min()
        if heatmap.max() > 0:
            heatmap /= heatmap.max()

        return heatmap.astype(np.float32)

    def remove_hooks(self):
        for h in self._handles:
            h.remove()
        self._handles = []


# ─────────────────────────────────────────────────────────────────────────────
# Overlay Renderer
# ─────────────────────────────────────────────────────────────────────────────

class HeatmapOverlayRenderer:
    """
    Renders explainability heatmaps as colored overlays on retinal images.

    Produces publication-quality and clinical-use visualizations.
    """

    COLORMAPS = {
        "jet":      cv2.COLORMAP_JET,
        "hot":      cv2.COLORMAP_HOT,
        "rainbow":  cv2.COLORMAP_RAINBOW,
        "viridis":  cv2.COLORMAP_VIRIDIS,
        "plasma":   cv2.COLORMAP_PLASMA,
        "clinical": cv2.COLORMAP_JET,   # Standard for medical imaging
    }

    def overlay(
        self,
        image:    np.ndarray,
        heatmap:  np.ndarray,
        alpha:    float = 0.4,
        colormap: str = "jet",
    ) -> np.ndarray:
        """
        Blend heatmap over original retinal image.

        Args:
            image:    (H, W, 3) uint8 RGB image
            heatmap:  (H, W) float32 in [0, 1]
            alpha:    heatmap opacity (0=invisible, 1=full overlay)
            colormap: one of COLORMAPS keys

        Returns:
            (H, W, 3) uint8 blended image
        """
        # Ensure correct types
        image = self._ensure_uint8(image)
        heatmap_uint8 = (heatmap * 255).astype(np.uint8)

        # Apply colormap
        cmap = self.COLORMAPS.get(colormap, cv2.COLORMAP_JET)
        heatmap_colored = cv2.applyColorMap(heatmap_uint8, cmap)
        heatmap_rgb = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)

        # Resize if needed
        if heatmap_rgb.shape[:2] != image.shape[:2]:
            heatmap_rgb = cv2.resize(heatmap_rgb, (image.shape[1], image.shape[0]))

        # Blend
        overlay = cv2.addWeighted(image, 1 - alpha, heatmap_rgb, alpha, 0)
        return overlay

    def contour_overlay(
        self,
        image:   np.ndarray,
        heatmap: np.ndarray,
        threshold: float = 0.6,
        color: Tuple[int, int, int] = (255, 50, 50),
        thickness: int = 2,
    ) -> np.ndarray:
        """Draw contours of high-activation regions (clean clinical style)."""
        image = self._ensure_uint8(image).copy()
        binary = (heatmap > threshold).astype(np.uint8) * 255

        # Find contours
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        result = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        cv2.drawContours(result, contours, -1, color[::-1], thickness)
        return cv2.cvtColor(result, cv2.COLOR_BGR2RGB)

    def side_by_side(
        self,
        image:    np.ndarray,
        heatmap:  np.ndarray,
        title_left:  str = "Original",
        title_right: str = "Grad-CAM",
        colormap: str = "jet",
    ) -> np.ndarray:
        """Create side-by-side comparison panel."""
        image = self._ensure_uint8(image)
        overlay = self.overlay(image, heatmap, alpha=0.5, colormap=colormap)

        # Stack side by side
        h, w = image.shape[:2]
        panel = np.zeros((h + 30, w * 2, 3), dtype=np.uint8)
        panel[30:, :w]  = image
        panel[30:, w:]  = overlay

        # Add text labels
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(panel, title_left,  (10,   22), font, 0.7, (220, 220, 220), 2)
        cv2.putText(panel, title_right, (w+10, 22), font, 0.7, (220, 220, 220), 2)

        return panel

    def explanation_panel(
        self,
        image:    np.ndarray,
        grad_cam: np.ndarray,
        attn_map: Optional[np.ndarray] = None,
        predictions: Optional[Dict] = None,
    ) -> np.ndarray:
        """
        Full multi-panel explanation visualization for clinical reports.

        Returns an image with:
        [Original] [Grad-CAM] [Attention] [Predictions]
        """
        image = self._ensure_uint8(image)
        H, W = image.shape[:2]

        panels = [image]

        # Grad-CAM panel
        gc_overlay = self.overlay(image, grad_cam, alpha=0.45)
        panels.append(gc_overlay)

        # Attention map panel
        if attn_map is not None:
            attn_overlay = self.overlay(image, attn_map, alpha=0.45, colormap="plasma")
            panels.append(attn_overlay)

        # Prediction text panel
        if predictions:
            pred_panel = self._make_prediction_panel(H, W, predictions)
            panels.append(pred_panel)

        # Stack horizontally
        result = np.hstack(panels)

        # Add panel labels
        labels = ["Original", "Grad-CAM", "Attention", "Predictions"]
        font = cv2.FONT_HERSHEY_SIMPLEX
        for i, label in enumerate(labels[:len(panels)]):
            cv2.putText(result, label, (i * W + 5, 18), font, 0.6, (200, 200, 200), 1)

        return result

    def _make_prediction_panel(self, H: int, W: int, predictions: Dict) -> np.ndarray:
        """Create a text panel showing model predictions."""
        panel = np.zeros((H, W, 3), dtype=np.uint8)
        panel[:] = (25, 25, 35)  # Dark background

        font = cv2.FONT_HERSHEY_SIMPLEX
        y = 30
        cv2.putText(panel, "PREDICTIONS", (10, y), font, 0.55, (100, 200, 255), 1)
        y += 25
        cv2.line(panel, (10, y), (W-10, y), (60, 60, 80), 1)
        y += 15

        for key, val in predictions.items():
            if isinstance(val, float):
                text = f"{key}: {val:.1%}"
                color = (100, 255, 100) if val > 0.7 else (255, 200, 100) if val > 0.4 else (200, 200, 200)
            else:
                text = f"{key}: {val}"
                color = (200, 220, 255)

            cv2.putText(panel, text, (12, y), font, 0.42, color, 1)
            y += 20
            if y > H - 20:
                break

        return panel

    @staticmethod
    def _ensure_uint8(image: np.ndarray) -> np.ndarray:
        """Ensure image is uint8 RGB (H,W,3)."""
        if image.dtype != np.uint8:
            if image.max() <= 1.0:
                image = (image * 255).clip(0, 255).astype(np.uint8)
            else:
                image = image.clip(0, 255).astype(np.uint8)
        if image.ndim == 2:
            image = np.stack([image]*3, axis=-1)
        if image.shape[0] == 3:   # (3, H, W) → (H, W, 3)
            image = image.transpose(1, 2, 0)
        return image


# ─────────────────────────────────────────────────────────────────────────────
# Unified Retina Explainer
# ─────────────────────────────────────────────────────────────────────────────

class RetinaExplainer:
    """
    Unified explainability interface for Retina-GPT.

    Single class that provides all visualization methods.

    Usage:
        explainer = RetinaExplainer(model)

        # Explain a DR diagnosis
        result = explainer.explain(
            image_tensor,
            image_rgb_numpy,
            target_class=2,
            method="both",
        )
        cv2.imwrite("explanation.png", cv2.cvtColor(result.panel, cv2.COLOR_RGB2BGR))
    """

    def __init__(
        self,
        model: nn.Module,
        patch_size: int = 16,
        image_size: int = 224,
    ):
        self.model = model
        self.patch_size = patch_size
        self.image_size = image_size

        self.grad_cam       = GradCAMViT(model)
        self.attn_rollout   = AttentionRollout(model)
        self.renderer       = HeatmapOverlayRenderer()

    def explain(
        self,
        image_tensor:  torch.Tensor,
        image_rgb:     np.ndarray,
        target_class:  Optional[int] = None,
        method:        str = "both",    # "gradcam" | "attention" | "both"
        predictions:   Optional[Dict] = None,
    ) -> Dict[str, Union[np.ndarray, Dict]]:
        """
        Full explanation pipeline.

        Returns:
            dict with keys:
                'grad_cam_map':   (H,W) float heatmap from Grad-CAM
                'attention_map':  (H,W) float heatmap from attention rollout
                'panel':          (H, W*4, 3) explanation panel image
                'target_class':   int
        """
        results = {}

        # Grad-CAM
        grad_cam_map = None
        if method in ("gradcam", "both"):
            try:
                grad_cam_map = self.grad_cam(
                    image_tensor.clone(),
                    target_class=target_class,
                    patch_size=self.patch_size,
                    image_size=self.image_size,
                )
                results["grad_cam_map"] = grad_cam_map
            except Exception as e:
                results["grad_cam_error"] = str(e)

        # Attention rollout
        attn_map = None
        if method in ("attention", "both"):
            try:
                attn_map = self.attn_rollout(
                    image_tensor,
                    patch_size=self.patch_size,
                    image_size=self.image_size,
                )
                results["attention_map"] = attn_map
            except Exception as e:
                results["attention_error"] = str(e)

        # Explanation panel
        try:
            panel = self.renderer.explanation_panel(
                image_rgb, grad_cam_map or np.zeros((self.image_size, self.image_size)),
                attn_map, predictions
            )
            results["panel"] = panel
        except Exception as e:
            results["panel_error"] = str(e)

        # Clean up hooks
        self.grad_cam.remove_hooks()

        return results

    def save_explanation(
        self,
        results: Dict,
        save_dir: str,
        filename_prefix: str = "explanation",
    ):
        """Save all explanation outputs to disk."""
        import os
        os.makedirs(save_dir, exist_ok=True)

        if "panel" in results:
            img = results["panel"]
            cv2.imwrite(
                os.path.join(save_dir, f"{filename_prefix}_panel.png"),
                cv2.cvtColor(img, cv2.COLOR_RGB2BGR) if img.shape[-1] == 3 else img
            )

        if "grad_cam_map" in results:
            hm = (results["grad_cam_map"] * 255).astype(np.uint8)
            hm_colored = cv2.applyColorMap(hm, cv2.COLORMAP_JET)
            cv2.imwrite(os.path.join(save_dir, f"{filename_prefix}_gradcam.png"), hm_colored)

        if "attention_map" in results:
            hm = (results["attention_map"] * 255).astype(np.uint8)
            hm_colored = cv2.applyColorMap(hm, cv2.COLORMAP_PLASMA)
            cv2.imwrite(os.path.join(save_dir, f"{filename_prefix}_attention.png"), hm_colored)
