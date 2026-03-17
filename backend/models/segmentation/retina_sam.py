"""
retina_sam.py — Retina Segment Anything Model
==============================================
Retina-SAM: A promptable segmentation engine for retinal structures.
Inspired by Meta's Segment Anything Model (SAM), adapted for fundus imaging.

Supports prompted segmentation of:
    • Retinal vessels (arteries, veins, capillaries)
    • Optic disc and optic cup
    • Lesions (microaneurysms, hemorrhages, exudates, cotton-wool spots)
    • Macula / fovea region
    • Any retinal structure via point, box, or text prompt

Architecture:
    ┌─────────────────────────────────────┐
    │   Retina Image Encoder (ViT)        │  ← Foundation encoder
    └────────────────┬────────────────────┘
                     │ image embedding
    ┌────────────────▼────────────────────┐
    │   Prompt Encoder                    │  ← point / box / mask / text
    └────────────────┬────────────────────┘
                     │ prompt tokens
    ┌────────────────▼────────────────────┐
    │   Mask Decoder (Transformer)        │  ← bidirectional cross-attention
    └────────────────┬────────────────────┘
                     │
    ┌────────────────▼────────────────────┐
    │   Multi-Scale Upsampler             │  ← pixel-accurate output
    └────────────────┬────────────────────┘
                     ↓
    Segmentation mask  +  IoU confidence score
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List, Dict, Any
import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# Prompt Types
# ─────────────────────────────────────────────────────────────────────────────

RETINA_STRUCTURES = {
    # Anatomy
    "vessel":       0,
    "artery":       1,
    "vein":         2,
    "optic_disc":   3,
    "optic_cup":    4,
    "macula":       5,
    "fovea":        6,
    # Pathology
    "microaneurysm":    7,
    "hemorrhage":       8,
    "hard_exudate":     9,
    "soft_exudate":     10,  # cotton wool spot
    "neovascularization": 11,
    "drusen":           12,
}


# ─────────────────────────────────────────────────────────────────────────────
# Positional Encoding (2D)
# ─────────────────────────────────────────────────────────────────────────────

class PositionalEncoding2D(nn.Module):
    """Sinusoidal 2D positional encoding for image feature maps."""

    def __init__(self, num_pos_feats: int = 128, temperature: float = 10000.0,
                 normalize: bool = True, scale: float = 2 * np.pi):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        self.scale = scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, H, W)
        Returns:
            pos: (B, num_pos_feats*2, H, W)
        """
        B, _, H, W = x.shape
        device = x.device

        y_embed = torch.arange(H, device=device, dtype=torch.float32).unsqueeze(1).expand(H, W)
        x_embed = torch.arange(W, device=device, dtype=torch.float32).unsqueeze(0).expand(H, W)

        if self.normalize:
            y_embed = y_embed / (H - 1) * self.scale
            x_embed = x_embed / (W - 1) * self.scale

        dim_t = torch.arange(self.num_pos_feats, device=device, dtype=torch.float32)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)

        pos_x = x_embed.unsqueeze(-1) / dim_t
        pos_y = y_embed.unsqueeze(-1) / dim_t

        pos_x = torch.stack([pos_x[..., 0::2].sin(), pos_x[..., 1::2].cos()], dim=-1).flatten(-2)
        pos_y = torch.stack([pos_y[..., 0::2].sin(), pos_y[..., 1::2].cos()], dim=-1).flatten(-2)

        pos = torch.cat([pos_y, pos_x], dim=-1).permute(2, 0, 1).unsqueeze(0)
        return pos.expand(B, -1, -1, -1)


# ─────────────────────────────────────────────────────────────────────────────
# Image Encoder (Adapter wrapping Foundation Encoder)
# ─────────────────────────────────────────────────────────────────────────────

class RetinaImageEncoderAdapter(nn.Module):
    """
    Wraps the Foundation Encoder and produces dense feature maps
    suitable for the SAM-style mask decoder.

    Foundation encoder outputs CLS token (global).
    We extract intermediate patch features to get spatial maps.
    """

    def __init__(self, foundation_encoder: nn.Module,
                 embed_dim: int = 1024, output_dim: int = 256,
                 image_size: int = 224, patch_size: int = 16):
        super().__init__()
        self.encoder = foundation_encoder
        self.embed_dim = embed_dim
        self.output_dim = output_dim
        self.num_patches_per_side = image_size // patch_size  # 14

        # Project from encoder dim to SAM decoder dim
        self.neck = nn.Sequential(
            nn.Conv2d(embed_dim, output_dim, kernel_size=1, bias=False),
            nn.LayerNorm([output_dim, self.num_patches_per_side, self.num_patches_per_side]),
            nn.Conv2d(output_dim, output_dim, kernel_size=3, padding=1, bias=False),
            nn.LayerNorm([output_dim, self.num_patches_per_side, self.num_patches_per_side]),
        )

        self.pos_enc = PositionalEncoding2D(num_pos_feats=output_dim // 2)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            image_embedding: (B, output_dim, H', W')
            pos_encoding:    (B, output_dim, H', W')
        """
        B = x.shape[0]

        # Extract patch tokens from encoder (skip CLS token and register tokens)
        tokens = self.encoder.patch_embed(x)                   # (B, N, D)
        cls = self.encoder.cls_token.expand(B, -1, -1)
        tokens = torch.cat([cls, tokens], dim=1)
        tokens = tokens + self.encoder.pos_embed
        for block in self.encoder.blocks:
            tokens = block(tokens)
        tokens = self.encoder.norm(tokens)

        # Reshape patch tokens to spatial map
        patch_tokens = tokens[:, 1:, :]                        # Remove CLS: (B, N, D)
        H = W = self.num_patches_per_side
        feat_map = patch_tokens.reshape(B, H, W, self.embed_dim).permute(0, 3, 1, 2)

        # Project to decoder dim
        image_embedding = self.neck(feat_map)                  # (B, output_dim, H, W)
        pos = self.pos_enc(image_embedding)

        return image_embedding, pos


# ─────────────────────────────────────────────────────────────────────────────
# Prompt Encoder
# ─────────────────────────────────────────────────────────────────────────────

class RetinaPromptEncoder(nn.Module):
    """
    Encodes various prompt types into tokens for the mask decoder:
        - Point prompts (foreground / background clicks)
        - Box prompts (bounding box around structure)
        - Structure label prompts (which structure to segment)
        - Dense mask prompts (previous iteration output)
    """

    def __init__(self, embed_dim: int = 256, input_image_size: Tuple[int, int] = (224, 224),
                 image_embedding_size: Tuple[int, int] = (14, 14)):
        super().__init__()
        self.embed_dim = embed_dim
        self.input_image_size = input_image_size
        self.image_embedding_size = image_embedding_size

        # Point embeddings: foreground and background
        self.point_embeddings = nn.Embedding(2, embed_dim)  # 0=bg, 1=fg

        # Special token for padding when no prompt given
        self.not_a_point_embed = nn.Embedding(1, embed_dim)

        # Box prompt: encode as 2 corner points
        self.box_embed_corner = nn.Embedding(2, embed_dim)  # top-left, bottom-right

        # Structure label embedding
        self.structure_embed = nn.Embedding(len(RETINA_STRUCTURES) + 1, embed_dim)

        # Dense mask input projection
        self.mask_input_conv = nn.Sequential(
            nn.Conv2d(1, embed_dim // 4, kernel_size=2, stride=2),
            nn.GELU(),
            nn.Conv2d(embed_dim // 4, embed_dim, kernel_size=2, stride=2),
            nn.GELU(),
            nn.Conv2d(embed_dim, embed_dim, kernel_size=1),
            nn.LayerNorm([embed_dim, *image_embedding_size]),
        )

        self.pos_enc = PositionalEncoding2D(num_pos_feats=embed_dim // 2)

    def _encode_points(self, points: torch.Tensor,
                       labels: torch.Tensor) -> torch.Tensor:
        """
        Args:
            points: (B, N, 2) normalized [0,1] coordinates
            labels: (B, N) — 1=foreground, 0=background

        Returns:
            (B, N, D) point tokens
        """
        B, N, _ = points.shape
        H_img, W_img = self.input_image_size

        # Sinusoidal positional encoding for point coordinates
        x_coords = points[..., 0] * W_img
        y_coords = points[..., 1] * H_img

        dim = self.embed_dim
        dim_t = torch.arange(dim // 2, device=points.device, dtype=torch.float32)
        dim_t = 10000.0 ** (2 * dim_t / dim)

        pe_x = x_coords.unsqueeze(-1) / dim_t
        pe_y = y_coords.unsqueeze(-1) / dim_t

        pe = torch.cat([pe_x.sin(), pe_y.sin()], dim=-1)

        # Add foreground/background embedding
        point_emb = self.point_embeddings(labels.clamp(0, 1))  # (B, N, D)
        return pe + point_emb

    def forward(
        self,
        image_embedding: torch.Tensor,
        points: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        boxes: Optional[torch.Tensor] = None,
        structure_label: Optional[torch.Tensor] = None,
        mask_input: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            sparse_embeddings: (B, N_prompts, D)  — point/box/structure tokens
            dense_embeddings:  (B, D, H', W')      — mask embedding
        """
        B = image_embedding.shape[0]
        sparse_tokens = []

        # 1. Point prompts
        if points is not None:
            coords, labels = points
            pt_tokens = self._encode_points(coords, labels)  # (B, N, D)
            sparse_tokens.append(pt_tokens)

        # 2. Box prompt
        if boxes is not None:
            # boxes: (B, 4) — x1, y1, x2, y2 normalized
            corner_coords = boxes.reshape(B, 2, 2)
            corner_labels = torch.tensor([0, 1], device=boxes.device).expand(B, -1)
            box_tokens = self._encode_points(corner_coords, corner_labels)
            box_tokens = box_tokens + self.box_embed_corner.weight.unsqueeze(0)
            sparse_tokens.append(box_tokens)

        # 3. Structure label
        if structure_label is not None:
            struct_tokens = self.structure_embed(structure_label).unsqueeze(1)  # (B, 1, D)
            sparse_tokens.append(struct_tokens)

        if sparse_tokens:
            sparse_embeddings = torch.cat(sparse_tokens, dim=1)
        else:
            sparse_embeddings = self.not_a_point_embed.weight.unsqueeze(0).expand(B, 1, -1)

        # 4. Dense mask prompt
        if mask_input is not None:
            dense_embeddings = self.mask_input_conv(mask_input)
        else:
            # Zero dense embedding + positional encoding
            H, W = self.image_embedding_size
            dense_embeddings = torch.zeros(B, self.embed_dim, H, W, device=image_embedding.device)

        return sparse_embeddings, dense_embeddings


# ─────────────────────────────────────────────────────────────────────────────
# Mask Decoder
# ─────────────────────────────────────────────────────────────────────────────

class TwoWayAttentionBlock(nn.Module):
    """Bidirectional cross-attention between image tokens and prompt tokens."""

    def __init__(self, embedding_dim: int = 256, num_heads: int = 8):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(embedding_dim, num_heads, batch_first=True)
        self.cross_attn_token_to_image = nn.MultiheadAttention(embedding_dim, num_heads, batch_first=True)
        self.cross_attn_image_to_token = nn.MultiheadAttention(embedding_dim, num_heads, batch_first=True)

        self.norm1 = nn.LayerNorm(embedding_dim)
        self.norm2 = nn.LayerNorm(embedding_dim)
        self.norm3 = nn.LayerNorm(embedding_dim)
        self.norm4 = nn.LayerNorm(embedding_dim)

        self.mlp = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim * 4),
            nn.GELU(),
            nn.Linear(embedding_dim * 4, embedding_dim),
        )

    def forward(
        self,
        queries: torch.Tensor,   # prompt tokens (B, N_q, D)
        keys: torch.Tensor,      # image tokens (B, N_k, D)
        query_pe: torch.Tensor,
        key_pe: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        # Self attention on queries
        q = queries + query_pe
        attn, _ = self.self_attn(q, q, q)
        queries = self.norm1(queries + attn)

        # Cross: query → image
        q = queries + query_pe
        k = keys + key_pe
        attn, _ = self.cross_attn_token_to_image(q, k, keys)
        queries = self.norm2(queries + attn)

        # MLP
        queries = self.norm3(queries + self.mlp(queries))

        # Cross: image → query
        q = queries + query_pe
        k = keys + key_pe
        attn, _ = self.cross_attn_image_to_token(k, q, queries)
        keys = self.norm4(keys + attn)

        return queries, keys


class RetinaMaskDecoder(nn.Module):
    """
    Transformer-based mask decoder.
    Produces multiple mask hypotheses with IoU confidence scores.
    """

    def __init__(self, transformer_dim: int = 256,
                 num_multimask_outputs: int = 3, num_heads: int = 8,
                 iou_head_depth: int = 3, iou_head_hidden_dim: int = 256):
        super().__init__()
        self.transformer_dim = transformer_dim
        self.num_masks = num_multimask_outputs + 1  # +1 for single-mask mode

        # Output tokens
        self.iou_token = nn.Embedding(1, transformer_dim)
        self.mask_tokens = nn.Embedding(self.num_masks, transformer_dim)

        # Transformer
        self.transformer = nn.ModuleList([
            TwoWayAttentionBlock(transformer_dim, num_heads)
            for _ in range(2)
        ])
        self.final_attn = nn.MultiheadAttention(transformer_dim, num_heads, batch_first=True)
        self.norm_final = nn.LayerNorm(transformer_dim)

        # Upsampling
        self.output_upscaling = nn.Sequential(
            nn.ConvTranspose2d(transformer_dim, transformer_dim // 4, kernel_size=2, stride=2),
            nn.LayerNorm([transformer_dim // 4]),
            nn.GELU(),
            nn.ConvTranspose2d(transformer_dim // 4, transformer_dim // 8, kernel_size=2, stride=2),
            nn.GELU(),
        )

        # Per-mask MLPs
        self.output_hypernetworks_mlps = nn.ModuleList([
            nn.Sequential(
                nn.Linear(transformer_dim, transformer_dim),
                nn.GELU(),
                nn.Linear(transformer_dim, transformer_dim // 8),
            )
            for _ in range(self.num_masks)
        ])

        # IoU prediction head
        self.iou_prediction_head = nn.Sequential(
            *[nn.Sequential(nn.Linear(transformer_dim, iou_head_hidden_dim), nn.GELU())
              for _ in range(iou_head_depth - 1)],
            nn.Linear(iou_head_hidden_dim, self.num_masks),
        )

    def forward(
        self,
        image_embeddings: torch.Tensor,  # (B, D, H', W')
        image_pe: torch.Tensor,          # (B, D, H', W')
        sparse_prompt_embeddings: torch.Tensor,  # (B, N, D)
        dense_prompt_embeddings: torch.Tensor,   # (B, D, H', W')
        multimask_output: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            masks:     (B, num_masks, H, W) upsampled predictions
            iou_pred:  (B, num_masks) confidence scores
        """
        B = image_embeddings.shape[0]

        # Concatenate output tokens
        output_tokens = torch.cat([
            self.iou_token.weight.unsqueeze(0).expand(B, -1, -1),
            self.mask_tokens.weight.unsqueeze(0).expand(B, -1, -1),
        ], dim=1)                                               # (B, num_masks+1, D)

        tokens = torch.cat([output_tokens, sparse_prompt_embeddings], dim=1)

        # Add dense prompt to image embedding
        src = image_embeddings + dense_prompt_embeddings        # (B, D, H', W')
        H, W = src.shape[-2:]

        # Flatten spatial dims for transformer
        src_flat = src.flatten(2).transpose(1, 2)              # (B, H'W', D)
        pos_flat = image_pe.flatten(2).transpose(1, 2)         # (B, H'W', D)

        # Run transformer
        q = tokens
        k = src_flat
        q_pe = torch.zeros_like(q)
        for block in self.transformer:
            q, k = block(q, k, q_pe, pos_flat)

        # Final cross-attention
        q_out, _ = self.final_attn(q, k + pos_flat, k)
        q = self.norm_final(q + q_out)

        # Extract IoU and mask tokens
        iou_token_out = q[:, 0, :]                             # (B, D)
        mask_tokens_out = q[:, 1:self.num_masks + 1, :]        # (B, num_masks, D)

        # Upsample image features
        upscaled = self.output_upscaling(
            k.transpose(1, 2).reshape(B, -1, H, W)
        )                                                       # (B, D//8, H*4, W*4)

        # Generate mask predictions
        hyper_in = torch.stack([
            mlp(mask_tokens_out[:, i, :])
            for i, mlp in enumerate(self.output_hypernetworks_mlps)
        ], dim=1)                                               # (B, num_masks, D//8)

        b, c, h, w = upscaled.shape
        masks = (hyper_in @ upscaled.view(b, c, -1)).view(b, -1, h, w)

        iou_pred = self.iou_prediction_head(iou_token_out)

        if multimask_output:
            return masks[:, 1:], iou_pred[:, 1:]
        else:
            return masks[:, :1], iou_pred[:, :1]


# ─────────────────────────────────────────────────────────────────────────────
# Full Retina-SAM
# ─────────────────────────────────────────────────────────────────────────────

class RetinaSAM(nn.Module):
    """
    Full Retina Segment Anything Model.

    Combines:
    - RetinaImageEncoderAdapter (wraps Foundation encoder)
    - RetinaPromptEncoder
    - RetinaMaskDecoder

    Usage:
        model = RetinaSAM(foundation_encoder)

        # Segment optic disc
        masks, iou = model.predict(
            image,
            structure="optic_disc"
        )

        # Segment via point click
        masks, iou = model.predict(
            image,
            points=([[0.5, 0.5]], [[1]])  # center point, foreground
        )
    """

    PIXEL_MEAN = [123.675, 116.28, 103.53]
    PIXEL_STD  = [58.395, 57.12, 57.375]

    def __init__(self, foundation_encoder: nn.Module,
                 encoder_embed_dim: int = 1024,
                 decoder_embed_dim: int = 256,
                 image_size: int = 224):
        super().__init__()

        self.image_encoder = RetinaImageEncoderAdapter(
            foundation_encoder,
            embed_dim=encoder_embed_dim,
            output_dim=decoder_embed_dim,
            image_size=image_size,
        )

        embedding_size = (image_size // 16, image_size // 16)  # (14, 14)

        self.prompt_encoder = RetinaPromptEncoder(
            embed_dim=decoder_embed_dim,
            input_image_size=(image_size, image_size),
            image_embedding_size=embedding_size,
        )

        self.mask_decoder = RetinaMaskDecoder(
            transformer_dim=decoder_embed_dim,
        )

        self.register_buffer(
            "pixel_mean",
            torch.tensor(self.PIXEL_MEAN).view(1, 3, 1, 1), persistent=False
        )
        self.register_buffer(
            "pixel_std",
            torch.tensor(self.PIXEL_STD).view(1, 3, 1, 1), persistent=False
        )

    def preprocess(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.pixel_mean) / self.pixel_std

    def forward(
        self,
        image: torch.Tensor,
        points: Optional[Tuple] = None,
        boxes: Optional[torch.Tensor] = None,
        structure_label: Optional[torch.Tensor] = None,
        mask_input: Optional[torch.Tensor] = None,
        multimask_output: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """
        Returns dict with keys:
            'masks':    (B, num_masks, H, W) float logits
            'iou_pred': (B, num_masks)
        """
        image = self.preprocess(image)

        image_embedding, image_pe = self.image_encoder(image)

        sparse_emb, dense_emb = self.prompt_encoder(
            image_embedding=image_embedding,
            points=points,
            boxes=boxes,
            structure_label=structure_label,
            mask_input=mask_input,
        )

        masks, iou_pred = self.mask_decoder(
            image_embeddings=image_embedding,
            image_pe=image_pe,
            sparse_prompt_embeddings=sparse_emb,
            dense_prompt_embeddings=dense_emb,
            multimask_output=multimask_output,
        )

        return {"masks": masks, "iou_pred": iou_pred}

    @torch.no_grad()
    def predict(
        self,
        image: torch.Tensor,
        structure: Optional[str] = None,
        points: Optional[Tuple] = None,
        boxes: Optional[torch.Tensor] = None,
        threshold: float = 0.0,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        High-level prediction interface.

        Args:
            image: (1, 3, H, W) tensor or (3, H, W)
            structure: structure name from RETINA_STRUCTURES dict
            points: (coords, labels) tuples
            boxes: (1, 4) bounding box
            threshold: mask logit threshold

        Returns:
            masks:     (num_masks, H, W) boolean numpy arrays
            scores:    (num_masks,) float IoU confidence scores
        """
        if image.dim() == 3:
            image = image.unsqueeze(0)

        device = next(self.parameters()).device
        image = image.to(device)

        struct_label = None
        if structure is not None:
            label_idx = RETINA_STRUCTURES.get(structure, 0)
            struct_label = torch.tensor([label_idx], device=device)

        out = self.forward(image, points=points, boxes=boxes,
                           structure_label=struct_label)

        masks = (out['masks'] > threshold).squeeze(0).cpu().numpy()
        scores = out['iou_pred'].squeeze(0).cpu().numpy()

        # Return best mask
        best_idx = scores.argmax()
        return masks[best_idx:best_idx+1], scores[best_idx:best_idx+1]
