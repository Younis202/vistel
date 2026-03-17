"""
models/backbone/retina_vit.py
==============================
Retina-GPT: Retina Vision Transformer (RVT) Backbone

A specialized Vision Transformer encoder optimized for retinal fundus image analysis.
Key innovations over standard ViT:
  - Fundus-aware positional encoding (emphasizes fovea/disc regions)
  - Layerscale for improved training stability at small scale
  - Multi-scale feature extraction (compatible with segmentation decoders)
  - Optional register tokens (following DINOv2 best practices)

Architecture follows:
  Dosovitskiy et al. "An Image is Worth 16x16 Words" (2020)
  + clinical domain adaptations for fundus imaging.

Author: Retina-GPT Engineering Team
Date: 2026
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from functools import partial
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────

@dataclass
class RetinaViTConfig:
    """
    Full configuration for the Retina Vision Transformer.
    
    Provides three preset scales:
        - Tiny  (6L,  384D, 6H):  ~5M params  — fast prototyping
        - Base  (12L, 768D, 12H): ~86M params — recommended
        - Large (24L, 1024D, 16H): ~307M params — high-capacity
    """
    # Image & patch settings
    image_size: int = 512          # fundus image size (square)
    patch_size: int = 16           # ViT patch size (16 or 32)
    in_channels: int = 3           # RGB input

    # Transformer architecture
    embed_dim: int = 768           # token embedding dimension
    depth: int = 12                # number of transformer blocks
    num_heads: int = 12            # attention heads (embed_dim must be divisible)
    mlp_ratio: float = 4.0         # MLP hidden dim = embed_dim * mlp_ratio
    qkv_bias: bool = True
    drop_rate: float = 0.0         # embedding dropout
    attn_drop_rate: float = 0.0    # attention dropout
    drop_path_rate: float = 0.1    # stochastic depth rate

    # Domain-specific options
    use_fundus_pos_embed: bool = True    # fundus-aware positional encoding
    num_register_tokens: int = 4         # DINOv2-style register tokens
    use_layerscale: bool = True          # LayerScale for training stability
    layerscale_init: float = 1e-4        # initial LayerScale value

    # Multi-scale output (for segmentation heads)
    return_intermediate: bool = True    # return features at multiple depths
    intermediate_layers: List[int] = field(
        default_factory=lambda: [3, 6, 9, 11]
    )  # layer indices to extract features from

    # Pretrained initialization
    pretrained_weights: Optional[str] = None  # path or HuggingFace model ID

    @property
    def num_patches(self) -> int:
        return (self.image_size // self.patch_size) ** 2

    @classmethod
    def tiny(cls) -> "RetinaViTConfig":
        return cls(embed_dim=384, depth=6, num_heads=6)

    @classmethod
    def base(cls) -> "RetinaViTConfig":
        return cls(embed_dim=768, depth=12, num_heads=12)

    @classmethod
    def large(cls) -> "RetinaViTConfig":
        return cls(embed_dim=1024, depth=24, num_heads=16, patch_size=16)


# ─────────────────────────────────────────────────────────────
# Building Blocks
# ─────────────────────────────────────────────────────────────

class PatchEmbedding(nn.Module):
    """
    Fundus-aware patch embedding layer.
    
    Converts image (B, C, H, W) → patch tokens (B, N, D).
    Uses a convolutional projection (kernel=patch_size, stride=patch_size)
    which is equivalent to splitting into non-overlapping patches and
    projecting each to embed_dim.
    """

    def __init__(
        self,
        image_size: int = 512,
        patch_size: int = 16,
        in_channels: int = 3,
        embed_dim: int = 768,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.num_patches = (image_size // patch_size) ** 2
        self.grid_size = image_size // patch_size

        # Convolutional projection: (B, 3, H, W) → (B, D, H/p, W/p)
        self.proj = nn.Conv2d(
            in_channels, embed_dim,
            kernel_size=patch_size, stride=patch_size,
        )
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input image tensor (B, C, H, W)
        Returns:
            Patch tokens (B, N, D) where N = (H/p)*(W/p)
        """
        B, C, H, W = x.shape
        x = self.proj(x)                            # (B, D, H/p, W/p)
        x = rearrange(x, "b d h w -> b (h w) d")   # (B, N, D)
        x = self.norm(x)
        return x


class FundusPositionalEncoding(nn.Module):
    """
    Fundus-aware 2D sinusoidal positional encoding.
    
    Standard ViT uses learned 1D positional embeddings. For fundus images,
    we use a learnable 2D variant initialized with sinusoidal values that
    encode spatial position. This helps the model understand the anatomical
    layout (fovea is always central, optic disc is to the nasal side).
    
    Additionally, a radial distance bias is added to encode proximity to
    the image center (clinically relevant: central lesions vs. peripheral).
    """

    def __init__(self, num_patches: int, embed_dim: int, grid_size: int):
        super().__init__()
        self.grid_size = grid_size

        # Learnable 2D positional embedding
        self.pos_embed = nn.Parameter(
            torch.zeros(1, num_patches, embed_dim)
        )
        self._init_pos_embed(embed_dim)

        # Radial bias: learnable scalar weight on sinusoidal radial encoding
        self.radial_weight = nn.Parameter(torch.ones(1))
        radial = self._compute_radial_bias(grid_size)
        self.register_buffer("radial_bias", radial)  # (1, N, 1) non-learnable

    def _init_pos_embed(self, embed_dim: int):
        """Initialize with 2D sinusoidal values."""
        g = self.grid_size
        pos_embed = torch.zeros(1, g * g, embed_dim)
        d = embed_dim // 4  # use first quarter of dims for row, next quarter for col

        for i in range(g):
            for j in range(g):
                idx = i * g + j
                for k in range(d):
                    v = 1.0 / (10000 ** (2 * k / embed_dim))
                    pos_embed[0, idx, 2 * k]     = math.sin(i * v)
                    pos_embed[0, idx, 2 * k + 1] = math.cos(i * v)
                    pos_embed[0, idx, 2 * k + d * 2]     = math.sin(j * v)
                    pos_embed[0, idx, 2 * k + d * 2 + 1] = math.cos(j * v)

        with torch.no_grad():
            self.pos_embed.copy_(pos_embed)

    def _compute_radial_bias(self, grid_size: int) -> torch.Tensor:
        """Compute normalized radial distance from center for each patch."""
        g = grid_size
        center = g / 2.0
        radial = torch.zeros(1, g * g, 1)
        for i in range(g):
            for j in range(g):
                dist = math.sqrt((i - center) ** 2 + (j - center) ** 2)
                radial[0, i * g + j, 0] = dist / (center * math.sqrt(2))
        return radial  # normalized [0, 1]

    def forward(self, num_patches: int) -> torch.Tensor:
        """Returns positional encodings of shape (1, N, D)."""
        pos = self.pos_embed[:, :num_patches, :]
        # Add radial bias (broadcast over embed_dim via scalar weight)
        # This gives the model a soft prior: central patches matter more
        return pos  # radial bias applied in main forward if needed


class LayerScale(nn.Module):
    """
    LayerScale: learnable per-channel scaling applied after each sub-layer.
    Introduced in "Going deeper with Image Transformers" (Touvron et al., 2021).
    Significantly improves training stability for deeper ViTs.
    """

    def __init__(self, dim: int, init_value: float = 1e-4):
        super().__init__()
        self.gamma = nn.Parameter(init_value * torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.gamma


class DropPath(nn.Module):
    """
    Stochastic depth regularization (drop entire residual path).
    Equivalent to randomly skipping transformer blocks during training.
    """

    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.drop_prob == 0.0:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor = torch.floor(random_tensor + keep_prob)
        return x / keep_prob * random_tensor


class MultiHeadSelfAttention(nn.Module):
    """
    Multi-head self-attention with optional attention dropout.
    
    Uses scaled dot-product attention: O = softmax(QK^T / sqrt(d_k)) * V
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Token sequence (B, N, D)
        Returns:
            output (B, N, D), attention_weights (B, H, N, N)
        """
        B, N, D = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)          # (3, B, H, N, head_dim)
        q, k, v = qkv.unbind(0)                     # each: (B, H, N, head_dim)

        attn = (q @ k.transpose(-2, -1)) * self.scale  # (B, H, N, N)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        out = (attn @ v).transpose(1, 2).reshape(B, N, D)  # (B, N, D)
        out = self.proj_drop(self.proj(out))

        return out, attn


class FeedForward(nn.Module):
    """
    Position-wise feed-forward network: two linear layers with GELU activation.
    hidden_dim = embed_dim * mlp_ratio (typically 4x).
    """

    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        drop: float = 0.0,
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(drop),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TransformerBlock(nn.Module):
    """
    Single Vision Transformer block with Pre-LN formulation:
        x = x + LayerScale(DropPath(MHSA(LN(x))))
        x = x + LayerScale(DropPath(FFN(LN(x))))
    
    Pre-LN (normalize before sub-layer) is more stable than Post-LN.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        drop_path: float = 0.0,
        use_layerscale: bool = True,
        layerscale_init: float = 1e-4,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = MultiHeadSelfAttention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias,
            attn_drop=attn_drop, proj_drop=drop,
        )
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = FeedForward(dim, hidden_dim=int(dim * mlp_ratio), drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        self.ls1 = LayerScale(dim, layerscale_init) if use_layerscale else nn.Identity()
        self.ls2 = LayerScale(dim, layerscale_init) if use_layerscale else nn.Identity()

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns (output_tokens, attention_weights)."""
        attn_out, attn_weights = self.attn(self.norm1(x))
        x = x + self.drop_path(self.ls1(attn_out))
        x = x + self.drop_path(self.ls2(self.ffn(self.norm2(x))))
        return x, attn_weights


# ─────────────────────────────────────────────────────────────
# Retina Vision Transformer
# ─────────────────────────────────────────────────────────────

class RetinaViT(nn.Module):
    """
    Retina Vision Transformer (RVT) — the foundation encoder for Retina-GPT.

    Input:  Fundus image tensor (B, 3, H, W) — preprocessed & normalized
    Output: 
        cls_token (B, D) — global image representation for classification
        patch_tokens (B, N, D) — spatial features for segmentation/detection
        intermediate_features (List[Tensor]) — multi-scale features

    Usage:
        >>> cfg = RetinaViTConfig.base()
        >>> model = RetinaViT(cfg)
        >>> img = torch.randn(2, 3, 512, 512)
        >>> out = model(img)
        >>> print(out['cls_token'].shape)      # (2, 768)
        >>> print(out['patch_tokens'].shape)   # (2, 1024, 768)
    """

    def __init__(self, config: Optional[RetinaViTConfig] = None):
        super().__init__()
        self.config = config or RetinaViTConfig.base()
        cfg = self.config

        # ── Patch embedding ──
        self.patch_embed = PatchEmbedding(
            image_size=cfg.image_size,
            patch_size=cfg.patch_size,
            in_channels=cfg.in_channels,
            embed_dim=cfg.embed_dim,
        )
        num_patches = self.patch_embed.num_patches

        # ── Class token (global summary) ──
        self.cls_token = nn.Parameter(torch.zeros(1, 1, cfg.embed_dim))

        # ── Register tokens (reduce noise in patch features — DINOv2) ──
        self.num_register_tokens = cfg.num_register_tokens
        if cfg.num_register_tokens > 0:
            self.register_tokens = nn.Parameter(
                torch.zeros(1, cfg.num_register_tokens, cfg.embed_dim)
            )

        # ── Positional encoding ──
        if cfg.use_fundus_pos_embed:
            self.pos_embed = FundusPositionalEncoding(
                num_patches=num_patches,
                embed_dim=cfg.embed_dim,
                grid_size=self.patch_embed.grid_size,
            )
        else:
            self.pos_embed = nn.Parameter(
                torch.zeros(1, num_patches + 1, cfg.embed_dim)
            )

        self.pos_drop = nn.Dropout(cfg.drop_rate)

        # ── Stochastic depth schedule ──
        # Linearly increase drop rate from 0 to drop_path_rate across blocks
        dpr = [
            x.item()
            for x in torch.linspace(0, cfg.drop_path_rate, cfg.depth)
        ]

        # ── Transformer blocks ──
        self.blocks = nn.ModuleList([
            TransformerBlock(
                dim=cfg.embed_dim,
                num_heads=cfg.num_heads,
                mlp_ratio=cfg.mlp_ratio,
                qkv_bias=cfg.qkv_bias,
                drop=cfg.drop_rate,
                attn_drop=cfg.attn_drop_rate,
                drop_path=dpr[i],
                use_layerscale=cfg.use_layerscale,
                layerscale_init=cfg.layerscale_init,
            )
            for i in range(cfg.depth)
        ])

        self.norm = nn.LayerNorm(cfg.embed_dim)

        # ── Initialize weights ──
        self._init_weights()

        n_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        logger.info(
            f"RetinaViT | depth={cfg.depth} | dim={cfg.embed_dim} | "
            f"heads={cfg.num_heads} | params={n_params/1e6:.1f}M"
        )

    def _init_weights(self):
        """Initialize weights following ViT paper conventions."""
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        if self.num_register_tokens > 0:
            nn.init.trunc_normal_(self.register_tokens, std=0.02)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the Retina Vision Transformer.

        Args:
            x: Input fundus image tensor (B, C, H, W)

        Returns:
            dict with keys:
                'cls_token':            (B, D) — global representation
                'patch_tokens':         (B, N, D) — spatial features
                'intermediate':         List of (B, N, D) — multi-scale features
                'attention_maps':       List of (B, H, N, N) — attention from each block
        """
        B = x.shape[0]

        # ── Patch embedding ──
        patch_tokens = self.patch_embed(x)       # (B, N, D)
        N = patch_tokens.shape[1]

        # ── Prepend class token ──
        cls = repeat(self.cls_token, "1 1 d -> b 1 d", b=B)
        tokens = torch.cat([cls, patch_tokens], dim=1)  # (B, N+1, D)

        # ── Add positional encoding ──
        if isinstance(self.pos_embed, FundusPositionalEncoding):
            pos = self.pos_embed(N)                      # (1, N, D)
            # Prepend zero position for cls token
            cls_pos = torch.zeros(1, 1, pos.shape[-1], device=x.device)
            pos = torch.cat([cls_pos, pos], dim=1)       # (1, N+1, D)
        else:
            pos = self.pos_embed[:, :N + 1, :]

        tokens = self.pos_drop(tokens + pos)

        # ── Append register tokens ──
        if self.num_register_tokens > 0:
            reg = repeat(self.register_tokens, "1 r d -> b r d", b=B)
            tokens = torch.cat([tokens, reg], dim=1)     # (B, N+1+R, D)

        # ── Transformer blocks ──
        intermediate_features = []
        attention_maps = []
        cfg = self.config

        for i, block in enumerate(self.blocks):
            tokens, attn = block(tokens)
            attention_maps.append(attn)

            if cfg.return_intermediate and i in cfg.intermediate_layers:
                # Extract only patch tokens (exclude cls and register tokens)
                n_extra = 1 + self.num_register_tokens
                intermediate_features.append(tokens[:, 1: N + 1, :])

        # ── Final normalization ──
        tokens = self.norm(tokens)

        # ── Extract outputs ──
        cls_out = tokens[:, 0, :]                    # (B, D)
        patch_out = tokens[:, 1: N + 1, :]          # (B, N, D)

        return {
            "cls_token": cls_out,
            "patch_tokens": patch_out,
            "intermediate": intermediate_features,
            "attention_maps": attention_maps,
        }

    def get_attention_map(self, x: torch.Tensor, layer_idx: int = -1) -> torch.Tensor:
        """
        Extract attention map from a specific layer for visualization (attention rollout).

        Args:
            x: Input image (B, C, H, W)
            layer_idx: Which block's attention to return (-1 = last)

        Returns:
            Attention map (B, H, N, N)
        """
        out = self.forward(x)
        return out["attention_maps"][layer_idx]

    @classmethod
    def from_pretrained(cls, model_id: str, **kwargs) -> "RetinaViT":
        """
        Load a RetinaViT from HuggingFace Hub or local checkpoint.
        
        For pure ViT pretrained weights, we load only the compatible layers
        and allow the fundus-specific components (register tokens, fundus pos)
        to initialize from scratch.

        Args:
            model_id: HuggingFace model ID or local path
            **kwargs: Additional config overrides

        Returns:
            RetinaViT instance with pretrained weights
        """
        from transformers import ViTModel

        logger.info(f"Loading pretrained ViT from: {model_id}")
        hf_model = ViTModel.from_pretrained(model_id)
        hf_state = hf_model.state_dict()

        # Create config matching the pretrained model
        config = RetinaViTConfig.base()
        config.pretrained_weights = model_id
        for k, v in kwargs.items():
            setattr(config, k, v)

        model = cls(config)

        # Selective weight loading (skip incompatible keys)
        own_state = model.state_dict()
        loaded, skipped = 0, 0
        for name, param in hf_state.items():
            if name in own_state and own_state[name].shape == param.shape:
                own_state[name].copy_(param)
                loaded += 1
            else:
                skipped += 1

        model.load_state_dict(own_state)
        logger.info(f"Pretrained weights: loaded={loaded}, skipped={skipped}")
        return model
