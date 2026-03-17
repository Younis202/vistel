"""
retina_dino.py — Retina Foundation Model Self-Supervised Pretraining
=====================================================================
Implements Retina-DINO: a DINOv2-inspired self-supervised pretraining
framework specialized for retinal fundus images.

Architecture:
    Student RetinaViT  ←──── multi-crop augmented views
    Teacher RetinaViT  ←──── EMA of student weights

Training philosophy (like DINOv2):
    - No labels required
    - Self-distillation across global + local crops
    - Teacher updated via Exponential Moving Average (EMA)
    - Centering prevents representation collapse

After pretraining the encoder learns:
    • optic disc / cup structure
    • vascular tree topology
    • lesion textures (microaneurysms, hemorrhages, exudates)
    • macula / fovea anatomy

This pretrained backbone becomes the Universal Retina Embedding engine
that powers ALL downstream tasks (classification, detection, segmentation).

Reference philosophies:
    - DINOv2 (Oquab et al., 2023)
    - DINO (Caron et al., 2021)
    - iBOT (Zhou et al., 2022)
"""

import math
import copy
from dataclasses import dataclass, field
from typing import List, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast

import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2


# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class RetinaDINOConfig:
    """Full configuration for Retina-DINO pretraining."""

    # Encoder dimensions
    embed_dim: int = 1024          # Universal retina embedding dimension
    depth: int = 12                # Transformer depth
    num_heads: int = 16            # Attention heads
    mlp_ratio: float = 4.0
    patch_size: int = 16
    image_size: int = 224

    # DINO-specific
    out_dim: int = 65536           # Projection head output dimension (prototype space)
    use_bn_in_head: bool = False
    norm_last_layer: bool = True

    # EMA teacher update
    momentum_teacher: float = 0.996   # Starting EMA momentum
    momentum_teacher_end: float = 1.0 # Final EMA momentum

    # Centering (prevents collapse)
    center_momentum: float = 0.9

    # Temperature schedule
    warmup_teacher_temp: float = 0.04
    teacher_temp: float = 0.04
    warmup_teacher_temp_epochs: int = 30
    student_temp: float = 0.1

    # Multi-crop
    global_crops_scale: Tuple[float, float] = (0.4, 1.0)
    local_crops_scale: Tuple[float, float] = (0.05, 0.4)
    local_crops_number: int = 8
    global_crop_size: int = 224
    local_crop_size: int = 96

    # Training
    epochs: int = 100
    warmup_epochs: int = 10
    base_lr: float = 5e-4
    min_lr: float = 1e-6
    weight_decay: float = 0.04
    weight_decay_end: float = 0.4
    clip_grad: float = 3.0
    batch_size: int = 64
    freeze_last_layer: int = 1     # Epochs to freeze last projection layer


# ─────────────────────────────────────────────────────────────────────────────
# Retina-Specific Multi-Crop Augmentation
# ─────────────────────────────────────────────────────────────────────────────

class RetinaMultiCropAugmentation:
    """
    Multi-crop augmentation strategy specialized for retinal fundus images.

    Key differences from natural image DINO:
    - Conservative color jitter (retinal pathology is color-sensitive)
    - No aggressive perspective distortion (preserves vascular topology)
    - CLAHE enhancement preserving lesion contrast
    - Circular crop awareness for fundus images
    """

    def __init__(self, config: RetinaDINOConfig):
        self.config = config

        # Global crop transforms (2 views, large scale)
        self.global_transform = A.Compose([
            A.RandomResizedCrop(
                height=config.global_crop_size,
                width=config.global_crop_size,
                scale=config.global_crops_scale,
                ratio=(0.9, 1.1),  # Near-square for fundus
            ),
            A.HorizontalFlip(p=0.5),
            A.OneOf([
                A.ColorJitter(
                    brightness=0.2, contrast=0.2,
                    saturation=0.1, hue=0.05, p=1.0
                ),
                A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=1.0),
            ], p=0.8),
            A.GaussianBlur(blur_limit=(3, 7), p=0.5),
            A.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ])

        # Local crop transforms (many views, small scale)
        self.local_transform = A.Compose([
            A.RandomResizedCrop(
                height=config.local_crop_size,
                width=config.local_crop_size,
                scale=config.local_crops_scale,
                ratio=(0.9, 1.1),
            ),
            A.HorizontalFlip(p=0.5),
            A.ColorJitter(
                brightness=0.2, contrast=0.2,
                saturation=0.1, hue=0.05, p=0.8
            ),
            A.GaussianBlur(blur_limit=(3, 5), p=0.5),
            A.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ])

    def __call__(self, image: np.ndarray) -> List[torch.Tensor]:
        """
        Generate multi-crop views from a single retinal image.

        Returns:
            List of tensors: [global_1, global_2, local_1, ..., local_N]
        """
        crops = []

        # 2 global crops
        for _ in range(2):
            aug = self.global_transform(image=image)
            crops.append(aug['image'])

        # N local crops
        for _ in range(self.config.local_crops_number):
            aug = self.local_transform(image=image)
            crops.append(aug['image'])

        return crops


# ─────────────────────────────────────────────────────────────────────────────
# Patch Embedding for Foundation Model
# ─────────────────────────────────────────────────────────────────────────────

class FoundationPatchEmbedding(nn.Module):
    """Patch embedding with fundus-aware initialization."""

    def __init__(self, image_size: int, patch_size: int, in_channels: int, embed_dim: int):
        super().__init__()
        self.patch_size = patch_size
        self.num_patches = (image_size // patch_size) ** 2

        self.projection = nn.Conv2d(
            in_channels, embed_dim,
            kernel_size=patch_size, stride=patch_size, bias=False
        )
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.projection(x)                           # (B, D, H/P, W/P)
        x = x.flatten(2).transpose(1, 2)                # (B, N, D)
        return self.norm(x)


# ─────────────────────────────────────────────────────────────────────────────
# Foundation Encoder (ViT with register tokens)
# ─────────────────────────────────────────────────────────────────────────────

class RetinaFoundationEncoder(nn.Module):
    """
    Vision Transformer encoder for retinal foundation model.

    Produces a 1024-dim Universal Retina Embedding from any fundus image.
    Incorporates register tokens (DINOv2-style) to reduce patch token noise.
    """

    def __init__(self, config: RetinaDINOConfig, num_register_tokens: int = 4):
        super().__init__()
        self.config = config
        self.embed_dim = config.embed_dim

        # Patch embedding
        self.patch_embed = FoundationPatchEmbedding(
            image_size=config.image_size,
            patch_size=config.patch_size,
            in_channels=3,
            embed_dim=config.embed_dim,
        )
        num_patches = self.patch_embed.num_patches

        # Special tokens
        self.cls_token = nn.Parameter(torch.zeros(1, 1, config.embed_dim))
        self.register_tokens = nn.Parameter(
            torch.zeros(1, num_register_tokens, config.embed_dim)
        )
        self.pos_embed = nn.Parameter(
            torch.zeros(1, num_patches + 1, config.embed_dim)
        )

        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(
                dim=config.embed_dim,
                num_heads=config.num_heads,
                mlp_ratio=config.mlp_ratio,
            )
            for _ in range(config.depth)
        ])
        self.norm = nn.LayerNorm(config.embed_dim)

        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.register_tokens, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]

        # Patch embedding
        tokens = self.patch_embed(x)                     # (B, N, D)

        # Prepend CLS token
        cls = self.cls_token.expand(B, -1, -1)
        tokens = torch.cat([cls, tokens], dim=1)         # (B, N+1, D)

        # Add positional embedding
        tokens = tokens + self.pos_embed

        # Append register tokens (no positional encoding — DINOv2 style)
        reg = self.register_tokens.expand(B, -1, -1)
        tokens = torch.cat([tokens, reg], dim=1)         # (B, N+1+R, D)

        # Transformer layers
        for block in self.blocks:
            tokens = block(tokens)

        tokens = self.norm(tokens)

        # Return CLS token as the Universal Retina Embedding
        return tokens[:, 0]                              # (B, D=1024)


class TransformerBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 4.0, drop: float = 0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=drop, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden), nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(hidden, dim), nn.Dropout(drop),
        )
        self.ls1 = LayerScale(dim)
        self.ls2 = LayerScale(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm1(x)
        attn_out, _ = self.attn(h, h, h)
        x = x + self.ls1(attn_out)
        x = x + self.ls2(self.mlp(self.norm2(x)))
        return x


class LayerScale(nn.Module):
    def __init__(self, dim: int, init_value: float = 1e-5):
        super().__init__()
        self.gamma = nn.Parameter(init_value * torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.gamma


# ─────────────────────────────────────────────────────────────────────────────
# DINO Projection Head
# ─────────────────────────────────────────────────────────────────────────────

class DINOProjectionHead(nn.Module):
    """
    MLP projection head mapping embedding → prototype space.
    Shared between student and teacher (separate copies).
    """

    def __init__(self, in_dim: int, out_dim: int, hidden_dim: int = 2048,
                 bottleneck_dim: int = 256, use_bn: bool = False,
                 norm_last_layer: bool = True):
        super().__init__()

        layers = [nn.Linear(in_dim, hidden_dim)]
        if use_bn:
            layers.append(nn.BatchNorm1d(hidden_dim))
        layers.append(nn.GELU())
        layers.append(nn.Linear(hidden_dim, hidden_dim))
        if use_bn:
            layers.append(nn.BatchNorm1d(hidden_dim))
        layers.append(nn.GELU())
        layers.append(nn.Linear(hidden_dim, bottleneck_dim))

        self.mlp = nn.Sequential(*layers)

        self.last_layer = nn.utils.weight_norm(
            nn.Linear(bottleneck_dim, out_dim, bias=False)
        )
        self.last_layer.weight_g.data.fill_(1)
        if norm_last_layer:
            self.last_layer.weight_g.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.mlp(x)
        x = F.normalize(x, dim=-1, p=2)
        return self.last_layer(x)


# ─────────────────────────────────────────────────────────────────────────────
# Retina-DINO Loss
# ─────────────────────────────────────────────────────────────────────────────

class RetinaDINOLoss(nn.Module):
    """
    Self-distillation loss for Retina-DINO.

    Student learns from teacher outputs across multiple crop scales.
    Centering prevents collapse without needing contrastive negatives.
    """

    def __init__(self, out_dim: int, num_crops: int,
                 warmup_teacher_temp: float, teacher_temp: float,
                 warmup_epochs: int, total_epochs: int,
                 student_temp: float = 0.1,
                 center_momentum: float = 0.9):
        super().__init__()
        self.student_temp = student_temp
        self.center_momentum = center_momentum
        self.num_crops = num_crops

        self.register_buffer("center", torch.zeros(1, out_dim))

        # Teacher temperature schedule
        self.teacher_temp_schedule = np.concatenate([
            np.linspace(warmup_teacher_temp, teacher_temp, warmup_epochs),
            np.ones(total_epochs - warmup_epochs) * teacher_temp
        ])

    def forward(
        self,
        student_output: torch.Tensor,
        teacher_output: torch.Tensor,
        epoch: int
    ) -> torch.Tensor:
        """
        Args:
            student_output: (num_crops * B, out_dim) — logits from student
            teacher_output: (2 * B, out_dim)         — logits from teacher (global only)
            epoch: current training epoch

        Returns:
            Scalar loss
        """
        student_out = student_output / self.student_temp
        student_out = student_out.chunk(self.num_crops)

        teacher_temp = self.teacher_temp_schedule[epoch]
        teacher_out = F.softmax((teacher_output - self.center) / teacher_temp, dim=-1)
        teacher_out = teacher_out.detach().chunk(2)  # 2 global crops

        total_loss = 0.0
        n_loss_terms = 0

        for iq, q in enumerate(teacher_out):
            for v in range(len(student_out)):
                if v == iq:
                    continue  # Skip same crop pair
                loss = torch.sum(
                    -q * F.log_softmax(student_out[v], dim=-1), dim=-1
                )
                total_loss += loss.mean()
                n_loss_terms += 1

        total_loss /= n_loss_terms
        self._update_center(teacher_output)
        return total_loss

    @torch.no_grad()
    def _update_center(self, teacher_output: torch.Tensor):
        """Update centering buffer using EMA."""
        batch_center = torch.mean(teacher_output, dim=0, keepdim=True)
        self.center = (
            self.center * self.center_momentum +
            batch_center * (1 - self.center_momentum)
        )


# ─────────────────────────────────────────────────────────────────────────────
# Retina-DINO Model (Student + Teacher)
# ─────────────────────────────────────────────────────────────────────────────

class RetinaDINO(nn.Module):
    """
    Full Retina-DINO self-supervised pretraining model.

    Wraps student encoder + teacher encoder (EMA copy) + projection heads.

    Usage:
        model = RetinaDINO(config)
        loss = model(crops, epoch)
        loss.backward()
        model.update_teacher(momentum)
    """

    def __init__(self, config: RetinaDINOConfig):
        super().__init__()
        self.config = config

        # Student
        self.student_encoder = RetinaFoundationEncoder(config)
        self.student_head = DINOProjectionHead(
            in_dim=config.embed_dim,
            out_dim=config.out_dim,
            use_bn=config.use_bn_in_head,
            norm_last_layer=config.norm_last_layer,
        )

        # Teacher (EMA copy — no gradients)
        self.teacher_encoder = copy.deepcopy(self.student_encoder)
        self.teacher_head = DINOProjectionHead(
            in_dim=config.embed_dim,
            out_dim=config.out_dim,
            use_bn=config.use_bn_in_head,
            norm_last_layer=False,  # Teacher head no weight norm freeze
        )

        # Teacher requires no gradients
        for p in self.teacher_encoder.parameters():
            p.requires_grad = False
        for p in self.teacher_head.parameters():
            p.requires_grad = False

        # Loss
        num_crops = 2 + config.local_crops_number
        self.loss_fn = RetinaDINOLoss(
            out_dim=config.out_dim,
            num_crops=num_crops,
            warmup_teacher_temp=config.warmup_teacher_temp,
            teacher_temp=config.teacher_temp,
            warmup_epochs=config.warmup_teacher_temp_epochs,
            total_epochs=config.epochs,
            student_temp=config.student_temp,
            center_momentum=config.center_momentum,
        )

    def forward(
        self,
        crops: List[torch.Tensor],
        epoch: int
    ) -> torch.Tensor:
        """
        Args:
            crops: list of [global_1, global_2, local_1..local_N] tensors
            epoch: current epoch

        Returns:
            DINO self-distillation loss
        """
        # ── Student forward (all crops)
        all_crops = torch.cat(crops, dim=0)              # (num_crops * B, C, H, W)
        student_embed = self.student_encoder(all_crops)  # (num_crops * B, D)
        student_out = self.student_head(student_embed)   # (num_crops * B, out_dim)

        # ── Teacher forward (global crops only, no grad)
        global_crops = torch.cat(crops[:2], dim=0)       # (2B, C, H, W)
        with torch.no_grad():
            teacher_embed = self.teacher_encoder(global_crops)
            teacher_out = self.teacher_head(teacher_embed)

        return self.loss_fn(student_out, teacher_out, epoch)

    @torch.no_grad()
    def update_teacher(self, momentum: float):
        """EMA update of teacher from student weights."""
        for param_s, param_t in zip(
            self.student_encoder.parameters(),
            self.teacher_encoder.parameters()
        ):
            param_t.data = param_t.data * momentum + param_s.data * (1.0 - momentum)

        for param_s, param_t in zip(
            self.student_head.parameters(),
            self.teacher_head.parameters()
        ):
            param_t.data = param_t.data * momentum + param_s.data * (1.0 - momentum)

    def get_student_encoder(self) -> RetinaFoundationEncoder:
        """Return student encoder for downstream fine-tuning."""
        return self.student_encoder

    def get_teacher_encoder(self) -> RetinaFoundationEncoder:
        """Return teacher encoder (usually better representations)."""
        return self.teacher_encoder


# ─────────────────────────────────────────────────────────────────────────────
# Retina-DINO Trainer
# ─────────────────────────────────────────────────────────────────────────────

def cosine_schedule(base_value: float, final_value: float,
                    total_steps: int, warmup_steps: int = 0) -> np.ndarray:
    """Cosine schedule with optional linear warmup."""
    warmup = np.linspace(0, base_value, warmup_steps)
    cos = final_value + 0.5 * (base_value - final_value) * (
        1 + np.cos(np.pi * np.arange(total_steps - warmup_steps) / (total_steps - warmup_steps))
    )
    return np.concatenate([warmup, cos])


class RetinaDINOTrainer:
    """
    Trainer for Retina-DINO self-supervised pretraining.

    Handles:
    - Learning rate scheduling (cosine + warmup)
    - Weight decay scheduling
    - EMA momentum scheduling
    - Gradient clipping
    - Last-layer freezing (early epochs)
    - Mixed precision training
    - Checkpoint saving
    """

    def __init__(self, model: RetinaDINO, config: RetinaDINOConfig,
                 dataloader, device: torch.device, save_dir: str = "checkpoints/dino"):
        self.model = model.to(device)
        self.config = config
        self.dataloader = dataloader
        self.device = device
        self.save_dir = save_dir

        # Steps
        steps_per_epoch = len(dataloader)
        total_steps = config.epochs * steps_per_epoch
        warmup_steps = config.warmup_epochs * steps_per_epoch

        # Schedules
        self.lr_schedule = cosine_schedule(
            config.base_lr, config.min_lr, total_steps, warmup_steps
        )
        self.wd_schedule = cosine_schedule(
            config.weight_decay, config.weight_decay_end, total_steps
        )
        self.momentum_schedule = cosine_schedule(
            config.momentum_teacher, config.momentum_teacher_end, total_steps
        )

        # Optimizer
        self.optimizer = torch.optim.AdamW(
            model.student_encoder.parameters(),
            lr=config.base_lr,
            weight_decay=config.weight_decay,
        )

        self.scaler = torch.cuda.amp.GradScaler()

    def train_epoch(self, epoch: int) -> float:
        self.model.train()
        total_loss = 0.0

        for step, (crops, _) in enumerate(self.dataloader):
            global_step = epoch * len(self.dataloader) + step

            # Update LR and WD
            for pg in self.optimizer.param_groups:
                pg['lr'] = self.lr_schedule[global_step]
                pg['weight_decay'] = self.wd_schedule[global_step]

            # Move crops to device
            crops = [c.to(self.device, non_blocking=True) for c in crops]

            # Forward + loss
            with autocast():
                loss = self.model(crops, epoch)

            # Backward
            self.optimizer.zero_grad()
            self.scaler.scale(loss).backward()

            # Gradient clipping
            if self.config.clip_grad:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.student_encoder.parameters(),
                    self.config.clip_grad
                )

            self.scaler.step(self.optimizer)
            self.scaler.update()

            # Freeze last layer for first N epochs
            if epoch < self.config.freeze_last_layer:
                for n, p in self.model.student_head.named_parameters():
                    if "last_layer" in n:
                        p.grad = None

            # EMA teacher update
            momentum = self.momentum_schedule[global_step]
            self.model.update_teacher(momentum)

            total_loss += loss.item()

        return total_loss / len(self.dataloader)

    def save_checkpoint(self, epoch: int, loss: float):
        import os
        os.makedirs(self.save_dir, exist_ok=True)
        torch.save({
            'epoch': epoch,
            'student_encoder': self.model.student_encoder.state_dict(),
            'teacher_encoder': self.model.teacher_encoder.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'loss': loss,
            'config': self.config,
        }, f"{self.save_dir}/retina_dino_epoch_{epoch:04d}.pt")

    def run(self):
        print("🔬 Starting Retina-DINO Self-Supervised Pretraining")
        print(f"   Encoder dim: {self.config.embed_dim}")
        print(f"   Prototype dim: {self.config.out_dim}")
        print(f"   Epochs: {self.config.epochs}")
        print(f"   Local crops: {self.config.local_crops_number}")
        print()

        for epoch in range(self.config.epochs):
            avg_loss = self.train_epoch(epoch)
            print(f"Epoch [{epoch+1:3d}/{self.config.epochs}]  Loss: {avg_loss:.6f}")
            if (epoch + 1) % 10 == 0:
                self.save_checkpoint(epoch, avg_loss)

        print("✅ Pretraining complete. Encoder ready for downstream tasks.")
