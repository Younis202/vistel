"""
universal_embedding.py — Universal Retina Embedding Engine
===========================================================
The Universal Retina Embedding is the central component of Retina-GPT.

It acts as a hub connecting:
    Retina-DINO (self-supervised pretraining)
    Retina-CLIP  (vision-language alignment)
    Retina-SAM   (promptable segmentation)
    Task Heads   (classification, detection, segmentation, QA)
    Report Generator (clinical text generation)

A single retinal image is compressed into a 1024-dimensional vector that
encodes the complete semantic content of the fundus:
    • Structural anatomy (vessels, disc, macula)
    • Pathological findings (lesions, hemorrhages, exudates)
    • Quality metrics
    • Disease state embeddings

This design mirrors how biological memory works — one compact representation
that can answer any question about the image downstream.

Usage:
    engine = UniversalRetinaEmbeddingEngine(config)
    result = engine.encode(image_tensor)

    result.embedding          # 1024-dim universal vector
    result.structural_features  # vascular / anatomical spatial features
    result.quality_score         # image quality (0-1)
    result.patch_features        # full (H/P, W/P, D) feature map for SAM
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Tuple
import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# Embedding Result Container
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class RetinaEmbeddingResult:
    """
    Rich output from the Universal Retina Embedding Engine.

    All fields are on the original device unless .cpu() is called.
    """

    # Core universal embedding — used by ALL downstream tasks
    embedding: torch.Tensor              # (B, 1024) normalized embedding

    # Spatial features for dense prediction tasks (SAM, segmentation)
    patch_features: torch.Tensor         # (B, N_patches, 1024)

    # Multi-scale feature maps for detection and segmentation
    scale4: Optional[torch.Tensor] = None   # 1/4 resolution: (B, D, H/4, W/4)
    scale8: Optional[torch.Tensor] = None   # 1/8 resolution: (B, D, H/8, W/8)
    scale16: Optional[torch.Tensor] = None  # 1/16 resolution: (B, D, H/16, W/16)

    # Quality assessment pre-embedding
    quality_score: Optional[torch.Tensor] = None   # (B,) in [0, 1]

    # Per-task projections (populated by task-specific encoders)
    clip_embedding: Optional[torch.Tensor] = None   # (B, 512) CLIP space
    dino_features: Optional[torch.Tensor] = None    # (B, 65536) DINO prototype space

    def to(self, device: torch.device) -> "RetinaEmbeddingResult":
        """Move all tensors to device."""
        def _move(t):
            return t.to(device) if t is not None else None
        return RetinaEmbeddingResult(
            embedding=_move(self.embedding),
            patch_features=_move(self.patch_features),
            scale4=_move(self.scale4),
            scale8=_move(self.scale8),
            scale16=_move(self.scale16),
            quality_score=_move(self.quality_score),
            clip_embedding=_move(self.clip_embedding),
            dino_features=_move(self.dino_features),
        )

    def cpu(self) -> "RetinaEmbeddingResult":
        return self.to(torch.device("cpu"))

    def numpy(self) -> Dict[str, Optional[np.ndarray]]:
        """Convert all tensors to numpy arrays."""
        def _np(t):
            return t.detach().cpu().numpy() if t is not None else None
        return {
            "embedding": _np(self.embedding),
            "patch_features": _np(self.patch_features),
            "quality_score": _np(self.quality_score),
            "clip_embedding": _np(self.clip_embedding),
        }


# ─────────────────────────────────────────────────────────────────────────────
# Image Quality Assessor (pre-embedding gate)
# ─────────────────────────────────────────────────────────────────────────────

class RetinaQualityAssessor(nn.Module):
    """
    Lightweight quality gate applied before embedding.

    Flags and scores images for:
        • Focus/blur quality
        • Illumination quality
        • Field of view completeness
        • Artifact presence (lens reflections, dust)

    Poor quality images (score < 0.3) are flagged so clinical
    systems can reject them before diagnosis.
    """

    def __init__(self, embed_dim: int = 1024):
        super().__init__()

        # Shallow quality head from patch statistics
        self.quality_head = nn.Sequential(
            nn.Linear(embed_dim, 256),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(256, 64),
            nn.GELU(),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

        # Per-aspect quality scores
        self.aspect_head = nn.Sequential(
            nn.Linear(embed_dim, 128),
            nn.GELU(),
            nn.Linear(128, 4),   # [focus, illumination, fov, artifacts]
            nn.Sigmoid(),
        )

        self.quality_threshold = 0.3

    def forward(self, cls_token: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            cls_token: (B, D) CLS token from encoder

        Returns:
            overall_quality: (B,) in [0, 1]
            aspect_scores:   (B, 4) per-aspect scores
            is_adequate:     (B,) bool mask — True if quality adequate for diagnosis
        """
        overall = self.quality_head(cls_token).squeeze(-1)    # (B,)
        aspects = self.aspect_head(cls_token)                  # (B, 4)
        is_adequate = overall >= self.quality_threshold        # (B,)
        return overall, aspects, is_adequate


# ─────────────────────────────────────────────────────────────────────────────
# Multi-Scale Feature Pyramid
# ─────────────────────────────────────────────────────────────────────────────

class RetinaFeaturePyramid(nn.Module):
    """
    Builds a multi-scale feature pyramid from ViT patch tokens.

    ViT produces flat patch tokens (no inherent scale hierarchy).
    We reconstruct spatial maps at multiple scales for dense tasks.

    Output scales:
        P4:  56x56  (4x upsampled from 14x14 patches)
        P8:  28x28
        P16: 14x14  (native patch grid)
    """

    def __init__(self, embed_dim: int = 1024, out_dim: int = 256,
                 patch_grid_size: int = 14):
        super().__init__()
        self.patch_grid_size = patch_grid_size

        # Lateral projections: embed_dim → out_dim
        self.lateral_p16 = nn.Conv2d(embed_dim, out_dim, 1)
        self.lateral_p8  = nn.Conv2d(embed_dim, out_dim, 1)
        self.lateral_p4  = nn.Conv2d(embed_dim, out_dim, 1)

        # Smooth outputs
        self.smooth_p16 = nn.Conv2d(out_dim, out_dim, 3, padding=1)
        self.smooth_p8  = nn.Conv2d(out_dim, out_dim, 3, padding=1)
        self.smooth_p4  = nn.Conv2d(out_dim, out_dim, 3, padding=1)

    def forward(self, patch_tokens: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            patch_tokens: (B, N, D) where N = patch_grid_size^2

        Returns:
            p16: (B, out_dim, 14, 14)
            p8:  (B, out_dim, 28, 28)
            p4:  (B, out_dim, 56, 56)
        """
        B, N, D = patch_tokens.shape
        H = W = self.patch_grid_size

        # Reshape to spatial
        feat = patch_tokens.reshape(B, H, W, D).permute(0, 3, 1, 2)  # (B, D, H, W)

        # P16 (native)
        p16 = self.smooth_p16(self.lateral_p16(feat))

        # P8 (2x upsampled)
        p8 = self.smooth_p8(
            self.lateral_p8(F.interpolate(feat, scale_factor=2, mode='bilinear', align_corners=False))
        )

        # P4 (4x upsampled)
        p4 = self.smooth_p4(
            self.lateral_p4(F.interpolate(feat, scale_factor=4, mode='bilinear', align_corners=False))
        )

        return p16, p8, p4


# ─────────────────────────────────────────────────────────────────────────────
# Universal Retina Embedding Engine
# ─────────────────────────────────────────────────────────────────────────────

class UniversalRetinaEmbeddingEngine(nn.Module):
    """
    The central hub of Retina-GPT.

    Takes a retinal fundus image and produces a rich, multi-purpose
    embedding structure that powers all downstream tasks:

    ┌────────────────────────────────────────────────────────────┐
    │                   Retinal Image                            │
    │                         ↓                                 │
    │             Foundation Encoder (ViT)                       │
    │                    ↙          ↘                           │
    │            CLS Token      Patch Tokens                     │
    │               ↓                ↓                          │
    │       Universal Emb.     Feature Pyramid                   │
    │       Quality Score      Multi-Scale Maps                  │
    │       CLIP Projection    SAM-ready features                │
    └────────────────────────────────────────────────────────────┘

    All downstream task heads receive their input from here.
    No task head ever touches the raw image — only the embedding.
    """

    def __init__(
        self,
        foundation_encoder: nn.Module,
        embed_dim: int = 1024,
        fpn_out_dim: int = 256,
        clip_dim: int = 512,
        patch_grid_size: int = 14,
        quality_gate: bool = True,
    ):
        super().__init__()

        self.encoder = foundation_encoder
        self.embed_dim = embed_dim
        self.patch_grid_size = patch_grid_size
        self.quality_gate = quality_gate

        # Quality assessor
        if quality_gate:
            self.quality_assessor = RetinaQualityAssessor(embed_dim=embed_dim)

        # Feature pyramid for dense tasks
        self.fpn = RetinaFeaturePyramid(
            embed_dim=embed_dim,
            out_dim=fpn_out_dim,
            patch_grid_size=patch_grid_size,
        )

        # CLIP projection (aligned with Retina-CLIP training)
        self.clip_projection = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, clip_dim),
            nn.LayerNorm(clip_dim),
        )

        # Embedding normalization
        self.embedding_norm = nn.LayerNorm(embed_dim)

    def _extract_tokens(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Run foundation encoder and extract CLS + patch tokens.

        Returns:
            cls_token:    (B, D)
            patch_tokens: (B, N, D)
        """
        B = x.shape[0]
        tokens = self.encoder.patch_embed(x)                # (B, N, D)
        cls = self.encoder.cls_token.expand(B, -1, -1)
        tokens = torch.cat([cls, tokens], dim=1)
        tokens = tokens + self.encoder.pos_embed

        # Register tokens (appended at end)
        reg = self.encoder.register_tokens.expand(B, -1, -1)
        tokens = torch.cat([tokens, reg], dim=1)

        for block in self.encoder.blocks:
            tokens = block(tokens)
        tokens = self.encoder.norm(tokens)

        # CLS token (index 0), patch tokens (indices 1 : 1+N)
        cls_token = tokens[:, 0, :]                         # (B, D)
        N = self.patch_grid_size ** 2
        patch_tokens = tokens[:, 1:1+N, :]                 # (B, N, D)

        return cls_token, patch_tokens

    def forward(self, image: torch.Tensor) -> RetinaEmbeddingResult:
        """
        Full embedding extraction pipeline.

        Args:
            image: (B, 3, H, W) preprocessed retinal image

        Returns:
            RetinaEmbeddingResult with all features populated
        """
        # Extract foundation features
        cls_token, patch_tokens = self._extract_tokens(image)  # (B, D), (B, N, D)

        # Universal embedding (normalized CLS)
        embedding = self.embedding_norm(cls_token)              # (B, D)

        # Feature pyramid
        p16, p8, p4 = self.fpn(patch_tokens)

        # CLIP projection
        clip_emb = F.normalize(self.clip_projection(cls_token), dim=-1)

        # Quality assessment
        quality_score = None
        if self.quality_gate:
            quality_score, _, _ = self.quality_assessor(cls_token)

        return RetinaEmbeddingResult(
            embedding=embedding,
            patch_features=patch_tokens,
            scale4=p4,
            scale8=p8,
            scale16=p16,
            quality_score=quality_score,
            clip_embedding=clip_emb,
        )

    @torch.no_grad()
    def encode(self, image: torch.Tensor) -> RetinaEmbeddingResult:
        """Inference-only encoding (no gradients)."""
        return self.forward(image)

    def compute_similarity(
        self,
        image_a: torch.Tensor,
        image_b: torch.Tensor,
        space: str = "clip"
    ) -> torch.Tensor:
        """
        Compute similarity between two retinal images.

        Args:
            image_a: (B, 3, H, W)
            image_b: (B, 3, H, W)
            space: 'clip' for normalized space, 'embedding' for raw

        Returns:
            (B,) cosine similarity scores
        """
        with torch.no_grad():
            emb_a = self.encode(image_a)
            emb_b = self.encode(image_b)

        if space == "clip":
            a = emb_a.clip_embedding
            b = emb_b.clip_embedding
        else:
            a = F.normalize(emb_a.embedding, dim=-1)
            b = F.normalize(emb_b.embedding, dim=-1)

        return (a * b).sum(dim=-1)  # (B,) cosine similarity


# ─────────────────────────────────────────────────────────────────────────────
# Retina Embedding Database (in-memory retrieval index)
# ─────────────────────────────────────────────────────────────────────────────

class RetinaEmbeddingDatabase:
    """
    Efficient in-memory embedding database for retinal image retrieval.

    Stores:
        - Embeddings (torch.Tensor)
        - Image IDs and metadata
        - Pre-computed norms for fast cosine similarity

    Useful for:
        - Patient monitoring (compare with baseline images)
        - Similar case retrieval for clinical decision support
        - Quality screening of large datasets
    """

    def __init__(self, embed_dim: int = 1024):
        self.embed_dim = embed_dim
        self.embeddings: Optional[torch.Tensor] = None
        self.ids: List = []
        self.metadata: List[Dict] = []

    def add(self, embedding: torch.Tensor, id_: str, metadata: Dict = None):
        """Add a single embedding to the database."""
        emb = F.normalize(embedding.detach().cpu(), dim=-1)

        if self.embeddings is None:
            self.embeddings = emb.unsqueeze(0)
        else:
            self.embeddings = torch.cat([self.embeddings, emb.unsqueeze(0)], dim=0)

        self.ids.append(id_)
        self.metadata.append(metadata or {})

    def add_batch(self, embeddings: torch.Tensor, ids: List[str],
                  metadata: List[Dict] = None):
        """Add a batch of embeddings."""
        embs = F.normalize(embeddings.detach().cpu(), dim=-1)

        if self.embeddings is None:
            self.embeddings = embs
        else:
            self.embeddings = torch.cat([self.embeddings, embs], dim=0)

        self.ids.extend(ids)
        self.metadata.extend(metadata or [{} for _ in ids])

    def search(self, query: torch.Tensor, k: int = 5) -> List[Dict]:
        """
        Search for k most similar embeddings.

        Returns:
            List of dicts: [{id, score, metadata}, ...]
        """
        if self.embeddings is None:
            return []

        q = F.normalize(query.detach().cpu(), dim=-1)
        sims = (q @ self.embeddings.T).squeeze(0)
        top_scores, top_indices = sims.topk(min(k, len(self.ids)))

        results = []
        for score, idx in zip(top_scores.tolist(), top_indices.tolist()):
            results.append({
                "id": self.ids[idx],
                "score": score,
                "metadata": self.metadata[idx],
            })

        return results

    def save(self, path: str):
        """Save database to disk."""
        torch.save({
            "embeddings": self.embeddings,
            "ids": self.ids,
            "metadata": self.metadata,
            "embed_dim": self.embed_dim,
        }, path)
        print(f"[RetinaEmbeddingDatabase] Saved {len(self.ids)} embeddings → {path}")

    @classmethod
    def load(cls, path: str) -> "RetinaEmbeddingDatabase":
        """Load database from disk."""
        data = torch.load(path, map_location='cpu')
        db = cls(embed_dim=data['embed_dim'])
        db.embeddings = data['embeddings']
        db.ids = data['ids']
        db.metadata = data['metadata']
        print(f"[RetinaEmbeddingDatabase] Loaded {len(db.ids)} embeddings from {path}")
        return db

    def __len__(self) -> int:
        return len(self.ids)

    def __repr__(self) -> str:
        return f"RetinaEmbeddingDatabase({len(self.ids)} entries, dim={self.embed_dim})"
