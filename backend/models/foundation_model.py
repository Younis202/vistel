"""
foundation_model.py — Retina-GPT Foundation Model Orchestrator
===============================================================
The master system that ties together ALL components of Retina-GPT:

    Retina-DINO     → Self-supervised pretraining backbone
    Retina-CLIP     → Vision-language alignment
    Retina-SAM      → Promptable segmentation
    Universal Embedding Engine → Central representation hub
    Multi-Task Heads → All downstream clinical tasks

Philosophy:
    Train once. Use everywhere.

    The foundation encoder is pretrained via Retina-DINO (no labels).
    Then frozen and used as the universal feature extractor.
    All task heads are fine-tuned on small labeled datasets.
    Retina-CLIP enables zero-shot disease queries via text.
    Retina-SAM enables any-structure segmentation via prompts.

Training stages:
    Stage 1: Retina-DINO pretraining (self-supervised, large unlabeled dataset)
    Stage 2: Retina-CLIP alignment (paired image-report data)
    Stage 3: Multi-task fine-tuning (labeled datasets, frozen backbone)
    Stage 4: Full system fine-tuning (end-to-end, small LR)

Clinical output per image:
    {
        "quality": {"score": 0.92, "adequate": True},
        "dr_grading": {"grade": 2, "label": "Moderate DR", "confidence": 0.87},
        "amd": {"present": False, "confidence": 0.94},
        "glaucoma_suspect": {"present": True, "confidence": 0.73},
        "lesions": {
            "microaneurysm": {"present": True, "count": 12, "mask": ...},
            "hemorrhage": {"present": True, "count": 3, "mask": ...},
            "hard_exudate": {"present": False},
            "cotton_wool": {"present": False},
        },
        "vessels": {"segmentation_mask": ...},
        "optic_disc": {"segmentation_mask": ..., "cup_disc_ratio": 0.62},
        "zero_shot_findings": {"moderate diabetic retinopathy": 0.87, ...},
        "report": "FINDINGS: Moderate diabetic retinopathy detected...",
    }
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Tuple, Any
import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# Foundation Model Config
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class RetinaFoundationConfig:
    """Master configuration for the full Retina-GPT Foundation Model."""

    # Vision backbone
    embed_dim: int = 1024
    depth: int = 12
    num_heads: int = 16
    patch_size: int = 16
    image_size: int = 224

    # Joint embedding space
    clip_dim: int = 512
    dino_out_dim: int = 65536

    # Feature pyramid
    fpn_out_dim: int = 256

    # Task heads
    num_dr_classes: int = 5     # Normal, Mild, Moderate, Severe, Proliferative
    num_amd_classes: int = 4    # Normal, Early, Intermediate, Late
    num_lesion_types: int = 6   # MA, hemorrhage, hard EX, soft EX, NV, drusen

    # Report generation
    report_lm_name: str = "microsoft/BioGPT-Large"
    max_report_length: int = 512
    num_visual_prefix_tokens: int = 16

    # Quality
    quality_threshold: float = 0.3

    # Training stages
    freeze_backbone_in_stage3: bool = True


# ─────────────────────────────────────────────────────────────────────────────
# Multi-Task Classification Heads
# ─────────────────────────────────────────────────────────────────────────────

class DRGradingHead(nn.Module):
    """
    Diabetic Retinopathy grading head.
    5-class ordinal classification: No DR → Proliferative DR.
    """

    GRADE_LABELS = {
        0: "No Diabetic Retinopathy",
        1: "Mild Non-Proliferative DR",
        2: "Moderate Non-Proliferative DR",
        3: "Severe Non-Proliferative DR",
        4: "Proliferative Diabetic Retinopathy",
    }

    REFERRAL_THRESHOLD = 2  # Grade ≥ 2 → refer to ophthalmologist

    def __init__(self, embed_dim: int = 1024, num_classes: int = 5, dropout: float = 0.2):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, 512),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(512, 128),
            nn.GELU(),
            nn.Linear(128, num_classes),
        )
        self.num_classes = num_classes

    def forward(self, embedding: torch.Tensor) -> Dict[str, torch.Tensor]:
        logits = self.classifier(embedding)
        probs = F.softmax(logits, dim=-1)
        grade = probs.argmax(dim=-1)
        confidence = probs.max(dim=-1).values
        return {
            "logits": logits,
            "probabilities": probs,
            "grade": grade,
            "confidence": confidence,
            "refer": grade >= self.REFERRAL_THRESHOLD,
        }

    def get_label(self, grade: int) -> str:
        return self.GRADE_LABELS.get(grade, "Unknown")


class AMDStagingHead(nn.Module):
    """Age-Related Macular Degeneration staging: Normal → Late AMD."""

    STAGE_LABELS = {
        0: "No AMD",
        1: "Early AMD (Small Drusen)",
        2: "Intermediate AMD (Medium Drusen / RPE Changes)",
        3: "Late AMD (Geographic Atrophy or Neovascular AMD)",
    }

    def __init__(self, embed_dim: int = 1024, num_classes: int = 4, dropout: float = 0.2):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

    def forward(self, embedding: torch.Tensor) -> Dict[str, torch.Tensor]:
        logits = self.classifier(embedding)
        probs = F.softmax(logits, dim=-1)
        stage = probs.argmax(dim=-1)
        return {
            "logits": logits,
            "probabilities": probs,
            "stage": stage,
            "confidence": probs.max(dim=-1).values,
        }


class GlaucomaSuspectHead(nn.Module):
    """Glaucoma suspect detection: binary with cup-to-disc ratio prediction."""

    def __init__(self, embed_dim: int = 1024, dropout: float = 0.2):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(embed_dim, 256),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.binary_head = nn.Linear(256, 2)   # Normal vs Suspect
        self.cdr_head = nn.Linear(256, 1)      # Cup-to-disc ratio regression

    def forward(self, embedding: torch.Tensor) -> Dict[str, torch.Tensor]:
        shared = self.shared(embedding)
        logits = self.binary_head(shared)
        probs = F.softmax(logits, dim=-1)
        cdr = torch.sigmoid(self.cdr_head(shared)).squeeze(-1) * 1.0   # CDR in [0,1]

        return {
            "logits": logits,
            "probabilities": probs,
            "suspect": probs[:, 1] > 0.5,
            "confidence": probs.max(dim=-1).values,
            "cup_disc_ratio": cdr,
        }


class MultiLesionDetectionHead(nn.Module):
    """
    Multi-label lesion presence head (global).
    Detects 6 lesion types simultaneously from the universal embedding.

    For spatial localization, use Retina-SAM with structure prompts.
    """

    LESION_TYPES = [
        "microaneurysm",
        "hemorrhage",
        "hard_exudate",
        "soft_exudate",      # cotton wool spot
        "neovascularization",
        "drusen",
    ]

    def __init__(self, embed_dim: int = 1024, num_lesions: int = 6, dropout: float = 0.2):
        super().__init__()
        self.heads = nn.ModuleDict({
            lesion: nn.Sequential(
                nn.Linear(embed_dim, 128),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(128, 1),
            )
            for lesion in self.LESION_TYPES
        })

    def forward(self, embedding: torch.Tensor) -> Dict[str, Dict]:
        results = {}
        for lesion, head in self.heads.items():
            logit = head(embedding).squeeze(-1)
            prob = torch.sigmoid(logit)
            results[lesion] = {
                "logit": logit,
                "probability": prob,
                "present": prob > 0.5,
            }
        return results


# ─────────────────────────────────────────────────────────────────────────────
# Clinical Report Generator
# ─────────────────────────────────────────────────────────────────────────────

class ClinicalReportGenerator(nn.Module):
    """
    Generates structured clinical reports from multi-task findings + visual embedding.

    Architecture:
        Universal Embedding
        + Task predictions (DR grade, AMD stage, lesion presence)
               ↓
        Visual prefix tokens (cross-attention Perceiver)
               ↓
        Medical Language Model (BioGPT)
               ↓
        Clinical report text

    Report structure:
        CLINICAL FINDINGS:
        [structured findings from multi-task heads]

        IMPRESSION:
        [LLM-generated clinical impression]

        RECOMMENDATION:
        [automated referral recommendation]
    """

    def __init__(self, embed_dim: int = 1024, lm_name: str = "microsoft/biogpt",
                 num_prefix_tokens: int = 16, max_length: int = 512,
                 use_lm: bool = True):
        super().__init__()
        self.num_prefix_tokens = num_prefix_tokens
        self.max_length = max_length
        self.use_lm = use_lm

        # Visual → LM projection (Perceiver-lite cross-attention)
        self.visual_queries = nn.Parameter(
            torch.randn(num_prefix_tokens, embed_dim)
        )
        self.cross_attention = nn.MultiheadAttention(
            embed_dim, num_heads=8, batch_first=True
        )

        # Load BioGPT if requested
        if use_lm:
            try:
                from transformers import AutoModelForCausalLM, AutoTokenizer
                self.tokenizer = AutoTokenizer.from_pretrained(lm_name)
                self.lm = AutoModelForCausalLM.from_pretrained(lm_name)
                lm_hidden = self.lm.config.hidden_size
                # Freeze most of LM
                for p in self.lm.parameters():
                    p.requires_grad = False
                # Unfreeze last 2 layers
                for layer in list(self.lm.parameters())[-20:]:
                    layer.requires_grad = True
                self.visual_projection = nn.Linear(embed_dim, lm_hidden)
            except Exception:
                self.tokenizer = None
                self.lm = None
                self.visual_projection = None
                self.use_lm = False
        else:
            self.tokenizer = None
            self.lm = None
            self.visual_projection = None

    def _build_structured_findings(self, task_results: Dict) -> str:
        """Build structured findings text from task head outputs."""
        lines = ["RETINAL FINDINGS:"]

        # DR grading
        if "dr" in task_results:
            dr = task_results["dr"]
            grade = dr.get("grade", 0)
            if isinstance(grade, torch.Tensor):
                grade = grade.item()
            conf = dr.get("confidence", 0)
            if isinstance(conf, torch.Tensor):
                conf = conf.item()
            label = DRGradingHead.GRADE_LABELS.get(grade, f"Grade {grade}")
            lines.append(f"  Diabetic Retinopathy: {label} (confidence: {conf:.1%})")
            if grade >= DRGradingHead.REFERRAL_THRESHOLD:
                lines.append("  ⚠ Ophthalmology referral recommended.")

        # AMD
        if "amd" in task_results:
            amd = task_results["amd"]
            stage = amd.get("stage", 0)
            if isinstance(stage, torch.Tensor):
                stage = stage.item()
            if stage > 0:
                label = AMDStagingHead.STAGE_LABELS.get(stage, f"Stage {stage}")
                lines.append(f"  AMD: {label}")

        # Glaucoma
        if "glaucoma" in task_results:
            g = task_results["glaucoma"]
            suspect = g.get("suspect", False)
            if isinstance(suspect, torch.Tensor):
                suspect = suspect.item()
            cdr = g.get("cup_disc_ratio", 0)
            if isinstance(cdr, torch.Tensor):
                cdr = cdr.item()
            status = "Suspect" if suspect else "No suspicion"
            lines.append(f"  Glaucoma: {status} | Cup-to-Disc Ratio: {cdr:.2f}")

        # Lesions
        if "lesions" in task_results:
            present_lesions = []
            for lesion, info in task_results["lesions"].items():
                present = info.get("present", False)
                if isinstance(present, torch.Tensor):
                    present = present.item()
                if present:
                    prob = info.get("probability", 0)
                    if isinstance(prob, torch.Tensor):
                        prob = prob.item()
                    present_lesions.append(f"{lesion.replace('_', ' ')} ({prob:.1%})")
            if present_lesions:
                lines.append(f"  Lesions detected: {', '.join(present_lesions)}")
            else:
                lines.append("  No significant lesions detected.")

        # Quality
        if "quality" in task_results:
            q = task_results["quality"]
            score = q.get("score", 1.0)
            if isinstance(score, torch.Tensor):
                score = score.item()
            adequate = q.get("adequate", True)
            if not adequate:
                lines.append(f"  ⚠ Image quality inadequate for reliable diagnosis (score: {score:.2f}).")

        return "\n".join(lines)

    def _build_recommendation(self, task_results: Dict) -> str:
        """Generate automated clinical recommendation."""
        urgent = False
        routine_referral = False
        rescreen = False

        if "dr" in task_results:
            grade = task_results["dr"].get("grade", 0)
            if isinstance(grade, torch.Tensor):
                grade = grade.item()
            if grade >= 4:
                urgent = True
            elif grade >= 2:
                routine_referral = True

        if "glaucoma" in task_results:
            suspect = task_results["glaucoma"].get("suspect", False)
            if isinstance(suspect, torch.Tensor):
                suspect = suspect.item()
            if suspect:
                routine_referral = True

        if "amd" in task_results:
            stage = task_results["amd"].get("stage", 0)
            if isinstance(stage, torch.Tensor):
                stage = stage.item()
            if stage >= 3:
                urgent = True
            elif stage >= 1:
                rescreen = True

        if urgent:
            return "RECOMMENDATION: Urgent ophthalmology referral required. Please expedite appointment within 1-2 weeks."
        elif routine_referral:
            return "RECOMMENDATION: Ophthalmology referral recommended within 3 months."
        elif rescreen:
            return "RECOMMENDATION: Repeat screening in 6-12 months."
        else:
            return "RECOMMENDATION: Routine screening in 12 months."

    def forward(
        self,
        embedding: torch.Tensor,
        patch_tokens: torch.Tensor,
        task_results: Dict,
    ) -> Dict[str, str]:
        """
        Generate clinical report.

        Args:
            embedding:    (B, D) universal retina embedding
            patch_tokens: (B, N, D) spatial patch features
            task_results: dict of task head outputs

        Returns:
            dict with keys: 'structured_findings', 'recommendation', 'full_report'
        """
        B = embedding.shape[0]

        # Build structured findings (rule-based, always available)
        findings = self._build_structured_findings(task_results)
        recommendation = self._build_recommendation(task_results)

        # LM-based impression (if model available)
        impression = ""
        if self.use_lm and self.lm is not None and self.tokenizer is not None:
            try:
                # Cross-attend visual queries to patch tokens
                queries = self.visual_queries.unsqueeze(0).expand(B, -1, -1)
                visual_prefix, _ = self.cross_attention(queries, patch_tokens, patch_tokens)
                visual_prefix = self.visual_projection(visual_prefix)  # (B, num_prefix, lm_hidden)

                prompt = (
                    f"{findings}\n\n"
                    "CLINICAL IMPRESSION:"
                )
                input_ids = self.tokenizer.encode(prompt, return_tensors='pt')
                input_ids = input_ids.to(embedding.device)

                output_ids = self.lm.generate(
                    input_ids,
                    max_new_tokens=200,
                    do_sample=False,
                    temperature=1.0,
                    pad_token_id=self.tokenizer.eos_token_id,
                )
                generated = self.tokenizer.decode(
                    output_ids[0, input_ids.shape[1]:],
                    skip_special_tokens=True
                ).strip()
                impression = f"\nCLINICAL IMPRESSION:\n{generated}"
            except Exception:
                impression = ""

        full_report = f"{findings}\n{impression}\n\n{recommendation}"

        return {
            "structured_findings": findings,
            "recommendation": recommendation,
            "impression": impression,
            "full_report": full_report,
        }


# ─────────────────────────────────────────────────────────────────────────────
# Retina-GPT Foundation Model
# ─────────────────────────────────────────────────────────────────────────────

class RetinaGPTFoundationModel(nn.Module):
    """
    Retina-GPT: The complete Retina AI Foundation Model.

    Single entry point for all retinal analysis tasks.

    Usage:
        model = RetinaGPTFoundationModel.from_pretrained("path/to/checkpoint")

        # Full analysis
        result = model.analyze(image)

        # Zero-shot query
        scores = model.query(image, "signs of diabetic macular edema")

        # Segment structure
        mask = model.segment(image, structure="optic_disc")
    """

    def __init__(
        self,
        config: RetinaFoundationConfig,
        report_generator: Optional[ClinicalReportGenerator] = None,
        use_sam: bool = False,
        use_clip: bool = False,
    ):
        super().__init__()
        self.config = config

        # ── Build Foundation Encoder ──────────────────────────────────────────
        # Import here to avoid circular imports
        from models.pretraining.retina_dino import RetinaFoundationEncoder, RetinaDINOConfig

        dino_config = RetinaDINOConfig(
            embed_dim=config.embed_dim,
            depth=config.depth,
            num_heads=config.num_heads,
            patch_size=config.patch_size,
            image_size=config.image_size,
        )
        self.encoder = RetinaFoundationEncoder(dino_config)

        # ── Universal Embedding Engine ────────────────────────────────────────
        from models.embedding.universal_embedding import UniversalRetinaEmbeddingEngine
        self.embedding_engine = UniversalRetinaEmbeddingEngine(
            foundation_encoder=self.encoder,
            embed_dim=config.embed_dim,
            fpn_out_dim=config.fpn_out_dim,
            clip_dim=config.clip_dim,
        )

        # ── Task Heads ────────────────────────────────────────────────────────
        self.dr_head = DRGradingHead(
            embed_dim=config.embed_dim,
            num_classes=config.num_dr_classes,
        )
        self.amd_head = AMDStagingHead(
            embed_dim=config.embed_dim,
            num_classes=config.num_amd_classes,
        )
        self.glaucoma_head = GlaucomaSuspectHead(embed_dim=config.embed_dim)
        self.lesion_head = MultiLesionDetectionHead(
            embed_dim=config.embed_dim,
            num_lesions=config.num_lesion_types,
        )

        # ── Report Generator ──────────────────────────────────────────────────
        self.report_generator = report_generator or ClinicalReportGenerator(
            embed_dim=config.embed_dim,
            lm_name=config.report_lm_name,
            num_prefix_tokens=config.num_visual_prefix_tokens,
            use_lm=False,  # Start with rule-based, enable when LM loaded
        )

        # ── Optional: Retina-SAM ──────────────────────────────────────────────
        self.sam = None
        if use_sam:
            from models.segmentation.retina_sam import RetinaSAM
            self.sam = RetinaSAM(
                foundation_encoder=self.encoder,
                encoder_embed_dim=config.embed_dim,
                decoder_embed_dim=config.fpn_out_dim,
                image_size=config.image_size,
            )

        # ── Optional: Retina-CLIP ─────────────────────────────────────────────
        self.clip = None
        if use_clip:
            from models.vision_language.retina_clip import RetinaCLIP
            self.clip = RetinaCLIP(
                foundation_encoder=self.encoder,
                encoder_embed_dim=config.embed_dim,
                joint_embed_dim=config.clip_dim,
            )

    def forward(self, image: torch.Tensor) -> Dict[str, Any]:
        """
        Full forward pass.

        Args:
            image: (B, 3, H, W) preprocessed retinal image

        Returns:
            dict with all task predictions and report
        """
        # Universal embedding
        emb_result = self.embedding_engine(image)
        embedding = emb_result.embedding
        patch_tokens = emb_result.patch_features

        # Task heads
        dr_out = self.dr_head(embedding)
        amd_out = self.amd_head(embedding)
        glaucoma_out = self.glaucoma_head(embedding)
        lesion_out = self.lesion_head(embedding)

        # Quality
        quality_out = {
            "score": emb_result.quality_score,
            "adequate": (emb_result.quality_score >= self.config.quality_threshold
                         if emb_result.quality_score is not None else True)
        }

        task_results = {
            "dr": dr_out,
            "amd": amd_out,
            "glaucoma": glaucoma_out,
            "lesions": lesion_out,
            "quality": quality_out,
        }

        # Clinical report
        report_out = self.report_generator(embedding, patch_tokens, task_results)

        return {
            "embedding": embedding,
            "dr": dr_out,
            "amd": amd_out,
            "glaucoma": glaucoma_out,
            "lesions": lesion_out,
            "quality": quality_out,
            "report": report_out,
            "multi_scale_features": {
                "p4": emb_result.scale4,
                "p8": emb_result.scale8,
                "p16": emb_result.scale16,
            }
        }

    @torch.no_grad()
    def analyze(self, image: torch.Tensor) -> Dict[str, Any]:
        """Inference-mode full analysis."""
        self.eval()
        return self.forward(image)

    @torch.no_grad()
    def query(self, image: torch.Tensor, text_query: str) -> float:
        """
        Zero-shot query: 'Does this retina show X?'
        Requires Retina-CLIP to be loaded.
        """
        if self.clip is None:
            raise RuntimeError("Retina-CLIP not loaded. Initialize with use_clip=True.")
        self.eval()
        sims = self.clip.similarity(image, [text_query])
        return sims[text_query]

    @torch.no_grad()
    def segment(
        self,
        image: torch.Tensor,
        structure: Optional[str] = None,
        points: Optional[Tuple] = None,
        boxes: Optional[torch.Tensor] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Promptable segmentation via Retina-SAM.
        Requires SAM to be loaded.
        """
        if self.sam is None:
            raise RuntimeError("Retina-SAM not loaded. Initialize with use_sam=True.")
        self.eval()
        return self.sam.predict(image, structure=structure, points=points, boxes=boxes)

    def freeze_backbone(self):
        """Freeze foundation encoder for task-head fine-tuning."""
        for p in self.encoder.parameters():
            p.requires_grad = False
        print("[RetinaGPT] Foundation encoder frozen.")

    def unfreeze_backbone(self, last_n_layers: int = 4):
        """Unfreeze last N transformer layers for full fine-tuning."""
        for p in self.encoder.parameters():
            p.requires_grad = False
        for block in self.encoder.blocks[-last_n_layers:]:
            for p in block.parameters():
                p.requires_grad = True
        print(f"[RetinaGPT] Unfroze last {last_n_layers} encoder layers.")

    def save_checkpoint(self, path: str, epoch: int = 0, metadata: Dict = None):
        """Save full model checkpoint."""
        torch.save({
            "epoch": epoch,
            "config": self.config,
            "model_state_dict": self.state_dict(),
            "metadata": metadata or {},
        }, path)
        print(f"[RetinaGPT] Checkpoint saved → {path}")

    @classmethod
    def from_pretrained(cls, checkpoint_path: str,
                        use_sam: bool = False, use_clip: bool = False) -> "RetinaGPTFoundationModel":
        """Load model from checkpoint."""
        ckpt = torch.load(checkpoint_path, map_location='cpu')
        config = ckpt.get("config", RetinaFoundationConfig())
        model = cls(config, use_sam=use_sam, use_clip=use_clip)
        model.load_state_dict(ckpt["model_state_dict"], strict=False)
        print(f"[RetinaGPT] Loaded checkpoint from {checkpoint_path}")
        return model

    @classmethod
    def build_base(cls) -> "RetinaGPTFoundationModel":
        """Quick factory: base-size foundation model."""
        return cls(RetinaFoundationConfig(
            embed_dim=768, depth=12, num_heads=12
        ))

    @classmethod
    def build_large(cls) -> "RetinaGPTFoundationModel":
        """Quick factory: large foundation model."""
        return cls(RetinaFoundationConfig(
            embed_dim=1024, depth=24, num_heads=16
        ))

    def parameter_count(self) -> Dict[str, int]:
        """Return parameter count breakdown."""
        def count(m):
            return sum(p.numel() for p in m.parameters())

        breakdown = {
            "encoder": count(self.encoder),
            "embedding_engine": count(self.embedding_engine),
            "dr_head": count(self.dr_head),
            "amd_head": count(self.amd_head),
            "glaucoma_head": count(self.glaucoma_head),
            "lesion_head": count(self.lesion_head),
            "report_generator": count(self.report_generator),
        }
        if self.sam:
            breakdown["sam"] = count(self.sam)
        if self.clip:
            breakdown["clip"] = count(self.clip)

        breakdown["total"] = sum(breakdown.values())
        return breakdown

    def __repr__(self) -> str:
        counts = self.parameter_count()
        total_M = counts["total"] / 1e6
        return (
            f"RetinaGPTFoundationModel(\n"
            f"  embed_dim={self.config.embed_dim}, "
            f"depth={self.config.depth}, "
            f"heads={self.config.num_heads}\n"
            f"  Components: DINO encoder + Universal Embedding + 4 Task Heads + Report\n"
            f"  SAM: {'✓' if self.sam else '✗'} | CLIP: {'✓' if self.clip else '✗'}\n"
            f"  Total parameters: {total_M:.1f}M\n"
            f")"
        )
