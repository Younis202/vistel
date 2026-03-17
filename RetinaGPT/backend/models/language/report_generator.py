"""
models/language/report_generator.py
=====================================
Retina-GPT: Medical Report Generator (Vision-Language Bridge)

Implements a vision-language model that generates structured clinical reports
from retinal embeddings. Architecture:

  ViT Embedding → Linear Projection → Visual Prefix Tokens
       → Concatenate with Clinical Prompt Tokens
       → Decoder-only Language Model (GPT-2 / BioMedLM / Llama-3)
       → Structured Clinical Report

The report follows the standard ophthalmology reporting structure:
  - Image Quality
  - Findings (vessel, disc, lesions)
  - Impression (DR grade, severity)
  - Recommendations

Author: Retina-GPT Engineering Team
Date: 2026
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    GenerationConfig,
    PreTrainedModel,
    PreTrainedTokenizer,
)

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────

@dataclass
class ReportGeneratorConfig:
    """
    Configuration for the medical report generator.
    
    Supports any HuggingFace CausalLM as the language backbone.
    Recommended models:
        - 'stanford-crfm/BioMedLM'          — biomedical domain (2.7B)
        - 'microsoft/BiomedNLP-BiomedBERT-*' — encoder (for retrieval augmented)
        - 'meta-llama/Llama-3.2-3B'          — general purpose LLM
        - 'gpt2-medium'                       — lightweight baseline
    """
    # Language model backbone
    lm_model_id: str = "gpt2-medium"

    # Visual encoder dimension (must match RetinaViT embed_dim)
    visual_embed_dim: int = 768

    # Number of visual prefix tokens (visual soft prompts)
    num_visual_tokens: int = 16

    # Generation settings
    max_new_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.9
    repetition_penalty: float = 1.2
    do_sample: bool = True

    # Training
    freeze_lm_backbone: bool = True      # Only train visual projection layer
    visual_proj_layers: int = 2          # Depth of visual→language projection MLP


# ─────────────────────────────────────────────────────────────
# Clinical Prompt Templates
# ─────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are an expert ophthalmologist generating a structured clinical report from a retinal fundus image analysis.

Generate a report with the following sections:
1. IMAGE QUALITY: [Gradable/Ungradable + justification]
2. FINDINGS: [Describe retinal structures — vessels, optic disc, macula, lesions]  
3. IMPRESSION: [Diabetic retinopathy grade and severity]
4. RECOMMENDATIONS: [Clinical action plan]

Be precise, clinical, and concise. Use standard ophthalmology terminology."""

DR_GRADE_DESCRIPTIONS = {
    0: "No diabetic retinopathy changes identified.",
    1: "Mild non-proliferative diabetic retinopathy (NPDR) — few microaneurysms.",
    2: "Moderate NPDR — microaneurysms, dot-blot hemorrhages, and/or hard exudates.",
    3: "Severe NPDR — extensive hemorrhages in 4+ quadrants or venous beading.",
    4: "Proliferative diabetic retinopathy (PDR) — neovascularization present.",
}


def build_clinical_prompt(
    dr_grade: Optional[int] = None,
    quality: Optional[int] = None,
    has_vessels: bool = True,
    lesion_types: Optional[List[str]] = None,
) -> str:
    """
    Build a structured clinical prompt incorporating structured predictions
    from the multi-task heads to guide report generation.

    Args:
        dr_grade: Predicted DR grade [0-4]
        quality: Predicted image quality [0=ungradable, 1=gradable]
        has_vessels: Whether vessel segmentation found vessels
        lesion_types: List of detected lesion types

    Returns:
        Formatted prompt string
    """
    prompt_parts = [SYSTEM_PROMPT, "\n\n--- RETINAL ANALYSIS REPORT ---\n"]

    # Quality section
    if quality is not None:
        q_str = "GRADABLE — adequate quality for clinical interpretation" if quality == 1 \
                else "UNGRADABLE — image quality insufficient for reliable interpretation"
        prompt_parts.append(f"\nIMAGE QUALITY: {q_str}")

    # Findings section
    findings = []
    if has_vessels:
        findings.append("Retinal vessel architecture is visible")
    if lesion_types:
        findings.append(f"Lesions identified: {', '.join(lesion_types)}")
    if findings:
        prompt_parts.append(f"\nFINDINGS: {'. '.join(findings)}.")

    # Impression
    if dr_grade is not None:
        grade_desc = DR_GRADE_DESCRIPTIONS.get(dr_grade, "Grade indeterminate")
        prompt_parts.append(f"\nIMPRESSION: DR Grade {dr_grade}/4. {grade_desc}")

    prompt_parts.append("\nDETAILED REPORT:\n")
    return "\n".join(prompt_parts)


# ─────────────────────────────────────────────────────────────
# Visual Projection Layer
# ─────────────────────────────────────────────────────────────

class VisualProjection(nn.Module):
    """
    Projects ViT visual embeddings into the language model's token space.
    
    Converts the high-dimensional visual representation into a sequence of
    'visual prefix tokens' that are prepended to the text prompt, allowing
    the language model to condition its generation on image content.
    
    Design: Perceiver-lite style compression
        - Cross-attend N_visual query tokens over all patch tokens
        - Project to LM embedding dim
    
    Input:  (B, N_patches, visual_dim) — patch tokens from ViT
    Output: (B, N_visual_tokens, lm_dim) — visual prefix for LM
    """

    def __init__(
        self,
        visual_dim: int,
        lm_dim: int,
        num_visual_tokens: int = 16,
        num_layers: int = 2,
    ):
        super().__init__()
        self.num_visual_tokens = num_visual_tokens

        # Learnable query vectors (one per visual token slot)
        self.queries = nn.Parameter(torch.randn(1, num_visual_tokens, visual_dim))

        # Cross-attention: queries attend to patch tokens
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=visual_dim,
            num_heads=8,
            batch_first=True,
        )
        self.norm_q = nn.LayerNorm(visual_dim)
        self.norm_kv = nn.LayerNorm(visual_dim)

        # MLP projection to LM embedding dimension
        hidden = max(visual_dim, lm_dim)
        projection_layers = []
        in_dim = visual_dim
        for i in range(num_layers):
            out_dim = lm_dim if i == num_layers - 1 else hidden
            projection_layers.extend([
                nn.Linear(in_dim, out_dim),
                nn.GELU() if i < num_layers - 1 else nn.Identity(),
            ])
            in_dim = out_dim
        self.proj = nn.Sequential(*projection_layers)
        self.norm_out = nn.LayerNorm(lm_dim)

        nn.init.trunc_normal_(self.queries, std=0.02)

    def forward(self, patch_tokens: torch.Tensor) -> torch.Tensor:
        """
        Args:
            patch_tokens: (B, N_patches, visual_dim)
        Returns:
            visual_prefix: (B, num_visual_tokens, lm_dim)
        """
        B = patch_tokens.shape[0]
        q = self.queries.expand(B, -1, -1)           # (B, V, D_vis)

        kv = self.norm_kv(patch_tokens)
        q  = self.norm_q(q)

        # Cross-attention: query tokens attend to all patch tokens
        attended, _ = self.cross_attn(query=q, key=kv, value=kv)  # (B, V, D_vis)

        # Project to LM embedding space
        visual_prefix = self.norm_out(self.proj(attended))          # (B, V, D_lm)
        return visual_prefix


# ─────────────────────────────────────────────────────────────
# Medical Report Generator
# ─────────────────────────────────────────────────────────────

class MedicalReportGenerator(nn.Module):
    """
    Vision-Language Medical Report Generator.
    
    Combines the RetinaViT backbone's visual embeddings with a pretrained
    language model to generate structured clinical ophthalmology reports.
    
    Architecture:
        ViT patch tokens → VisualProjection → visual prefix (V tokens)
        Clinical prompt → tokenize → prompt embeddings (T tokens)
        [visual prefix | prompt embeddings] → CausalLM → report tokens
    
    Training strategy:
        - Phase 1: Train only VisualProjection (LM frozen) — visual alignment
        - Phase 2: Fine-tune full model with LoRA — report quality
    
    Usage:
        >>> model = MedicalReportGenerator(config)
        >>> backbone_out = retina_vit(images)
        >>> report = model.generate_report(backbone_out, dr_grade=2)
        >>> print(report)
    """

    def __init__(self, config: Optional[ReportGeneratorConfig] = None):
        super().__init__()
        self.config = config or ReportGeneratorConfig()
        cfg = self.config

        logger.info(f"Loading language model: {cfg.lm_model_id}")

        # ── Load tokenizer ──
        self.tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(
            cfg.lm_model_id
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # ── Load language model ──
        self.lm: PreTrainedModel = AutoModelForCausalLM.from_pretrained(
            cfg.lm_model_id,
            torch_dtype=torch.float32,
        )
        lm_embed_dim = self.lm.config.hidden_size

        # ── Freeze LM if configured ──
        if cfg.freeze_lm_backbone:
            for param in self.lm.parameters():
                param.requires_grad = False
            logger.info("LM backbone frozen. Only training VisualProjection.")

        # ── Visual projection layer (always trainable) ──
        self.visual_proj = VisualProjection(
            visual_dim=cfg.visual_embed_dim,
            lm_dim=lm_embed_dim,
            num_visual_tokens=cfg.num_visual_tokens,
            num_layers=cfg.visual_proj_layers,
        )

        n_trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        logger.info(f"MedicalReportGenerator ready | trainable params: {n_trainable/1e6:.2f}M")

    def forward(
        self,
        backbone_output: Dict[str, torch.Tensor],
        prompt_ids: torch.Tensor,
        prompt_mask: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Training forward pass.

        Args:
            backbone_output: ViT output dict (needs 'patch_tokens')
            prompt_ids: Tokenized prompt (B, T) — text context
            prompt_mask: Attention mask for prompt (B, T)
            labels: Target token ids for language modeling loss (B, T), -100 for ignored

        Returns:
            dict with 'loss' (if labels provided) and 'logits' (B, V+T, vocab_size)
        """
        patch_tokens = backbone_output["patch_tokens"]  # (B, N, D_vis)

        # Project visual tokens to LM embedding space
        visual_prefix = self.visual_proj(patch_tokens)  # (B, V, D_lm)

        # Get LM token embeddings for text prompt
        embed_layer = self.lm.get_input_embeddings()
        text_embeds = embed_layer(prompt_ids)            # (B, T, D_lm)

        # Concatenate: [visual prefix | text prompt]
        full_embeds = torch.cat([visual_prefix, text_embeds], dim=1)   # (B, V+T, D_lm)

        # Build extended attention mask
        B, V = visual_prefix.shape[:2]
        visual_mask = torch.ones(B, V, device=prompt_mask.device, dtype=prompt_mask.dtype)
        full_mask = torch.cat([visual_mask, prompt_mask], dim=1)        # (B, V+T)

        # Adjust labels if provided (shift by V to account for visual prefix)
        if labels is not None:
            visual_labels = torch.full((B, V), -100, device=labels.device, dtype=labels.dtype)
            full_labels = torch.cat([visual_labels, labels], dim=1)     # (B, V+T)
        else:
            full_labels = None

        # Language model forward pass
        lm_output = self.lm(
            inputs_embeds=full_embeds,
            attention_mask=full_mask,
            labels=full_labels,
            return_dict=True,
        )

        return {
            "loss": lm_output.loss,
            "logits": lm_output.logits,
        }

    @torch.no_grad()
    def generate_report(
        self,
        backbone_output: Dict[str, torch.Tensor],
        dr_grade: Optional[int] = None,
        quality: Optional[int] = None,
        lesion_types: Optional[List[str]] = None,
        device: Optional[torch.device] = None,
    ) -> str:
        """
        Generate a clinical report from backbone embeddings.

        Args:
            backbone_output: ViT output dict
            dr_grade: Predicted DR grade for prompt conditioning
            quality: Predicted image quality
            lesion_types: Detected lesion type names
            device: Target device

        Returns:
            Generated clinical report as string
        """
        self.eval()
        cfg = self.config

        if device is None:
            device = next(self.parameters()).device

        # Build structured prompt
        prompt_text = build_clinical_prompt(
            dr_grade=dr_grade,
            quality=quality,
            lesion_types=lesion_types,
        )

        # Tokenize prompt
        prompt_enc = self.tokenizer(
            prompt_text,
            return_tensors="pt",
            truncation=True,
            max_length=512,
        )
        prompt_ids = prompt_enc["input_ids"].to(device)
        prompt_mask = prompt_enc["attention_mask"].to(device)

        # Build input embeddings
        patch_tokens = backbone_output["patch_tokens"].to(device)
        visual_prefix = self.visual_proj(patch_tokens)           # (B, V, D_lm)
        embed_layer = self.lm.get_input_embeddings()
        text_embeds = embed_layer(prompt_ids)                    # (B, T, D_lm)
        full_embeds = torch.cat([visual_prefix, text_embeds], dim=1)

        B, V = visual_prefix.shape[:2]
        visual_mask = torch.ones(B, V, device=device, dtype=torch.long)
        full_mask = torch.cat([visual_mask, prompt_mask], dim=1)

        # Generate
        gen_config = GenerationConfig(
            max_new_tokens=cfg.max_new_tokens,
            temperature=cfg.temperature,
            top_p=cfg.top_p,
            repetition_penalty=cfg.repetition_penalty,
            do_sample=cfg.do_sample,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )

        output_ids = self.lm.generate(
            inputs_embeds=full_embeds,
            attention_mask=full_mask,
            generation_config=gen_config,
        )

        # Decode: skip prompt tokens (first T+V tokens)
        generated = output_ids[0, full_embeds.shape[1]:]
        report_text = self.tokenizer.decode(generated, skip_special_tokens=True)

        return report_text.strip()
