"""
retina_time.py — Retina-TIME: Temporal Intelligence for Medical Evolution
=========================================================================
Retina-TIME analyzes longitudinal retinal data — multiple patient visits
over time — to detect disease PROGRESSION, not just current status.

This is the capability that makes Retina-GPT genuinely useful for
ongoing patient management rather than single-point screening.

Architecture:
    Visit 1 image  →  RetinaViT  →  embedding_t1
    Visit 2 image  →  RetinaViT  →  embedding_t2
    Visit 3 image  →  RetinaViT  →  embedding_t3
                             ↓
            Temporal Transformer (processes sequence of embeddings)
                             ↓
            Disease Progression Head
                             ↓
    ┌────────────────────────────────────────────┐
    │  Progression Report:                       │
    │  DR Grade: Mild → Moderate (WORSENING)     │
    │  New lesions: 3 new microaneurysms detected│
    │  Risk 12-month: HIGH (0.82)                │
    └────────────────────────────────────────────┘

Capabilities:
    • Progression detection (stable / improving / worsening)
    • New lesion detection between visits
    • Rate-of-change estimation
    • 12-month risk prediction
    • Temporal attention visualization (which visits matter most)
    • Longitudinal clinical report generation
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Tuple
from datetime import datetime, date


# ─────────────────────────────────────────────────────────────────────────────
# Data Structures
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class VisitData:
    """Single patient visit record."""
    visit_date:   str          # ISO format: "2024-03-15"
    image_path:   str          # Path to fundus image
    embedding:    Optional[torch.Tensor] = None   # (1024,) — filled after encoding
    diagnosis:    Optional[Dict] = None           # Task head outputs
    quality_score: float = 1.0
    notes:        str = ""

    def days_since(self, other: "VisitData") -> int:
        """Days elapsed since another visit."""
        try:
            d1 = datetime.fromisoformat(self.visit_date).date()
            d2 = datetime.fromisoformat(other.visit_date).date()
            return abs((d1 - d2).days)
        except Exception:
            return 0


@dataclass
class ProgressionReport:
    """Clinical progression report for a patient across visits."""

    patient_id:       str
    num_visits:       int
    visit_dates:      List[str]
    date_range_days:  int

    # Progression assessment
    overall_trend:    str = "stable"   # "stable" | "worsening" | "improving"
    progression_score: float = 0.0    # 0=stable, 1=maximal progression
    rate_per_year:    float = 0.0     # Estimated grade change per year

    # DR-specific
    dr_grades:        List[int] = field(default_factory=list)
    dr_progression:   str = "stable"
    grade_change:     int = 0         # Latest - first grade

    # New findings
    new_lesions:      List[str] = field(default_factory=list)
    resolved_lesions: List[str] = field(default_factory=list)

    # Risk prediction
    risk_12m:        float = 0.0     # Risk of progression in 12 months
    risk_level:      str = "low"     # "low" | "moderate" | "high" | "critical"

    # Attention weights (which visits were most informative)
    visit_importance: List[float] = field(default_factory=list)

    # Clinical output
    findings_text:    str = ""
    recommendation:   str = ""
    full_report:      str = ""

    def to_dict(self) -> Dict:
        return {
            "patient_id":       self.patient_id,
            "num_visits":       self.num_visits,
            "visit_dates":      self.visit_dates,
            "overall_trend":    self.overall_trend,
            "progression_score": self.progression_score,
            "dr_grades":        self.dr_grades,
            "grade_change":     self.grade_change,
            "risk_12m":         self.risk_12m,
            "risk_level":       self.risk_level,
            "new_lesions":      self.new_lesions,
            "recommendation":   self.recommendation,
        }


# ─────────────────────────────────────────────────────────────────────────────
# Temporal Positional Encoding
# ─────────────────────────────────────────────────────────────────────────────

class TemporalPositionalEncoding(nn.Module):
    """
    Positional encoding for clinical visit sequences.

    Unlike standard sinusoidal PE (which assumes uniform spacing),
    this uses actual inter-visit time gaps in days.

    A visit 6 months after baseline gets a different encoding
    than a visit 3 months after baseline.
    """

    def __init__(self, embed_dim: int, max_days: int = 3650):
        super().__init__()
        self.embed_dim = embed_dim
        self.max_days = max_days

        # Learnable time-gap embedding
        self.time_embed = nn.Sequential(
            nn.Linear(1, embed_dim // 4),
            nn.SiLU(),
            nn.Linear(embed_dim // 4, embed_dim),
        )

        # Standard sinusoidal PE as base
        pe = torch.zeros(max_days + 1, embed_dim)
        position = torch.arange(0, max_days + 1).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, embed_dim, 2).float() * (-math.log(10000.0) / embed_dim)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term[:embed_dim//2])
        self.register_buffer("pe", pe)

    def forward(
        self,
        embeddings: torch.Tensor,      # (B, T, D)
        days_since_baseline: List[int] # [0, 90, 365, 730, ...]
    ) -> torch.Tensor:
        """
        Add temporal position encoding to visit embeddings.

        Args:
            embeddings:           (B, T, D) visit embeddings
            days_since_baseline:  list of T day offsets

        Returns:
            (B, T, D) temporally encoded embeddings
        """
        B, T, D = embeddings.shape

        # Clamp to max_days
        days = [min(d, self.max_days) for d in days_since_baseline]

        # Sinusoidal base PE
        day_indices = torch.tensor(days, device=embeddings.device)
        sin_pe = self.pe[day_indices].unsqueeze(0)  # (1, T, D)

        # Learnable time-gap PE
        normalized_days = torch.tensor(
            [[d / self.max_days] for d in days],
            device=embeddings.device, dtype=torch.float32
        ).unsqueeze(0)   # (1, T, 1)
        learnable_pe = self.time_embed(normalized_days)  # (1, T, D)

        return embeddings + sin_pe + learnable_pe


# ─────────────────────────────────────────────────────────────────────────────
# Temporal Transformer
# ─────────────────────────────────────────────────────────────────────────────

class TemporalTransformer(nn.Module):
    """
    Transformer that models disease evolution across clinical visits.

    Causal masking ensures the model can only look at past visits
    (useful for real-time prediction during ongoing care).

    Self-attention weights reveal which visits are most influential
    for the current progression assessment.
    """

    def __init__(
        self,
        embed_dim: int = 1024,
        num_heads: int = 8,
        depth: int = 4,
        mlp_ratio: float = 4.0,
        max_visits: int = 20,
        dropout: float = 0.1,
        causal: bool = False,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.causal = causal
        self.max_visits = max_visits

        # Temporal PE
        self.temporal_pe = TemporalPositionalEncoding(embed_dim)

        # Visit-level projection (normalize input scale)
        self.input_proj = nn.Linear(embed_dim, embed_dim)
        self.input_norm = nn.LayerNorm(embed_dim)

        # Transformer layers
        self.layers = nn.ModuleList([
            TemporalBlock(embed_dim, num_heads, mlp_ratio, dropout)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)

        # [PROGRESSION] summary token (like CLS)
        self.progression_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        nn.init.trunc_normal_(self.progression_token, std=0.02)

        # Store attention weights for visualization
        self._attention_weights: Optional[torch.Tensor] = None

    def forward(
        self,
        visit_embeddings: torch.Tensor,           # (B, T, D)
        days_since_baseline: Optional[List[int]] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            visit_embeddings:    (B, T, D) — T = number of visits
            days_since_baseline: [d0, d1, ..., dT] day offsets
            attention_mask:      (B, T) — 1=valid, 0=padding

        Returns:
            progression_repr: (B, D) — progression summary vector
            visit_tokens:     (B, T+1, D) — per-visit refined tokens
        """
        B, T, D = visit_embeddings.shape

        # Temporal positional encoding
        if days_since_baseline is None:
            days_since_baseline = [i * 180 for i in range(T)]  # Assume 6-month intervals

        x = self.input_norm(self.input_proj(visit_embeddings))
        x = self.temporal_pe(x, days_since_baseline)

        # Prepend progression token
        prog_tok = self.progression_token.expand(B, -1, -1)
        x = torch.cat([prog_tok, x], dim=1)   # (B, T+1, D)

        # Build causal mask if needed
        mask = None
        if self.causal:
            n = T + 1
            mask = torch.triu(torch.ones(n, n, device=x.device), diagonal=1).bool()

        # Transformer
        attn_weights_all = []
        for layer in self.layers:
            x, attn_w = layer(x, mask)
            attn_weights_all.append(attn_w)

        x = self.norm(x)

        # Store last layer attention for visualization
        if attn_weights_all:
            self._attention_weights = attn_weights_all[-1]

        progression_repr = x[:, 0, :]     # Progression token output
        visit_tokens     = x[:, 1:, :]    # Per-visit tokens

        return progression_repr, visit_tokens

    def get_visit_importance(self) -> Optional[torch.Tensor]:
        """
        Return attention weights showing which visits were most important.
        Returns: (T,) tensor of importance scores summing to 1.
        """
        if self._attention_weights is None:
            return None
        # Attention from progression token (index 0) to visit tokens (indices 1:)
        # Shape: (B, H, T+1, T+1) — take row 0 (prog token), cols 1: (visit tokens)
        attn = self._attention_weights
        if attn.dim() == 4:
            visit_attn = attn[0].mean(0)[0, 1:]   # (T,)
        elif attn.dim() == 3:
            visit_attn = attn[0][0, 1:]
        else:
            return None
        return visit_attn / visit_attn.sum().clamp(min=1e-9)


class TemporalBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, mlp_ratio: float, dropout: float):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn  = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, dim), nn.Dropout(dropout),
        )

    def forward(
        self, x: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        h = self.norm1(x)
        attn_out, attn_weights = self.attn(h, h, h, attn_mask=attn_mask,
                                            need_weights=True, average_attn_weights=False)
        x = x + attn_out
        x = x + self.mlp(self.norm2(x))
        return x, attn_weights


# ─────────────────────────────────────────────────────────────────────────────
# Progression Heads
# ─────────────────────────────────────────────────────────────────────────────

class DiseaseProgressionHead(nn.Module):
    """
    Multi-output head predicting disease evolution.

    Outputs:
        trend:        3-class (stable / improving / worsening)
        speed:        regression — grade changes per year
        risk_12m:     probability of progression in next 12 months
        new_lesions:  multi-label binary — which new lesions appeared
    """

    TREND_LABELS = {0: "stable", 1: "improving", 2: "worsening"}

    LESION_TYPES = [
        "microaneurysm", "hemorrhage", "hard_exudate",
        "soft_exudate", "neovascularization", "drusen"
    ]

    def __init__(self, embed_dim: int = 1024, dropout: float = 0.1):
        super().__init__()
        shared_dim = 512

        self.shared = nn.Sequential(
            nn.Linear(embed_dim, shared_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # Trend classification (stable / improving / worsening)
        self.trend_head = nn.Sequential(
            nn.Linear(shared_dim, 128),
            nn.GELU(),
            nn.Linear(128, 3),
        )

        # Progression speed (regression)
        self.speed_head = nn.Sequential(
            nn.Linear(shared_dim, 64),
            nn.GELU(),
            nn.Linear(64, 1),
            nn.Tanh(),   # → [-1, 1] where + = worsening, - = improving
        )

        # 12-month risk prediction
        self.risk_head = nn.Sequential(
            nn.Linear(shared_dim, 128),
            nn.GELU(),
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )

        # New lesion detection
        self.lesion_head = nn.Sequential(
            nn.Linear(shared_dim, 128),
            nn.GELU(),
            nn.Linear(128, len(self.LESION_TYPES)),
        )

    def forward(self, progression_repr: torch.Tensor) -> Dict:
        """
        Args:
            progression_repr: (B, D) from TemporalTransformer

        Returns:
            dict with: trend, speed, risk_12m, new_lesions, trend_label, risk_level
        """
        shared = self.shared(progression_repr)

        # Trend
        trend_logits = self.trend_head(shared)
        trend_probs  = F.softmax(trend_logits, dim=-1)
        trend_class  = trend_probs.argmax(dim=-1)

        # Speed
        speed = self.speed_head(shared).squeeze(-1)

        # Risk
        risk  = self.risk_head(shared).squeeze(-1)

        # New lesions
        lesion_logits = self.lesion_head(shared)
        lesion_probs  = torch.sigmoid(lesion_logits)
        lesion_present = lesion_probs > 0.5

        # Risk level
        risk_level = self._risk_level(risk)

        return {
            "trend_logits":    trend_logits,
            "trend_probs":     trend_probs,
            "trend":           trend_class,
            "trend_label":     [self.TREND_LABELS[t.item()] for t in trend_class],
            "speed":           speed,
            "risk_12m":        risk,
            "risk_level":      risk_level,
            "new_lesion_probs": lesion_probs,
            "new_lesions":     lesion_present,
        }

    def _risk_level(self, risk: torch.Tensor) -> List[str]:
        levels = []
        for r in risk:
            v = r.item()
            if v >= 0.75:     levels.append("critical")
            elif v >= 0.50:   levels.append("high")
            elif v >= 0.25:   levels.append("moderate")
            else:             levels.append("low")
        return levels


# ─────────────────────────────────────────────────────────────────────────────
# Retina-TIME Model
# ─────────────────────────────────────────────────────────────────────────────

class RetinaTimeModel(nn.Module):
    """
    Retina-TIME: Complete longitudinal retinal intelligence system.

    Takes a sequence of patient visit embeddings and produces
    a disease progression analysis.

    Usage:
        time_model = RetinaTimeModel(foundation_encoder)

        # Multiple visits from one patient
        visits = [
            VisitData("2022-01-15", "visit1.jpg"),
            VisitData("2023-01-20", "visit2.jpg"),
            VisitData("2024-02-10", "visit3.jpg"),
        ]

        report = time_model.analyze_patient(patient_id, visits, foundation_model)
    """

    def __init__(
        self,
        embed_dim:   int = 1024,
        temporal_depth: int = 4,
        temporal_heads: int = 8,
        max_visits:  int = 20,
        causal:      bool = False,
    ):
        super().__init__()
        self.embed_dim = embed_dim

        self.temporal_transformer = TemporalTransformer(
            embed_dim=embed_dim,
            num_heads=temporal_heads,
            depth=temporal_depth,
            max_visits=max_visits,
            causal=causal,
        )

        self.progression_head = DiseaseProgressionHead(embed_dim=embed_dim)

        # Change detection: pairwise difference analysis
        self.change_detector = nn.Sequential(
            nn.Linear(embed_dim * 2, 512),
            nn.GELU(),
            nn.Linear(512, 128),
            nn.GELU(),
            nn.Linear(128, len(DiseaseProgressionHead.LESION_TYPES)),
            nn.Sigmoid(),
        )

    def forward(
        self,
        visit_embeddings: torch.Tensor,
        days_since_baseline: Optional[List[int]] = None,
    ) -> Dict:
        """
        Args:
            visit_embeddings:    (B, T, D) — T visit embeddings
            days_since_baseline: [d1, d2, ..., dT] time offsets in days

        Returns:
            dict with progression outputs + temporal attention weights
        """
        # Temporal modeling
        prog_repr, visit_tokens = self.temporal_transformer(
            visit_embeddings, days_since_baseline
        )

        # Progression outputs
        prog_out = self.progression_head(prog_repr)

        # Pairwise change detection (latest vs. earliest)
        if visit_embeddings.shape[1] >= 2:
            first = visit_embeddings[:, 0, :]
            last  = visit_embeddings[:, -1, :]
            change_input = torch.cat([first, last], dim=-1)
            change_probs = self.change_detector(change_input)
            prog_out["change_detection"] = change_probs

        # Visit importance from attention
        visit_importance = self.temporal_transformer.get_visit_importance()
        prog_out["visit_importance"] = visit_importance

        return prog_out

    @torch.no_grad()
    def analyze_patient(
        self,
        patient_id: str,
        visits: List[VisitData],
        foundation_model,        # RetinaGPTFoundationModel
        device: torch.device = None,
    ) -> ProgressionReport:
        """
        Full patient longitudinal analysis.

        Args:
            patient_id:        patient identifier
            visits:            list of VisitData (sorted chronologically)
            foundation_model:  RetinaGPTFoundationModel for embedding
            device:            compute device

        Returns:
            ProgressionReport with all progression findings
        """
        if device is None:
            device = next(self.parameters()).device

        # Sort visits chronologically
        visits = sorted(visits, key=lambda v: v.visit_date)
        T = len(visits)

        if T < 2:
            return ProgressionReport(
                patient_id=patient_id, num_visits=T,
                visit_dates=[v.visit_date for v in visits],
                date_range_days=0,
                overall_trend="insufficient_data",
                findings_text="Minimum 2 visits required for progression analysis.",
                recommendation="Schedule follow-up visit.",
            )

        # Compute days since first visit
        first_visit = visits[0]
        days = [v.days_since(first_visit) for v in visits]
        date_range = max(days)

        # Encode all visits
        embeddings = []
        dr_grades  = []

        for visit in visits:
            if visit.embedding is not None:
                emb = visit.embedding
            else:
                # Load and encode image
                import cv2, numpy as np
                import torchvision.transforms.functional as TF
                img = cv2.imread(visit.image_path)
                if img is None:
                    emb = torch.zeros(self.embed_dim)
                else:
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    img = cv2.resize(img, (224, 224))
                    tensor = TF.to_tensor(img).unsqueeze(0).to(device)
                    result = foundation_model.analyze(tensor)
                    emb = result["embedding"].squeeze(0).cpu()
                    visit.embedding = emb
                    visit.diagnosis = result

                    # Extract DR grade
                    dr = result.get("dr", {})
                    grade = dr.get("grade", None)
                    if grade is not None:
                        if isinstance(grade, torch.Tensor):
                            grade = grade.item()
                        dr_grades.append(int(grade))

            embeddings.append(emb)

        # Stack into sequence
        seq = torch.stack(embeddings).unsqueeze(0).to(device)   # (1, T, D)

        # Forward pass
        self.eval()
        outputs = self.forward(seq, days)

        # Build report
        trend_label = outputs["trend_label"][0]
        risk_val    = outputs["risk_12m"][0].item()
        risk_level  = outputs["risk_level"][0]
        speed       = outputs["speed"][0].item()

        # New lesions from change detection
        new_lesions = []
        if "change_detection" in outputs:
            change_probs = outputs["change_detection"][0]
            for i, lesion_name in enumerate(DiseaseProgressionHead.LESION_TYPES):
                if change_probs[i].item() > 0.5:
                    new_lesions.append(lesion_name.replace("_", " "))

        # Visit importance
        visit_importance = []
        if outputs.get("visit_importance") is not None:
            visit_importance = outputs["visit_importance"].cpu().tolist()

        # Grade change
        grade_change = (dr_grades[-1] - dr_grades[0]) if len(dr_grades) >= 2 else 0

        # Build text report
        findings = self._build_findings(visits, trend_label, dr_grades,
                                        new_lesions, speed, date_range)
        recommendation = self._build_recommendation(risk_level, trend_label, grade_change)

        full_report = (
            f"RETINA-TIME LONGITUDINAL ANALYSIS REPORT\n"
            f"{'='*50}\n"
            f"Patient ID: {patient_id}\n"
            f"Analysis date: {datetime.now().strftime('%Y-%m-%d')}\n"
            f"Visits analyzed: {T}\n"
            f"Period: {visits[0].visit_date} → {visits[-1].visit_date} "
            f"({date_range} days / {date_range/365:.1f} years)\n\n"
            f"{findings}\n\n"
            f"RISK ASSESSMENT:\n"
            f"  12-month progression risk: {risk_val:.1%} [{risk_level.upper()}]\n\n"
            f"{recommendation}"
        )

        return ProgressionReport(
            patient_id=patient_id,
            num_visits=T,
            visit_dates=[v.visit_date for v in visits],
            date_range_days=date_range,
            overall_trend=trend_label,
            progression_score=outputs["risk_12m"][0].item(),
            rate_per_year=speed * 2.0,
            dr_grades=dr_grades,
            dr_progression=trend_label,
            grade_change=grade_change,
            new_lesions=new_lesions,
            risk_12m=risk_val,
            risk_level=risk_level,
            visit_importance=visit_importance,
            findings_text=findings,
            recommendation=recommendation,
            full_report=full_report,
        )

    def _build_findings(self, visits, trend, grades, new_lesions, speed, days):
        lines = ["LONGITUDINAL FINDINGS:"]

        trend_icon = {"worsening": "⚠ WORSENING", "improving": "✓ IMPROVING", "stable": "→ STABLE"}
        lines.append(f"  Overall trend: {trend_icon.get(trend, trend)}")

        if len(grades) >= 2:
            dr_labels = ["No DR", "Mild", "Moderate", "Severe", "Proliferative DR"]
            first_label = dr_labels[min(grades[0], 4)]
            last_label  = dr_labels[min(grades[-1], 4)]
            lines.append(f"  DR Grade: {first_label} → {last_label}")

        if new_lesions:
            lines.append(f"  New lesions detected: {', '.join(new_lesions)}")
        else:
            lines.append("  No new lesions detected since baseline.")

        if days > 365:
            annual_rate = speed * (365 / days) * 4.0  # Normalize to grade/year
            direction = "worsening" if annual_rate > 0 else "improving"
            lines.append(f"  Estimated rate: {abs(annual_rate):.2f} grade units/year ({direction})")

        lines.append(f"\n  Visit timeline ({len(visits)} visits over {days} days):")
        for i, v in enumerate(visits):
            grade_str = f" | DR Grade {grades[i]}" if i < len(grades) else ""
            lines.append(f"    [{v.visit_date}]{grade_str}")

        return "\n".join(lines)

    def _build_recommendation(self, risk_level, trend, grade_change):
        rec = ["RECOMMENDATION:"]

        if risk_level == "critical" or trend == "worsening":
            rec.append("  ⚠ URGENT: Immediate ophthalmology referral required.")
            rec.append("  Consider treatment intensification and 1-month follow-up.")
        elif risk_level == "high":
            rec.append("  HIGH RISK: Ophthalmology review within 4 weeks.")
            rec.append("  Consider additional diagnostic testing (OCT, fluorescein angiography).")
        elif risk_level == "moderate":
            rec.append("  MODERATE RISK: Follow-up screening in 3-6 months.")
        else:
            rec.append("  LOW RISK: Continue annual screening.")

        if grade_change >= 2:
            rec.append("  NOTE: Significant grade progression detected — specialist review essential.")

        return "\n".join(rec)
