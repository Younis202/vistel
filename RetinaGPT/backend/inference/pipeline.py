"""
inference/pipeline.py — Retina-GPT Complete Inference Pipeline
===============================================================
Connects ALL modules: Foundation Model + Explainability + PDF + Temporal.

Usage:
    pipeline = RetinaGPTPipeline.from_checkpoint("checkpoints/multitask/best.pt")
    result = pipeline.analyze("retina.jpg", explain=True)
    print(result.dr_label, result.recommendation)
    result.save_pdf("report.pdf")
"""

from __future__ import annotations
import base64, io, logging, os, time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import cv2, numpy as np, torch
from PIL import Image

logger = logging.getLogger(__name__)

# ── Pre-inference Quality Gate ────────────────────────────────────────────────

class _PreInferenceQualityGate:
    """
    Rule-based image quality check BEFORE the model runs.
    Catches truly bad images (blurry, dark, artifact-heavy) before
    wasting model inference time on them.

    Uses ImageQualityAssessor from data_engine if available,
    falls back to a lightweight OpenCV check.
    """

    def __init__(self, min_score: float = 0.25):
        self.min_score = min_score
        self._assessor = None
        try:
            from data_engine.data_engine import ImageQualityAssessor
            self._assessor = ImageQualityAssessor(min_quality_score=min_score)
        except Exception:
            pass  # Fallback to simple check

    def check(self, image_rgb: np.ndarray) -> Tuple[bool, float, str]:
        """
        Returns:
            (is_adequate, quality_score, rejection_reason)
        """
        if self._assessor is not None:
            scores = self._assessor.assess(image_rgb)
            return (
                bool(scores["is_adequate"]),
                float(scores["overall_score"]),
                str(scores["rejection_reason"]),
            )

        # Lightweight fallback: just check blur + brightness
        gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
        blur = cv2.Laplacian(gray, cv2.CV_64F).var()
        brightness = gray.mean()

        if blur < 30:
            return False, 0.1, f"Image too blurry (Laplacian={blur:.1f})"
        if brightness < 15:
            return False, 0.1, "Image too dark"
        if brightness > 240:
            return False, 0.1, "Image overexposed"

        score = min(1.0, blur / 300.0) * 0.5 + 0.5
        return True, float(score), ""


DR_GRADE_LABELS = {
    0: "No Diabetic Retinopathy", 1: "Mild Non-Proliferative DR",
    2: "Moderate Non-Proliferative DR", 3: "Severe Non-Proliferative DR",
    4: "Proliferative Diabetic Retinopathy",
}
AMD_STAGE_LABELS = {0: "No AMD", 1: "Early AMD", 2: "Intermediate AMD", 3: "Late AMD"}


# ─────────────────────────────────────────────────────────────────────────────
# Result Container
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class FullAnalysisResult:
    image_id: str = ""; image_path: str = ""
    # Quality
    quality_score: float = 1.0; quality_adequate: bool = True
    # Disease
    dr_grade: int = 0; dr_label: str = "No DR"; dr_confidence: float = 0.0
    dr_probabilities: List[float] = field(default_factory=list); dr_refer: bool = False
    amd_stage: int = 0; amd_label: str = "No AMD"; amd_confidence: float = 0.0
    glaucoma_suspect: bool = False; cup_disc_ratio: float = 0.0; glaucoma_confidence: float = 0.0
    # Lesions
    lesions: Dict[str, Dict] = field(default_factory=dict)
    # Segmentation (base64 PNG)
    vessel_mask_b64: Optional[str] = None; optic_disc_mask_b64: Optional[str] = None
    # Explainability (base64 PNG)
    gradcam_b64: Optional[str] = None; attention_b64: Optional[str] = None
    explanation_panel_b64: Optional[str] = None
    # Embedding
    embedding: Optional[np.ndarray] = None
    # Report
    structured_findings: str = ""; recommendation: str = ""
    clinical_impression: str = ""; full_report_text: str = ""
    pdf_path: Optional[str] = None
    # Meta
    inference_time_ms: float = 0.0; model_version: str = "retina-gpt-foundation"
    _raw_outputs: Optional[Dict] = field(default=None, repr=False)

    def to_api_dict(self) -> Dict:
        return {
            "image_id": self.image_id,
            "quality":  {"score": round(self.quality_score, 4), "adequate": self.quality_adequate},
            "dr_grading": {
                "grade": self.dr_grade, "label": self.dr_label,
                "confidence": round(self.dr_confidence, 4),
                "probabilities": [round(p, 4) for p in self.dr_probabilities],
                "refer": self.dr_refer,
            },
            "amd": {"stage": self.amd_stage, "label": self.amd_label,
                    "confidence": round(self.amd_confidence, 4)},
            "glaucoma": {"suspect": self.glaucoma_suspect,
                         "cup_disc_ratio": round(self.cup_disc_ratio, 3),
                         "confidence": round(self.glaucoma_confidence, 4)},
            "lesions": {k: {"present": v.get("present", False),
                            "probability": round(float(v.get("probability", 0)), 4)}
                        for k, v in self.lesions.items()},
            "report": {"structured_findings": self.structured_findings,
                       "recommendation": self.recommendation,
                       "full_text": self.full_report_text},
            "explainability": {"gradcam_image": self.gradcam_b64,
                               "attention_image": self.attention_b64,
                               "explanation_panel": self.explanation_panel_b64},
            "segmentation": {"vessel_mask": self.vessel_mask_b64,
                             "optic_disc_mask": self.optic_disc_mask_b64},
            "inference_time_ms": round(self.inference_time_ms, 2),
            "model_version": self.model_version,
        }

    def save_pdf(self, output_path: str, patient_info: Optional[Dict] = None,
                 original_image: Optional[np.ndarray] = None) -> str:
        try:
            from reporting.pdf_report import ClinicalPDFGenerator
            path = ClinicalPDFGenerator().generate(
                output_path=output_path,
                patient_info=patient_info or {"id": self.image_id},
                analysis_result=self._raw_outputs or {},
                original_image=original_image,
            )
            self.pdf_path = path
            return path
        except Exception as e:
            logger.error(f"PDF generation failed: {e}")
            return ""

    def save_explanation(self, output_dir: str, prefix: str = ""):
        os.makedirs(output_dir, exist_ok=True)
        prefix = prefix or self.image_id or "result"
        def _save(b64, name):
            if b64:
                (Path(output_dir) / f"{prefix}_{name}.png").write_bytes(base64.b64decode(b64))
        _save(self.gradcam_b64, "gradcam")
        _save(self.attention_b64, "attention")
        _save(self.explanation_panel_b64, "explanation_panel")
        _save(self.vessel_mask_b64, "vessel_mask")


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _load_image(src, size=224):
    if isinstance(src, (str, Path)):
        raw = cv2.imread(str(src))
        if raw is None: raise FileNotFoundError(f"Cannot load: {src}")
        rgb = cv2.cvtColor(raw, cv2.COLOR_BGR2RGB)
    elif isinstance(src, bytes):
        rgb = cv2.cvtColor(cv2.imdecode(np.frombuffer(src, np.uint8), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    else:
        rgb = src.copy()
        if rgb.ndim == 2: rgb = np.stack([rgb]*3, axis=-1)

    try:
        from utils.preprocessing import RetinaPreprocessor, PreprocessingConfig
        t = RetinaPreprocessor(PreprocessingConfig(output_size=size)).preprocess(rgb)
        if not isinstance(t, torch.Tensor): t = torch.from_numpy(t).float()
    except Exception:
        r = cv2.resize(rgb, (size, size))
        mean, std = np.array([0.485,0.456,0.406]), np.array([0.229,0.224,0.225])
        t = torch.from_numpy(((r/255.0 - mean)/std).transpose(2,0,1)).float()

    return (t.unsqueeze(0) if t.dim()==3 else t), rgb


def _t(v, d=0):
    if v is None: return d
    return v.item() if isinstance(v, torch.Tensor) and v.numel()==1 else (v.tolist() if isinstance(v, torch.Tensor) else v)


def _to_b64(img):
    if img is None: return None
    if img.dtype != np.uint8:
        img = (img*255).clip(0,255).astype(np.uint8)
    if img.ndim == 2: img = np.stack([img]*3, axis=-1)
    buf = io.BytesIO()
    Image.fromarray(img).save(buf, "PNG")
    return base64.b64encode(buf.getvalue()).decode()


# ─────────────────────────────────────────────────────────────────────────────
# Main Pipeline
# ─────────────────────────────────────────────────────────────────────────────

class RetinaGPTPipeline:
    """
    Retina-GPT Complete Inference Pipeline.

    Orchestrates: Foundation Model + Explainability + Segmentation + Temporal + PDF.
    """

    def __init__(self, model, device=None, image_size=224,
                 enable_explainability=True, enable_temporal=True,
                 model_version="retina-gpt-foundation"):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device).eval()
        self.image_size = image_size
        self.model_version = model_version
        self._explainer = None
        self._temporal = None
        self._pdf = None
        self._quality_gate = _PreInferenceQualityGate(min_score=0.25)

        if enable_explainability:
            try:
                from interpretability.grad_cam import RetinaExplainer
                self._explainer = RetinaExplainer(self.model, 16, image_size)
                logger.info("[Pipeline] Explainer ✓")
            except Exception as e:
                logger.warning(f"[Pipeline] Explainer unavailable: {e}")

        if enable_temporal:
            try:
                from models.temporal.retina_time import RetinaTimeModel
                self._temporal = RetinaTimeModel(embed_dim=1024).to(self.device).eval()
                logger.info("[Pipeline] Temporal ✓")
            except Exception as e:
                logger.warning(f"[Pipeline] Temporal unavailable: {e}")

        try:
            from reporting.pdf_report import ClinicalPDFGenerator
            self._pdf = ClinicalPDFGenerator()
        except Exception: pass

        logger.info(f"[RetinaGPTPipeline] device={self.device} | "
                    f"explainer={'✓' if self._explainer else '✗'} | "
                    f"temporal={'✓' if self._temporal else '✗'}")

    @torch.no_grad()
    def analyze(self, image_input, image_id=None, explain=True,
                segment=False, generate_pdf=None, patient_info=None) -> FullAnalysisResult:
        t0 = time.perf_counter()
        image_id = image_id or (Path(image_input).stem if isinstance(image_input, str) else "image")

        tensor, rgb = _load_image(image_input, self.image_size)
        tensor = tensor.to(self.device)

        # ── Pre-inference quality gate (rule-based, before model) ──────────
        gate_ok, gate_score, gate_reason = self._quality_gate.check(rgb)
        if not gate_ok:
            logger.warning(f"[Pipeline] Quality gate REJECTED {image_id}: {gate_reason}")
            t_end = time.perf_counter()
            return FullAnalysisResult(
                image_id=image_id,
                image_path=str(image_input) if isinstance(image_input, str) else "",
                quality_score=gate_score,
                quality_adequate=False,
                structured_findings=f"IMAGE QUALITY INSUFFICIENT: {gate_reason}",
                recommendation="Please resubmit with a higher quality fundus image.",
                full_report_text=f"QUALITY REJECTION\n{gate_reason}\nResubmit with better image.",
                inference_time_ms=(time.perf_counter() - t0) * 1000,
                model_version=self.model_version,
            )

        res = FullAnalysisResult(image_id=image_id,
                                  image_path=str(image_input) if isinstance(image_input, str) else "",
                                  model_version=self.model_version)

        # ── Foundation model ──
        raw = self.model(tensor)
        res._raw_outputs = raw

        # Quality
        q = raw.get("quality", {})
        res.quality_score    = float(_t(q.get("score"), 1.0))
        res.quality_adequate = bool(_t(q.get("adequate"), True))

        # DR
        dr = raw.get("dr", {})
        res.dr_grade      = int(_t(dr.get("grade"), 0))
        res.dr_label      = DR_GRADE_LABELS.get(res.dr_grade, str(res.dr_grade))
        res.dr_confidence = float(_t(dr.get("confidence"), 0.0))
        res.dr_refer      = res.dr_grade >= 2
        probs = dr.get("probabilities")
        if probs is not None:
            res.dr_probabilities = (probs.squeeze().cpu().tolist()
                                    if isinstance(probs, torch.Tensor) else list(probs))

        # AMD
        amd = raw.get("amd", {})
        res.amd_stage      = int(_t(amd.get("stage"), 0))
        res.amd_label      = AMD_STAGE_LABELS.get(res.amd_stage, str(res.amd_stage))
        res.amd_confidence = float(_t(amd.get("confidence"), 0.0))

        # Glaucoma
        g = raw.get("glaucoma", {})
        res.glaucoma_suspect    = bool(_t(g.get("suspect"), False))
        res.cup_disc_ratio      = float(_t(g.get("cup_disc_ratio"), 0.0))
        res.glaucoma_confidence = float(_t(g.get("confidence"), 0.0))

        # Lesions
        for name, info in raw.get("lesions", {}).items():
            res.lesions[name] = {
                "present":     bool(_t(info.get("present"), False)),
                "probability": float(_t(info.get("probability"), 0.0)),
            }

        # Embedding
        emb = raw.get("embedding")
        if emb is not None:
            res.embedding = emb.squeeze(0).cpu().numpy()

        # Report
        rpt = raw.get("report", {})
        res.structured_findings = rpt.get("structured_findings", "")
        res.recommendation      = rpt.get("recommendation", "")
        res.clinical_impression = rpt.get("impression", "")
        res.full_report_text    = rpt.get("full_report", "") or self._fallback_report(res)

        # ── Explainability ──
        if explain and self._explainer is not None:
            try:
                from interpretability.grad_cam import HeatmapOverlayRenderer
                renderer = HeatmapOverlayRenderer()
                exp = self._explainer.explain(
                    tensor, rgb, target_class=res.dr_grade,
                    predictions={"DR": res.dr_label,
                                 "Confidence": f"{res.dr_confidence:.0%}",
                                 "Refer": "YES" if res.dr_refer else "NO"},
                )
                if "grad_cam_map" in exp:
                    res.gradcam_b64 = _to_b64(renderer.overlay(rgb, exp["grad_cam_map"], 0.45))
                if "attention_map" in exp:
                    res.attention_b64 = _to_b64(renderer.overlay(rgb, exp["attention_map"], 0.45, "plasma"))
                if "panel" in exp:
                    res.explanation_panel_b64 = _to_b64(exp["panel"])
            except Exception as e:
                logger.warning(f"[Pipeline] Explain failed: {e}")

        # ── Segmentation (SAM) ──
        if segment and hasattr(self.model, "sam") and self.model.sam is not None:
            try:
                masks, _ = self.model.segment(tensor, structure="vessel")
                if masks is not None and len(masks):
                    res.vessel_mask_b64 = _to_b64((masks[0]*255).astype(np.uint8))
                masks, _ = self.model.segment(tensor, structure="optic_disc")
                if masks is not None and len(masks):
                    res.optic_disc_mask_b64 = _to_b64((masks[0]*255).astype(np.uint8))
            except Exception as e:
                logger.warning(f"[Pipeline] Segment failed: {e}")

        # ── PDF ──
        if generate_pdf and self._pdf is not None:
            try:
                res.save_pdf(generate_pdf, patient_info, rgb)
            except Exception as e:
                logger.warning(f"[Pipeline] PDF failed: {e}")

        res.inference_time_ms = (time.perf_counter() - t0) * 1000
        return res

    def analyze_batch(self, images, explain=False, **kw):
        return [self.analyze(img, explain=explain, **kw) for img in images]

    def analyze_progression(self, patient_id, visits):
        if self._temporal is None:
            raise RuntimeError("Temporal model not initialized")
        return self._temporal.analyze_patient(patient_id, visits, self.model, self.device)

    def _fallback_report(self, r: FullAnalysisResult) -> str:
        lines = [f"DR: {r.dr_label} ({r.dr_confidence:.0%})"]
        if r.dr_refer: lines.append("⚠ Referral recommended")
        present = [k for k, v in r.lesions.items() if v.get("present")]
        if present: lines.append(f"Lesions: {', '.join(present)}")
        if r.dr_grade >= 4: lines.append("RECOMMENDATION: Urgent referral.")
        elif r.dr_grade >= 2: lines.append("RECOMMENDATION: Referral within 3 months.")
        else: lines.append("RECOMMENDATION: Annual screening.")
        return "\n".join(lines)

    def model_info(self) -> Dict:
        return {
            "version": self.model_version, "device": str(self.device),
            "image_size": self.image_size,
            "capabilities": {
                "dr_grading": True, "amd_staging": True, "glaucoma": True,
                "lesion_detection": True,
                "explainability": self._explainer is not None,
                "temporal": self._temporal is not None,
                "segmentation": hasattr(self.model, "sam") and self.model.sam is not None,
            },
        }

    @classmethod
    def from_checkpoint(cls, path: str, device=None, **kw) -> "RetinaGPTPipeline":
        device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        try:
            from models.foundation_model import RetinaGPTFoundationModel
            model = RetinaGPTFoundationModel.from_pretrained(path)
        except Exception as e:
            logger.warning(f"[Pipeline] Checkpoint load failed: {e}. Using demo.")
            return cls.demo(device=device, **kw)
        return cls(model, device=device, **kw)

    @classmethod
    def demo(cls, device=None, **kw) -> "RetinaGPTPipeline":
        logger.warning("[Pipeline] DEMO MODE — random weights. Not clinically valid.")
        from models.foundation_model import RetinaGPTFoundationModel, RetinaFoundationConfig
        model = RetinaGPTFoundationModel(
            RetinaFoundationConfig(embed_dim=768, depth=6, num_heads=12)
        )
        return cls(model, device=device, model_version="retina-gpt-demo", **kw)
