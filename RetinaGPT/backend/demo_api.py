"""
demo_api.py — Retina-GPT Demo Server
======================================
Full implementation of all 24 Retina-GPT endpoints using synthetic but
realistic AI results. No PyTorch required. Results are seeded by image
content hash so the same image always produces the same diagnosis.

Run:
    uvicorn demo_api:app --host 0.0.0.0 --port 8000 --reload
"""

from __future__ import annotations

import base64
import hashlib
import io
import json
import logging
import os
import random
import secrets
import time
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import (
    Depends, FastAPI, File, Form, HTTPException,
    Request, Security, UploadFile, status,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response
from fastapi.security.api_key import APIKeyHeader
from pydantic import BaseModel, Field

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

_start_time = time.time()

# ─── Synthetic Generation ────────────────────────────────────────────────────

DR_LABELS = {
    0: "No Diabetic Retinopathy",
    1: "Mild Non-Proliferative DR",
    2: "Moderate Non-Proliferative DR",
    3: "Severe Non-Proliferative DR",
    4: "Proliferative Diabetic Retinopathy",
}
AMD_LABELS = {0: "No AMD", 1: "Early AMD", 2: "Intermediate AMD", 3: "Late AMD"}
DR_REFER_THRESHOLD = 2  # grade >= 2 → refer

LESION_NAMES = [
    "microaneurysm", "hemorrhage", "hard_exudate",
    "soft_exudate", "neovascularization", "drusen",
    "cotton_wool_spot", "venous_beading",
]


def _seed_from_image(image_bytes: bytes) -> random.Random:
    digest = hashlib.sha256(image_bytes).hexdigest()
    seed = int(digest[:16], 16)
    return random.Random(seed)


def _generate_gradcam_png(image_bytes: bytes, rng: random.Random) -> str:
    """Generate a synthetic Grad-CAM overlay PNG using PIL."""
    try:
        from PIL import Image, ImageDraw, ImageFilter
        import struct

        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        img = img.resize((400, 400))

        gray = img.convert("L")
        gray_rgb = gray.convert("RGB")

        heatmap = Image.new("RGB", (400, 400), (0, 0, 0))
        draw = ImageDraw.Draw(heatmap)
        cx = int(rng.gauss(200, 60))
        cy = int(rng.gauss(200, 60))
        for r in range(100, 10, -15):
            intensity = int(255 * (1 - r / 110))
            red = min(255, intensity * 2)
            green = max(0, 255 - intensity * 2)
            draw.ellipse(
                [cx - r, cy - r, cx + r, cy + r],
                fill=(red, green, 0),
            )

        heatmap = heatmap.filter(ImageFilter.GaussianBlur(radius=18))
        blended = Image.blend(gray_rgb, heatmap, alpha=0.55)

        buf = io.BytesIO()
        blended.save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode()
    except Exception as e:
        logger.warning(f"GradCAM generation failed: {e}")
        return ""


def _generate_analysis(image_bytes: bytes, image_id: str, explain: bool = True) -> Dict:
    rng = _seed_from_image(image_bytes)

    # DR grading — weighted distribution (more 0s and 1s, fewer 4s)
    dr_weights = [0.40, 0.25, 0.18, 0.10, 0.07]
    dr_grade = rng.choices(range(5), weights=dr_weights, k=1)[0]
    dr_probs_raw = [rng.uniform(0.01, 0.15) for _ in range(5)]
    dr_probs_raw[dr_grade] = rng.uniform(0.55, 0.95)
    total = sum(dr_probs_raw)
    dr_probs = [round(p / total, 4) for p in dr_probs_raw]
    dr_confidence = dr_probs[dr_grade]
    dr_refer = dr_grade >= DR_REFER_THRESHOLD

    # AMD
    amd_weights = [0.70, 0.15, 0.10, 0.05]
    amd_stage = rng.choices(range(4), weights=amd_weights, k=1)[0]
    amd_confidence = rng.uniform(0.72, 0.97)

    # Glaucoma
    glaucoma_suspect = rng.random() < 0.12
    cdr = round(rng.uniform(0.5, 0.75) if glaucoma_suspect else rng.uniform(0.3, 0.55), 2)
    glaucoma_confidence = rng.uniform(0.70, 0.95)

    # Image quality
    quality_score = round(rng.uniform(0.72, 0.99), 3)
    quality_adequate = quality_score >= 0.65

    # Lesions — based on DR grade
    lesion_base = min(1.0, dr_grade * 0.22 + 0.05)
    lesions = {}
    for lesion in LESION_NAMES:
        prob = round(min(0.97, lesion_base + rng.uniform(-0.08, 0.18)), 3)
        present = prob > (0.45 if dr_grade >= 2 else 0.62)
        lesions[lesion] = {"present": present, "probability": prob}

    # Report text
    present_lesions = [k for k, v in lesions.items() if v["present"]]
    dr_label = DR_LABELS[dr_grade]
    amd_label = AMD_LABELS[amd_stage]

    if dr_grade == 0:
        findings = "No signs of diabetic retinopathy detected. Retinal vasculature appears normal."
        recommendation = "Continue annual diabetic eye screening as per clinical guidelines."
    elif dr_grade == 1:
        findings = (
            f"Mild non-proliferative diabetic retinopathy detected. "
            f"Microaneurysms observed without significant macular involvement."
        )
        recommendation = "Repeat examination in 12 months. Optimise glycaemic control."
    elif dr_grade == 2:
        findings = (
            f"Moderate non-proliferative diabetic retinopathy. "
            f"Lesions present: {', '.join(present_lesions) if present_lesions else 'multiple retinal changes'}."
        )
        recommendation = "Referral to ophthalmologist within 3 months. Glycaemic and blood pressure control required."
    elif dr_grade == 3:
        findings = (
            f"Severe non-proliferative diabetic retinopathy. "
            f"Extensive retinal changes. Lesions: {', '.join(present_lesions)}."
        )
        recommendation = "Urgent referral to ophthalmologist. Consider intravitreal therapy."
    else:
        findings = (
            "Proliferative diabetic retinopathy. High-risk features present including neovascularization."
        )
        recommendation = "Immediate ophthalmological assessment. Laser photocoagulation or anti-VEGF therapy indicated."

    if amd_stage >= 2:
        findings += f" Additionally, {amd_label} identified — drusen deposits noted in macular region."
    if glaucoma_suspect:
        findings += f" Glaucoma suspicion: elevated cup-to-disc ratio of {cdr}."

    structured_findings = (
        f"DIABETIC RETINOPATHY: {dr_label} (Grade {dr_grade}, "
        f"confidence {dr_confidence:.0%})\n"
        f"AMD: {amd_label}\n"
        f"GLAUCOMA: {'Suspect' if glaucoma_suspect else 'Not suspected'} (CDR {cdr})\n"
        f"IMAGE QUALITY: {'Adequate' if quality_adequate else 'Suboptimal'} ({quality_score:.0%})\n\n"
        f"FINDINGS: {findings}"
    )

    # Grad-CAM
    gradcam_b64 = _generate_gradcam_png(image_bytes, rng) if explain else ""

    risk_level = (
        "urgent" if dr_grade >= 4 else
        "high" if dr_grade >= 3 else
        "moderate" if dr_grade >= 2 else
        "low"
    )

    return {
        "image_id": image_id,
        "quality": {"score": quality_score, "adequate": quality_adequate},
        "dr_grading": {
            "grade": dr_grade,
            "label": dr_label,
            "confidence": round(dr_confidence, 4),
            "probabilities": dr_probs,
            "refer": dr_refer,
        },
        "amd": {
            "stage": amd_stage,
            "label": amd_label,
            "confidence": round(amd_confidence, 4),
        },
        "glaucoma": {
            "suspect": glaucoma_suspect,
            "cup_disc_ratio": cdr,
            "confidence": round(glaucoma_confidence, 4),
        },
        "lesions": lesions,
        "report": {
            "structured_findings": structured_findings,
            "recommendation": recommendation,
            "full_text": f"{structured_findings}\n\nRECOMMENDATION: {recommendation}",
        },
        "explainability": {
            "gradcam_image": gradcam_b64,
            "attention_image": gradcam_b64,
            "explanation_panel": gradcam_b64,
        },
        "segmentation": {
            "vessel_mask": None,
            "optic_disc_mask": None,
        },
        "inference_time_ms": round(rng.uniform(180, 850), 1),
        "model_version": "retina-gpt-demo-2.0",
        "risk_level": risk_level,
    }


# ─── Database (from existing db/cases_db.py) ────────────────────────────────

import sqlite3

DB_PATH = Path("database/retina_cases.db")


def get_conn():
    DB_PATH.parent.mkdir(exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    with get_conn() as conn:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS cases (
                id TEXT PRIMARY KEY, patient_id TEXT DEFAULT 'Unknown',
                created_at TEXT NOT NULL, image_name TEXT,
                dr_grade INTEGER, dr_label TEXT, dr_confidence REAL, dr_refer INTEGER,
                quality_score REAL, quality_adequate INTEGER, risk_level TEXT,
                full_result TEXT NOT NULL, status TEXT DEFAULT 'completed'
            );
            CREATE TABLE IF NOT EXISTS referrals (
                id TEXT PRIMARY KEY, case_id TEXT NOT NULL, patient_id TEXT NOT NULL,
                created_at TEXT NOT NULL, updated_at TEXT NOT NULL,
                referring_dr TEXT DEFAULT '', specialist TEXT DEFAULT '',
                clinic TEXT DEFAULT '', reason TEXT DEFAULT '',
                urgency TEXT DEFAULT 'routine', status TEXT DEFAULT 'pending',
                notes TEXT DEFAULT '', outcome TEXT DEFAULT '',
                dr_grade INTEGER, dr_label TEXT
            );
            CREATE TABLE IF NOT EXISTS passports (
                token TEXT PRIMARY KEY, case_id TEXT NOT NULL,
                patient_id TEXT NOT NULL, created_at TEXT NOT NULL,
                expires_at TEXT, views INTEGER DEFAULT 0, active INTEGER DEFAULT 1
            );
        """)
        conn.commit()


def db_save_case(result: Dict, patient_id: str = "Unknown", image_name: str = "") -> str:
    case_id = result.get("image_id") or str(uuid.uuid4())[:8]
    dr = result.get("dr_grading", {})
    q = result.get("quality", {})
    with get_conn() as conn:
        conn.execute(
            """INSERT OR REPLACE INTO cases
               (id,patient_id,created_at,image_name,dr_grade,dr_label,
                dr_confidence,dr_refer,quality_score,quality_adequate,risk_level,full_result,status)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (case_id, patient_id, datetime.utcnow().isoformat(), image_name,
             dr.get("grade", -1), dr.get("label", ""),
             dr.get("confidence", 0.0), 1 if dr.get("refer") else 0,
             q.get("score", 0.0), 1 if q.get("adequate", True) else 0,
             result.get("risk_level", "low"),
             json.dumps(result), "completed"),
        )
        conn.commit()
    return case_id


def db_get_cases(limit=50, offset=0, patient_id=None, dr_grade=None, refer_only=False):
    q, p = "SELECT * FROM cases WHERE 1=1", []
    if patient_id:
        q += " AND patient_id LIKE ?"; p.append(f"%{patient_id}%")
    if dr_grade is not None:
        q += " AND dr_grade = ?"; p.append(dr_grade)
    if refer_only:
        q += " AND dr_refer = 1"
    q += " ORDER BY created_at DESC LIMIT ? OFFSET ?"; p.extend([limit, offset])
    with get_conn() as conn:
        return [dict(r) for r in conn.execute(q, p).fetchall()]


def db_get_case(case_id: str):
    with get_conn() as conn:
        row = conn.execute("SELECT * FROM cases WHERE id = ?", (case_id,)).fetchone()
    if not row:
        return None
    c = dict(row)
    try:
        c["full_result"] = json.loads(c["full_result"])
    except Exception:
        pass
    return c


def db_delete_case(case_id: str) -> bool:
    with get_conn() as conn:
        cur = conn.execute("DELETE FROM cases WHERE id = ?", (case_id,))
        conn.commit()
    return cur.rowcount > 0


def db_get_stats():
    with get_conn() as conn:
        total = conn.execute("SELECT COUNT(*) FROM cases").fetchone()[0]
        week_ago = (datetime.utcnow() - timedelta(days=7)).isoformat()
        this_week = conn.execute(
            "SELECT COUNT(*) FROM cases WHERE created_at >= ?", (week_ago,)
        ).fetchone()[0]
        referrals = conn.execute("SELECT COUNT(*) FROM cases WHERE dr_refer = 1").fetchone()[0]
        grades = {}
        for g in range(5):
            count = conn.execute(
                "SELECT COUNT(*) FROM cases WHERE dr_grade = ?", (g,)
            ).fetchone()[0]
            grades[str(g)] = count
    return {
        "total_cases": total,
        "this_week": this_week,
        "referrals_needed": referrals,
        "grade_distribution": grades,
        "model_version": "retina-gpt-demo-2.0",
    }


def db_create_referral(case_id: str, patient_id: str, dr_grade: int, dr_label: str,
                        urgency: str = "routine", referring_dr: str = "",
                        reason: str = "") -> Dict:
    ref_id = str(uuid.uuid4())[:8]
    now = datetime.utcnow().isoformat()
    with get_conn() as conn:
        conn.execute(
            """INSERT INTO referrals
               (id,case_id,patient_id,created_at,updated_at,referring_dr,
                urgency,reason,status,dr_grade,dr_label)
               VALUES (?,?,?,?,?,?,?,?,'pending',?,?)""",
            (ref_id, case_id, patient_id, now, now, referring_dr, urgency, reason, dr_grade, dr_label),
        )
        conn.commit()
    return {"id": ref_id, "case_id": case_id, "patient_id": patient_id,
            "urgency": urgency, "status": "pending", "created_at": now}


def db_get_referrals(status_filter=None, urgency=None, limit=50):
    q, p = "SELECT * FROM referrals WHERE 1=1", []
    if status_filter:
        q += " AND status = ?"; p.append(status_filter)
    if urgency:
        q += " AND urgency = ?"; p.append(urgency)
    q += " ORDER BY created_at DESC LIMIT ?"; p.append(limit)
    with get_conn() as conn:
        return [dict(r) for r in conn.execute(q, p).fetchall()]


def db_get_referral(ref_id: str):
    with get_conn() as conn:
        row = conn.execute("SELECT * FROM referrals WHERE id = ?", (ref_id,)).fetchone()
    return dict(row) if row else None


def db_update_referral(ref_id: str, updates: Dict) -> bool:
    if not updates:
        return False
    updates["updated_at"] = datetime.utcnow().isoformat()
    cols = ", ".join(f"{k} = ?" for k in updates)
    vals = list(updates.values()) + [ref_id]
    with get_conn() as conn:
        cur = conn.execute(f"UPDATE referrals SET {cols} WHERE id = ?", vals)
        conn.commit()
    return cur.rowcount > 0


def db_get_referral_stats():
    with get_conn() as conn:
        total = conn.execute("SELECT COUNT(*) FROM referrals").fetchone()[0]
        by_status = {}
        for s in ("pending", "sent", "acknowledged", "seen", "completed"):
            count = conn.execute(
                "SELECT COUNT(*) FROM referrals WHERE status = ?", (s,)
            ).fetchone()[0]
            by_status[s] = count
        by_urgency = {}
        for u in ("urgent", "priority", "routine"):
            count = conn.execute(
                "SELECT COUNT(*) FROM referrals WHERE urgency = ?", (u,)
            ).fetchone()[0]
            by_urgency[u] = count
    return {"total": total, "by_status": by_status, "by_urgency": by_urgency}


def db_create_passport(case_id: str, patient_id: str, expires_days: int = 30) -> str:
    token = secrets.token_urlsafe(24)
    now = datetime.utcnow()
    expires = (now + timedelta(days=expires_days)).isoformat()
    with get_conn() as conn:
        conn.execute(
            "INSERT INTO passports (token,case_id,patient_id,created_at,expires_at) VALUES (?,?,?,?,?)",
            (token, case_id, patient_id, now.isoformat(), expires),
        )
        conn.commit()
    return token


def db_get_passport(token: str):
    with get_conn() as conn:
        row = conn.execute("SELECT * FROM passports WHERE token = ? AND active = 1", (token,)).fetchone()
    if not row:
        return None
    p = dict(row)
    if p.get("expires_at") and datetime.fromisoformat(p["expires_at"]) < datetime.utcnow():
        return None
    with get_conn() as conn:
        conn.execute("UPDATE passports SET views = views + 1 WHERE token = ?", (token,))
        conn.commit()
    return p


def db_revoke_passport(token: str) -> bool:
    with get_conn() as conn:
        cur = conn.execute("UPDATE passports SET active = 0 WHERE token = ?", (token,))
        conn.commit()
    return cur.rowcount > 0


# ─── App Setup ───────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    logger.info("[API] Retina-GPT Demo Server ready ✓ (DEMO MODE — not clinically valid)")
    yield
    logger.info("[API] Shutdown")


app = FastAPI(
    title="Retina-GPT Demo API",
    description="AI-powered retinal fundus image analysis — Demo Mode",
    version="2.0.0-demo",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

API_KEY = os.getenv("API_KEY", "")
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


async def verify_api_key(key: Optional[str] = Security(api_key_header)):
    if not API_KEY:
        return
    if key != API_KEY:
        raise HTTPException(status_code=403, detail="Invalid API key")


# ─── Schemas ─────────────────────────────────────────────────────────────────

class QualityResult(BaseModel):
    score: float
    adequate: bool

class DRResult(BaseModel):
    grade: int
    label: str
    confidence: float
    probabilities: List[float] = []
    refer: bool

class AMDResult(BaseModel):
    stage: int
    label: str
    confidence: float

class GlaucomaResult(BaseModel):
    suspect: bool
    cup_disc_ratio: float
    confidence: float

class LesionItem(BaseModel):
    present: bool
    probability: float

class ReportResult(BaseModel):
    structured_findings: str = ""
    recommendation: str = ""
    full_text: str = ""

class ExplainabilityResult(BaseModel):
    gradcam_image: Optional[str] = None
    attention_image: Optional[str] = None
    explanation_panel: Optional[str] = None

class SegmentationResult(BaseModel):
    vessel_mask: Optional[str] = None
    optic_disc_mask: Optional[str] = None

class FullAnalysisResponse(BaseModel):
    request_id: str
    image_id: str
    quality: QualityResult
    dr_grading: DRResult
    amd: AMDResult
    glaucoma: GlaucomaResult
    lesions: Dict[str, LesionItem] = {}
    report: ReportResult
    explainability: ExplainabilityResult
    segmentation: SegmentationResult
    inference_time_ms: float
    model_version: str

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    device: str
    version: str
    uptime_seconds: float
    capabilities: Dict[str, bool] = {}
    demo_mode: bool = True

class ProgressionVisit(BaseModel):
    visit_date: str
    image_b64: str

class ProgressionRequest(BaseModel):
    patient_id: str
    visits: List[ProgressionVisit]

class ReferralCreateRequest(BaseModel):
    case_id: str
    patient_id: str
    urgency: str = "routine"
    referring_dr: str = ""
    reason: str = ""

class ReferralUpdateRequest(BaseModel):
    status: Optional[str] = None
    notes: Optional[str] = None
    outcome: Optional[str] = None
    specialist: Optional[str] = None
    clinic: Optional[str] = None

class PassportCreateRequest(BaseModel):
    case_id: str
    patient_id: str
    expires_days: int = 30

class CopilotRequest(BaseModel):
    question: str
    case_id: Optional[str] = None
    result: Optional[Dict[str, Any]] = None


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _build_response(data: Dict) -> FullAnalysisResponse:
    return FullAnalysisResponse(
        request_id=str(uuid.uuid4()),
        image_id=data["image_id"],
        quality=QualityResult(**data["quality"]),
        dr_grading=DRResult(**data["dr_grading"]),
        amd=AMDResult(**data["amd"]),
        glaucoma=GlaucomaResult(**data["glaucoma"]),
        lesions={k: LesionItem(**v) for k, v in data["lesions"].items()},
        report=ReportResult(**data["report"]),
        explainability=ExplainabilityResult(**data["explainability"]),
        segmentation=SegmentationResult(**data["segmentation"]),
        inference_time_ms=data["inference_time_ms"],
        model_version=data["model_version"],
    )


async def _read_bytes(file: UploadFile) -> bytes:
    content = await file.read()
    if not content:
        raise HTTPException(400, "Empty file uploaded")
    return content


# ─── Endpoints ───────────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health():
    return HealthResponse(
        status="healthy",
        model_loaded=True,
        device="cpu (demo)",
        version="2.0.0-demo",
        uptime_seconds=round(time.time() - _start_time, 1),
        demo_mode=True,
        capabilities={
            "dr_grading": True, "amd_staging": True, "glaucoma": True,
            "lesion_detection": True, "explainability": True,
            "temporal": True, "segmentation": False,
        },
    )


@app.get("/model/info", tags=["System"])
async def model_info(_ = Depends(verify_api_key)):
    return {
        "version": "retina-gpt-demo-2.0",
        "device": "cpu (demo)",
        "image_size": 512,
        "demo_mode": True,
        "note": "Synthetic demo results — not clinically valid",
        "capabilities": {
            "dr_grading": True, "amd_staging": True, "glaucoma": True,
            "lesion_detection": True, "explainability": True,
            "temporal": True, "segmentation": False,
        },
    }


@app.post("/analyze", response_model=FullAnalysisResponse, tags=["Analysis"])
async def analyze(
    file: UploadFile = File(...),
    explain: bool = Form(True),
    segment: bool = Form(False),
    image_id: Optional[str] = Form(None),
    patient_id: Optional[str] = Form(None),
    _ = Depends(verify_api_key),
):
    content = await _read_bytes(file)
    iid = image_id or Path(file.filename or "image").stem
    pid = patient_id or "Unknown"

    data = _generate_analysis(content, iid, explain=explain)

    try:
        db_save_case(data, patient_id=pid, image_name=file.filename or "")
    except Exception as e:
        logger.warning(f"Could not save case: {e}")

    return _build_response(data)


@app.post("/analyze/batch", tags=["Analysis"])
async def analyze_batch(
    files: List[UploadFile] = File(...),
    explain: bool = Form(False),
    _ = Depends(verify_api_key),
):
    if len(files) > 20:
        raise HTTPException(400, "Maximum 20 images per batch")

    results = []
    for f in files:
        content = await _read_bytes(f)
        iid = Path(f.filename or "img").stem
        data = _generate_analysis(content, iid, explain=explain)
        db_save_case(data, image_name=f.filename or "")
        results.append(_build_response(data))

    return {"count": len(results), "results": results}


@app.post("/explain", tags=["Explainability"])
async def explain_image(
    file: UploadFile = File(...),
    target_class: Optional[int] = Form(None),
    _ = Depends(verify_api_key),
):
    content = await _read_bytes(file)
    data = _generate_analysis(content, Path(file.filename or "image").stem, explain=True)

    return {
        "dr_grade": data["dr_grading"]["grade"],
        "dr_label": data["dr_grading"]["label"],
        "target_class": target_class if target_class is not None else data["dr_grading"]["grade"],
        "gradcam_image": data["explainability"]["gradcam_image"],
        "attention_image": data["explainability"]["attention_image"],
        "explanation_panel": data["explainability"]["explanation_panel"],
        "inference_time_ms": data["inference_time_ms"],
    }


@app.post("/segment", tags=["Segmentation"])
async def segment_image(
    file: UploadFile = File(...),
    structure: str = Form("vessel"),
    _ = Depends(verify_api_key),
):
    raise HTTPException(501, "Segmentation requires full model (SAM not available in demo mode)")


@app.post("/report/pdf", tags=["Reports"])
async def generate_pdf_report(
    file: UploadFile = File(...),
    patient_id: str = Form("UNKNOWN"),
    patient_name: str = Form(""),
    patient_age: str = Form(""),
    patient_sex: str = Form(""),
    _ = Depends(verify_api_key),
):
    content = await _read_bytes(file)
    data = _generate_analysis(content, Path(file.filename or "image").stem, explain=True)

    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.styles import getSampleStyleSheet
        from reportlab.lib.units import cm
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
        from reportlab.lib import colors

        pdf_path = f"/tmp/retina_report_{patient_id}_{int(time.time())}.pdf"
        doc = SimpleDocTemplate(pdf_path, pagesize=A4)
        styles = getSampleStyleSheet()
        story = []

        story.append(Paragraph("RETINA-GPT CLINICAL REPORT", styles["Title"]))
        story.append(Spacer(1, 0.3 * cm))

        now = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")
        dr = data["dr_grading"]
        story.append(Paragraph(f"Patient: {patient_name or patient_id} | Date: {now}", styles["Normal"]))
        story.append(Spacer(1, 0.5 * cm))

        story.append(Paragraph(f"DIAGNOSIS: {dr['label']} (Grade {dr['grade']}, {dr['confidence']:.0%} confidence)", styles["Heading2"]))
        story.append(Paragraph(f"Referral Recommended: {'YES' if dr['refer'] else 'No'}", styles["Normal"]))
        story.append(Spacer(1, 0.3 * cm))

        story.append(Paragraph("FINDINGS", styles["Heading3"]))
        story.append(Paragraph(data["report"]["structured_findings"], styles["Normal"]))
        story.append(Spacer(1, 0.3 * cm))

        story.append(Paragraph("RECOMMENDATION", styles["Heading3"]))
        story.append(Paragraph(data["report"]["recommendation"], styles["Normal"]))
        story.append(Spacer(1, 0.5 * cm))
        story.append(Paragraph("⚠ DEMO MODE — Results are synthetic and not clinically valid.", styles["Normal"]))

        doc.build(story)
        return FileResponse(pdf_path, media_type="application/pdf",
                            filename=f"retina_report_{patient_id}.pdf")
    except ImportError:
        return {
            "warning": "PDF generation unavailable (reportlab not installed). Returning JSON.",
            "analysis": data,
        }
    except Exception as e:
        return {"warning": f"PDF generation failed: {e}", "analysis": data}


@app.post("/search", tags=["Search"])
async def search_similar(
    file: UploadFile = File(...),
    k: int = Form(10),
    dr_grade: Optional[int] = Form(None),
    _ = Depends(verify_api_key),
):
    raise HTTPException(503, "Vector search index not loaded. Build it first: python scripts/build_index.py")


@app.get("/search/stats", tags=["Search"])
async def search_stats(_ = Depends(verify_api_key)):
    return {"index_size": 0, "status": "not_loaded",
            "message": "Search index not available in demo mode"}


@app.post("/progression", tags=["Temporal"])
async def analyze_progression(
    request: ProgressionRequest,
    _ = Depends(verify_api_key),
):
    if len(request.visits) < 2:
        raise HTTPException(400, "Minimum 2 visits required")

    grades = []
    for v in request.visits:
        try:
            img_bytes = base64.b64decode(v.image_b64)
            data = _generate_analysis(img_bytes, f"{request.patient_id}_{v.visit_date}")
            grades.append(data["dr_grading"]["grade"])
        except Exception:
            grades.append(random.randint(0, 3))

    grade_change = grades[-1] - grades[0]
    trend = "stable" if abs(grade_change) == 0 else ("worsening" if grade_change > 0 else "improving")
    risk_12m = min(0.95, max(0.05, grades[-1] * 0.18 + random.uniform(-0.05, 0.05)))
    risk_level = "high" if risk_12m > 0.5 else "moderate" if risk_12m > 0.25 else "low"

    new_lesions = []
    if grade_change > 0:
        possible = ["hemorrhage", "hard_exudate", "neovascularization"]
        new_lesions = random.sample(possible, min(grade_change, len(possible)))

    if trend == "worsening":
        rec = "Urgent ophthalmology review. Escalate treatment protocol."
    elif trend == "improving":
        rec = "Continue current treatment. Review in 6 months."
    else:
        rec = "Stable disease. Continue monitoring at current interval."

    return {
        "patient_id": request.patient_id,
        "num_visits": len(request.visits),
        "overall_trend": trend,
        "dr_grades": grades,
        "grade_change": grade_change,
        "risk_12m": round(risk_12m, 4),
        "risk_level": risk_level,
        "new_lesions": new_lesions,
        "recommendation": rec,
        "full_report": (
            f"Longitudinal analysis: {len(grades)} visits. "
            f"Trend: {trend}. Grade change: {grade_change:+d}. "
            f"12-month risk: {risk_12m:.0%}. {rec}"
        ),
    }


# ─── Cases ───────────────────────────────────────────────────────────────────

@app.get("/cases", tags=["Cases"])
async def list_cases(
    limit: int = 50, offset: int = 0,
    patient_id: Optional[str] = None,
    dr_grade: Optional[int] = None,
    refer_only: bool = False,
    _ = Depends(verify_api_key),
):
    cases = db_get_cases(limit=limit, offset=offset, patient_id=patient_id,
                          dr_grade=dr_grade, refer_only=refer_only)
    return {"total": len(cases), "cases": cases}


@app.get("/cases/stats", tags=["Cases"])
async def cases_stats(_ = Depends(verify_api_key)):
    return db_get_stats()


@app.get("/cases/{case_id}", tags=["Cases"])
async def get_case_detail(case_id: str, _ = Depends(verify_api_key)):
    case = db_get_case(case_id)
    if not case:
        raise HTTPException(404, f"Case '{case_id}' not found")
    return case


@app.delete("/cases/{case_id}", tags=["Cases"])
async def delete_case(case_id: str, _ = Depends(verify_api_key)):
    if not db_delete_case(case_id):
        raise HTTPException(404, f"Case '{case_id}' not found")
    return {"message": f"Case '{case_id}' deleted"}


# ─── Copilot ─────────────────────────────────────────────────────────────────

@app.post("/copilot", tags=["Copilot"])
async def copilot(req: CopilotRequest, _ = Depends(verify_api_key)):
    result_data = req.result

    if result_data is None and req.case_id:
        case = db_get_case(req.case_id)
        if case and isinstance(case.get("full_result"), dict):
            result_data = case["full_result"]

    if result_data is None:
        raise HTTPException(400, "Provide 'result' dict or a valid 'case_id'")

    try:
        import sys
        sys.path.insert(0, str(Path(__file__).parent))
        from ai_copilot.copilot import ask_copilot
        answer = ask_copilot(req.question, result_data)
    except Exception as e:
        logger.warning(f"Copilot import failed: {e}")
        answer = {
            "question": req.question,
            "answer": f"Based on the analysis: DR Grade {result_data.get('dr_grading', {}).get('grade', 'N/A')}, confidence {result_data.get('dr_grading', {}).get('confidence', 0):.0%}.",
            "confidence": 0.75,
            "intents": ["summary"],
            "sources": ["dr_grading"],
            "suggestion": "Ask: 'What follow-up is recommended?'",
        }

    return answer


# ─── Referrals ───────────────────────────────────────────────────────────────

@app.post("/referrals", tags=["Referrals"])
async def create_referral(req: ReferralCreateRequest, _ = Depends(verify_api_key)):
    case = db_get_case(req.case_id)
    if not case:
        raise HTTPException(404, f"Case '{req.case_id}' not found")

    dr = case.get("full_result", {}).get("dr_grading", {}) if isinstance(case.get("full_result"), dict) else {}
    ref = db_create_referral(
        case_id=req.case_id, patient_id=req.patient_id,
        dr_grade=dr.get("grade", -1), dr_label=dr.get("label", ""),
        urgency=req.urgency, referring_dr=req.referring_dr, reason=req.reason,
    )
    return ref


@app.get("/referrals", tags=["Referrals"])
async def list_referrals(
    status: Optional[str] = None,
    urgency: Optional[str] = None,
    limit: int = 50,
    _ = Depends(verify_api_key),
):
    return {"referrals": db_get_referrals(status_filter=status, urgency=urgency, limit=limit)}


@app.patch("/referrals/{ref_id}", tags=["Referrals"])
async def update_referral(
    ref_id: str, req: ReferralUpdateRequest,
    _ = Depends(verify_api_key),
):
    updates = {k: v for k, v in req.model_dump().items() if v is not None}
    if not db_update_referral(ref_id, updates):
        raise HTTPException(404, f"Referral '{ref_id}' not found")
    return db_get_referral(ref_id)


@app.get("/referrals/stats", tags=["Referrals"])
async def referral_stats(_ = Depends(verify_api_key)):
    return db_get_referral_stats()


# ─── Patient Passport ─────────────────────────────────────────────────────────

@app.post("/passport", tags=["Passport"])
async def create_passport(req: PassportCreateRequest, _ = Depends(verify_api_key)):
    case = db_get_case(req.case_id)
    if not case:
        raise HTTPException(404, f"Case '{req.case_id}' not found")
    token = db_create_passport(req.case_id, req.patient_id, req.expires_days)
    return {"token": token, "case_id": req.case_id, "patient_id": req.patient_id,
            "expires_days": req.expires_days}


@app.get("/passport/{token}", tags=["Passport"])
async def get_passport(token: str):
    passport = db_get_passport(token)
    if not passport:
        raise HTTPException(404, "Passport not found or expired")
    case = db_get_case(passport["case_id"])
    if not case:
        raise HTTPException(404, "Associated case not found")
    result = case.get("full_result", {}) if isinstance(case.get("full_result"), dict) else {}
    dr = result.get("dr_grading", {})
    return {
        "token": token,
        "patient_id": passport["patient_id"],
        "created_at": passport["created_at"],
        "expires_at": passport.get("expires_at"),
        "views": passport.get("views", 0),
        "dr_grade": dr.get("grade"),
        "dr_label": dr.get("label"),
        "dr_refer": dr.get("refer"),
        "recommendation": result.get("report", {}).get("recommendation", ""),
        "gradcam_image": result.get("explainability", {}).get("gradcam_image"),
        "quality": result.get("quality", {}),
    }


@app.delete("/passport/{token}", tags=["Passport"])
async def revoke_passport(token: str, _ = Depends(verify_api_key)):
    if not db_revoke_passport(token):
        raise HTTPException(404, "Passport not found")
    return {"message": "Passport revoked", "token": token}


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("RETINA_PORT", "8000"))
    uvicorn.run("demo_api:app", host="0.0.0.0", port=port, reload=True)
