"""
api/main.py — Retina-GPT Production API
=========================================
Complete REST API for the Retina-GPT clinical AI platform.

Endpoints:
    GET  /health                 — system health + model status
    GET  /model/info             — capabilities and version
    POST /analyze                — full analysis (image → all results)
    POST /analyze/batch          — batch analysis
    POST /explain                — Grad-CAM + attention maps only
    POST /segment                — vessel + optic disc segmentation
    POST /report/pdf             — generate PDF clinical report
    POST /progression            — longitudinal patient progression

All image endpoints accept:
    - multipart/form-data with image file
    - application/json with base64-encoded image

Auth: X-API-Key header (set via API_KEY env variable)

Run:
    uvicorn api.main:app --host 0.0.0.0 --port 8000 --workers 2
"""

from __future__ import annotations

import base64
import io
import logging
import os
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from fastapi import Depends, FastAPI, File, Form, HTTPException, Request, Security, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from db.cases_db import (
    save_case, get_cases, get_case, get_stats, init_db,
    create_referral, get_referral, update_referral, get_referrals, get_referral_stats,
    create_passport, get_passport, revoke_passport, get_passports_for_case,
)
from ai_copilot.copilot import ask_copilot
from fastapi.security.api_key import APIKeyHeader
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# App State
# ─────────────────────────────────────────────────────────────────────────────

_pipeline = None
_search_engine = None   # VectorSearchEngine — loaded if index exists
_start_time = time.time()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model on startup."""
    global _pipeline
    checkpoint = os.getenv("RETINA_CHECKPOINT", "")
    logger.info(f"[API] Starting Retina-GPT | checkpoint={checkpoint or 'demo'}")

    from inference.pipeline import RetinaGPTPipeline
    if checkpoint and Path(checkpoint).exists():
        _pipeline = RetinaGPTPipeline.from_checkpoint(checkpoint)
    else:
        logger.warning("[API] No checkpoint found — running in DEMO mode")
        _pipeline = RetinaGPTPipeline.demo()

    # Load vector search index if available
    global _search_engine
    index_path = os.getenv("RETINA_INDEX", "indexes/retina_index.bin")
    if os.path.exists(index_path):
        try:
            from retrieval.vector_search import VectorSearchEngine
            _search_engine = VectorSearchEngine.load(index_path)
            logger.info(f"[API] Search index loaded: {len(_search_engine):,} vectors ✓")
        except Exception as e:
            logger.warning(f"[API] Search index load failed: {e}")

    # Initialize cases database
    init_db()
    logger.info("[API] Cases database ready ✓")
    logger.info("[API] Pipeline ready ✓")
    yield
    logger.info("[API] Shutdown")


app = FastAPI(
    title="Retina-GPT API",
    description="AI-powered retinal fundus image analysis platform",
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─────────────────────────────────────────────────────────────────────────────
# Auth
# ─────────────────────────────────────────────────────────────────────────────

API_KEY = os.getenv("API_KEY", "")
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

async def verify_api_key(key: Optional[str] = Security(api_key_header)):
    if not API_KEY:
        return  # No auth configured
    if key != API_KEY:
        raise HTTPException(status_code=403, detail="Invalid API key")


def get_pipeline():
    if _pipeline is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return _pipeline


def get_search_engine():
    if _search_engine is None:
        raise HTTPException(
            status_code=503,
            detail="Search index not loaded. Build it first: python scripts/build_index.py"
        )
    return _search_engine


# ─────────────────────────────────────────────────────────────────────────────
# Pydantic Schemas
# ─────────────────────────────────────────────────────────────────────────────

class QualityResult(BaseModel):
    score:    float = Field(..., description="Quality score [0-1]")
    adequate: bool  = Field(..., description="Adequate for clinical analysis?")

class DRResult(BaseModel):
    grade:         int        = Field(..., description="DR grade 0-4")
    label:         str        = Field(..., description="DR grade name")
    confidence:    float      = Field(..., description="Confidence [0-1]")
    probabilities: List[float]= Field(default_factory=list)
    refer:         bool       = Field(..., description="Referral recommended?")

class AMDResult(BaseModel):
    stage:      int   = Field(..., description="AMD stage 0-3")
    label:      str   = Field(...)
    confidence: float = Field(...)

class GlaucomaResult(BaseModel):
    suspect:        bool  = Field(...)
    cup_disc_ratio: float = Field(...)
    confidence:     float = Field(...)

class LesionItem(BaseModel):
    present:     bool  = Field(...)
    probability: float = Field(...)

class ReportResult(BaseModel):
    structured_findings: str = Field(default="")
    recommendation:      str = Field(default="")
    full_text:           str = Field(default="")

class ExplainabilityResult(BaseModel):
    gradcam_image:     Optional[str] = Field(None, description="Base64 PNG Grad-CAM overlay")
    attention_image:   Optional[str] = Field(None, description="Base64 PNG attention map")
    explanation_panel: Optional[str] = Field(None, description="Base64 PNG full explanation panel")

class SegmentationResult(BaseModel):
    vessel_mask:     Optional[str] = Field(None, description="Base64 PNG vessel segmentation")
    optic_disc_mask: Optional[str] = Field(None, description="Base64 PNG optic disc mask")

class FullAnalysisResponse(BaseModel):
    request_id:       str
    image_id:         str
    quality:          QualityResult
    dr_grading:       DRResult
    amd:              AMDResult
    glaucoma:         GlaucomaResult
    lesions:          Dict[str, LesionItem] = Field(default_factory=dict)
    report:           ReportResult
    explainability:   ExplainabilityResult
    segmentation:     SegmentationResult
    inference_time_ms: float
    model_version:    str

class HealthResponse(BaseModel):
    status:          str
    model_loaded:    bool
    device:          str
    version:         str
    uptime_seconds:  float
    capabilities:    Dict[str, bool] = Field(default_factory=dict)

class ProgressionVisit(BaseModel):
    visit_date:  str  = Field(..., description="ISO date: 2024-03-15")
    image_b64:   str  = Field(..., description="Base64 encoded image")

class ProgressionRequest(BaseModel):
    patient_id:  str
    visits:      List[ProgressionVisit]

class SearchResultItem(BaseModel):
    rank:        int
    image_id:    str
    score:       float    = Field(..., description="Cosine similarity [0-1]")
    distance:    float    = Field(..., description="L2 distance")
    dr_grade:    Optional[int]   = None
    dr_label:    Optional[str]   = None
    dataset:     Optional[str]   = None
    image_path:  Optional[str]   = None

class SearchResponse(BaseModel):
    query_id:         str
    num_results:      int
    search_time_ms:   float
    index_size:       int
    results:          List[SearchResultItem]

class ProgressionResponse(BaseModel):
    patient_id:       str
    num_visits:       int
    overall_trend:    str
    dr_grades:        List[int]
    grade_change:     int
    risk_12m:         float
    risk_level:       str
    new_lesions:      List[str]
    recommendation:   str
    full_report:      str


# ─────────────────────────────────────────────────────────────────────────────
# Helper: Load image from upload
# ─────────────────────────────────────────────────────────────────────────────

async def _read_image_bytes(file: UploadFile) -> bytes:
    content = await file.read()
    if len(content) == 0:
        raise HTTPException(400, "Empty file uploaded")
    return content


def _build_response(result) -> FullAnalysisResponse:
    """Convert FullAnalysisResult → FullAnalysisResponse."""
    d = result.to_api_dict()
    return FullAnalysisResponse(
        request_id=str(uuid.uuid4()),
        image_id=d["image_id"],
        quality=QualityResult(**d["quality"]),
        dr_grading=DRResult(**d["dr_grading"]),
        amd=AMDResult(**d["amd"]),
        glaucoma=GlaucomaResult(**d["glaucoma"]),
        lesions={k: LesionItem(**v) for k, v in d["lesions"].items()},
        report=ReportResult(**d["report"]),
        explainability=ExplainabilityResult(**d["explainability"]),
        segmentation=SegmentationResult(**d["segmentation"]),
        inference_time_ms=d["inference_time_ms"],
        model_version=d["model_version"],
    )


# ─────────────────────────────────────────────────────────────────────────────
# Endpoints
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health():
    """System health check."""
    loaded = _pipeline is not None
    device = str(_pipeline.device) if loaded else "N/A"
    caps   = _pipeline.model_info().get("capabilities", {}) if loaded else {}
    return HealthResponse(
        status="healthy" if loaded else "initializing",
        model_loaded=loaded,
        device=device,
        version="2.0.0",
        uptime_seconds=time.time() - _start_time,
        capabilities=caps,
    )


@app.get("/model/info", tags=["System"])
async def model_info(pipeline=Depends(get_pipeline), _=Depends(verify_api_key)):
    """Return model capabilities, version, parameter count."""
    return pipeline.model_info()


@app.post("/analyze", response_model=FullAnalysisResponse, tags=["Analysis"])
async def analyze(
    file:    UploadFile = File(..., description="Retinal fundus image"),
    explain: bool       = Form(True,  description="Run Grad-CAM explainability"),
    segment: bool       = Form(False, description="Run SAM segmentation"),
    image_id: Optional[str] = Form(None, description="Optional image identifier"),
    pipeline = Depends(get_pipeline),
    _ = Depends(verify_api_key),
):
    """
    **Full retinal analysis** — the main endpoint.

    Returns:
    - Image quality assessment
    - DR grading (0-4) with probabilities
    - AMD staging
    - Glaucoma suspect detection
    - Lesion detection (microaneurysms, hemorrhages, exudates...)
    - Clinical report with recommendation
    - Grad-CAM explainability images (base64 PNG)
    - Segmentation masks (optional, base64 PNG)
    """
    content = await _read_image_bytes(file)
    iid = image_id or Path(file.filename or "image").stem

    result = pipeline.analyze(
        content, image_id=iid,
        explain=explain, segment=segment,
    )
    # Save to database (Data Flywheel)
    try:
        patient_id_val = patient_info.get("id", "Unknown") if patient_info else (image_id or "Unknown")
        save_case(result.to_api_dict(), patient_id=patient_id_val, image_name=file.filename or "")
    except Exception as e:
        logger.warning(f"[API] Could not save case to DB: {e}")

    return _build_response(result)


@app.post("/analyze/batch", tags=["Analysis"])
async def analyze_batch(
    files:   List[UploadFile] = File(...),
    explain: bool = Form(False),
    pipeline = Depends(get_pipeline),
    _ = Depends(verify_api_key),
):
    """Analyze multiple retinal images in one request."""
    if len(files) > 20:
        raise HTTPException(400, "Maximum 20 images per batch")

    results = []
    for f in files:
        content = await _read_image_bytes(f)
        result  = pipeline.analyze(content, image_id=Path(f.filename or "img").stem,
                                   explain=explain)
        results.append(_build_response(result))

    return {"count": len(results), "results": results}


@app.post("/explain", tags=["Explainability"])
async def explain_image(
    file:         UploadFile = File(...),
    target_class: Optional[int] = Form(None, description="Target DR grade to explain (0-4). None = predicted."),
    pipeline = Depends(get_pipeline),
    _ = Depends(verify_api_key),
):
    """
    **Explainability only** — Grad-CAM + attention maps.

    Returns base64 images showing WHICH regions caused the diagnosis.
    Use this to understand model decisions before clinical deployment.
    """
    content = await _read_image_bytes(file)
    result  = pipeline.analyze(content, explain=True, segment=False)

    return {
        "dr_grade":          result.dr_grade,
        "dr_label":          result.dr_label,
        "target_class":      target_class or result.dr_grade,
        "gradcam_image":     result.gradcam_b64,
        "attention_image":   result.attention_b64,
        "explanation_panel": result.explanation_panel_b64,
        "inference_time_ms": result.inference_time_ms,
    }


@app.post("/segment", tags=["Segmentation"])
async def segment_image(
    file:      UploadFile = File(...),
    structure: str        = Form("vessel",
                                  description="Structure to segment: vessel | optic_disc | macula"),
    pipeline = Depends(get_pipeline),
    _ = Depends(verify_api_key),
):
    """
    **Retinal structure segmentation** via Retina-SAM.

    Segments: vessels, optic disc, optic cup, macula.
    Returns binary mask as base64 PNG.
    """
    content = await _read_image_bytes(file)

    caps = pipeline.model_info().get("capabilities", {})
    if not caps.get("segmentation"):
        raise HTTPException(501, "SAM segmentation not enabled in this deployment")

    result = pipeline.analyze(content, explain=False, segment=True)

    mask_b64 = (result.vessel_mask_b64 if structure == "vessel"
                else result.optic_disc_mask_b64)

    if mask_b64 is None:
        raise HTTPException(422, f"Segmentation failed for structure: {structure}")

    return {
        "structure":      structure,
        "mask_image":     mask_b64,   # base64 PNG binary mask
        "inference_time_ms": result.inference_time_ms,
    }


@app.post("/report/pdf", tags=["Reports"])
async def generate_pdf_report(
    file:       UploadFile = File(...),
    patient_id: str        = Form("UNKNOWN"),
    patient_name: str      = Form(""),
    patient_age:  str      = Form(""),
    patient_sex:  str      = Form(""),
    pipeline = Depends(get_pipeline),
    _ = Depends(verify_api_key),
):
    """
    **Generate PDF clinical report**.

    Analyzes the image and generates a professional medical PDF report
    including findings, severity grading, lesion summary, and recommendations.

    Returns the PDF file for download.
    """
    content = await _read_image_bytes(file)

    patient_info = {
        "id":   patient_id,
        "name": patient_name or "—",
        "age":  patient_age  or "—",
        "sex":  patient_sex  or "—",
    }

    pdf_path = f"/tmp/retina_report_{patient_id}_{int(time.time())}.pdf"
    result   = pipeline.analyze(content, explain=True, generate_pdf=pdf_path,
                                 patient_info=patient_info)

    if result.pdf_path and Path(result.pdf_path).exists():
        return FileResponse(
            result.pdf_path,
            media_type="application/pdf",
            filename=f"retina_report_{patient_id}.pdf",
        )

    # Fallback: return JSON if PDF failed
    return {
        "warning": "PDF generation unavailable (install reportlab). Returning JSON.",
        "analysis": _build_response(result),
    }


@app.post("/search", response_model=SearchResponse, tags=["Search"])
async def search_similar(
    file:       UploadFile = File(..., description="Query retinal image"),
    k:          int        = Form(10,  description="Number of similar cases to return (max 50)"),
    dr_grade:   Optional[int] = Form(None, description="Filter results by DR grade (0-4)"),
    pipeline    = Depends(get_pipeline),
    engine      = Depends(get_search_engine),
    _           = Depends(verify_api_key),
):
    """
    **Semantic similar-case retrieval**.

    Encodes the query image and finds the most similar retinal images
    in the database using Approximate Nearest Neighbor (FAISS) search.

    Use for:
    - Clinical decision support ("show me cases like this")
    - Quality audit ("find all Moderate DR cases")
    - Research queries ("find similar lesion patterns")
    """
    k = min(k, 50)
    content = await _read_image_bytes(file)
    iid = Path(file.filename or "query").stem

    # Encode query image
    result = pipeline.analyze(content, image_id=iid, explain=False)
    if result.embedding is None:
        raise HTTPException(422, "Could not generate embedding for query image")

    import torch
    query_emb = torch.from_numpy(result.embedding)

    # Apply filter if requested
    filter_fn = None
    if dr_grade is not None:
        filter_fn = lambda m: m.get("dr_grade") == dr_grade or m.get("label") == dr_grade

    search_resp = engine.search(query_emb, k=k, filter_fn=filter_fn, query_id=iid)

    return SearchResponse(
        query_id=search_resp.query_id,
        num_results=search_resp.num_results,
        search_time_ms=search_resp.search_time_ms,
        index_size=search_resp.index_size,
        results=[
            SearchResultItem(
                rank=r.rank, image_id=r.image_id, score=r.score, distance=r.distance,
                dr_grade=r.dr_grade, dr_label=r.dr_label,
                dataset=r.dataset, image_path=r.image_path,
            )
            for r in search_resp.results
        ],
    )


@app.get("/search/stats", tags=["Search"])
async def search_stats(
    engine = Depends(get_search_engine),
    _      = Depends(verify_api_key),
):
    """Return statistics about the search index (size, DR distribution, index type)."""
    return engine.stats()


@app.post("/progression", response_model=ProgressionResponse, tags=["Temporal"])
async def analyze_progression(
    request: ProgressionRequest,
    pipeline = Depends(get_pipeline),
    _ = Depends(verify_api_key),
):
    """
    **Longitudinal disease progression analysis** via Retina-TIME.

    Analyzes multiple visits from one patient over time to detect:
    - Disease progression trend (stable / worsening / improving)
    - New lesions since last visit
    - 12-month progression risk
    - Clinical progression report

    Minimum 2 visits required.
    """
    if len(request.visits) < 2:
        raise HTTPException(400, "Minimum 2 visits required for progression analysis")

    caps = pipeline.model_info().get("capabilities", {})
    if not caps.get("temporal"):
        raise HTTPException(501, "Temporal model not enabled in this deployment")

    from models.temporal.retina_time import VisitData
    import tempfile

    visits = []
    for v in request.visits:
        # Decode image and save to temp file
        img_bytes = base64.b64decode(v.image_b64)
        tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
        tmp.write(img_bytes)
        tmp.close()
        visits.append(VisitData(visit_date=v.visit_date, image_path=tmp.name))

    try:
        prog = pipeline.analyze_progression(request.patient_id, visits)
    finally:
        # Clean up temp files
        for v in visits:
            try: os.unlink(v.image_path)
            except Exception: pass

    return ProgressionResponse(
        patient_id=prog.patient_id,
        num_visits=prog.num_visits,
        overall_trend=prog.overall_trend,
        dr_grades=prog.dr_grades,
        grade_change=prog.grade_change,
        risk_12m=round(prog.risk_12m, 4),
        risk_level=prog.risk_level,
        new_lesions=prog.new_lesions,
        recommendation=prog.recommendation,
        full_report=prog.full_report,
    )


@app.get("/cases", tags=["Cases"])
async def list_cases(
    limit:      int  = 50,
    offset:     int  = 0,
    patient_id: Optional[str] = None,
    dr_grade:   Optional[int] = None,
    refer_only: bool = False,
    _ = Depends(verify_api_key),
):
    """List all analyzed cases — for the dashboard."""
    cases = get_cases(limit=limit, offset=offset,
                      patient_id=patient_id, dr_grade=dr_grade,
                      refer_only=refer_only)
    return {"total": len(cases), "cases": cases}


@app.get("/cases/stats", tags=["Cases"])
async def cases_stats(_ = Depends(verify_api_key)):
    """Dashboard statistics: totals, this week, grade distribution."""
    return get_stats()


@app.get("/cases/{case_id}", tags=["Cases"])
async def get_case_detail(
    case_id: str,
    _ = Depends(verify_api_key),
):
    """Get full details of a single case."""
    case = get_case(case_id)
    if not case:
        raise HTTPException(404, f"Case '{case_id}' not found")
    return case


@app.delete("/cases/{case_id}", tags=["Cases"])
async def remove_case(
    case_id: str,
    _ = Depends(verify_api_key),
):
    """Delete a case from the database."""
    from db.cases_db import delete_case
    if not delete_case(case_id):
        raise HTTPException(404, f"Case '{case_id}' not found")
    return {"deleted": case_id}


# ─────────────────────────────────────────────────────────────────────────────
# AI Copilot
# ─────────────────────────────────────────────────────────────────────────────

class CopilotRequest(BaseModel):
    case_id:    str
    question:   str
    include_progression: bool = False

class CopilotResponse(BaseModel):
    question:   str
    answer:     str
    confidence: float
    intents:    List[str]
    sources:    List[str]
    suggestion: str

@app.post("/copilot", response_model=CopilotResponse, tags=["Copilot"])
async def copilot(
    req: CopilotRequest,
    _ = Depends(verify_api_key),
):
    """
    **AI Copilot** — ask natural language questions about a retinal scan.

    Examples:
    - "Should I refer this patient?"
    - "What lesions do you see?"
    - "Explain why you graded this as Moderate DR."
    - "How confident are you?"
    - "Is this worse than before?"
    """
    case = get_case(req.case_id)
    if not case:
        raise HTTPException(404, f"Case {req.case_id} not found")

    result = case.get("full_result", {})
    answer = ask_copilot(req.question, result)
    return CopilotResponse(**answer)


# ─────────────────────────────────────────────────────────────────────────────
# Referral Workflow
# ─────────────────────────────────────────────────────────────────────────────

class ReferralCreate(BaseModel):
    case_id:      str
    patient_id:   str
    referring_dr: str = ""
    specialist:   str = ""
    clinic:       str = ""
    reason:       str = ""
    urgency:      str = "routine"
    notes:        str = ""

class ReferralStatusUpdate(BaseModel):
    status:  str
    notes:   str = ""
    outcome: str = ""

class ReferralResponse(BaseModel):
    id:           str
    case_id:      str
    patient_id:   str
    created_at:   str
    updated_at:   str
    referring_dr: str
    specialist:   str
    clinic:       str
    reason:       str
    urgency:      str
    status:       str
    notes:        str
    outcome:      str
    dr_grade:     Optional[int]
    dr_label:     Optional[str]

@app.post("/referrals", response_model=ReferralResponse, tags=["Referrals"])
async def create_referral_endpoint(
    body: ReferralCreate,
    _ = Depends(verify_api_key),
):
    """Create a new referral for a case."""
    case = get_case(body.case_id)
    if not case:
        raise HTTPException(404, f"Case {body.case_id} not found")
    dr = case.get("full_result", {}).get("dr_grading", {})
    ref = create_referral(
        case_id=body.case_id, patient_id=body.patient_id,
        referring_dr=body.referring_dr, specialist=body.specialist,
        clinic=body.clinic, reason=body.reason, urgency=body.urgency,
        notes=body.notes, dr_grade=dr.get("grade"), dr_label=dr.get("label",""),
    )
    return ReferralResponse(**ref)

@app.get("/referrals", tags=["Referrals"])
async def list_referrals(
    patient_id: Optional[str] = None,
    case_id:    Optional[str] = None,
    status:     Optional[str] = None,
    _ = Depends(verify_api_key),
):
    """List referrals with optional filters."""
    refs = get_referrals(patient_id=patient_id, case_id=case_id, status=status)
    return {"total": len(refs), "referrals": refs}

@app.get("/referrals/stats", tags=["Referrals"])
async def referral_stats(_ = Depends(verify_api_key)):
    """Referral pipeline statistics."""
    return get_referral_stats()

@app.get("/referrals/{referral_id}", response_model=ReferralResponse, tags=["Referrals"])
async def get_referral_endpoint(referral_id: str, _ = Depends(verify_api_key)):
    ref = get_referral(referral_id)
    if not ref: raise HTTPException(404, f"Referral {referral_id} not found")
    return ReferralResponse(**ref)

@app.patch("/referrals/{referral_id}", response_model=ReferralResponse, tags=["Referrals"])
async def update_referral_endpoint(
    referral_id: str,
    body: ReferralStatusUpdate,
    _ = Depends(verify_api_key),
):
    """
    Update referral status.
    Valid transitions: pending → sent → acknowledged → seen → completed | cancelled
    """
    ref = update_referral(referral_id, body.status, body.notes, body.outcome)
    if not ref: raise HTTPException(404, f"Referral {referral_id} not found")
    return ReferralResponse(**ref)


# ─────────────────────────────────────────────────────────────────────────────
# Patient Passport
# ─────────────────────────────────────────────────────────────────────────────

class PassportCreate(BaseModel):
    case_id:      str
    patient_id:   str
    expires_days: Optional[int] = 30

class PassportResponse(BaseModel):
    token:       str
    share_url:   str
    case_id:     str
    patient_id:  str
    created_at:  str
    expires_at:  Optional[str]
    views:       int
    active:      bool

@app.post("/passport", response_model=PassportResponse, tags=["Passport"])
async def create_passport_endpoint(
    body: PassportCreate,
    request: Request,
    _ = Depends(verify_api_key),
):
    """Create a shareable Patient Passport link for a case."""
    if not get_case(body.case_id):
        raise HTTPException(404, f"Case {body.case_id} not found")
    token    = create_passport(body.case_id, body.patient_id, body.expires_days)
    base_url = str(request.base_url).rstrip("/")
    return PassportResponse(
        token=token,
        share_url=f"{base_url}/passport/{token}",
        case_id=body.case_id,
        patient_id=body.patient_id,
        created_at=__import__('datetime').datetime.utcnow().isoformat(),
        expires_at=None,
        views=0,
        active=True,
    )

@app.get("/passport/{token}", tags=["Passport"])
async def view_passport(token: str):
    """
    **Public endpoint** — no auth required.
    Returns patient-friendly view of scan results.
    """
    data = get_passport(token)
    if not data:
        raise HTTPException(404, "This link has expired or is no longer valid.")

    case   = data["case"]
    result = case.get("full_result", {})
    dr     = result.get("dr_grading", {})

    return {
        "patient_id":   data["passport"]["patient_id"],
        "scan_date":    case.get("created_at", ""),
        "views":        data["passport"]["views"],
        "dr_grade":     dr.get("grade", 0),
        "dr_label":     dr.get("label", ""),
        "dr_refer":     dr.get("refer", False),
        "recommendation": result.get("report", {}).get("recommendation", ""),
        "quality_adequate": result.get("quality", {}).get("adequate", True),
        "gradcam_image": result.get("explainability", {}).get("gradcam_image"),
    }

@app.delete("/passport/{token}", tags=["Passport"])
async def revoke_passport_endpoint(token: str, _ = Depends(verify_api_key)):
    """Revoke a passport link."""
    if not revoke_passport(token):
        raise HTTPException(404, "Passport not found")
    return {"revoked": token}

@app.get("/passport/case/{case_id}", tags=["Passport"])
async def list_passports(case_id: str, _ = Depends(verify_api_key)):
    """List all passport links for a case."""
    return {"passports": get_passports_for_case(case_id)}
