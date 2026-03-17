# RetinaGPT — Complete AI Ophthalmology Platform

## What's inside

```
backend/    FastAPI AI engine — 24 endpoints
frontend/   Next.js 14 — premium UI
```

## Start

```bash
# Backend
cd backend
pip install -r requirements.txt
uvicorn api.main:app --port 8000

# Frontend (new terminal)
cd frontend
npm install && npm run dev
```

## Full API (24 endpoints)

| Tag | Endpoints |
|-----|-----------|
| Analysis | POST /analyze, /analyze/batch, /explain, /segment |
| Reports | POST /report/pdf |
| Search | POST /search, GET /search/stats |
| Cases | GET/DELETE /cases, GET /cases/{id}, GET /cases/stats |
| Temporal | POST /progression |
| **Copilot** | **POST /copilot** |
| **Referrals** | **POST/GET /referrals, PATCH /referrals/{id}, GET /referrals/stats** |
| **Passport** | **POST /passport, GET /passport/{token}, DELETE /passport/{token}** |

## New features

### AI Copilot — /copilot
Natural language Q&A over any scan result. Ask:
- "Should I refer this patient?"
- "What lesions are present?"
- "Explain why you graded this as Moderate DR."

### Referral Workflow — /referrals
Full pipeline: pending → sent → acknowledged → seen → completed
Urgency levels: urgent / priority / routine

### Patient Passport — /passport/{token}
Shareable public link. Patient sees:
- DR grade with plain-language explanation
- Grad-CAM image
- Recommendation
- Image quality note

Passport page: http://localhost:3000/passport/{token}
