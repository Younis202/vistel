# EyeWisdom — AI Retinal Diagnostics Platform

## Overview

Full-stack AI-powered retinal fundus image diagnosis system powered by the user's custom **Retina-GPT v2** Python backend. Analyzes fundus retinal images and generates professional medical diagnostic reports covering 13+ retinal pathologies including DR grading, AMD staging, glaucoma screening, and lesion detection with Grad-CAM explainability.

## Architecture

### Services (3 running)

| Service | Command | Port | Path |
|---------|---------|------|------|
| Retina-GPT Python Backend | `cd RetinaGPT/backend && uvicorn demo_api:app` | 8000 | internal |
| Node.js API Server | `pnpm --filter @workspace/api-server run dev` | 8080 | `/api/` |
| React Frontend | `pnpm --filter @workspace/eye-diagnosis run dev` | dynamic | `/` |

### Request Flow

```
Browser → Vite Frontend (/)
         → Node.js API (/api/)
              → /api/retina/* → Python Backend (localhost:8000)
              → /api/patients, /api/analyses → PostgreSQL via Drizzle
```

The Node.js API server acts as a reverse proxy for all Python backend requests at `/api/retina/*`.

## Stack

- **Monorepo**: pnpm workspaces
- **Frontend**: React 18, Vite, TanStack Query, Wouter routing, Tailwind CSS, shadcn/ui
- **Node.js Backend**: Express 5, TypeScript, Drizzle ORM, PostgreSQL
- **Python AI Backend**: FastAPI, Uvicorn, PIL (custom Retina-GPT system)
- **Python version**: 3.11
- **Node.js version**: 24

## Python Backend (RetinaGPT)

Located at `RetinaGPT/backend/`. The active server is `demo_api.py` which runs in demo mode (synthetic but realistic results seeded by image content hash).

### Key Endpoints

| Endpoint | Purpose |
|----------|---------|
| `POST /analyze` | Full retinal analysis — main endpoint |
| `POST /analyze/batch` | Batch analysis (up to 20 images) |
| `POST /explain` | Grad-CAM explainability only |
| `POST /report/pdf` | Generate PDF clinical report |
| `POST /copilot` | AI clinical Q&A (rule-based NLP) |
| `GET /cases` | List all analyzed cases |
| `GET /cases/stats` | Dashboard statistics |
| `GET /cases/{id}` | Case detail |
| `POST /referrals` | Create referral workflow |
| `POST /passport` | Patient-shareable passport link |
| `POST /progression` | Longitudinal analysis |

### Response Data Per Analysis

- **DR Grading** (grades 0-4): grade, label, confidence, probabilities, refer flag
- **AMD Staging** (stages 0-3): stage, label, confidence
- **Glaucoma**: suspect boolean, cup-disc ratio, confidence
- **Lesions** (8 types): microaneurysm, hemorrhage, hard_exudate, soft_exudate, neovascularization, drusen, cotton_wool_spot, venous_beading
- **Image Quality**: score, adequate boolean
- **Grad-CAM**: base64 PNG heatmap overlay
- **Clinical Report**: structured findings + recommendation text
- **SQLite Database**: auto-saved cases with full result JSON

## Frontend Pages

- `/` — Dashboard (live stats from Python backend + DR grade distribution chart)
- `/patients` — Patient list (PostgreSQL)
- `/patients/:id` — Patient detail
- `/analyses/new` — New analysis (uploads to Python backend via multipart form)
- `/retina-analyses/:caseId` — **New Retina case detail** with:
  - Grad-CAM explainability heatmap
  - DR grade with probability distribution bar
  - AMD staging, Glaucoma screening
  - Lesion detection (8 categories)
  - Clinical report with recommendations
  - **AI Copilot**: Ask clinical Q&A in natural language

## Database

### PostgreSQL (via Drizzle ORM)
- `patients` table: id, name, age, gender, medicalHistory, createdAt
- `analyses` table: legacy Node.js analyses (GPT-based)

### SQLite (Python Backend)
- Located at `RetinaGPT/backend/database/retina_cases.db`
- Tables: `cases`, `referrals`, `passports`

## Key Files

- `RetinaGPT/backend/demo_api.py` — Active Python FastAPI server (24 endpoints)
- `RetinaGPT/backend/api/main.py` — Original full backend with real PyTorch models
- `RetinaGPT/backend/ai_copilot/copilot.py` — Rule-based clinical NLP
- `artifacts/api-server/src/app.ts` — Express proxy to Python backend
- `artifacts/eye-diagnosis/src/pages/RetinaAnalysisDetail.tsx` — Full case detail page
- `artifacts/eye-diagnosis/src/pages/NewAnalysis.tsx` — Upload page
- `artifacts/eye-diagnosis/src/pages/Dashboard.tsx` — Live stats dashboard

## Upgrading to Full Model

To switch from demo mode to real PyTorch models:
1. Install: `pip install torch torchvision timm einops`
2. Place trained checkpoint at a path of your choice
3. Set env var: `RETINA_CHECKPOINT=/path/to/checkpoint`
4. Update workflow command to use `api/main.py` instead of `demo_api.py`
