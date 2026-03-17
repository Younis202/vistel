# Retina-GPT — Retina Foundation Model

Production-grade AI platform for retinal fundus image analysis.

## What it does

Upload a retinal image → get back:
- DR grading (0-4) with confidence
- AMD staging
- Glaucoma detection
- Lesion detection (microaneurysms, hemorrhages, exudates)
- Grad-CAM explanation (which region caused the diagnosis)
- PDF clinical report
- Similar cases from database

---

## Actual Project Structure

```
retina_gpt/
│
├── data/                          # Put your dataset here
│   └── aptos/
│       ├── train_images/          ← APTOS images go here
│       └── train.csv              ← APTOS labels go here
│
├── models/                        # AI models
│   ├── foundation_model.py        ← Main model (DR+AMD+Glaucoma+Lesions)
│   ├── backbone/
│   │   └── retina_vit.py          ← Vision Transformer backbone
│   ├── heads/
│   │   └── classification_head.py ← Task heads
│   ├── embedding/
│   │   └── universal_embedding.py ← 1024-dim retina embedding
│   ├── pretraining/
│   │   └── retina_dino.py         ← Self-supervised pretraining
│   ├── segmentation/
│   │   └── retina_sam.py          ← Segment Anything for retina
│   ├── vision_language/
│   │   └── retina_clip.py         ← Vision-language alignment
│   ├── temporal/
│   │   └── retina_time.py         ← Longitudinal progression
│   └── language/
│       └── report_generator.py    ← Clinical report generation
│
├── training/
│   ├── trainer.py                 ← Training loop
│   ├── dataset_manager.py         ← Loads APTOS/EyePACS/IDRiD
│   ├── distributed.py             ← Multi-GPU support
│   ├── experiment_tracker.py      ← W&B + TensorBoard
│   └── model_registry.py          ← Checkpoint versioning
│
├── inference/
│   └── pipeline.py                ← Main inference entry point
│
├── evaluation/
│   ├── metrics.py                 ← AUC, Kappa, Dice, IoU
│   └── clinical_eval.py           ← Clinical benchmark
│
├── interpretability/
│   └── grad_cam.py                ← Grad-CAM + Attention maps
│
├── retrieval/
│   └── vector_search.py           ← FAISS similarity search
│
├── reporting/
│   └── pdf_report.py              ← PDF report generator
│
├── data_engine/
│   └── data_engine.py             ← Quality control + versioning
│
├── db/
│   └── cases_db.py               ← SQLite database (auto-saves every analysis)
│
├── api/
│   └── main.py                    ← FastAPI (14 endpoints)
│
├── scripts/
│   ├── train.py                   ← START HERE to train
│   ├── train_foundation.py        ← Full training orchestrator
│   └── build_index.py             ← Build FAISS search index
│
├── configs/
│   ├── model_config.yaml          ← Model settings
│   └── training_config.yaml       ← Training settings (RTX 4050 ready)
│
├── utils/
│   └── preprocessing.py           ← Image preprocessing
│
├── requirements.txt
├── setup.py
├── Dockerfile
├── docker-compose.yml
└── SETUP_AND_RUN.md               ← READ THIS FIRST
```

---

## Quick Start

**Read `SETUP_AND_RUN.md` — full step-by-step guide.**

```bash
# 1. Install
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt

# 2. Put data at: data/aptos/train_images/ and data/aptos/train.csv

# 3. Train
python scripts/train.py --stage multitask --data_dir data/ --epochs 50

# 4. Run API (Windows)
set RETINA_CHECKPOINT=checkpoints/multitask/multitask_best.pt
uvicorn api.main:app --port 8000

# 5. Open: http://localhost:8000/docs
```

---

## API Endpoints

| Method | Endpoint | What it does |
|--------|----------|-------------|
| GET | /health | System status |
| GET | /model/info | Model capabilities |
| POST | /analyze | Full analysis + Grad-CAM |
| POST | /analyze/batch | Multiple images |
| POST | /explain | Grad-CAM only |
| POST | /segment | Vessel + disc masks |
| POST | /report/pdf | Download PDF |
| POST | /search | Find similar cases |
| GET | /search/stats | Index statistics |
| POST | /progression | Patient history analysis |

---

## Hardware Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| GPU | RTX 3060 6GB | RTX 4050+ |
| RAM | 16 GB | 32 GB |
| Storage | 20 GB | 50 GB |
| Python | 3.9+ | 3.11 |

Config is pre-tuned for RTX 4050 6GB:
- image_size = 224
- batch_size = 16
- grad_accum = 4 (effective batch = 64)
- precision = fp16
