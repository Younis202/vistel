
============================================================
 NOTE — CASES DATABASE
============================================================

Every analysis is automatically saved to:
    database/retina_cases.db    ← SQLite, created automatically

Access via API:
    GET /cases              ← All cases (for dashboard)
    GET /cases/stats        ← Statistics
    GET /cases/{id}         ← Single case

No setup needed — file is created on first /analyze call.

# Retina-GPT — Complete Setup & Run Guide
# RTX 4050 · APTOS 2019 · Windows/Linux

============================================================
 STEP 1 — INSTALL EVERYTHING (run once)
============================================================

Open terminal inside the retina_gpt folder, then:

    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
    pip install -e .
    pip install reportlab faiss-cpu rich wandb

Verify GPU works:

    python -c "import torch; print('GPU:', torch.cuda.get_device_name(0)); print('VRAM:', round(torch.cuda.get_device_properties(0).total_memory/1e9,1), 'GB')"

Expected output:
    GPU: NVIDIA GeForce RTX 4050 Laptop GPU
    VRAM: 6.0 GB


============================================================
 STEP 2 — ORGANIZE APTOS DATA
============================================================

Your APTOS zip has these files inside:
    train_images/     ← 3,662 PNG images
    train.csv         ← labels (id_code, diagnosis)
    test_images/      ← (ignore for now)

Create this structure:

    retina_gpt/
    └── data/
        └── aptos/
            ├── train_images/    ← PASTE FOLDER HERE
            └── train.csv        ← PASTE FILE HERE

Quick check — run this to confirm:

    python -c "
    import pandas as pd
    from pathlib import Path
    df = pd.read_csv('data/aptos/train.csv')
    imgs = list(Path('data/aptos/train_images').glob('*.png'))
    print(f'Labels: {len(df)} rows')
    print(f'Images: {len(imgs)} files')
    print(f'DR distribution:\n{df.diagnosis.value_counts().sort_index()}')
    "

Expected output:
    Labels: 3662 rows
    Images: 3662 files
    DR distribution:
    0    1805   ← No DR
    1     370   ← Mild
    2     999   ← Moderate
    3     193   ← Severe
    4     295   ← Proliferative


============================================================
 STEP 3 — TRAIN THE MODEL
============================================================

Single command — nothing else needed:

    python scripts/train.py --stage multitask --data_dir data/ --epochs 50

What happens:
  - Epoch 1-5:   Warmup — numbers low, that's normal
  - Epoch 10+:   Kappa should pass 0.6
  - Epoch 30+:   Kappa should reach 0.75+
  - Best model auto-saved to: checkpoints/multitask/multitask_best.pt

Watch for this on screen:
    Epoch [  1/50]  Train kappa: 0.31  Val kappa: 0.28
    Epoch [  5/50]  Train kappa: 0.58  Val kappa: 0.51
    Epoch [15/50]  Train kappa: 0.74  Val kappa: 0.71
    Epoch [30/50]  Train kappa: 0.83  Val kappa: 0.79
    ✅ New best! kappa=0.791

Time on RTX 4050: ~3-5 hours for 50 epochs.

If you see OOM (Out of Memory) error:
    python scripts/train.py --stage multitask --data_dir data/ --epochs 50 --batch_size 8


============================================================
 STEP 4 — EVALUATE THE TRAINED MODEL
============================================================

After training finishes:

    python -c "
    import sys; sys.path.insert(0, '.')
    from inference.pipeline import RetinaGPTPipeline
    from evaluation.clinical_eval import ClinicalEvaluator

    pipeline = RetinaGPTPipeline.from_checkpoint(
        'checkpoints/multitask/multitask_best.pt'
    )
    evaluator = ClinicalEvaluator(pipeline)
    report = evaluator.evaluate_dr_grading(
        'data/aptos/train_images',
        'data/aptos/train.csv',
        max_samples=500,   # Quick eval on 500 images
        dataset_name='APTOS-2019',
    )
    print(report.summary())
    "

Clinical standards you're targeting:
    AUC    >= 0.93   ← World-class
    Kappa  >= 0.75   ← Strong agreement
    Sensitivity (referable DR) >= 0.87  ← FDA standard


============================================================
 STEP 5 — BUILD SEARCH INDEX
============================================================

    python scripts/build_index.py \
        --data_dir data/aptos/train_images \
        --labels_csv data/aptos/train.csv \
        --checkpoint checkpoints/multitask/multitask_best.pt \
        --output indexes/aptos_index.bin

Takes ~10 minutes. Creates:
    indexes/aptos_index.bin        ← FAISS index
    indexes/aptos_index.bin.meta.json  ← metadata


============================================================
 STEP 6 — RUN THE API
============================================================

    set RETINA_CHECKPOINT=checkpoints/multitask/multitask_best.pt
    set RETINA_INDEX=indexes/aptos_index.bin
    uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload

Linux/Mac:
    RETINA_CHECKPOINT=checkpoints/multitask/multitask_best.pt \
    RETINA_INDEX=indexes/aptos_index.bin \
    uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload

Open your browser: http://localhost:8000/docs
(Auto-generated API documentation — all endpoints with test buttons)


============================================================
 STEP 7 — TEST THE API
============================================================

Analyze a retinal image:

    curl -X POST "http://localhost:8000/analyze" \
         -F "file=@data/aptos/train_images/000c1434d8d7.png" \
         -F "explain=true"

Expected response:
    {
      "dr_grading": {
        "grade": 2,
        "label": "Moderate Non-Proliferative DR",
        "confidence": 0.87,
        "refer": true
      },
      "report": {
        "recommendation": "Ophthalmology referral within 3 months."
      },
      "explainability": {
        "gradcam_image": "iVBORw0KGgo..."  ← base64 PNG
      }
    }

Search similar cases:
    curl -X POST "http://localhost:8000/search" \
         -F "file=@data/aptos/train_images/000c1434d8d7.png" \
         -F "k=5"

Generate PDF report:
    curl -X POST "http://localhost:8000/report/pdf" \
         -F "file=@data/aptos/train_images/000c1434d8d7.png" \
         -F "patient_id=P-001" \
         --output report.pdf


============================================================
 WHAT YOU BUILT — FULL SYSTEM MAP
============================================================

Foundation Model              models/foundation_model.py
  ├── RetinaViT Backbone       models/backbone/retina_vit.py
  ├── Universal Embedding      models/embedding/universal_embedding.py
  ├── DR Grading Head          (inside foundation_model.py)
  ├── AMD Staging Head         (inside foundation_model.py)
  ├── Glaucoma Head            (inside foundation_model.py)
  └── Lesion Detection Head    (inside foundation_model.py)

Self-Supervised Learning      models/pretraining/retina_dino.py
Vision-Language Alignment     models/vision_language/retina_clip.py
Promptable Segmentation       models/segmentation/retina_sam.py
Temporal Progression          models/temporal/retina_time.py

Training System
  ├── Distributed Training     training/distributed.py
  ├── Experiment Tracking      training/experiment_tracker.py
  ├── Model Registry           training/model_registry.py
  └── Dataset Manager          training/dataset_manager.py

Evaluation                    evaluation/clinical_eval.py
Explainability                interpretability/grad_cam.py
PDF Reports                   reporting/pdf_report.py
Vector Search (FAISS)         retrieval/vector_search.py
Data Engine                   data_engine/data_engine.py

API (10 endpoints)            api/main.py
  GET  /health
  GET  /model/info
  POST /analyze          ← Main: image → full diagnosis + Grad-CAM
  POST /analyze/batch    ← Multiple images
  POST /explain          ← Grad-CAM only
  POST /segment          ← Vessel + disc segmentation
  POST /report/pdf       ← Download PDF clinical report
  POST /search           ← Find similar cases (FAISS)
  GET  /search/stats     ← Index statistics
  POST /progression      ← Longitudinal patient analysis

Deployment
  ├── Dockerfile
  └── docker-compose.yml


============================================================
 COMMON ERRORS & SOLUTIONS
============================================================

Error: CUDA out of memory
  Fix: python scripts/train.py ... --batch_size 8

Error: No module named 'torch'
  Fix: pip install torch --index-url https://download.pytorch.org/whl/cu121

Error: FileNotFoundError: data/aptos/train_images
  Fix: Make sure APTOS is at data/aptos/ (see Step 2)

Error: Search index not loaded
  Fix: Run Step 5 first (build_index.py)

Error: Model not loaded (API 503)
  Fix: Set RETINA_CHECKPOINT env variable before uvicorn

Tensorboard (watch training):
  tensorboard --logdir logs/
  Open: http://localhost:6006
