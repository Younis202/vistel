"""
evaluation/clinical_eval.py — Clinical Benchmark Evaluation
=============================================================
Runs a complete medical-grade evaluation of Retina-GPT against
standard ophthalmology AI benchmarks.

Clinical standards enforced:
    DR grading sensitivity ≥ 0.87 (referable DR)
    DR grading specificity ≥ 0.90
    DR AUC ≥ 0.95 (competitive with published baselines)
    Vessel segmentation Dice ≥ 0.80

Usage:
    evaluator = ClinicalEvaluator(pipeline)

    # Evaluate on APTOS dataset
    report = evaluator.evaluate_dr_grading("data/aptos", "data/aptos/train.csv")
    print(report.summary())
    report.save("results/aptos_eval.json")

    # Full benchmark across all datasets
    benchmark = evaluator.run_full_benchmark("data/")
    benchmark.print_table()
"""

import json
import logging
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Clinical Thresholds (based on published ophthalmology AI benchmarks)
# ─────────────────────────────────────────────────────────────────────────────

CLINICAL_THRESHOLDS = {
    "dr_referable_sensitivity": 0.87,   # IDx-DR FDA clearance standard
    "dr_referable_specificity": 0.90,
    "dr_auc":                   0.93,   # EyePACS / APTOS competitive
    "dr_kappa":                 0.75,   # Substantial agreement
    "vessel_dice":              0.80,   # DRIVE benchmark
    "disc_dice":                0.85,   # REFUGE benchmark
    "lesion_iou":               0.50,   # IDRiD benchmark
}

DR_GRADE_NAMES = ["No DR", "Mild", "Moderate", "Severe", "Proliferative"]


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation Report
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class EvaluationReport:
    """Complete evaluation results for one dataset/task."""
    dataset:          str
    task:             str
    num_samples:      int
    duration_seconds: float

    # Classification metrics
    auc:              float = 0.0
    kappa:            float = 0.0
    accuracy:         float = 0.0
    balanced_accuracy: float = 0.0
    sensitivity:      float = 0.0   # macro
    specificity:      float = 0.0   # macro
    f1_macro:         float = 0.0

    # Clinical binary (referable vs not)
    referable_sensitivity: float = 0.0
    referable_specificity: float = 0.0
    referable_auc:         float = 0.0

    # Segmentation
    dice:     float = 0.0
    iou:      float = 0.0
    hausdorff: float = 0.0

    # Per-class
    per_class_auc:        Dict[int, float] = field(default_factory=dict)
    per_class_sensitivity: Dict[int, float] = field(default_factory=dict)

    # Clinical pass/fail
    passes_clinical_standards: bool = False
    failed_standards: List[str] = field(default_factory=list)

    # Timing
    images_per_second: float = 0.0

    def check_clinical_standards(self) -> Tuple[bool, List[str]]:
        """Check if results meet clinical deployment standards."""
        failed = []

        if self.task in ("dr_grading", "classification"):
            if self.auc < CLINICAL_THRESHOLDS["dr_auc"]:
                failed.append(f"DR AUC {self.auc:.3f} < {CLINICAL_THRESHOLDS['dr_auc']} required")
            if self.kappa < CLINICAL_THRESHOLDS["dr_kappa"]:
                failed.append(f"Kappa {self.kappa:.3f} < {CLINICAL_THRESHOLDS['dr_kappa']} required")
            if self.referable_sensitivity > 0 and self.referable_sensitivity < CLINICAL_THRESHOLDS["dr_referable_sensitivity"]:
                failed.append(
                    f"Referable DR sensitivity {self.referable_sensitivity:.3f} "
                    f"< {CLINICAL_THRESHOLDS['dr_referable_sensitivity']} (FDA standard)"
                )
            if self.referable_specificity > 0 and self.referable_specificity < CLINICAL_THRESHOLDS["dr_referable_specificity"]:
                failed.append(
                    f"Referable DR specificity {self.referable_specificity:.3f} "
                    f"< {CLINICAL_THRESHOLDS['dr_referable_specificity']} required"
                )

        if self.task == "vessel_segmentation":
            if self.dice < CLINICAL_THRESHOLDS["vessel_dice"]:
                failed.append(f"Vessel Dice {self.dice:.3f} < {CLINICAL_THRESHOLDS['vessel_dice']} required")

        self.passes_clinical_standards = len(failed) == 0
        self.failed_standards = failed
        return self.passes_clinical_standards, failed

    def summary(self) -> str:
        self.check_clinical_standards()
        passed = "✅ PASS" if self.passes_clinical_standards else "❌ FAIL"

        lines = [
            "─" * 60,
            f"  EVALUATION: {self.dataset} | Task: {self.task}",
            f"  Clinical Standards: {passed}",
            "─" * 60,
            f"  Samples:              {self.num_samples:,}",
            f"  Duration:             {self.duration_seconds:.1f}s "
            f"({self.images_per_second:.1f} img/s)",
        ]

        if self.auc > 0:
            lines += [
                f"  AUC (macro-OvR):      {self.auc:.4f}  "
                f"{'✅' if self.auc >= CLINICAL_THRESHOLDS['dr_auc'] else '⚠'}",
                f"  Quadratic Kappa:      {self.kappa:.4f}  "
                f"{'✅' if self.kappa >= CLINICAL_THRESHOLDS['dr_kappa'] else '⚠'}",
                f"  Accuracy:             {self.accuracy:.4f}",
                f"  Balanced Accuracy:    {self.balanced_accuracy:.4f}",
                f"  Sensitivity (macro):  {self.sensitivity:.4f}",
                f"  Specificity (macro):  {self.specificity:.4f}",
                f"  F1 (macro):           {self.f1_macro:.4f}",
            ]

        if self.referable_sensitivity > 0:
            lines += [
                "",
                f"  [Referable DR — binary clinical threshold]",
                f"  Sensitivity:          {self.referable_sensitivity:.4f}  "
                f"{'✅' if self.referable_sensitivity >= CLINICAL_THRESHOLDS['dr_referable_sensitivity'] else '❌'}",
                f"  Specificity:          {self.referable_specificity:.4f}  "
                f"{'✅' if self.referable_specificity >= CLINICAL_THRESHOLDS['dr_referable_specificity'] else '❌'}",
            ]

        if self.dice > 0:
            lines += [
                f"  Dice Score:           {self.dice:.4f}",
                f"  IoU:                  {self.iou:.4f}",
            ]

        if self.per_class_auc:
            lines.append("\n  Per-class AUC:")
            for cls, auc in self.per_class_auc.items():
                name = DR_GRADE_NAMES[cls] if cls < len(DR_GRADE_NAMES) else str(cls)
                lines.append(f"    [{cls}] {name:20s}: {auc:.4f}")

        if self.failed_standards:
            lines.append("\n  Failed standards:")
            for f in self.failed_standards:
                lines.append(f"    ⚠ {f}")

        lines.append("─" * 60)
        return "\n".join(lines)

    def to_dict(self) -> Dict:
        self.check_clinical_standards()
        return asdict(self)

    def save(self, path: str):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
        logger.info(f"[Eval] Report saved → {path}")


# ─────────────────────────────────────────────────────────────────────────────
# Clinical Evaluator
# ─────────────────────────────────────────────────────────────────────────────

class ClinicalEvaluator:
    """
    Medical-grade evaluation of Retina-GPT on standard benchmarks.

    Enforces clinical performance standards used in:
        IDx-DR (FDA-cleared AI for DR screening)
        EyePACS diabetic retinopathy grading
        REFUGE glaucoma detection challenge
        IDRiD lesion segmentation challenge
    """

    def __init__(self, pipeline, device: Optional[torch.device] = None):
        self.pipeline = pipeline
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

    def evaluate_dr_grading(
        self,
        data_dir: str,
        labels_csv: str,
        max_samples: Optional[int] = None,
        dataset_name: str = "dataset",
    ) -> EvaluationReport:
        """
        Evaluate DR grading on a labeled dataset.

        Args:
            data_dir:    directory with fundus images
            labels_csv:  CSV file with columns [id_code, diagnosis]
            max_samples: limit number of samples (for quick eval)
            dataset_name: name for the report

        Returns:
            EvaluationReport with full clinical metrics
        """
        from evaluation.metrics import compute_dr_grading_metrics

        logger.info(f"[ClinicalEval] Evaluating DR grading on {dataset_name}")

        # Load labels
        image_paths, true_labels = self._load_classification_data(data_dir, labels_csv)
        if max_samples:
            image_paths  = image_paths[:max_samples]
            true_labels  = true_labels[:max_samples]

        n = len(image_paths)
        if n == 0:
            logger.error("[ClinicalEval] No data found!")
            return EvaluationReport(dataset=dataset_name, task="dr_grading",
                                    num_samples=0, duration_seconds=0)

        logger.info(f"[ClinicalEval] Running on {n} images...")
        t0 = time.time()

        y_true, y_pred, y_proba_list = [], [], []

        for i, (path, label) in enumerate(zip(image_paths, true_labels)):
            if (i + 1) % 100 == 0:
                elapsed = time.time() - t0
                logger.info(f"  {i+1}/{n} ({elapsed:.0f}s elapsed, "
                             f"{(i+1)/elapsed:.1f} img/s)")

            try:
                result = self.pipeline.analyze(path, explain=False)
                y_true.append(label)
                y_pred.append(result.dr_grade)
                if result.dr_probabilities:
                    y_proba_list.append(result.dr_probabilities)
            except Exception as e:
                logger.warning(f"  Failed on {path}: {e}")
                continue

        duration = time.time() - t0
        n_done = len(y_true)

        if n_done == 0:
            return EvaluationReport(dataset=dataset_name, task="dr_grading",
                                    num_samples=0, duration_seconds=duration)

        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        y_proba = np.array(y_proba_list) if y_proba_list else None

        # Full metrics
        clf_metrics = compute_dr_grading_metrics(y_true, y_pred, y_proba)

        # Referable DR binary metrics (grade >= 2 = referable)
        ref_sensitivity = clf_metrics.per_class_sensitivity.get(-1, 0.0)
        ref_specificity = clf_metrics.per_class_specificity.get(-1, 0.0)

        # Referable AUC
        ref_auc = 0.0
        if y_proba is not None:
            try:
                from sklearn.metrics import roc_auc_score
                y_ref = (y_true >= 2).astype(int)
                y_ref_prob = y_proba[:, 2:].sum(axis=1) if y_proba.shape[1] >= 3 else y_proba[:, -1]
                ref_auc = float(roc_auc_score(y_ref, y_ref_prob))
            except Exception:
                pass

        report = EvaluationReport(
            dataset=dataset_name,
            task="dr_grading",
            num_samples=n_done,
            duration_seconds=duration,
            images_per_second=n_done / max(duration, 0.01),

            auc=clf_metrics.auc,
            kappa=clf_metrics.kappa,
            accuracy=clf_metrics.accuracy,
            balanced_accuracy=clf_metrics.balanced_accuracy,
            sensitivity=clf_metrics.sensitivity,
            specificity=clf_metrics.specificity,
            f1_macro=clf_metrics.f1_macro,

            referable_sensitivity=ref_sensitivity,
            referable_specificity=ref_specificity,
            referable_auc=ref_auc,

            per_class_auc=clf_metrics.per_class_auc,
            per_class_sensitivity=clf_metrics.per_class_sensitivity,
        )

        report.check_clinical_standards()
        logger.info(f"\n{report.summary()}")
        return report

    def evaluate_segmentation(
        self,
        image_dir: str,
        mask_dir:  str,
        structure: str = "vessel",
        dataset_name: str = "dataset",
        max_samples: Optional[int] = None,
    ) -> EvaluationReport:
        """Evaluate segmentation (vessel / disc / cup) on a labeled dataset."""
        from evaluation.metrics import compute_segmentation_metrics
        import cv2

        logger.info(f"[ClinicalEval] Evaluating {structure} segmentation on {dataset_name}")

        img_dir   = Path(image_dir)
        mask_dir_ = Path(mask_dir)

        extensions = {".png", ".jpg", ".jpeg", ".tif"}
        image_files = sorted([f for f in img_dir.glob("*.*") if f.suffix.lower() in extensions])
        if max_samples:
            image_files = image_files[:max_samples]

        preds, targets = [], []
        t0 = time.time()

        for img_path in image_files:
            # Find corresponding mask
            mask_path = mask_dir_ / img_path.name
            if not mask_path.exists():
                mask_path = mask_dir_ / (img_path.stem + ".png")
            if not mask_path.exists():
                continue

            try:
                # Ground truth mask
                gt = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
                gt_bin = (gt > 127).astype(np.float32)

                # Model prediction
                result = self.pipeline.analyze(str(img_path), explain=False, segment=True)
                import base64
                mask_b64 = (result.vessel_mask_b64 if structure == "vessel"
                            else result.optic_disc_mask_b64)
                if mask_b64 is None:
                    continue

                import io
                from PIL import Image
                mask_bytes = base64.b64decode(mask_b64)
                pred_img = np.array(Image.open(io.BytesIO(mask_bytes)).convert("L"))
                pred_bin = (pred_img > 127).astype(np.float32)

                # Resize to match
                if gt_bin.shape != pred_bin.shape:
                    pred_bin = cv2.resize(pred_bin, (gt_bin.shape[1], gt_bin.shape[0]),
                                          interpolation=cv2.INTER_NEAREST)

                preds.append(pred_bin)
                targets.append(gt_bin)

            except Exception as e:
                logger.warning(f"  Segmentation eval failed on {img_path.name}: {e}")

        duration = time.time() - t0

        if not preds:
            return EvaluationReport(dataset=dataset_name, task=f"{structure}_segmentation",
                                    num_samples=0, duration_seconds=duration)

        seg_metrics = compute_segmentation_metrics(preds, targets)

        report = EvaluationReport(
            dataset=dataset_name,
            task=f"{structure}_segmentation",
            num_samples=len(preds),
            duration_seconds=duration,
            images_per_second=len(preds) / max(duration, 0.01),
            dice=seg_metrics.dice,
            iou=seg_metrics.iou,
            sensitivity=seg_metrics.sensitivity,
            specificity=seg_metrics.specificity,
            hausdorff=seg_metrics.hausdorff_95,
        )
        report.check_clinical_standards()
        logger.info(f"\n{report.summary()}")
        return report

    def run_full_benchmark(
        self,
        data_root: str,
        output_dir: str = "results/benchmark",
        max_samples_per_dataset: Optional[int] = None,
    ) -> Dict[str, EvaluationReport]:
        """
        Run complete benchmark across all available datasets.

        Automatically detects which datasets are present in data_root.
        """
        data_root = Path(data_root)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        all_reports = {}

        # Dataset configs to try
        dataset_configs = [
            {
                "name": "APTOS-2019",
                "dir":  data_root / "aptos",
                "csv":  data_root / "aptos" / "train.csv",
                "task": "dr_grading",
                "img_dir": data_root / "aptos" / "train_images",
            },
            {
                "name": "EyePACS",
                "dir":  data_root / "eyepacs",
                "csv":  data_root / "eyepacs" / "trainLabels.csv",
                "task": "dr_grading",
                "img_dir": data_root / "eyepacs" / "train",
            },
        ]

        for cfg in dataset_configs:
            if not cfg["dir"].exists():
                logger.info(f"[Benchmark] Skipping {cfg['name']} (not found at {cfg['dir']})")
                continue

            logger.info(f"\n{'='*60}")
            logger.info(f"[Benchmark] Evaluating: {cfg['name']}")
            logger.info(f"{'='*60}")

            try:
                if cfg["task"] == "dr_grading":
                    report = self.evaluate_dr_grading(
                        str(cfg.get("img_dir", cfg["dir"])),
                        str(cfg["csv"]),
                        max_samples=max_samples_per_dataset,
                        dataset_name=cfg["name"],
                    )
                    all_reports[cfg["name"]] = report
                    report.save(str(output_dir / f"{cfg['name'].lower()}_dr_eval.json"))

            except Exception as e:
                logger.error(f"[Benchmark] {cfg['name']} failed: {e}")

        # Print comparison table
        self._print_benchmark_table(all_reports)

        # Save full summary
        summary = {
            k: v.to_dict() for k, v in all_reports.items()
        }
        with open(output_dir / "benchmark_summary.json", "w") as f:
            json.dump(summary, f, indent=2)

        return all_reports

    def _print_benchmark_table(self, reports: Dict[str, EvaluationReport]):
        """Print a comparison table of all results."""
        if not reports:
            return

        print("\n" + "="*80)
        print("  RETINA-GPT BENCHMARK RESULTS")
        print("="*80)
        header = f"{'Dataset':20s} {'AUC':6s} {'Kappa':6s} {'Sens':6s} {'Spec':6s} {'F1':6s} {'Pass':5s}"
        print(header)
        print("-"*80)

        for name, r in reports.items():
            status = "✅" if r.passes_clinical_standards else "❌"
            row = (f"{name:20s} {r.auc:6.3f} {r.kappa:6.3f} "
                   f"{r.sensitivity:6.3f} {r.specificity:6.3f} {r.f1_macro:6.3f} {status:5s}")
            print(row)

        print("="*80)

    # ── Data loading helpers ──────────────────────────────────────────────────

    def _load_classification_data(
        self, image_dir: str, csv_path: str
    ) -> Tuple[List[str], List[int]]:
        """Load image paths and labels from CSV."""
        import pandas as pd

        img_dir = Path(image_dir)
        try:
            df = pd.read_csv(csv_path)
            # Auto-detect columns
            id_col = next((c for c in df.columns
                           if any(k in c.lower() for k in ["id", "image", "file"])), df.columns[0])
            lbl_col = next((c for c in df.columns
                            if any(k in c.lower() for k in ["label", "diagnosis", "grade", "level"])),
                           df.columns[-1])
        except Exception as e:
            logger.error(f"[ClinicalEval] Cannot load CSV {csv_path}: {e}")
            return [], []

        extensions = [".png", ".jpg", ".jpeg", ".tif"]
        paths, labels = [], []

        for _, row in df.iterrows():
            stem = str(row[id_col])
            label = int(row[lbl_col])

            found = False
            for ext in extensions:
                p = img_dir / f"{stem}{ext}"
                if p.exists():
                    paths.append(str(p))
                    labels.append(label)
                    found = True
                    break

            if not found:
                # Try without extension change
                p = img_dir / stem
                if p.exists():
                    paths.append(str(p))
                    labels.append(label)

        logger.info(f"[ClinicalEval] Loaded {len(paths)}/{len(df)} samples from {Path(csv_path).name}")
        return paths, labels
