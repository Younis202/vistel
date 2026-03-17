"""
metrics.py — Clinical Evaluation Metrics for Retina-GPT
=========================================================
Medical-grade performance metrics for ophthalmology AI.

Covers:
    Classification:   AUC, sensitivity, specificity, F1, kappa, balanced accuracy
    Segmentation:     Dice, IoU, Hausdorff distance, pixel accuracy
    Detection:        Lesion-level precision, recall, F1
    Clinical:         Grade agreement, clinically significant disagreement

All functions accept numpy arrays or torch tensors.
"""

import numpy as np
import torch
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field


# ─────────────────────────────────────────────────────────────────────────────
# Result Containers
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ClassificationMetrics:
    """Complete classification performance report."""
    auc:                  float = 0.0
    sensitivity:          float = 0.0   # Recall for positive class
    specificity:          float = 0.0   # True negative rate
    accuracy:             float = 0.0
    balanced_accuracy:    float = 0.0
    f1_macro:             float = 0.0
    f1_weighted:          float = 0.0
    kappa:                float = 0.0   # Cohen's quadratic weighted kappa
    precision_macro:      float = 0.0
    per_class_auc:        Dict[int, float] = field(default_factory=dict)
    per_class_sensitivity: Dict[int, float] = field(default_factory=dict)
    per_class_specificity: Dict[int, float] = field(default_factory=dict)
    confusion_matrix:     Optional[np.ndarray] = None
    roc_curves:           Optional[Dict] = None   # class → (fpr, tpr, thresholds)
    optimal_thresholds:   Dict[int, float] = field(default_factory=dict)
    num_samples:          int = 0
    class_names:          List[str] = field(default_factory=list)

    def summary(self) -> str:
        lines = [
            "─" * 52,
            "  CLASSIFICATION PERFORMANCE SUMMARY",
            "─" * 52,
            f"  AUC (macro-OvR):       {self.auc:.4f}",
            f"  Sensitivity:           {self.sensitivity:.4f}",
            f"  Specificity:           {self.specificity:.4f}",
            f"  Accuracy:              {self.accuracy:.4f}",
            f"  Balanced Accuracy:     {self.balanced_accuracy:.4f}",
            f"  Quadratic Kappa:       {self.kappa:.4f}",
            f"  F1 (macro):            {self.f1_macro:.4f}",
            f"  F1 (weighted):         {self.f1_weighted:.4f}",
            f"  Samples evaluated:     {self.num_samples}",
        ]
        if self.per_class_auc:
            lines.append("\n  Per-class AUC:")
            names = self.class_names or [str(i) for i in self.per_class_auc]
            for i, (cls, auc) in enumerate(self.per_class_auc.items()):
                name = names[cls] if cls < len(names) else str(cls)
                lines.append(f"    [{cls}] {name:30s}: {auc:.4f}")
        lines.append("─" * 52)
        return "\n".join(lines)

    def to_dict(self) -> Dict:
        return {
            "auc": self.auc,
            "sensitivity": self.sensitivity,
            "specificity": self.specificity,
            "accuracy": self.accuracy,
            "balanced_accuracy": self.balanced_accuracy,
            "kappa": self.kappa,
            "f1_macro": self.f1_macro,
            "f1_weighted": self.f1_weighted,
            "num_samples": self.num_samples,
        }


@dataclass
class SegmentationMetrics:
    """Segmentation performance metrics."""
    dice:              float = 0.0
    iou:               float = 0.0    # Jaccard index
    pixel_accuracy:    float = 0.0
    sensitivity:       float = 0.0   # Recall
    specificity:       float = 0.0
    precision:         float = 0.0
    hausdorff_95:      float = 0.0   # 95th percentile Hausdorff distance
    per_class_dice:    Dict[int, float] = field(default_factory=dict)
    per_class_iou:     Dict[int, float] = field(default_factory=dict)
    num_samples:       int = 0
    class_names:       List[str] = field(default_factory=list)

    def summary(self) -> str:
        lines = [
            "─" * 52,
            "  SEGMENTATION PERFORMANCE SUMMARY",
            "─" * 52,
            f"  Dice Score:            {self.dice:.4f}",
            f"  IoU (Jaccard):         {self.iou:.4f}",
            f"  Pixel Accuracy:        {self.pixel_accuracy:.4f}",
            f"  Sensitivity:           {self.sensitivity:.4f}",
            f"  Specificity:           {self.specificity:.4f}",
            f"  Precision:             {self.precision:.4f}",
            f"  Hausdorff 95%:         {self.hausdorff_95:.2f} px",
            f"  Samples evaluated:     {self.num_samples}",
            "─" * 52,
        ]
        return "\n".join(lines)

    def to_dict(self) -> Dict:
        return {
            "dice": self.dice, "iou": self.iou,
            "pixel_accuracy": self.pixel_accuracy,
            "sensitivity": self.sensitivity, "specificity": self.specificity,
            "hausdorff_95": self.hausdorff_95,
        }


# ─────────────────────────────────────────────────────────────────────────────
# Classification Metrics
# ─────────────────────────────────────────────────────────────────────────────

def compute_classification_metrics(
    y_true:  Union[np.ndarray, List[int]],
    y_pred:  Union[np.ndarray, List[int]],
    y_proba: Optional[Union[np.ndarray, torch.Tensor]] = None,
    class_names: Optional[List[str]] = None,
    average:  str = "macro",
) -> ClassificationMetrics:
    """
    Compute full classification metrics for retinal disease grading.

    Args:
        y_true:   ground truth class labels (N,)
        y_pred:   predicted class labels (N,)
        y_proba:  predicted probabilities (N, C) — needed for AUC
        class_names: optional list of class names
        average:  averaging strategy for multi-class metrics

    Returns:
        ClassificationMetrics dataclass
    """
    from sklearn.metrics import (
        roc_auc_score, cohen_kappa_score, f1_score,
        balanced_accuracy_score, accuracy_score,
        confusion_matrix, precision_score,
        roc_curve,
    )

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    n = len(y_true)
    classes = np.unique(y_true)
    n_classes = len(classes)

    metrics = ClassificationMetrics(num_samples=n, class_names=class_names or [])

    # Basic metrics
    metrics.accuracy          = float(accuracy_score(y_true, y_pred))
    metrics.balanced_accuracy = float(balanced_accuracy_score(y_true, y_pred))
    metrics.kappa             = float(cohen_kappa_score(y_true, y_pred, weights="quadratic"))
    metrics.f1_macro          = float(f1_score(y_true, y_pred, average="macro",    zero_division=0))
    metrics.f1_weighted       = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))
    metrics.precision_macro   = float(precision_score(y_true, y_pred, average="macro", zero_division=0))

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=classes)
    metrics.confusion_matrix = cm

    # Per-class sensitivity & specificity from confusion matrix
    for i, cls in enumerate(classes):
        tp = cm[i, i]
        fn = cm[i, :].sum() - tp
        fp = cm[:, i].sum() - tp
        tn = cm.sum() - tp - fn - fp

        sens = tp / (tp + fn + 1e-9)
        spec = tn / (tn + fp + 1e-9)
        metrics.per_class_sensitivity[int(cls)] = float(sens)
        metrics.per_class_specificity[int(cls)] = float(spec)

    # Macro sensitivity & specificity
    metrics.sensitivity = float(np.mean(list(metrics.per_class_sensitivity.values())))
    metrics.specificity = float(np.mean(list(metrics.per_class_specificity.values())))

    # AUC (needs probabilities)
    if y_proba is not None:
        if isinstance(y_proba, torch.Tensor):
            y_proba = y_proba.detach().cpu().numpy()
        y_proba = np.array(y_proba)

        try:
            if n_classes == 2:
                metrics.auc = float(roc_auc_score(y_true, y_proba[:, 1]))
            else:
                metrics.auc = float(roc_auc_score(
                    y_true, y_proba, multi_class="ovr", average="macro"
                ))

            # Per-class AUC (one-vs-rest)
            roc_curves = {}
            optimal_thresholds = {}
            for i, cls in enumerate(classes):
                y_bin = (y_true == cls).astype(int)
                if len(np.unique(y_bin)) < 2:
                    continue
                cls_proba = y_proba[:, i] if y_proba.ndim > 1 else y_proba
                auc_i = float(roc_auc_score(y_bin, cls_proba))
                metrics.per_class_auc[int(cls)] = auc_i

                # ROC curve + Youden's J optimal threshold
                fpr, tpr, thresholds = roc_curve(y_bin, cls_proba)
                j_stat = tpr - fpr
                opt_idx = np.argmax(j_stat)
                roc_curves[int(cls)] = (fpr, tpr, thresholds)
                optimal_thresholds[int(cls)] = float(thresholds[opt_idx])

            metrics.roc_curves = roc_curves
            metrics.optimal_thresholds = optimal_thresholds

        except Exception as e:
            metrics.auc = 0.0

    return metrics


def compute_dr_grading_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: Optional[np.ndarray] = None,
) -> ClassificationMetrics:
    """
    Specialized DR grading evaluation with clinical thresholds.
    Also computes: referable DR rate, sight-threatening DR rate.
    """
    dr_class_names = [
        "No DR", "Mild NPDR", "Moderate NPDR", "Severe NPDR", "PDR"
    ]
    metrics = compute_classification_metrics(
        y_true, y_pred, y_proba, class_names=dr_class_names
    )

    # Clinical binary metrics: referable (grade >= 2) vs non-referable
    y_ref_true = (y_true >= 2).astype(int)
    y_ref_pred = (y_pred >= 2).astype(int)
    tp = ((y_ref_pred == 1) & (y_ref_true == 1)).sum()
    tn = ((y_ref_pred == 0) & (y_ref_true == 0)).sum()
    fp = ((y_ref_pred == 1) & (y_ref_true == 0)).sum()
    fn = ((y_ref_pred == 0) & (y_ref_true == 1)).sum()

    ref_sensitivity = tp / (tp + fn + 1e-9)
    ref_specificity = tn / (tn + fp + 1e-9)

    # Add to metrics as extra fields stored in dict
    metrics.per_class_sensitivity[-1] = float(ref_sensitivity)   # -1 = referable
    metrics.per_class_specificity[-1] = float(ref_specificity)

    return metrics


# ─────────────────────────────────────────────────────────────────────────────
# Segmentation Metrics
# ─────────────────────────────────────────────────────────────────────────────

def dice_score(
    pred: Union[np.ndarray, torch.Tensor],
    target: Union[np.ndarray, torch.Tensor],
    smooth: float = 1e-6,
) -> float:
    """Binary Dice coefficient: 2*TP / (2*TP + FP + FN)."""
    if isinstance(pred, torch.Tensor):
        pred = pred.detach().cpu().numpy()
    if isinstance(target, torch.Tensor):
        target = target.detach().cpu().numpy()

    pred   = (pred   > 0.5).astype(float).flatten()
    target = (target > 0.5).astype(float).flatten()

    intersection = (pred * target).sum()
    return float((2.0 * intersection + smooth) / (pred.sum() + target.sum() + smooth))


def iou_score(
    pred: Union[np.ndarray, torch.Tensor],
    target: Union[np.ndarray, torch.Tensor],
    smooth: float = 1e-6,
) -> float:
    """Intersection-over-Union (Jaccard index)."""
    if isinstance(pred, torch.Tensor):
        pred = pred.detach().cpu().numpy()
    if isinstance(target, torch.Tensor):
        target = target.detach().cpu().numpy()

    pred   = (pred   > 0.5).astype(float).flatten()
    target = (target > 0.5).astype(float).flatten()

    intersection = (pred * target).sum()
    union = pred.sum() + target.sum() - intersection
    return float((intersection + smooth) / (union + smooth))


def hausdorff_distance_95(
    pred: np.ndarray,
    target: np.ndarray,
) -> float:
    """95th percentile Hausdorff distance between binary masks."""
    try:
        from scipy.spatial.distance import directed_hausdorff
        pred_pts   = np.argwhere(pred   > 0.5)
        target_pts = np.argwhere(target > 0.5)
        if len(pred_pts) == 0 or len(target_pts) == 0:
            return float("inf")
        d1 = directed_hausdorff(pred_pts,   target_pts)[0]
        d2 = directed_hausdorff(target_pts, pred_pts)[0]
        return float(max(d1, d2))
    except Exception:
        return -1.0


def compute_segmentation_metrics(
    preds:   List[Union[np.ndarray, torch.Tensor]],
    targets: List[Union[np.ndarray, torch.Tensor]],
    class_names: Optional[List[str]] = None,
) -> SegmentationMetrics:
    """
    Compute segmentation metrics over a list of prediction/target pairs.

    Args:
        preds:   list of (H,W) binary prediction masks
        targets: list of (H,W) binary ground truth masks

    Returns:
        SegmentationMetrics dataclass
    """
    dices, ious, hds = [], [], []
    tp_total = fp_total = fn_total = tn_total = 0

    for pred, target in zip(preds, targets):
        if isinstance(pred,   torch.Tensor): pred   = pred.detach().cpu().numpy()
        if isinstance(target, torch.Tensor): target = target.detach().cpu().numpy()

        pred_bin   = (pred   > 0.5).astype(float)
        target_bin = (target > 0.5).astype(float)

        dices.append(dice_score(pred_bin, target_bin))
        ious.append(iou_score(pred_bin, target_bin))
        hds.append(hausdorff_distance_95(pred_bin, target_bin))

        pred_flat   = pred_bin.flatten()
        target_flat = target_bin.flatten()
        tp_total += (pred_flat * target_flat).sum()
        fp_total += (pred_flat * (1 - target_flat)).sum()
        fn_total += ((1 - pred_flat) * target_flat).sum()
        tn_total += ((1 - pred_flat) * (1 - target_flat)).sum()

    valid_hds = [h for h in hds if h >= 0]

    metrics = SegmentationMetrics(
        dice          = float(np.mean(dices)),
        iou           = float(np.mean(ious)),
        sensitivity   = float(tp_total / (tp_total + fn_total + 1e-9)),
        specificity   = float(tn_total / (tn_total + fp_total + 1e-9)),
        precision     = float(tp_total / (tp_total + fp_total + 1e-9)),
        pixel_accuracy= float((tp_total + tn_total) / (tp_total + tn_total + fp_total + fn_total + 1e-9)),
        hausdorff_95  = float(np.percentile(valid_hds, 95)) if valid_hds else -1.0,
        num_samples   = len(preds),
        class_names   = class_names or [],
    )
    return metrics


# ─────────────────────────────────────────────────────────────────────────────
# ROC Curve Plotting
# ─────────────────────────────────────────────────────────────────────────────

def plot_roc_curves(
    metrics: ClassificationMetrics,
    save_path: Optional[str] = None,
    show: bool = False,
) -> Optional[object]:
    """Plot multi-class ROC curves."""
    try:
        import matplotlib.pyplot as plt
        import matplotlib.colors as mcolors

        if not metrics.roc_curves:
            return None

        fig, ax = plt.subplots(figsize=(8, 6))
        colors = list(mcolors.TABLEAU_COLORS.values())
        names = metrics.class_names or [str(i) for i in metrics.roc_curves]

        for i, (cls, (fpr, tpr, _)) in enumerate(metrics.roc_curves.items()):
            auc_val = metrics.per_class_auc.get(cls, 0)
            name    = names[cls] if cls < len(names) else str(cls)
            ax.plot(fpr, tpr, color=colors[i % len(colors)], lw=2,
                    label=f"{name} (AUC={auc_val:.3f})")

        ax.plot([0, 1], [0, 1], "k--", lw=1, label="Random")
        ax.set_xlim([0, 1]); ax.set_ylim([0, 1.02])
        ax.set_xlabel("False Positive Rate", fontsize=12)
        ax.set_ylabel("True Positive Rate", fontsize=12)
        ax.set_title(f"ROC Curves — Macro AUC={metrics.auc:.3f}", fontsize=14)
        ax.legend(loc="lower right", fontsize=9)
        ax.grid(alpha=0.3)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
        if show:
            plt.show()

        return fig

    except ImportError:
        return None


def plot_confusion_matrix(
    metrics: ClassificationMetrics,
    save_path: Optional[str] = None,
    normalize: bool = True,
    show: bool = False,
) -> Optional[object]:
    """Plot normalized confusion matrix heatmap."""
    try:
        import matplotlib.pyplot as plt
        import matplotlib.colors as mcolors

        cm = metrics.confusion_matrix
        if cm is None:
            return None

        if normalize:
            cm_plot = cm.astype(float) / (cm.sum(axis=1, keepdims=True) + 1e-9)
        else:
            cm_plot = cm

        names = metrics.class_names or [str(i) for i in range(len(cm))]
        fig, ax = plt.subplots(figsize=(8, 7))
        im = ax.imshow(cm_plot, interpolation="nearest", cmap="Blues",
                       vmin=0, vmax=1 if normalize else None)
        plt.colorbar(im, ax=ax)

        ax.set_xticks(range(len(names))); ax.set_xticklabels(names, rotation=45, ha="right")
        ax.set_yticks(range(len(names))); ax.set_yticklabels(names)
        ax.set_xlabel("Predicted Label", fontsize=12)
        ax.set_ylabel("True Label", fontsize=12)
        ax.set_title("Confusion Matrix" + (" (Normalized)" if normalize else ""), fontsize=14)

        thresh = cm_plot.max() / 2.0
        for i in range(len(names)):
            for j in range(len(names)):
                val = f"{cm_plot[i, j]:.2f}" if normalize else str(cm[i, j])
                ax.text(j, i, val, ha="center", va="center",
                        color="white" if cm_plot[i, j] > thresh else "black", fontsize=9)

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
        if show:
            plt.show()
        return fig

    except ImportError:
        return None
