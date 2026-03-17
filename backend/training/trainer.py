"""
training/trainer.py
====================
Retina-GPT: Production Trainer

Full-featured multi-task training orchestrator for Retina-GPT.
Supports:
  - Multi-task loss weighting (classification + segmentation + detection + report)
  - Mixed precision training (torch.amp)
  - Gradient accumulation (for large models on smaller GPUs)
  - Checkpoint management (best/last)
  - Experiment logging (Weights & Biases + TensorBoard)
  - Learning rate scheduling with warmup
  - Early stopping

Author: Retina-GPT Engineering Team
Date: 2026
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────
# Training Configuration
# ─────────────────────────────────────────────────────────────

@dataclass
class TrainingConfig:
    """
    Full training configuration for Retina-GPT multi-task training.
    """
    # Experiment
    experiment_name: str = "retina_gpt_v1"
    output_dir: str = "checkpoints"
    log_dir: str = "logs"
    seed: int = 42

    # Training loop
    num_epochs: int = 100
    batch_size: int = 16
    grad_accumulation_steps: int = 2     # effective batch = batch_size × grad_accum
    max_grad_norm: float = 1.0           # gradient clipping

    # Optimizer
    learning_rate: float = 1e-4
    weight_decay: float = 0.05
    betas: Tuple[float, float] = (0.9, 0.999)

    # LR Schedule: linear warmup → cosine decay
    warmup_epochs: int = 5
    min_lr: float = 1e-6

    # Mixed precision
    use_amp: bool = True
    amp_dtype: str = "float16"          # 'float16' or 'bfloat16'

    # Multi-task loss weights
    loss_weight_quality: float = 0.5
    loss_weight_classification: float = 1.0
    loss_weight_segmentation: float = 1.0
    loss_weight_detection: float = 1.0
    loss_weight_report: float = 0.5

    # Which tasks are active
    task_quality: bool = True
    task_classification: bool = True
    task_segmentation: bool = False      # requires mask labels
    task_detection: bool = False         # requires box labels
    task_report: bool = False            # requires report text labels

    # Checkpointing
    save_every_n_epochs: int = 5
    keep_last_n_checkpoints: int = 3
    monitor_metric: str = "val_auc"     # metric to determine best checkpoint
    monitor_mode: str = "max"           # 'max' or 'min'

    # Early stopping
    early_stopping_patience: int = 15

    # Logging
    use_wandb: bool = False
    use_tensorboard: bool = True
    log_every_n_steps: int = 50

    # Hardware
    num_workers: int = 4
    pin_memory: bool = True
    device: str = "auto"               # 'auto', 'cuda', 'mps', 'cpu'


# ─────────────────────────────────────────────────────────────
# Retina-GPT Model Wrapper
# ─────────────────────────────────────────────────────────────

class RetinaGPTModel(nn.Module):
    """
    Unified multi-task model wrapper that combines:
        - RetinaViT backbone
        - QualityHead
        - ClassificationHead
        - SegmentationHead
        - DetectionHead
        - MedicalReportGenerator (optional)

    This is the full Retina-GPT system as a single nn.Module.
    """

    def __init__(
        self,
        backbone,
        quality_head=None,
        cls_head=None,
        seg_head=None,
        det_head=None,
        report_gen=None,
    ):
        super().__init__()
        self.backbone     = backbone
        self.quality_head = quality_head
        self.cls_head     = cls_head
        self.seg_head     = seg_head
        self.det_head     = det_head
        self.report_gen   = report_gen

    def forward(
        self,
        images: torch.Tensor,
        prompt_ids: Optional[torch.Tensor] = None,
        prompt_mask: Optional[torch.Tensor] = None,
        report_labels: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        """
        Full forward pass through all active heads.

        Args:
            images: (B, C, H, W) preprocessed fundus tensors
            prompt_ids: (B, T) prompt token ids for report generation
            prompt_mask: (B, T) attention mask for prompts
            report_labels: (B, T) target token ids for report LM loss

        Returns:
            dict with all head outputs merged
        """
        # Backbone
        backbone_out = self.backbone(images)
        outputs = {"backbone": backbone_out}

        # Quality head
        if self.quality_head is not None:
            outputs["quality"] = self.quality_head(backbone_out)

        # Classification head
        if self.cls_head is not None:
            outputs["classification"] = self.cls_head(backbone_out)

        # Segmentation head
        if self.seg_head is not None:
            outputs["segmentation"] = self.seg_head(backbone_out)

        # Detection head
        if self.det_head is not None:
            outputs["detection"] = self.det_head(backbone_out)

        # Report generation (training forward)
        if self.report_gen is not None and prompt_ids is not None:
            outputs["report"] = self.report_gen(
                backbone_out, prompt_ids, prompt_mask, report_labels
            )

        return outputs


# ─────────────────────────────────────────────────────────────
# Multi-Task Loss Computation
# ─────────────────────────────────────────────────────────────

class MultiTaskLoss(nn.Module):
    """
    Computes weighted multi-task loss for Retina-GPT.
    
    Individual losses:
        - Quality:        CrossEntropyLoss (binary)
        - Classification: CrossEntropyLoss + Ordinal loss
        - Segmentation:   Dice loss + Binary CE
        - Detection:      Focal loss + GIoU loss + Centerness BCE
        - Report:         Language modeling CE loss (from HF LM)
    """

    def __init__(self, config: TrainingConfig):
        super().__init__()
        self.config = config

        self.quality_loss_fn = nn.CrossEntropyLoss(ignore_index=-1)
        self.cls_loss_fn     = nn.CrossEntropyLoss(ignore_index=-1)
        self.seg_loss_fn     = nn.BCEWithLogitsLoss()
        self.det_cls_fn      = nn.BCEWithLogitsLoss()

    def dice_loss(self, pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
        """Differentiable Dice loss for binary segmentation."""
        pred_sig = torch.sigmoid(pred)
        intersection = (pred_sig * target).sum(dim=(2, 3))
        union = pred_sig.sum(dim=(2, 3)) + target.sum(dim=(2, 3))
        dice = (2 * intersection + eps) / (union + eps)
        return 1 - dice.mean()

    def forward(
        self,
        outputs: Dict[str, Any],
        batch: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """
        Compute all task losses and weighted total.

        Args:
            outputs: Forward pass output dict from RetinaGPTModel
            batch: Data batch dict with labels

        Returns:
            dict with individual losses and 'total_loss'
        """
        cfg = self.config
        losses = {}
        total = torch.tensor(0.0, device=next(iter(batch.values())).device)

        # Quality loss
        if "quality" in outputs and cfg.task_quality:
            q_labels = batch.get("quality_label")
            if q_labels is not None and (q_labels >= 0).any():
                loss_q = self.quality_loss_fn(
                    outputs["quality"]["logits"], q_labels
                )
                losses["quality"] = loss_q
                total = total + cfg.loss_weight_quality * loss_q

        # Classification loss
        if "classification" in outputs and cfg.task_classification:
            dr_labels = batch.get("dr_grade")
            if dr_labels is not None and (dr_labels >= 0).any():
                loss_cls = self.cls_loss_fn(
                    outputs["classification"]["class_logits"], dr_labels
                )
                losses["classification"] = loss_cls
                total = total + cfg.loss_weight_classification * loss_cls

        # Segmentation loss (Dice + BCE)
        if "segmentation" in outputs and cfg.task_segmentation:
            seg_mask = batch.get("vessel_mask")
            if seg_mask is not None:
                seg_logits = outputs["segmentation"]["logits"]
                loss_bce  = self.seg_loss_fn(seg_logits, seg_mask)
                loss_dice = self.dice_loss(seg_logits, seg_mask)
                loss_seg  = 0.5 * loss_bce + 0.5 * loss_dice
                losses["segmentation"] = loss_seg
                total = total + cfg.loss_weight_segmentation * loss_seg

        # Report generation loss (from HF LM)
        if "report" in outputs and cfg.task_report:
            report_loss = outputs["report"].get("loss")
            if report_loss is not None:
                losses["report"] = report_loss
                total = total + cfg.loss_weight_report * report_loss

        losses["total"] = total
        return losses


# ─────────────────────────────────────────────────────────────
# Trainer
# ─────────────────────────────────────────────────────────────

class RetinaGPTTrainer:
    """
    Production trainer for the Retina-GPT multi-task model.

    Handles the complete training lifecycle:
        setup → train loop → eval loop → checkpoint → log → repeat

    Usage:
        >>> trainer = RetinaGPTTrainer(model, train_loader, val_loader, config)
        >>> trainer.train()
    """

    def __init__(
        self,
        model: RetinaGPTModel,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: Optional[TrainingConfig] = None,
    ):
        self.config = config or TrainingConfig()
        cfg = self.config

        # ── Device ──
        if cfg.device == "auto":
            self.device = (
                torch.device("cuda") if torch.cuda.is_available()
                else torch.device("mps") if torch.backends.mps.is_available()
                else torch.device("cpu")
            )
        else:
            self.device = torch.device(cfg.device)
        logger.info(f"Training device: {self.device}")

        # ── Model ──
        self.model = model.to(self.device)

        # ── Data ──
        self.train_loader = train_loader
        self.val_loader   = val_loader

        # ── Optimizer ──
        self.optimizer = AdamW(
            [p for p in self.model.parameters() if p.requires_grad],
            lr=cfg.learning_rate,
            weight_decay=cfg.weight_decay,
            betas=cfg.betas,
        )

        # ── LR Schedule: warmup + cosine decay ──
        warmup_steps = cfg.warmup_epochs * len(train_loader)
        total_steps  = cfg.num_epochs * len(train_loader)
        self.scheduler = SequentialLR(
            self.optimizer,
            schedulers=[
                LinearLR(self.optimizer, start_factor=1e-3, end_factor=1.0, total_iters=warmup_steps),
                CosineAnnealingLR(self.optimizer, T_max=total_steps - warmup_steps, eta_min=cfg.min_lr),
            ],
            milestones=[warmup_steps],
        )

        # ── Loss ──
        self.loss_fn = MultiTaskLoss(cfg)

        # ── Mixed precision ──
        self.use_amp = cfg.use_amp and self.device.type == "cuda"
        self.scaler  = GradScaler(enabled=self.use_amp)
        self.amp_dtype = torch.float16 if cfg.amp_dtype == "float16" else torch.bfloat16

        # ── Paths ──
        self.output_dir = Path(cfg.output_dir) / cfg.experiment_name
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir = Path(cfg.log_dir) / cfg.experiment_name
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # ── Logging ──
        self.writer: Optional[SummaryWriter] = None
        if cfg.use_tensorboard:
            self.writer = SummaryWriter(log_dir=str(self.log_dir))

        if cfg.use_wandb:
            try:
                import wandb
                wandb.init(project="retina-gpt", name=cfg.experiment_name, config=asdict(cfg))
                self.wandb = wandb
            except ImportError:
                logger.warning("wandb not installed. Skipping W&B logging.")
                self.wandb = None
        else:
            self.wandb = None

        # ── State tracking ──
        self.global_step   = 0
        self.best_metric   = float("-inf") if cfg.monitor_mode == "max" else float("inf")
        self.patience_ctr  = 0
        self.checkpoint_paths: List[Path] = []

        logger.info(f"Trainer initialized | {cfg.experiment_name}")
        logger.info(f"Output dir: {self.output_dir}")

    # ──────────────────────────────────────────────────────────
    # Training Loop
    # ──────────────────────────────────────────────────────────

    def train(self):
        """Main training loop."""
        cfg = self.config
        logger.info(f"Starting training for {cfg.num_epochs} epochs")

        for epoch in range(1, cfg.num_epochs + 1):
            logger.info(f"\n{'='*60}\nEpoch {epoch}/{cfg.num_epochs}\n{'='*60}")

            train_metrics = self._train_epoch(epoch)
            val_metrics   = self._eval_epoch(epoch)

            # ── Log ──
            all_metrics = {**{f"train/{k}": v for k, v in train_metrics.items()},
                           **{f"val/{k}": v for k, v in val_metrics.items()}}
            self._log_metrics(all_metrics, step=epoch)

            # ── Checkpoint ──
            monitor_val = val_metrics.get(cfg.monitor_metric.replace("val_", ""), 0.0)
            is_best = (
                monitor_val > self.best_metric if cfg.monitor_mode == "max"
                else monitor_val < self.best_metric
            )

            if is_best:
                self.best_metric = monitor_val
                self.patience_ctr = 0
                self._save_checkpoint(epoch, is_best=True)
            else:
                self.patience_ctr += 1

            if epoch % cfg.save_every_n_epochs == 0:
                self._save_checkpoint(epoch, is_best=False)

            # ── Early stopping ──
            if self.patience_ctr >= cfg.early_stopping_patience:
                logger.info(f"Early stopping triggered at epoch {epoch}. Best: {self.best_metric:.4f}")
                break

        logger.info(f"Training complete. Best {cfg.monitor_metric}: {self.best_metric:.4f}")
        if self.writer:
            self.writer.close()

    def _train_epoch(self, epoch: int) -> Dict[str, float]:
        """Run one training epoch."""
        self.model.train()
        cfg = self.config
        total_loss = 0.0
        n_batches = 0

        pbar = tqdm(self.train_loader, desc=f"Train [{epoch}]", leave=False)

        for step, batch in enumerate(pbar):
            # Move to device
            images = batch["image"].to(self.device, non_blocking=True)
            batch_gpu = {
                k: v.to(self.device, non_blocking=True) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }

            # Forward with AMP
            with autocast(device_type=self.device.type, dtype=self.amp_dtype, enabled=self.use_amp):
                outputs = self.model(images)
                losses  = self.loss_fn(outputs, batch_gpu)
                loss    = losses["total"] / cfg.grad_accumulation_steps

            # Backward
            self.scaler.scale(loss).backward()

            # Gradient accumulation step
            if (step + 1) % cfg.grad_accumulation_steps == 0:
                self.scaler.unscale_(self.optimizer)
                nn.utils.clip_grad_norm_(self.model.parameters(), cfg.max_grad_norm)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()
                self.scheduler.step()
                self.global_step += 1

            total_loss += losses["total"].item()
            n_batches  += 1

            # Logging
            if self.global_step % cfg.log_every_n_steps == 0:
                lr = self.optimizer.param_groups[0]["lr"]
                pbar.set_postfix({"loss": f"{losses['total'].item():.4f}", "lr": f"{lr:.2e}"})
                if self.writer:
                    self.writer.add_scalar("train/step_loss", losses["total"].item(), self.global_step)
                    self.writer.add_scalar("train/lr", lr, self.global_step)

        return {"loss": total_loss / max(n_batches, 1)}

    @torch.no_grad()
    def _eval_epoch(self, epoch: int) -> Dict[str, float]:
        """Run one evaluation epoch."""
        self.model.eval()
        total_loss = 0.0
        all_preds, all_labels = [], []
        n_batches = 0

        pbar = tqdm(self.val_loader, desc=f"  Val [{epoch}]", leave=False)

        for batch in pbar:
            images = batch["image"].to(self.device, non_blocking=True)
            batch_gpu = {
                k: v.to(self.device, non_blocking=True) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }

            with autocast(device_type=self.device.type, dtype=self.amp_dtype, enabled=self.use_amp):
                outputs = self.model(images)
                losses  = self.loss_fn(outputs, batch_gpu)

            total_loss += losses["total"].item()
            n_batches  += 1

            # Collect predictions for metrics
            if "classification" in outputs:
                probs  = outputs["classification"]["probabilities"].cpu()
                labels = batch.get("dr_grade", torch.full((images.shape[0],), -1))
                mask   = labels >= 0
                if mask.any():
                    all_preds.append(probs[mask])
                    all_labels.append(labels[mask])

        metrics = {"loss": total_loss / max(n_batches, 1)}

        # Compute AUC if predictions collected
        if all_preds and all_labels:
            from sklearn.metrics import roc_auc_score
            preds_np  = torch.cat(all_preds).numpy()
            labels_np = torch.cat(all_labels).numpy()
            try:
                auc = roc_auc_score(labels_np, preds_np, multi_class="ovr", average="macro")
                metrics["auc"] = float(auc)
            except Exception:
                pass

        logger.info(f"  Val | " + " | ".join(f"{k}={v:.4f}" for k, v in metrics.items()))
        return metrics

    # ──────────────────────────────────────────────────────────
    # Checkpoint Management
    # ──────────────────────────────────────────────────────────

    def _save_checkpoint(self, epoch: int, is_best: bool = False):
        """Save model checkpoint with full training state."""
        cfg = self.config
        state = {
            "epoch": epoch,
            "global_step": self.global_step,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "scaler_state_dict": self.scaler.state_dict(),
            "best_metric": self.best_metric,
            "config": asdict(cfg),
        }

        # Save latest
        latest_path = self.output_dir / "checkpoint_latest.pth"
        torch.save(state, latest_path)

        # Save best
        if is_best:
            best_path = self.output_dir / "checkpoint_best.pth"
            torch.save(state, best_path)
            logger.info(f"  ✅ New best checkpoint saved | metric={self.best_metric:.4f}")

        # Save periodic
        if epoch % cfg.save_every_n_epochs == 0:
            periodic_path = self.output_dir / f"checkpoint_epoch_{epoch:04d}.pth"
            torch.save(state, periodic_path)
            self.checkpoint_paths.append(periodic_path)

            # Prune old checkpoints
            while len(self.checkpoint_paths) > cfg.keep_last_n_checkpoints:
                old = self.checkpoint_paths.pop(0)
                if old.exists():
                    old.unlink()
                    logger.debug(f"Removed old checkpoint: {old}")

    def load_checkpoint(self, checkpoint_path: Union[str, Path]):
        """Resume training from a saved checkpoint."""
        path = Path(checkpoint_path)
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {path}")

        state = torch.load(path, map_location=self.device)
        self.model.load_state_dict(state["model_state_dict"])
        self.optimizer.load_state_dict(state["optimizer_state_dict"])
        self.scheduler.load_state_dict(state["scheduler_state_dict"])
        self.scaler.load_state_dict(state["scaler_state_dict"])
        self.global_step = state["global_step"]
        self.best_metric = state["best_metric"]

        logger.info(f"Resumed from checkpoint: {path} | epoch={state['epoch']}")
        return state["epoch"]

    # ──────────────────────────────────────────────────────────
    # Logging
    # ──────────────────────────────────────────────────────────

    def _log_metrics(self, metrics: Dict[str, float], step: int):
        """Log metrics to all configured backends."""
        if self.writer:
            for k, v in metrics.items():
                self.writer.add_scalar(k, v, step)

        if self.wandb:
            self.wandb.log(metrics, step=step)

        # Also save to JSON
        log_file = self.log_dir / "metrics.jsonl"
        with open(log_file, "a") as f:
            f.write(json.dumps({"step": step, **metrics}) + "\n")
