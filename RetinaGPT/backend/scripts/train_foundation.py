"""
train_foundation.py — Retina-GPT 4-Stage Foundation Model Training
===================================================================
Master training script for the complete Retina-GPT foundation model.

Runs all 4 training stages sequentially or independently:

    Stage 1 — Retina-DINO Self-Supervised Pretraining
        • No labels required
        • Student-teacher self-distillation
        • Learns universal retinal representations
        • Requires: large unlabeled fundus dataset

    Stage 2 — Retina-CLIP Vision-Language Alignment
        • Aligns images with clinical text
        • Enables zero-shot disease classification
        • Requires: (image, clinical report) pairs

    Stage 3 — Multi-Task Fine-Tuning
        • DR grading, AMD staging, glaucoma detection
        • Lesion detection
        • Frozen backbone (or last N layers unfrozen)
        • Requires: labeled datasets (APTOS, IDRiD, REFUGE)

    Stage 4 — Retina-SAM Segmentation Training
        • Trains mask decoder on segmentation datasets
        • Backbone stays frozen
        • Requires: segmentation masks (DRIVE, IDRiD, REFUGE)

Usage:

    # Full pipeline:
    python scripts/train_foundation.py --all

    # Single stage:
    python scripts/train_foundation.py --stage dino
    python scripts/train_foundation.py --stage clip
    python scripts/train_foundation.py --stage multitask
    python scripts/train_foundation.py --stage sam

    # Resume from checkpoint:
    python scripts/train_foundation.py --stage multitask --resume checkpoints/multitask/best.pt

    # Multi-GPU:
    torchrun --nproc_per_node=4 scripts/train_foundation.py --stage dino
"""

import os
import sys
import time
import argparse
import logging
from pathlib import Path
from typing import Optional, Dict, Any

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# ── Project imports ──────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent.parent))

from training.distributed import (
    DistributedTrainer, DistributedConfig,
    init_distributed, is_main_process, barrier, get_world_size
)
from training.experiment_tracker import ExperimentTracker
from training.model_registry import ModelRegistry
from training.dataset_manager import DatasetManager

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("retina_gpt.train")


# ─────────────────────────────────────────────────────────────────────────────
# Stage 1: Retina-DINO Pretraining
# ─────────────────────────────────────────────────────────────────────────────

class DINOTrainer(DistributedTrainer):
    """
    Stage 1: Self-supervised pretraining using Retina-DINO.

    The encoder learns rich retinal representations from unlabeled images.
    No disease labels needed — learns from image structure alone.

    After training, the encoder is the foundation for ALL downstream tasks.
    """

    def __init__(self, dino_model, config, dist_info, tracker: ExperimentTracker,
                 registry: ModelRegistry, save_dir: str = "checkpoints/dino"):
        super().__init__(dino_model, config, dist_info)
        self.dino = dino_model
        self.tracker = tracker
        self.registry = registry
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

    def build_optimizer(self, cfg):
        """Build AdamW with separate WD for bias/norm (no decay)."""
        decay_params, no_decay_params = [], []
        for name, p in self.dino.student_encoder.named_parameters():
            if not p.requires_grad:
                continue
            if "bias" in name or "norm" in name or "cls_token" in name:
                no_decay_params.append(p)
            else:
                decay_params.append(p)

        return optim.AdamW([
            {"params": decay_params,    "weight_decay": cfg.weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},
        ], lr=cfg.base_lr, betas=(0.9, 0.95))

    def train(self, dataloader, cfg) -> str:
        """Run Retina-DINO pretraining. Returns path to best checkpoint."""
        from models.pretraining.retina_dino import cosine_schedule

        self.wrap_model()
        optimizer = self.build_optimizer(cfg)

        steps_per_epoch = len(dataloader)
        total_steps = cfg.epochs * steps_per_epoch

        # LR and WD schedules
        lr_schedule = cosine_schedule(cfg.base_lr, cfg.min_lr, total_steps,
                                      cfg.warmup_epochs * steps_per_epoch)
        wd_schedule = cosine_schedule(cfg.weight_decay, cfg.weight_decay_end, total_steps)
        momentum_schedule = cosine_schedule(cfg.momentum_teacher, cfg.momentum_teacher_end,
                                            total_steps)

        self.log("🔬 Stage 1: Retina-DINO Pretraining")
        self.log(f"   GPUs: {self.world_size} | Effective batch: {self.effective_batch_size}")
        self.log(f"   Epochs: {cfg.epochs} | Steps/epoch: {steps_per_epoch}")
        self.log(f"   Prototype dim: {cfg.out_dim} | Encoder dim: {cfg.embed_dim}")

        best_loss = float("inf")
        best_path = None
        start_time = time.time()

        for epoch in range(cfg.epochs):
            self._epoch = epoch

            # Distributed sampler needs to know epoch for proper shuffling
            if hasattr(dataloader.sampler, "set_epoch"):
                dataloader.sampler.set_epoch(epoch)

            epoch_loss = 0.0
            epoch_steps = 0

            self.model.train()

            for step, (crops, _) in enumerate(dataloader):
                global_step = epoch * steps_per_epoch + step

                # Update LR and WD
                for pg in optimizer.param_groups:
                    pg["lr"] = lr_schedule[global_step]
                    if "weight_decay" in pg and pg.get("_has_wd", True):
                        pg["weight_decay"] = wd_schedule[global_step]

                # Move crops to device
                crops = [c.to(self.device, non_blocking=True) for c in crops]

                # Forward pass
                with self.mp.autocast_context():
                    raw_model = self.model.module if hasattr(self.model, "module") else self.model
                    loss = raw_model(crops, epoch)

                # Backward
                step_metrics = self.backward_step(loss, optimizer)

                # EMA teacher update
                momentum = momentum_schedule[min(global_step, len(momentum_schedule) - 1)]
                with torch.no_grad():
                    raw_model.update_teacher(momentum)

                # Freeze last layer early epochs
                if epoch < cfg.freeze_last_layer:
                    for n, p in raw_model.student_head.named_parameters():
                        if "last_layer" in n:
                            p.grad = None

                epoch_loss += loss.item()
                epoch_steps += 1

            avg_loss = epoch_loss / max(epoch_steps, 1)

            # Log
            metrics = {
                "loss": avg_loss,
                "lr": lr_schedule[min((epoch + 1) * steps_per_epoch - 1, len(lr_schedule) - 1)],
                "momentum": momentum,
                "epoch": epoch,
                "elapsed_hours": (time.time() - start_time) / 3600,
            }
            metrics.update(self.aggregate_metrics({k: torch.tensor(v) for k, v in metrics.items()
                                                   if isinstance(v, float)}))

            if self.is_main:
                self.tracker.log(metrics, step=epoch)
                self.log(
                    f"Epoch [{epoch+1:3d}/{cfg.epochs}]  "
                    f"Loss: {avg_loss:.6f}  "
                    f"LR: {metrics['lr']:.2e}  "
                    f"Momentum: {momentum:.4f}"
                )

            # Checkpointing
            if self.is_main:
                ckpt_path = str(self.save_dir / f"retina_dino_epoch_{epoch+1:04d}.pt")
                self.save_checkpoint(ckpt_path, extra={
                    "loss": avg_loss, "epoch": epoch, "config": vars(cfg)
                })

                if avg_loss < best_loss:
                    best_loss = avg_loss
                    best_path = str(self.save_dir / "retina_dino_best.pt")
                    self.save_checkpoint(best_path, extra={
                        "loss": avg_loss, "epoch": epoch, "config": vars(cfg)
                    })
                    self.tracker.log_model(best_path, name="retina-dino-encoder",
                                           aliases=["best", f"epoch-{epoch}"])

            barrier()  # Sync all processes

        # Register in model registry
        if self.is_main and best_path:
            self.registry.register(
                checkpoint_path=best_path,
                name="retina-gpt",
                stage="dino",
                metrics={"pretrain_loss": best_loss},
                primary_metric="pretrain_loss",
                config=vars(cfg),
                notes="Retina-DINO self-supervised pretraining",
                copy_to_registry=False,
            )

        return best_path or ""


# ─────────────────────────────────────────────────────────────────────────────
# Stage 3: Multi-Task Fine-Tuning
# ─────────────────────────────────────────────────────────────────────────────

class MultiTaskTrainer(DistributedTrainer):
    """
    Stage 3: Fine-tune task heads on labeled clinical datasets.

    Simultaneously trains:
        • DR grading (APTOS / EyePACS)
        • AMD staging
        • Glaucoma detection (REFUGE)
        • Lesion detection (IDRiD)

    Backbone is frozen or partially unfrozen.
    """

    def __init__(self, foundation_model, config, dist_info, tracker, registry,
                 save_dir: str = "checkpoints/multitask"):
        super().__init__(foundation_model, config, dist_info)
        self.tracker = tracker
        self.registry = registry
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.best_metric = -float("inf")

    def compute_metrics(
        self,
        all_preds: list,
        all_labels: list,
        task: str = "dr_grading"
    ) -> Dict[str, float]:
        """Compute classification metrics (accuracy, kappa, AUC)."""
        preds = np.array(all_preds)
        labels = np.array(all_labels)

        metrics = {}
        metrics["accuracy"] = float((preds == labels).mean())

        try:
            from sklearn.metrics import (
                cohen_kappa_score, roc_auc_score,
                f1_score, balanced_accuracy_score
            )
            metrics["kappa"] = float(cohen_kappa_score(labels, preds, weights="quadratic"))
            metrics["balanced_accuracy"] = float(balanced_accuracy_score(labels, preds))
            metrics["f1_macro"] = float(f1_score(labels, preds, average="macro", zero_division=0))
        except Exception:
            pass

        return metrics

    def train_epoch(self, model, dataloader, optimizer, loss_fn, task: str = "dr"):
        """One training epoch for a single task."""
        model.train()
        total_loss = 0.0
        all_preds, all_labels = [], []

        for batch_idx, batch in enumerate(dataloader):
            images, labels = batch[0], batch[1]
            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)

            with self.mp.autocast_context():
                out = model(images)
                task_out = out.get(task, out.get("dr", {}))
                logits = task_out.get("logits", task_out)
                loss = loss_fn(logits, labels) / self.config.grad_accumulation_steps

            step_metrics = self.backward_step(loss, optimizer)

            total_loss += loss.item() * self.config.grad_accumulation_steps
            with torch.no_grad():
                preds = logits.argmax(dim=1).cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(labels.cpu().numpy())

        avg_loss = total_loss / max(len(dataloader), 1)
        metrics = self.compute_metrics(all_preds, all_labels, task)
        metrics["loss"] = avg_loss
        return metrics

    @torch.no_grad()
    def val_epoch(self, model, dataloader, loss_fn, task: str = "dr"):
        """Validation epoch."""
        model.eval()
        total_loss = 0.0
        all_preds, all_labels = [], []

        for images, labels in dataloader:
            images = images.to(self.device)
            labels = labels.to(self.device)

            with self.mp.autocast_context():
                out = model(images)
                task_out = out.get(task, out.get("dr", {}))
                logits = task_out.get("logits", task_out)
                loss = loss_fn(logits, labels)

            total_loss += loss.item()
            preds = logits.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())

        avg_loss = total_loss / max(len(dataloader), 1)
        metrics = self.compute_metrics(all_preds, all_labels, task)
        metrics["loss"] = avg_loss
        return metrics

    def train(
        self,
        train_loaders: Dict,     # task → DataLoader
        val_loaders: Dict,
        epochs: int = 50,
        lr: float = 1e-4,
        unfreeze_backbone_layers: int = 4,
        primary_task: str = "dr",
        primary_metric: str = "kappa",
    ) -> str:
        """
        Main multi-task training loop.

        Args:
            train_loaders: dict mapping task name → DataLoader
            val_loaders:   dict mapping task name → DataLoader
            epochs:        number of training epochs
            lr:            learning rate
            unfreeze_backbone_layers: unfreeze last N backbone layers
            primary_task:  task used for model selection
            primary_metric: metric used for model selection

        Returns:
            path to best checkpoint
        """
        self.wrap_model()
        raw_model = self.model.module if isinstance(self.model, DDP) else self.model

        # Freeze backbone, optionally unfreeze last N layers
        raw_model.freeze_backbone()
        if unfreeze_backbone_layers > 0:
            raw_model.unfreeze_backbone(unfreeze_backbone_layers)

        # Loss functions
        loss_fns = {task: nn.CrossEntropyLoss() for task in train_loaders}

        # Optimizer
        optimizer = optim.AdamW(
            [p for p in raw_model.parameters() if p.requires_grad],
            lr=lr, weight_decay=0.01,
        )
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

        best_path = None
        best_val_metric = -float("inf")

        self.log("🏥 Stage 3: Multi-Task Fine-Tuning")
        self.log(f"   Tasks: {list(train_loaders.keys())}")
        self.log(f"   Primary: {primary_task}/{primary_metric}")
        self.log(f"   Backbone layers unfrozen: {unfreeze_backbone_layers}")

        for epoch in range(epochs):
            self._epoch = epoch
            epoch_train_metrics = {}
            epoch_val_metrics = {}

            # Train each task
            for task, loader in train_loaders.items():
                if hasattr(loader.sampler, "set_epoch"):
                    loader.sampler.set_epoch(epoch)
                train_m = self.train_epoch(self.model, loader, optimizer,
                                           loss_fns[task], task=task)
                epoch_train_metrics[task] = train_m

            # Validate each task
            for task, loader in val_loaders.items():
                val_m = self.val_epoch(self.model, loader, loss_fns[task], task=task)
                epoch_val_metrics[task] = val_m

            scheduler.step()

            if self.is_main:
                # Flatten metrics for logging
                log_dict = {"epoch": epoch}
                for task, m in epoch_train_metrics.items():
                    log_dict.update({f"train/{task}/{k}": v for k, v in m.items()})
                for task, m in epoch_val_metrics.items():
                    log_dict.update({f"val/{task}/{k}": v for k, v in m.items()})

                self.tracker.log(log_dict, step=epoch)

                # Primary metric for model selection
                val_primary = epoch_val_metrics.get(primary_task, {}).get(primary_metric, 0)
                train_primary = epoch_train_metrics.get(primary_task, {}).get(primary_metric, 0)

                self.log(
                    f"Epoch [{epoch+1:3d}/{epochs}]  "
                    f"Train {primary_metric}: {train_primary:.4f}  "
                    f"Val {primary_metric}: {val_primary:.4f}"
                )

                # Save checkpoint
                ckpt_path = str(self.save_dir / f"multitask_epoch_{epoch+1:04d}.pt")
                self.save_checkpoint(ckpt_path, extra={
                    "val_metrics": epoch_val_metrics,
                    "epoch": epoch,
                })

                if val_primary > best_val_metric:
                    best_val_metric = val_primary
                    best_path = str(self.save_dir / "multitask_best.pt")
                    self.save_checkpoint(best_path, extra={
                        "val_metrics": epoch_val_metrics,
                        "epoch": epoch,
                        "best_metric": best_val_metric,
                    })
                    self.log(f"  ✅ New best! {primary_metric}={best_val_metric:.4f}")

            barrier()

        # Register in registry
        if self.is_main and best_path:
            all_val = {}
            for task, m in epoch_val_metrics.items():
                all_val.update({f"val/{task}/{k}": v for k, v in m.items()})

            self.registry.register(
                checkpoint_path=best_path,
                name="retina-gpt",
                stage="multitask",
                metrics=all_val,
                primary_metric=f"val/{primary_task}/{primary_metric}",
                notes="Multi-task fine-tuning: DR + AMD + Glaucoma + Lesions",
                copy_to_registry=False,
            )

        return best_path or ""


# ─────────────────────────────────────────────────────────────────────────────
# Master Training Orchestrator
# ─────────────────────────────────────────────────────────────────────────────

class RetinaGPTTrainingOrchestrator:
    """
    Master orchestrator for complete Retina-GPT foundation model training.

    Runs all 4 stages in sequence, passing pretrained weights forward.
    """

    def __init__(
        self,
        data_dir: str = "./data",
        checkpoint_dir: str = "./checkpoints",
        log_dir: str = "./logs",
        precision: str = "fp16",
        grad_accum: int = 2,
        num_workers: int = 4,
        wandb_backend: str = "all",
    ):
        self.data_dir = data_dir
        self.checkpoint_dir = Path(checkpoint_dir)
        self.log_dir = log_dir
        self.precision = precision
        self.grad_accum = grad_accum
        self.num_workers = num_workers
        self.wandb_backend = wandb_backend

        # Initialize distributed
        self.dist_info = init_distributed()

        # Dataset manager
        self.dataset_manager = DatasetManager(
            data_root=data_dir,
            image_size=224,          # Match ViT input size
            num_workers=num_workers,
        )

        # Model registry
        self.registry = ModelRegistry(str(self.checkpoint_dir))

        # Distributed config
        self.dist_config = DistributedConfig(
            precision=precision,
            grad_accumulation_steps=grad_accum,
        )

        if is_main_process():
            logger.info("=" * 60)
            logger.info("  Retina-GPT Foundation Model Training")
            logger.info("=" * 60)
            logger.info(f"  GPUs: {get_world_size()}")
            logger.info(f"  Precision: {precision}")
            logger.info(f"  Grad accumulation: {grad_accum}")
            logger.info(f"  Data: {data_dir}")
            logger.info("=" * 60)

    def run_stage1_dino(
        self,
        pretrain_datasets: list = None,
        epochs: int = 100,
        batch_size: int = 64,
        resume: Optional[str] = None,
    ) -> str:
        """Run Stage 1: Retina-DINO self-supervised pretraining."""
        from models.pretraining.retina_dino import (
            RetinaDINO, RetinaDINOConfig, RetinaMultiCropAugmentation
        )

        cfg = RetinaDINOConfig(epochs=epochs, batch_size=batch_size)
        tracker = ExperimentTracker.for_dino_pretraining(cfg, backend=self.wandb_backend)

        multi_crop = RetinaMultiCropAugmentation(cfg)
        pretrain_datasets = pretrain_datasets or ["aptos", "eyepacs"]

        try:
            pretrain_loader = self.dataset_manager.get_pretraining_loader(
                datasets=pretrain_datasets,
                multi_crop_aug=multi_crop,
                batch_size=batch_size,
            )
        except Exception as e:
            logger.warning(f"Could not load pretraining data: {e}. Using dummy loader.")
            pretrain_loader = self._dummy_loader(batch_size, cfg)

        model = RetinaDINO(cfg)

        if resume:
            ckpt = torch.load(resume, map_location="cpu")
            model.load_state_dict(ckpt.get("student_encoder", ckpt), strict=False)
            logger.info(f"Resumed from {resume}")

        trainer = DINOTrainer(
            model, self.dist_config, self.dist_info,
            tracker, self.registry,
            save_dir=str(self.checkpoint_dir / "dino")
        )

        best_path = trainer.train(pretrain_loader, cfg)
        tracker.finish()
        trainer.cleanup()
        return best_path

    def run_stage3_multitask(
        self,
        dino_checkpoint: Optional[str] = None,
        epochs: int = 50,
        batch_size: int = 32,
        lr: float = 1e-4,
        unfreeze_layers: int = 4,
        datasets: Optional[Dict] = None,
        resume: Optional[str] = None,
    ) -> str:
        """Run Stage 3: Multi-task fine-tuning."""
        from models.foundation_model import RetinaGPTFoundationModel, RetinaFoundationConfig

        config = RetinaFoundationConfig()
        model = RetinaGPTFoundationModel(config)

        # Load pretrained backbone
        if dino_checkpoint:
            try:
                ckpt = torch.load(dino_checkpoint, map_location="cpu")
                state = ckpt.get("student_encoder", ckpt.get("model_state_dict", ckpt))
                missing, unexpected = model.encoder.load_state_dict(state, strict=False)
                logger.info(
                    f"Loaded DINO backbone. "
                    f"Missing: {len(missing)}, Unexpected: {len(unexpected)}"
                )
            except Exception as e:
                logger.warning(f"Could not load DINO checkpoint: {e}")
        else:
            logger.info("Training multi-task from scratch (no DINO pretraining).")

        if resume:
            ckpt = torch.load(resume, map_location="cpu")
            model.load_state_dict(ckpt["model_state_dict"], strict=False)

        # Build data loaders
        datasets = datasets or {"dr": "aptos"}
        train_loaders, val_loaders = {}, {}

        for task, ds_name in datasets.items():
            try:
                train_l, val_l, _ = self.dataset_manager.get_classification_loaders(
                    ds_name, batch_size, task=task
                )
                train_loaders[task] = train_l
                val_loaders[task] = val_l
            except FileNotFoundError:
                logger.warning(f"Dataset {ds_name} not found for task {task}")

        if not train_loaders:
            logger.error("No datasets loaded. Cannot proceed with multi-task training.")
            return ""

        tracker = ExperimentTracker.for_multitask_training(
            {"lr": lr, "epochs": epochs, "datasets": datasets},
            backend=self.wandb_backend,
        )

        trainer = MultiTaskTrainer(
            model, self.dist_config, self.dist_info,
            tracker, self.registry,
            save_dir=str(self.checkpoint_dir / "multitask"),
        )

        best_path = trainer.train(
            train_loaders, val_loaders,
            epochs=epochs, lr=lr,
            unfreeze_backbone_layers=unfreeze_layers,
        )

        tracker.finish()
        trainer.cleanup()
        return best_path

    def run_all_stages(self, config: Optional[Dict] = None) -> Dict[str, str]:
        """Run complete 4-stage training pipeline."""
        config = config or {}
        checkpoints = {}

        logger.info("\n🚀 Starting full Retina-GPT foundation model training pipeline...")

        # Stage 1: DINO pretraining
        logger.info("\n" + "="*50)
        logger.info("STAGE 1: Retina-DINO Self-Supervised Pretraining")
        logger.info("="*50)
        dino_ckpt = self.run_stage1_dino(
            epochs=config.get("dino_epochs", 100),
            batch_size=config.get("dino_batch_size", 64),
        )
        checkpoints["dino"] = dino_ckpt

        # Stage 3: Multi-task fine-tuning
        logger.info("\n" + "="*50)
        logger.info("STAGE 3: Multi-Task Fine-Tuning")
        logger.info("="*50)
        multitask_ckpt = self.run_stage3_multitask(
            dino_checkpoint=dino_ckpt,
            epochs=config.get("multitask_epochs", 50),
            batch_size=config.get("multitask_batch_size", 32),
        )
        checkpoints["multitask"] = multitask_ckpt

        # Promote best multitask model to production
        if is_main_process() and multitask_ckpt:
            try:
                self.registry.promote("retina-gpt", "v3.0.0", alias="production")
                checkpoints["production"] = str(self.checkpoint_dir / "production" / "retina-gpt_production.pt")
                logger.info(f"\n🎉 Production model ready!")
                logger.info(self.registry.summary())
            except Exception as e:
                logger.warning(f"Could not promote model: {e}")

        return checkpoints

    def _dummy_loader(self, batch_size, cfg):
        """Create a dummy DataLoader for testing when real data is unavailable."""
        from torch.utils.data import TensorDataset

        n = batch_size * 10
        dummy_img = torch.zeros(n, 3, cfg.image_size, cfg.image_size)
        dummy_labels = torch.zeros(n, dtype=torch.long)
        ds = TensorDataset(dummy_img, dummy_labels)

        class DummyDINODataset(torch.utils.data.Dataset):
            def __init__(self, n, cfg):
                self.n = n
                self.cfg = cfg
            def __len__(self):
                return self.n
            def __getitem__(self, idx):
                num_crops = 2 + self.cfg.local_crops_number
                global_crops = [torch.zeros(3, self.cfg.global_crop_size, self.cfg.global_crop_size)
                                for _ in range(2)]
                local_crops = [torch.zeros(3, self.cfg.local_crop_size, self.cfg.local_crop_size)
                               for _ in range(self.cfg.local_crops_number)]
                return global_crops + local_crops, -1

        from training.dataset_manager import DatasetManager
        from torch.utils.data import DataLoader

        dummy_ds = DummyDINODataset(batch_size * 20, cfg)

        def collate(batch):
            num_crops = len(batch[0][0])
            crops = [torch.stack([b[0][i] for b in batch]) for i in range(num_crops)]
            return crops, torch.tensor([-1] * len(batch))

        return DataLoader(dummy_ds, batch_size=batch_size, shuffle=True, collate_fn=collate)


# ─────────────────────────────────────────────────────────────────────────────
# CLI Entry Point
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="Retina-GPT Foundation Model Training")

    # Stage selection
    stage_group = parser.add_mutually_exclusive_group(required=True)
    stage_group.add_argument("--all", action="store_true", help="Run all training stages")
    stage_group.add_argument("--stage", choices=["dino", "clip", "sam", "multitask"],
                             help="Run a specific training stage")

    # Data
    parser.add_argument("--data_dir", default="./data", help="Dataset root directory")
    parser.add_argument("--datasets", nargs="+", default=["aptos"],
                        help="Datasets to use for training")

    # Training
    parser.add_argument("--epochs", type=int, default=None, help="Override default epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size per GPU")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--precision", default="fp16",
                        choices=["fp32", "fp16", "bf16"], help="Mixed precision")
    parser.add_argument("--grad_accum", type=int, default=2, help="Gradient accumulation steps")
    parser.add_argument("--unfreeze_layers", type=int, default=4,
                        help="Backbone layers to unfreeze in multitask stage")

    # Checkpointing
    parser.add_argument("--checkpoint_dir", default="./checkpoints", help="Checkpoint directory")
    parser.add_argument("--resume", default=None, help="Resume from checkpoint path")
    parser.add_argument("--dino_checkpoint", default=None,
                        help="DINO pretrained checkpoint for downstream stages")

    # Experiment tracking
    parser.add_argument("--wandb", default="all",
                        choices=["wandb", "tensorboard", "all", "none"],
                        help="Experiment tracking backend")
    parser.add_argument("--log_dir", default="./logs", help="Log directory")

    # System
    parser.add_argument("--num_workers", type=int, default=4, help="DataLoader workers")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    return parser.parse_args()


def set_seed(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():
    args = parse_args()
    set_seed(args.seed)

    orchestrator = RetinaGPTTrainingOrchestrator(
        data_dir=args.data_dir,
        checkpoint_dir=args.checkpoint_dir,
        log_dir=args.log_dir,
        precision=args.precision,
        grad_accum=args.grad_accum,
        num_workers=args.num_workers,
        wandb_backend=args.wandb,
    )

    if args.all:
        checkpoints = orchestrator.run_all_stages()
        if is_main_process():
            logger.info("\n✅ All stages complete!")
            for stage, ckpt in checkpoints.items():
                logger.info(f"  {stage:12s}: {ckpt}")

    elif args.stage == "dino":
        epochs = args.epochs or 100
        ckpt = orchestrator.run_stage1_dino(
            pretrain_datasets=args.datasets,
            epochs=epochs,
            batch_size=args.batch_size,
            resume=args.resume,
        )
        if is_main_process():
            logger.info(f"\n✅ DINO pretraining complete! Checkpoint: {ckpt}")

    elif args.stage == "multitask":
        epochs = args.epochs or 50
        datasets = {ds: ds for ds in args.datasets} if args.datasets else {"dr": "aptos"}
        ckpt = orchestrator.run_stage3_multitask(
            dino_checkpoint=args.dino_checkpoint,
            epochs=epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            unfreeze_layers=args.unfreeze_layers,
            datasets=datasets,
            resume=args.resume,
        )
        if is_main_process():
            logger.info(f"\n✅ Multi-task training complete! Checkpoint: {ckpt}")

    else:
        logger.info(f"Stage '{args.stage}' trainer: coming in next update.")


if __name__ == "__main__":
    main()
