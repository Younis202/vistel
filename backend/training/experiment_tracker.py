"""
experiment_tracker.py — Unified Experiment Tracking for Retina-GPT
====================================================================
Single interface that logs to:
    • Weights & Biases  (primary — rich dashboards, artifact versioning)
    • TensorBoard       (local fallback)
    • JSONL log file    (always — offline backup)

Usage:
    tracker = ExperimentTracker(
        project="retina-gpt",
        run_name="dino-pretrain-v1",
        config={"embed_dim": 1024, "epochs": 100},
        backend="wandb",   # "wandb" | "tensorboard" | "all" | "none"
    )

    tracker.log({"loss": 0.42, "lr": 1e-4}, step=100)
    tracker.log_image("val/predictions", image_tensor, step=100)
    tracker.log_model("best_model.pt", aliases=["best", "v1"])
    tracker.finish()

All backends are optional — gracefully degrades to JSONL if W&B unavailable.
"""

import os
import json
import time
import logging
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any, List, Union

import torch
import numpy as np

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Backend Implementations
# ─────────────────────────────────────────────────────────────────────────────

class WandbBackend:
    """Weights & Biases logging backend."""

    def __init__(self, project: str, run_name: str, config: Dict,
                 tags: List[str] = None, group: str = None, dir: str = "./logs"):
        try:
            import wandb
            self.wandb = wandb
            self.run = wandb.init(
                project=project,
                name=run_name,
                config=config,
                tags=tags or [],
                group=group,
                dir=dir,
                resume="allow",
            )
            self.available = True
            logger.info(f"[W&B] Run started: {self.run.url}")
        except Exception as e:
            logger.warning(f"[W&B] Unavailable: {e}. Falling back to local logging.")
            self.available = False
            self.run = None

    def log(self, metrics: Dict, step: int):
        if self.available:
            self.wandb.log(metrics, step=step)

    def log_image(self, key: str, image: Any, step: int, caption: str = ""):
        if self.available:
            if isinstance(image, torch.Tensor):
                image = image.detach().cpu().permute(1, 2, 0).numpy()
                image = (image * 255).clip(0, 255).astype(np.uint8)
            self.wandb.log({key: self.wandb.Image(image, caption=caption)}, step=step)

    def log_model(self, path: str, name: str = "model", aliases: List[str] = None):
        if self.available:
            artifact = self.wandb.Artifact(name, type="model")
            artifact.add_file(path)
            self.wandb.log_artifact(artifact, aliases=aliases or ["latest"])

    def log_table(self, key: str, data: Dict[str, list]):
        if self.available:
            cols = list(data.keys())
            rows = list(zip(*data.values()))
            self.wandb.log({key: self.wandb.Table(columns=cols, data=rows)})

    def finish(self):
        if self.available and self.run:
            self.run.finish()


class TensorBoardBackend:
    """TensorBoard logging backend."""

    def __init__(self, log_dir: str):
        try:
            from torch.utils.tensorboard import SummaryWriter
            self.writer = SummaryWriter(log_dir=log_dir)
            self.available = True
            logger.info(f"[TensorBoard] Logging to {log_dir}")
        except Exception as e:
            logger.warning(f"[TensorBoard] Unavailable: {e}")
            self.available = False
            self.writer = None

    def log(self, metrics: Dict, step: int):
        if self.available:
            for k, v in metrics.items():
                if isinstance(v, (int, float)):
                    self.writer.add_scalar(k, v, global_step=step)

    def log_image(self, key: str, image: Any, step: int, caption: str = ""):
        if self.available and isinstance(image, torch.Tensor):
            img = image.detach().cpu()
            if img.dim() == 3:
                img = img.unsqueeze(0)
            self.writer.add_images(key, img, global_step=step)

    def log_model(self, path: str, name: str = "model", aliases: List[str] = None):
        pass  # TensorBoard doesn't do model registry

    def log_table(self, key: str, data: Dict[str, list]):
        pass  # Not supported

    def finish(self):
        if self.available and self.writer:
            self.writer.close()


class JSONLBackend:
    """Always-on JSONL file logging — works offline, zero dependencies."""

    def __init__(self, log_dir: str, run_name: str):
        Path(log_dir).mkdir(parents=True, exist_ok=True)
        self.path = Path(log_dir) / f"{run_name}.jsonl"
        self.available = True

    def log(self, metrics: Dict, step: int):
        record = {
            "step": step,
            "timestamp": datetime.utcnow().isoformat(),
            **{k: float(v) if isinstance(v, (int, float, torch.Tensor)) else v
               for k, v in metrics.items()},
        }
        with open(self.path, "a") as f:
            f.write(json.dumps(record) + "\n")

    def log_image(self, key: str, image: Any, step: int, caption: str = ""):
        pass  # Can't store images in JSONL

    def log_model(self, path: str, name: str = "model", aliases: List[str] = None):
        record = {
            "event": "model_saved",
            "path": path,
            "name": name,
            "aliases": aliases or [],
            "timestamp": datetime.utcnow().isoformat(),
        }
        with open(self.path, "a") as f:
            f.write(json.dumps(record) + "\n")

    def log_table(self, key: str, data: Dict[str, list]):
        record = {"table": key, "data": data, "timestamp": datetime.utcnow().isoformat()}
        with open(self.path, "a") as f:
            f.write(json.dumps(record) + "\n")

    def finish(self):
        pass  # Nothing to close

    def read_metrics(self) -> List[Dict]:
        """Read all logged metrics from JSONL."""
        if not self.path.exists():
            return []
        records = []
        with open(self.path) as f:
            for line in f:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
        return records


# ─────────────────────────────────────────────────────────────────────────────
# Unified Experiment Tracker
# ─────────────────────────────────────────────────────────────────────────────

class ExperimentTracker:
    """
    Unified experiment tracking interface for Retina-GPT training.

    Auto-detects available backends.
    Always logs to JSONL as a safety net.

    Example:
        tracker = ExperimentTracker.for_dino_pretraining(config)
        for epoch in range(config.epochs):
            loss = train_epoch(...)
            tracker.log({"train/loss": loss, "lr": lr}, step=epoch)
        tracker.finish()
    """

    def __init__(
        self,
        project: str,
        run_name: str,
        config: Dict,
        backend: str = "all",    # "wandb" | "tensorboard" | "all" | "none"
        log_dir: str = "./logs",
        tags: List[str] = None,
        group: str = None,
    ):
        self.project = project
        self.run_name = run_name
        self.config = config
        self.log_dir = Path(log_dir)
        self._step = 0
        self._start_time = time.time()

        run_log_dir = str(self.log_dir / project / run_name)
        os.makedirs(run_log_dir, exist_ok=True)

        # Always create JSONL backend
        self._jsonl = JSONLBackend(run_log_dir, run_name)

        # Conditional backends
        self._wandb = None
        self._tb = None

        if backend in ("wandb", "all"):
            self._wandb = WandbBackend(project, run_name, config, tags, group, run_log_dir)

        if backend in ("tensorboard", "all"):
            self._tb = TensorBoardBackend(run_log_dir)

        # Log config
        self._jsonl.log({"event": "run_start", "config": config}, step=0)
        logger.info(f"[ExperimentTracker] Started run '{run_name}' | project='{project}'")

    # ── Logging ──────────────────────────────────────────────────────────────

    def log(self, metrics: Dict[str, Any], step: Optional[int] = None):
        """Log scalar metrics to all backends."""
        if step is None:
            step = self._step
            self._step += 1
        else:
            self._step = step

        # Add elapsed time
        metrics["_elapsed_s"] = time.time() - self._start_time

        self._jsonl.log(metrics, step)
        if self._wandb:
            self._wandb.log(metrics, step)
        if self._tb:
            self._tb.log(metrics, step)

    def log_image(self, key: str, image: Any, step: Optional[int] = None,
                  caption: str = ""):
        """Log an image (tensor or numpy array)."""
        step = step or self._step
        if self._wandb:
            self._wandb.log_image(key, image, step, caption)
        if self._tb:
            self._tb.log_image(key, image, step, caption)

    def log_model(self, path: str, name: str = "model", aliases: List[str] = None):
        """Log a model artifact/checkpoint."""
        self._jsonl.log_model(path, name, aliases)
        if self._wandb:
            self._wandb.log_model(path, name, aliases)

    def log_table(self, key: str, data: Dict[str, list]):
        """Log a structured table."""
        self._jsonl.log_table(key, data)
        if self._wandb:
            self._wandb.log_table(key, data)

    def log_epoch_summary(self, epoch: int, train_metrics: Dict, val_metrics: Dict):
        """Convenience: log train + val metrics for one epoch."""
        merged = {
            **{f"train/{k}": v for k, v in train_metrics.items()},
            **{f"val/{k}": v for k, v in val_metrics.items()},
            "epoch": epoch,
        }
        self.log(merged, step=epoch)

    def log_confusion_matrix(self, y_true: List, y_pred: List,
                              class_names: List[str], step: int):
        """Log confusion matrix as a table."""
        if self._wandb and self._wandb.available:
            import wandb
            cm = wandb.plot.confusion_matrix(
                y_true=y_true, preds=y_pred, class_names=class_names
            )
            self._wandb.wandb.log({"confusion_matrix": cm}, step=step)

    def finish(self):
        """Finalize all backends."""
        elapsed = time.time() - self._start_time
        self._jsonl.log({"event": "run_end", "elapsed_s": elapsed}, step=self._step)
        self._jsonl.finish()
        if self._wandb:
            self._wandb.finish()
        if self._tb:
            self._tb.finish()
        logger.info(f"[ExperimentTracker] Run '{self.run_name}' finished ({elapsed:.0f}s)")

    # ── Factory Methods ───────────────────────────────────────────────────────

    @classmethod
    def for_dino_pretraining(cls, config, backend: str = "all") -> "ExperimentTracker":
        return cls(
            project="retina-gpt",
            run_name=f"dino-pretrain-{datetime.now().strftime('%Y%m%d-%H%M')}",
            config=vars(config) if hasattr(config, '__dict__') else config,
            backend=backend,
            tags=["dino", "pretraining", "self-supervised"],
            group="stage1-pretraining",
        )

    @classmethod
    def for_clip_training(cls, config, backend: str = "all") -> "ExperimentTracker":
        return cls(
            project="retina-gpt",
            run_name=f"clip-train-{datetime.now().strftime('%Y%m%d-%H%M')}",
            config=vars(config) if hasattr(config, '__dict__') else config,
            backend=backend,
            tags=["clip", "vision-language"],
            group="stage2-clip",
        )

    @classmethod
    def for_multitask_training(cls, config, backend: str = "all") -> "ExperimentTracker":
        return cls(
            project="retina-gpt",
            run_name=f"multitask-{datetime.now().strftime('%Y%m%d-%H%M')}",
            config=vars(config) if hasattr(config, '__dict__') else config,
            backend=backend,
            tags=["multitask", "fine-tuning", "classification"],
            group="stage3-multitask",
        )

    @classmethod
    def for_sam_training(cls, config, backend: str = "all") -> "ExperimentTracker":
        return cls(
            project="retina-gpt",
            run_name=f"sam-train-{datetime.now().strftime('%Y%m%d-%H%M')}",
            config=vars(config) if hasattr(config, '__dict__') else config,
            backend=backend,
            tags=["sam", "segmentation"],
            group="stage4-sam",
        )

    # ── Utilities ─────────────────────────────────────────────────────────────

    @property
    def run_url(self) -> Optional[str]:
        if self._wandb and self._wandb.available:
            return self._wandb.run.url
        return None

    def get_best_metric(self, metric_name: str, mode: str = "min") -> float:
        """Read JSONL history and return best value of a metric."""
        records = self._jsonl.read_metrics()
        values = [r[metric_name] for r in records if metric_name in r]
        if not values:
            return float("inf") if mode == "min" else float("-inf")
        return min(values) if mode == "min" else max(values)
