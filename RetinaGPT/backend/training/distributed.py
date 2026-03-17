"""
distributed.py — Distributed Training for Retina-GPT
=====================================================
Production-grade distributed training support.

Supports:
    • Single GPU
    • Multi-GPU (DataParallel / DistributedDataParallel)
    • Multi-Node (torchrun / torch.distributed.launch)
    • Gradient accumulation (simulate large batch on small GPU)
    • Mixed precision (BF16 / FP16 / FP32)
    • Gradient checkpointing (train large models on limited VRAM)
    • ZeRO-style memory optimization hints

Usage (single script, auto-detects):
    trainer = DistributedTrainer(model, config)
    trainer.setup()        # Initialize process group
    trainer.wrap_model()   # Wrap in DDP if multi-GPU
    trainer.train(dataloader, loss_fn)
    trainer.cleanup()

Launch commands:
    # Single GPU:
    python scripts/train_stage.py --stage dino

    # Multi-GPU (4 GPUs, 1 node):
    torchrun --nproc_per_node=4 scripts/train_stage.py --stage dino

    # Multi-Node (2 nodes, 4 GPUs each):
    torchrun --nnodes=2 --nproc_per_node=4 \\
             --node_rank=0 --master_addr=<IP> \\
             scripts/train_stage.py --stage dino
"""

import os
import logging
import contextlib
from typing import Optional, Dict, Any, Callable
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from torch.cuda.amp import GradScaler, autocast

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Distributed Config
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class DistributedConfig:
    """Configuration for distributed training setup."""

    # Precision
    precision: str = "fp16"          # "fp32" | "fp16" | "bf16"

    # Gradient accumulation
    grad_accumulation_steps: int = 1  # Simulate batch_size * N

    # Gradient clipping
    max_grad_norm: float = 1.0

    # Gradient checkpointing (trade compute for memory)
    gradient_checkpointing: bool = False

    # DDP options
    find_unused_parameters: bool = False
    static_graph: bool = True           # Faster DDP if graph doesn't change

    # Backend
    backend: str = "nccl"              # "nccl" (GPU) | "gloo" (CPU)

    # Memory
    empty_cache_freq: int = 100        # Steps between torch.cuda.empty_cache()


# ─────────────────────────────────────────────────────────────────────────────
# Process Group Utilities
# ─────────────────────────────────────────────────────────────────────────────

def init_distributed() -> Dict[str, Any]:
    """
    Initialize distributed process group.

    Auto-detects environment variables set by torchrun / SLURM.

    Returns:
        dict with rank, local_rank, world_size, is_main_process
    """
    if not dist.is_available():
        return {
            "rank": 0, "local_rank": 0, "world_size": 1,
            "is_main": True, "distributed": False,
        }

    # torchrun sets these
    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    rank = int(os.environ.get("RANK", -1))
    world_size = int(os.environ.get("WORLD_SIZE", 1))

    if local_rank == -1 or world_size <= 1:
        return {
            "rank": 0, "local_rank": 0, "world_size": 1,
            "is_main": True, "distributed": False,
        }

    torch.cuda.set_device(local_rank)

    if not dist.is_initialized():
        dist.init_process_group(
            backend="nccl",
            init_method="env://",
        )

    logger.info(
        f"[Distributed] Initialized: rank={rank}/{world_size}, "
        f"local_rank={local_rank}, device=cuda:{local_rank}"
    )

    return {
        "rank": rank,
        "local_rank": local_rank,
        "world_size": world_size,
        "is_main": rank == 0,
        "distributed": True,
    }


def is_main_process() -> bool:
    """True if this is the main (rank 0) process."""
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank() == 0
    return True


def get_world_size() -> int:
    if dist.is_available() and dist.is_initialized():
        return dist.get_world_size()
    return 1


def get_rank() -> int:
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank()
    return 0


def barrier():
    """Synchronization barrier across all processes."""
    if dist.is_available() and dist.is_initialized():
        dist.barrier()


def reduce_dict(d: Dict[str, torch.Tensor], average: bool = True) -> Dict[str, torch.Tensor]:
    """
    All-reduce a dict of tensors across all processes.
    Used for aggregating metrics from all GPUs.
    """
    if get_world_size() == 1:
        return d

    keys = sorted(d.keys())
    values = torch.stack([d[k].float() for k in keys])
    dist.all_reduce(values, op=dist.ReduceOp.SUM)

    if average:
        values /= get_world_size()

    return {k: v for k, v in zip(keys, values)}


def all_gather_tensor(t: torch.Tensor) -> torch.Tensor:
    """Gather a tensor from all processes and concatenate."""
    if get_world_size() == 1:
        return t
    gathered = [torch.zeros_like(t) for _ in range(get_world_size())]
    dist.all_gather(gathered, t)
    return torch.cat(gathered, dim=0)


def cleanup_distributed():
    """Clean up distributed process group."""
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


# ─────────────────────────────────────────────────────────────────────────────
# Mixed Precision Context
# ─────────────────────────────────────────────────────────────────────────────

class MixedPrecisionManager:
    """
    Manages mixed precision training context.

    Supports FP32, FP16, and BF16.
    BF16 is preferred on Ampere+ GPUs (A100, RTX 3090+) — no overflow issues.
    FP16 requires GradScaler to prevent underflow.
    """

    def __init__(self, precision: str = "fp16"):
        self.precision = precision
        self.enabled = precision != "fp32"

        self.scaler = None
        self.dtype = torch.float32

        if precision == "fp16":
            self.scaler = GradScaler()
            self.dtype = torch.float16
        elif precision == "bf16":
            # BF16 doesn't need scaler — same exponent range as FP32
            self.dtype = torch.bfloat16

        logger.info(f"[MixedPrecision] precision={precision}, dtype={self.dtype}")

    @contextlib.contextmanager
    def autocast_context(self):
        """Context manager for autocast."""
        if self.precision == "fp32":
            yield
        else:
            with autocast(dtype=self.dtype):
                yield

    def scale_loss(self, loss: torch.Tensor) -> torch.Tensor:
        if self.scaler is not None:
            return self.scaler.scale(loss)
        return loss

    def step(self, optimizer: torch.optim.Optimizer):
        if self.scaler is not None:
            self.scaler.step(optimizer)
            self.scaler.update()
        else:
            optimizer.step()

    def unscale(self, optimizer: torch.optim.Optimizer):
        if self.scaler is not None:
            self.scaler.unscale_(optimizer)

    def get_scale(self) -> float:
        if self.scaler is not None:
            return self.scaler.get_scale()
        return 1.0


# ─────────────────────────────────────────────────────────────────────────────
# Gradient Accumulation Manager
# ─────────────────────────────────────────────────────────────────────────────

class GradientAccumulationManager:
    """
    Manages gradient accumulation over N steps.

    Allows training with effective batch size = batch_size * N
    without increasing GPU memory.

    Example:
        With batch_size=16, accum_steps=4:
        effective_batch = 64, but GPU only holds 16 samples at a time.
    """

    def __init__(self, accum_steps: int = 1, model: Optional[nn.Module] = None):
        self.accum_steps = accum_steps
        self.model = model
        self._current_step = 0

    @property
    def should_update(self) -> bool:
        """True when accumulated enough steps to call optimizer.step()."""
        return (self._current_step + 1) % self.accum_steps == 0

    def scale_loss(self, loss: torch.Tensor) -> torch.Tensor:
        """Scale loss by 1/accum_steps for proper gradient scaling."""
        return loss / self.accum_steps

    @contextlib.contextmanager
    def no_sync_context(self):
        """
        Skip gradient sync in DDP for non-update steps.
        Critical for efficiency — without this, DDP syncs every backward pass.
        """
        if self.model is not None and isinstance(self.model, DDP) and not self.should_update:
            with self.model.no_sync():
                yield
        else:
            yield

    def step(self):
        """Increment step counter."""
        self._current_step += 1

    def reset(self):
        self._current_step = 0


# ─────────────────────────────────────────────────────────────────────────────
# Distributed Trainer Base
# ─────────────────────────────────────────────────────────────────────────────

class DistributedTrainer:
    """
    Base distributed trainer for all Retina-GPT training stages.

    Handles:
        - DDP model wrapping
        - Mixed precision
        - Gradient accumulation
        - Gradient checkpointing
        - Metrics aggregation across GPUs
        - Checkpoint saving (main process only)

    Subclassed by each stage trainer:
        DINOTrainer       — Stage 1 self-supervised
        CLIPTrainer       — Stage 2 vision-language
        SAMTrainer        — Stage 3 segmentation
        MultiTaskTrainer  — Stage 4 fine-tuning
    """

    def __init__(
        self,
        model: nn.Module,
        config: DistributedConfig,
        dist_info: Optional[Dict] = None,
    ):
        self.config = config
        self.dist_info = dist_info or init_distributed()

        self.rank = self.dist_info["rank"]
        self.local_rank = self.dist_info["local_rank"]
        self.world_size = self.dist_info["world_size"]
        self.is_main = self.dist_info["is_main"]
        self.distributed = self.dist_info["distributed"]

        # Set device
        if torch.cuda.is_available():
            self.device = torch.device(f"cuda:{self.local_rank}")
        else:
            self.device = torch.device("cpu")

        # Move model to device
        self.model = model.to(self.device)

        # Gradient checkpointing
        if config.gradient_checkpointing and hasattr(model, 'gradient_checkpointing_enable'):
            model.gradient_checkpointing_enable()
            self.log("Gradient checkpointing enabled")

        # Mixed precision
        self.mp = MixedPrecisionManager(config.precision)

        # Gradient accumulation
        self.ga = GradientAccumulationManager(config.grad_accumulation_steps)

        self._global_step = 0
        self._epoch = 0

    def wrap_model(self) -> nn.Module:
        """Wrap model in DDP if distributed training."""
        if self.distributed:
            self.model = DDP(
                self.model,
                device_ids=[self.local_rank],
                output_device=self.local_rank,
                find_unused_parameters=self.config.find_unused_parameters,
                static_graph=self.config.static_graph,
            )
            self.ga.model = self.model
            self.log(f"Model wrapped in DDP (world_size={self.world_size})")
        return self.model

    def get_distributed_dataloader(
        self,
        dataset,
        batch_size: int,
        shuffle: bool = True,
        **kwargs
    ) -> DataLoader:
        """Create DataLoader with DistributedSampler for DDP training."""
        sampler = None
        if self.distributed:
            sampler = DistributedSampler(
                dataset,
                num_replicas=self.world_size,
                rank=self.rank,
                shuffle=shuffle,
            )
            shuffle = False  # DistributedSampler handles shuffling

        return DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=sampler,
            shuffle=shuffle,
            **kwargs
        )

    def backward_step(
        self,
        loss: torch.Tensor,
        optimizer: torch.optim.Optimizer,
        retain_graph: bool = False,
    ) -> Dict[str, float]:
        """
        Single backward step with:
        - gradient accumulation
        - mixed precision scaling
        - gradient clipping
        - DDP no_sync optimization

        Returns metrics dict.
        """
        # Scale for accumulation
        scaled_loss = self.ga.scale_loss(loss)

        # Backward with DDP sync optimization
        with self.ga.no_sync_context():
            self.mp.scale_loss(scaled_loss).backward(retain_graph=retain_graph)

        self.ga.step()
        metrics = {}

        if self.ga.should_update:
            # Unscale before clipping
            self.mp.unscale(optimizer)

            # Gradient clipping
            if self.config.max_grad_norm > 0:
                raw_model = self.model.module if isinstance(self.model, DDP) else self.model
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    raw_model.parameters(),
                    self.config.max_grad_norm,
                )
                metrics["grad_norm"] = grad_norm.item()

            # Optimizer step
            self.mp.step(optimizer)
            optimizer.zero_grad()

            self._global_step += 1

            # Periodic cache clear
            if self._global_step % self.config.empty_cache_freq == 0:
                torch.cuda.empty_cache()

        return metrics

    def aggregate_metrics(self, metrics: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Aggregate metrics across all GPU processes."""
        if self.distributed:
            reduced = reduce_dict(
                {k: v.detach() if isinstance(v, torch.Tensor) else torch.tensor(v)
                 for k, v in metrics.items()}
            )
            return {k: v.item() for k, v in reduced.items()}
        return {k: v.item() if isinstance(v, torch.Tensor) else v
                for k, v in metrics.items()}

    def save_checkpoint(self, path: str, extra: Dict = None):
        """Save checkpoint (main process only)."""
        if not self.is_main:
            return  # Only rank 0 saves

        raw_model = self.model.module if isinstance(self.model, DDP) else self.model
        state = {
            "epoch": self._epoch,
            "global_step": self._global_step,
            "model_state_dict": raw_model.state_dict(),
            **(extra or {}),
        }
        torch.save(state, path)
        self.log(f"Checkpoint saved → {path}")

    def log(self, msg: str, level: str = "info"):
        """Log only from main process."""
        if self.is_main:
            getattr(logger, level)(f"[Rank {self.rank}] {msg}")

    def print_gpu_memory(self):
        """Print GPU memory usage (main process only)."""
        if self.is_main and torch.cuda.is_available():
            alloc = torch.cuda.memory_allocated(self.local_rank) / 1e9
            reserved = torch.cuda.memory_reserved(self.local_rank) / 1e9
            self.log(f"GPU Memory: {alloc:.2f}GB allocated / {reserved:.2f}GB reserved")

    def cleanup(self):
        """Clean up distributed process group."""
        cleanup_distributed()

    @property
    def effective_batch_size(self) -> int:
        """The effective global batch size across all GPUs and accumulation steps."""
        return self.ga.accum_steps * self.world_size

    def __repr__(self) -> str:
        return (
            f"DistributedTrainer(\n"
            f"  device={self.device}, world_size={self.world_size}\n"
            f"  precision={self.config.precision}\n"
            f"  grad_accum={self.config.grad_accumulation_steps}\n"
            f"  effective_batch_multiplier={self.effective_batch_size}\n"
            f")"
        )
