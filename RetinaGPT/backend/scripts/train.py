"""
scripts/train.py — Retina-GPT Training Entry Point
====================================================
Unified CLI for all training stages.

Usage:
    # Stage 1: Self-supervised pretraining (no labels needed)
    python scripts/train.py --stage dino --data_dir data/ --epochs 100

    # Stage 3: Multi-task fine-tuning (labeled data)
    python scripts/train.py --stage multitask --data_dir data/ --epochs 50
    python scripts/train.py --stage multitask --dino_checkpoint checkpoints/dino/best.pt

    # Multi-GPU (torchrun)
    torchrun --nproc_per_node=4 scripts/train.py --stage multitask

    # Quick test run
    python scripts/train.py --stage multitask --max_samples 200 --epochs 3
"""

from __future__ import annotations
import argparse, logging, os, random, sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("retina_gpt.train")


def set_seed(seed: int = 42):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def parse_args():
    p = argparse.ArgumentParser(description="Retina-GPT Training")
    p.add_argument("--stage",    required=True,
                   choices=["dino", "clip", "multitask", "sam", "all"],
                   help="Training stage to run")
    p.add_argument("--data_dir",    default="./data",         help="Dataset root")
    p.add_argument("--checkpoint_dir", default="./checkpoints", help="Where to save checkpoints")
    p.add_argument("--dino_checkpoint", default=None,          help="Pretrained DINO checkpoint path")
    p.add_argument("--resume",      default=None,              help="Resume from checkpoint")
    p.add_argument("--epochs",      type=int, default=None)
    p.add_argument("--batch_size",  type=int, default=32)
    p.add_argument("--lr",          type=float, default=1e-4)
    p.add_argument("--precision",   default="fp16", choices=["fp32", "fp16", "bf16"])
    p.add_argument("--grad_accum",  type=int, default=2)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--seed",        type=int, default=42)
    p.add_argument("--wandb",       default="all",
                   choices=["wandb", "tensorboard", "all", "none"])
    p.add_argument("--max_samples", type=int, default=None,
                   help="Limit dataset size (for quick testing)")
    return p.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)

    logger.info("=" * 60)
    logger.info("  Retina-GPT Foundation Model Training")
    logger.info("=" * 60)
    logger.info(f"  Stage:       {args.stage}")
    logger.info(f"  Data:        {args.data_dir}")
    logger.info(f"  Checkpoints: {args.checkpoint_dir}")
    logger.info(f"  Precision:   {args.precision}")
    logger.info(f"  GPUs:        {torch.cuda.device_count()}")
    logger.info("=" * 60)

    from scripts.train_foundation import RetinaGPTTrainingOrchestrator

    orchestrator = RetinaGPTTrainingOrchestrator(
        data_dir=args.data_dir,
        checkpoint_dir=args.checkpoint_dir,
        precision=args.precision,
        grad_accum=args.grad_accum,
        num_workers=args.num_workers,
        wandb_backend=args.wandb,
    )

    if args.stage == "all":
        checkpoints = orchestrator.run_all_stages({
            "dino_epochs":         args.epochs or 100,
            "dino_batch_size":     args.batch_size,
            "multitask_epochs":    args.epochs or 50,
            "multitask_batch_size": args.batch_size,
        })
        logger.info("\n✅ All stages complete!")
        for stage, ckpt in checkpoints.items():
            logger.info(f"  {stage:12s}: {ckpt}")

    elif args.stage == "dino":
        ckpt = orchestrator.run_stage1_dino(
            epochs=args.epochs or 100,
            batch_size=args.batch_size,
            resume=args.resume,
        )
        logger.info(f"\n✅ DINO pretraining done! → {ckpt}")

    elif args.stage == "multitask":
        ckpt = orchestrator.run_stage3_multitask(
            dino_checkpoint=args.dino_checkpoint,
            epochs=args.epochs or 50,
            batch_size=args.batch_size,
            lr=args.lr,
            resume=args.resume,
        )
        logger.info(f"\n✅ Multi-task training done! → {ckpt}")

    else:
        logger.info(f"Stage '{args.stage}' — use train_foundation.py for full control.")


if __name__ == "__main__":
    main()
