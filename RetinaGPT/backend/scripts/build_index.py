"""
scripts/build_index.py — Build Retina Vector Search Index
==========================================================
One-time script to encode all images in a dataset and build
the FAISS search index.

Usage:
    # Build from APTOS dataset
    python scripts/build_index.py \\
        --data_dir data/aptos/train_images \\
        --labels_csv data/aptos/train.csv \\
        --checkpoint checkpoints/multitask/best.pt \\
        --output indexes/aptos_index.bin

    # Build from multiple datasets
    python scripts/build_index.py \\
        --data_dir data/ \\
        --datasets aptos eyepacs \\
        --output indexes/combined_index.bin

    # Quick test (100 images)
    python scripts/build_index.py \\
        --data_dir data/aptos/train_images \\
        --max_images 100 \\
        --output indexes/test_index.bin
"""

import argparse
import logging
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s"
)
logger = logging.getLogger("retina_gpt.build_index")


def parse_args():
    p = argparse.ArgumentParser(description="Build Retina-GPT Vector Search Index")
    p.add_argument("--data_dir",     required=True,   help="Image directory or dataset root")
    p.add_argument("--datasets",     nargs="+",       help="Sub-datasets to index (e.g. aptos eyepacs)")
    p.add_argument("--labels_csv",   default=None,    help="CSV with image labels")
    p.add_argument("--checkpoint",   default=None,    help="Model checkpoint path")
    p.add_argument("--output",       default="indexes/retina_index.bin", help="Output index path")
    p.add_argument("--index_type",   default="ivfflat",
                   choices=["flat", "ivfflat", "ivfpq", "hnsw"])
    p.add_argument("--project_dim",  type=int, default=256, help="PCA projection dimension")
    p.add_argument("--max_images",   type=int, default=None, help="Max images (for testing)")
    p.add_argument("--batch_size",   type=int, default=1)
    return p.parse_args()


def main():
    args = parse_args()

    # Load pipeline
    from inference.pipeline import RetinaGPTPipeline
    from retrieval.vector_search import VectorSearchEngine

    if args.checkpoint and Path(args.checkpoint).exists():
        logger.info(f"Loading checkpoint: {args.checkpoint}")
        pipeline = RetinaGPTPipeline.from_checkpoint(
            args.checkpoint,
            enable_explainability=False,  # Not needed for indexing
            enable_temporal=False,
        )
    else:
        logger.warning("No checkpoint provided — using DEMO mode (random weights)")
        pipeline = RetinaGPTPipeline.demo(
            enable_explainability=False, enable_temporal=False
        )

    # Build search engine
    engine = VectorSearchEngine(
        embed_dim=1024,
        index_type=args.index_type,
        project_dim=args.project_dim,
    )

    data_root = Path(args.data_dir)

    if args.datasets:
        # Multi-dataset: index each and combine
        from retrieval.vector_search import VectorSearchEngine
        import numpy as np

        all_embeddings, all_ids, all_meta = [], [], []

        for dataset_name in args.datasets:
            img_dir  = data_root / dataset_name
            csv_path = data_root / dataset_name / "train.csv"

            if not img_dir.exists():
                logger.warning(f"Dataset not found: {img_dir}")
                continue

            # Temporary engine for this dataset
            tmp = VectorSearchEngine(embed_dim=1024, index_type="flat")
            n = tmp.build_from_pipeline(
                pipeline,
                image_dir=str(img_dir),
                labels_csv=str(csv_path) if csv_path.exists() else None,
                dataset_name=dataset_name,
                max_images=args.max_images,
            )

            if hasattr(tmp, "_torch_embeddings"):
                all_embeddings.append(tmp._torch_embeddings.numpy())
            all_ids.extend(tmp._ids)
            all_meta.extend(tmp._meta)
            logger.info(f"  {dataset_name}: {n} images")

        if all_embeddings:
            combined = np.vstack(all_embeddings)
            engine.build(combined, all_ids, all_meta)
            engine.save(args.output)

    else:
        # Single directory
        img_dir  = data_root
        csv_path = args.labels_csv

        # Look for default CSV if not specified
        if not csv_path:
            for candidate in ["train.csv", "labels.csv", "trainLabels.csv"]:
                p = data_root / candidate
                if p.exists():
                    csv_path = str(p)
                    logger.info(f"Auto-detected labels: {csv_path}")
                    break

        n = engine.build_from_pipeline(
            pipeline,
            image_dir=str(img_dir),
            labels_csv=csv_path,
            dataset_name=data_root.name,
            max_images=args.max_images,
            save_path=args.output,
        )

    # Print stats
    if engine._trained:
        stats = engine.stats()
        logger.info("\n" + "="*50)
        logger.info("  INDEX BUILT SUCCESSFULLY")
        logger.info("="*50)
        logger.info(f"  Vectors:     {stats['total_vectors']:,}")
        logger.info(f"  Index type:  {stats['index_type']}")
        logger.info(f"  Dimensions:  {stats['embed_dim']} → {stats['index_dim']}")
        logger.info(f"  FAISS:       {'✓' if stats['faiss'] else '✗'}")
        logger.info(f"  Output:      {args.output}")
        if stats["dr_distribution"]:
            logger.info("  DR distribution:")
            for label, count in stats["dr_distribution"].items():
                pct = 100 * count / max(stats["total_vectors"], 1)
                bar = "█" * int(pct / 3)
                logger.info(f"    {label:28s}: {count:5,} ({pct:5.1f}%) {bar}")
        logger.info("="*50)
        logger.info(f"\nTo use in API: set env RETINA_INDEX={args.output}")
    else:
        logger.error("Index building failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()
