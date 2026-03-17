"""
model_registry.py — Retina-GPT Model Registry
===============================================
Versioned model management system.

Tracks every trained checkpoint with:
    • Stage (dino / clip / sam / multitask / full)
    • Performance metrics
    • Training config
    • Dataset info
    • Git commit hash

Enables:
    model = registry.load("retina-gpt", stage="multitask", alias="best")
    registry.promote("retina-gpt", version="v1.2.0", alias="production")
    registry.compare(["v1.0.0", "v1.1.0", "v1.2.0"], metric="val/dr_kappa")

Registry is stored as a JSON file alongside checkpoints.
No external database required — everything is local and portable.

Directory structure:
    checkpoints/
        registry.json                 ← master registry
        dino/
            retina_dino_epoch_0099.pt
            retina_dino_best.pt
        clip/
            retina_clip_best.pt
        sam/
            retina_sam_best.pt
        multitask/
            retina_multitask_v1.0.0.pt
            retina_multitask_best.pt
        production/
            retina_gpt_production.pt  ← promoted model
"""

import os
import json
import shutil
import hashlib
import logging
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, List, Any, Tuple
from dataclasses import dataclass, asdict, field

import torch

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Model Entry
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ModelEntry:
    """Metadata entry for a registered model checkpoint."""

    # Identity
    name: str                    # e.g., "retina-gpt"
    version: str                 # e.g., "v1.2.0"
    stage: str                   # dino | clip | sam | multitask | full

    # File info
    path: str                    # Absolute path to checkpoint
    file_size_mb: float = 0.0
    sha256: str = ""

    # Metrics
    metrics: Dict[str, float] = field(default_factory=dict)
    primary_metric: str = ""     # e.g., "val/dr_kappa"
    primary_metric_value: float = 0.0

    # Training info
    epoch: int = 0
    total_steps: int = 0
    training_time_hours: float = 0.0
    config: Dict[str, Any] = field(default_factory=dict)

    # Dataset info
    dataset_name: str = ""
    dataset_size: int = 0

    # Provenance
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    git_commit: str = ""
    aliases: List[str] = field(default_factory=list)
    notes: str = ""
    is_production: bool = False

    def to_dict(self) -> Dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict) -> "ModelEntry":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


# ─────────────────────────────────────────────────────────────────────────────
# Model Registry
# ─────────────────────────────────────────────────────────────────────────────

class ModelRegistry:
    """
    Versioned model registry for Retina-GPT.

    All checkpoints tracked, compared, promoted, and loaded from here.

    Usage:
        registry = ModelRegistry("./checkpoints")

        # Register after training
        registry.register(
            checkpoint_path="checkpoints/multitask/best.pt",
            name="retina-gpt",
            stage="multitask",
            version="v1.0.0",
            metrics={"val/dr_kappa": 0.87, "val/dr_auc": 0.94},
            primary_metric="val/dr_kappa",
        )

        # Load best model
        ckpt = registry.load("retina-gpt", stage="multitask")

        # Promote to production
        registry.promote("retina-gpt", "v1.0.0")
    """

    REGISTRY_FILE = "registry.json"

    STAGE_DIRS = {
        "dino":      "dino",
        "clip":      "clip",
        "sam":       "sam",
        "multitask": "multitask",
        "full":      "full",
        "production": "production",
    }

    def __init__(self, base_dir: str = "./checkpoints"):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

        for stage_dir in self.STAGE_DIRS.values():
            (self.base_dir / stage_dir).mkdir(exist_ok=True)

        self.registry_path = self.base_dir / self.REGISTRY_FILE
        self._entries: Dict[str, ModelEntry] = {}
        self._load_registry()

    # ── Registry I/O ─────────────────────────────────────────────────────────

    def _load_registry(self):
        if self.registry_path.exists():
            with open(self.registry_path) as f:
                data = json.load(f)
            self._entries = {
                k: ModelEntry.from_dict(v)
                for k, v in data.get("entries", {}).items()
            }
            logger.info(f"[Registry] Loaded {len(self._entries)} entries from {self.registry_path}")

    def _save_registry(self):
        data = {
            "last_updated": datetime.utcnow().isoformat(),
            "total_entries": len(self._entries),
            "entries": {k: v.to_dict() for k, v in self._entries.items()},
        }
        with open(self.registry_path, "w") as f:
            json.dump(data, f, indent=2)

    def _entry_key(self, name: str, version: str) -> str:
        return f"{name}:{version}"

    # ── Registration ─────────────────────────────────────────────────────────

    def register(
        self,
        checkpoint_path: str,
        name: str,
        stage: str,
        version: Optional[str] = None,
        metrics: Dict[str, float] = None,
        primary_metric: str = "",
        config: Dict = None,
        dataset_name: str = "",
        dataset_size: int = 0,
        epoch: int = 0,
        training_time_hours: float = 0.0,
        aliases: List[str] = None,
        notes: str = "",
        copy_to_registry: bool = True,
    ) -> ModelEntry:
        """
        Register a new model checkpoint.

        Args:
            checkpoint_path: Path to the .pt file to register
            name: Model name (e.g., "retina-gpt")
            stage: Training stage (dino/clip/sam/multitask/full)
            version: Version string (auto-generated if None)
            metrics: Dict of eval metrics
            primary_metric: Key metric name for comparison
            ...

        Returns:
            ModelEntry for the registered model
        """
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        # Auto-version
        if version is None:
            version = self._next_version(name, stage)

        metrics = metrics or {}
        primary_value = metrics.get(primary_metric, 0.0) if primary_metric else 0.0

        # Compute SHA256 for integrity
        sha256 = self._sha256(checkpoint_path)
        file_size_mb = checkpoint_path.stat().st_size / 1e6

        # Copy to organized registry directory
        registered_path = checkpoint_path
        if copy_to_registry:
            stage_dir = self.base_dir / self.STAGE_DIRS.get(stage, stage)
            dest = stage_dir / f"{name.replace('/', '_')}_{version}.pt"
            if not dest.exists():
                shutil.copy2(checkpoint_path, dest)
            registered_path = dest

        entry = ModelEntry(
            name=name,
            version=version,
            stage=stage,
            path=str(registered_path),
            file_size_mb=round(file_size_mb, 2),
            sha256=sha256,
            metrics=metrics,
            primary_metric=primary_metric,
            primary_metric_value=primary_value,
            epoch=epoch,
            training_time_hours=training_time_hours,
            config=config or {},
            dataset_name=dataset_name,
            dataset_size=dataset_size,
            aliases=aliases or [],
            notes=notes,
            git_commit=self._git_commit(),
        )

        key = self._entry_key(name, version)
        self._entries[key] = entry
        self._save_registry()

        logger.info(f"[Registry] Registered {name}:{version} ({stage}) → {registered_path}")
        return entry

    # ── Loading ───────────────────────────────────────────────────────────────

    def load(
        self,
        name: str,
        version: Optional[str] = None,
        stage: Optional[str] = None,
        alias: Optional[str] = None,
        map_location: str = "cpu",
    ) -> Dict:
        """
        Load a checkpoint from the registry.

        Priority: alias > version > (best by primary metric in stage)
        """
        entry = self.get_entry(name, version=version, stage=stage, alias=alias)
        if entry is None:
            raise KeyError(f"No model found: name={name}, version={version}, stage={stage}, alias={alias}")

        logger.info(f"[Registry] Loading {entry.name}:{entry.version} from {entry.path}")
        checkpoint = torch.load(entry.path, map_location=map_location)
        checkpoint["_registry_entry"] = entry.to_dict()
        return checkpoint

    def get_entry(
        self,
        name: str,
        version: Optional[str] = None,
        stage: Optional[str] = None,
        alias: Optional[str] = None,
    ) -> Optional[ModelEntry]:
        """Find a registry entry matching the criteria."""

        if version:
            return self._entries.get(self._entry_key(name, version))

        # Search by alias
        if alias:
            for entry in self._entries.values():
                if entry.name == name and alias in entry.aliases:
                    return entry

        # Find best in stage
        candidates = [
            e for e in self._entries.values()
            if e.name == name and (stage is None or e.stage == stage)
        ]
        if not candidates:
            return None

        # Sort by primary metric (higher = better)
        candidates.sort(key=lambda e: e.primary_metric_value, reverse=True)
        return candidates[0]

    # ── Promotion ─────────────────────────────────────────────────────────────

    def promote(self, name: str, version: str, alias: str = "production",
                copy_to_production: bool = True) -> ModelEntry:
        """
        Promote a model version to production.

        Copies checkpoint to production/ directory and tags it.
        """
        entry = self._entries.get(self._entry_key(name, version))
        if entry is None:
            raise KeyError(f"Model {name}:{version} not found in registry.")

        # Update alias
        if alias not in entry.aliases:
            entry.aliases.append(alias)
        entry.is_production = True

        # Copy to production directory
        if copy_to_production:
            prod_dir = self.base_dir / "production"
            dest = prod_dir / f"{name.replace('/', '_')}_production.pt"
            shutil.copy2(entry.path, dest)
            entry.path = str(dest)
            logger.info(f"[Registry] Promoted {name}:{version} → {dest}")

        self._save_registry()
        return entry

    # ── Comparison ────────────────────────────────────────────────────────────

    def compare(
        self,
        name: str,
        versions: Optional[List[str]] = None,
        stage: Optional[str] = None,
        metric: Optional[str] = None,
        top_k: int = 10,
    ) -> List[Dict]:
        """
        Compare multiple model versions by metrics.

        Returns list of dicts sorted by primary_metric_value descending.
        """
        candidates = [
            e for e in self._entries.values()
            if e.name == name
            and (stage is None or e.stage == stage)
            and (versions is None or e.version in versions)
        ]

        if metric:
            candidates.sort(
                key=lambda e: e.metrics.get(metric, float("-inf")),
                reverse=True
            )
        else:
            candidates.sort(key=lambda e: e.primary_metric_value, reverse=True)

        results = []
        for e in candidates[:top_k]:
            row = {
                "version": e.version,
                "stage": e.stage,
                "epoch": e.epoch,
                "primary_metric": e.primary_metric,
                "primary_value": e.primary_metric_value,
                **e.metrics,
                "aliases": ", ".join(e.aliases),
                "created_at": e.created_at[:10],
            }
            results.append(row)

        return results

    def leaderboard(self, name: str, metric: str, stage: Optional[str] = None) -> str:
        """Print a leaderboard table to stdout."""
        rows = self.compare(name, stage=stage, metric=metric)
        if not rows:
            return f"No entries found for {name}"

        header = f"{'Version':10} {'Stage':12} {'Metric':30} {'Value':8} {'Epoch':6} {'Aliases'}"
        lines = [header, "-" * 80]
        for r in rows:
            lines.append(
                f"{r['version']:10} {r['stage']:12} {metric:30} "
                f"{r.get(metric, 0):8.4f} {r['epoch']:6} {r.get('aliases','')}"
            )
        table = "\n".join(lines)
        print(table)
        return table

    # ── Utilities ─────────────────────────────────────────────────────────────

    def _next_version(self, name: str, stage: str) -> str:
        """Auto-generate version string like v0.1.0, v0.2.0..."""
        existing = [
            e for e in self._entries.values()
            if e.name == name and e.stage == stage
        ]
        patch = len(existing)
        stage_major = {"dino": 0, "clip": 1, "sam": 2, "multitask": 3, "full": 4}.get(stage, 9)
        return f"v{stage_major}.0.{patch}"

    def _sha256(self, path: Path) -> str:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                h.update(chunk)
        return h.hexdigest()[:16]

    def _git_commit(self) -> str:
        try:
            import subprocess
            result = subprocess.run(
                ["git", "rev-parse", "--short", "HEAD"],
                capture_output=True, text=True, timeout=3
            )
            return result.stdout.strip()
        except Exception:
            return ""

    def list_all(self, name: Optional[str] = None) -> List[ModelEntry]:
        entries = list(self._entries.values())
        if name:
            entries = [e for e in entries if e.name == name]
        entries.sort(key=lambda e: e.created_at, reverse=True)
        return entries

    def summary(self) -> str:
        lines = [
            f"Retina-GPT Model Registry",
            f"Location: {self.registry_path}",
            f"Total entries: {len(self._entries)}",
            "",
        ]
        stages = {}
        for e in self._entries.values():
            stages.setdefault(e.stage, []).append(e)

        for stage, entries in sorted(stages.items()):
            best = max(entries, key=lambda x: x.primary_metric_value)
            lines.append(f"  [{stage:12}] {len(entries):3d} versions | "
                          f"best: {best.version} ({best.primary_metric}={best.primary_metric_value:.4f})")

        return "\n".join(lines)

    def __repr__(self) -> str:
        return f"ModelRegistry({self.base_dir}, {len(self._entries)} entries)"
