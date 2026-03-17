"""
retrieval/vector_search.py — Retina-GPT Vector Search Engine
=============================================================
Production-grade semantic search over retinal image embeddings.

The Universal Retina Embedding (1024-dim) represents the complete
semantic content of a fundus image. This module builds an index over
millions of embeddings for sub-millisecond similar-case retrieval.

Architecture:
    Flat L2 (exact, <10K images)        → cosine brute-force
    IVF-Flat (exact clusters, <500K)    → faiss.IndexIVFFlat
    IVF-PQ (compressed, millions)       → faiss.IndexIVFPQ
    HNSW (graph-based, best recall)     → faiss.IndexHNSWFlat

Clinical use cases:
    • "Find 5 cases similar to this retina" — clinical decision support
    • "Show me all severe DR cases in the database" — quality audit
    • "Find cases where this lesion pattern appeared" — research query
    • Patient baseline comparison — compare current vs. stored embedding

Usage:
    engine = VectorSearchEngine(embed_dim=1024, index_type="ivfflat")

    # Build index from a dataset
    engine.build_from_pipeline(pipeline, image_dir="data/aptos/train_images/")

    # Search
    results = engine.search(query_embedding, k=10)
    for r in results:
        print(r.image_id, r.score, r.dr_grade, r.visit_date)

    # Save / load
    engine.save("indexes/aptos_index.bin")
    engine = VectorSearchEngine.load("indexes/aptos_index.bin")
"""

import json
import logging
import os
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Search Result
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class SearchResult:
    """Single result from a vector search query."""
    rank:          int
    image_id:      str
    score:         float          # cosine similarity [0, 1] — higher = more similar
    distance:      float          # L2 distance — lower = more similar

    # Clinical metadata (populated from stored metadata)
    dr_grade:      Optional[int]  = None
    dr_label:      Optional[str]  = None
    amd_stage:     Optional[int]  = None
    dataset:       Optional[str]  = None
    visit_date:    Optional[str]  = None
    image_path:    Optional[str]  = None
    extra:         Dict           = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return {k: v for k, v in asdict(self).items() if v is not None}


@dataclass
class SearchResponse:
    """Complete search response for an API call."""
    query_id:         str
    num_results:      int
    search_time_ms:   float
    index_size:       int
    results:          List[SearchResult]

    def to_api_dict(self) -> Dict:
        return {
            "query_id":       self.query_id,
            "num_results":    self.num_results,
            "search_time_ms": round(self.search_time_ms, 3),
            "index_size":     self.index_size,
            "results": [r.to_dict() for r in self.results],
        }


# ─────────────────────────────────────────────────────────────────────────────
# Index Types
# ─────────────────────────────────────────────────────────────────────────────

INDEX_TYPES = {
    "flat":     "Exact cosine (brute force) — best for <10K images",
    "ivfflat":  "IVF exact clusters — best for 10K–500K images",
    "ivfpq":    "IVF + Product Quantization (compressed) — millions of images",
    "hnsw":     "HNSW graph index — best recall/speed tradeoff",
}

DR_GRADE_LABELS = {
    0: "No DR", 1: "Mild NPDR", 2: "Moderate NPDR",
    3: "Severe NPDR", 4: "Proliferative DR",
}


# ─────────────────────────────────────────────────────────────────────────────
# Vector Search Engine
# ─────────────────────────────────────────────────────────────────────────────

class VectorSearchEngine:
    """
    Production-grade vector search over retinal image embeddings.

    Scales from small datasets (cosine brute-force) to millions of
    images (FAISS IVF-PQ compressed index).

    The index stores 1024-dim embeddings projected to 256-dim for speed
    (with negligible recall loss due to PCA whitening).
    """

    def __init__(
        self,
        embed_dim:   int  = 1024,
        index_type:  str  = "ivfflat",   # flat | ivfflat | ivfpq | hnsw
        project_dim: int  = 256,          # Reduce to this dim before indexing
        n_clusters:  int  = 100,          # IVF clusters (sqrt of dataset size)
        n_probe:     int  = 10,           # IVF clusters to search at query time
        m_subvecs:   int  = 32,           # PQ sub-vectors (for ivfpq)
        hnsw_m:      int  = 32,           # HNSW connections per node
    ):
        self.embed_dim   = embed_dim
        self.index_type  = index_type
        self.project_dim = project_dim
        self.n_clusters  = n_clusters
        self.n_probe     = n_probe
        self.m_subvecs   = m_subvecs
        self.hnsw_m      = hnsw_m

        self._index      = None     # FAISS index
        self._pca        = None     # Optional PCA projection matrix
        self._ids:  List[str] = []
        self._meta: List[Dict] = []
        self._trained    = False

        self._has_faiss  = self._check_faiss()

        logger.info(
            f"[VectorSearch] index_type={index_type} | dim={embed_dim}→{project_dim} | "
            f"faiss={'✓' if self._has_faiss else '✗ (using torch fallback)'}"
        )

    def _check_faiss(self) -> bool:
        try:
            import faiss
            return True
        except ImportError:
            logger.warning("[VectorSearch] faiss not installed. Using PyTorch cosine fallback. "
                           "Install: pip install faiss-cpu")
            return False

    # ── Index Building ────────────────────────────────────────────────────────

    def build(
        self,
        embeddings: np.ndarray,      # (N, embed_dim) float32
        ids:        List[str],
        metadata:   Optional[List[Dict]] = None,
    ):
        """
        Build the search index from a numpy array of embeddings.

        Args:
            embeddings: (N, D) float32 — D = embed_dim
            ids:        list of N string identifiers
            metadata:   list of N dicts with clinical metadata
        """
        assert len(embeddings) == len(ids), "embeddings and ids must have same length"
        N, D = embeddings.shape
        logger.info(f"[VectorSearch] Building index: {N:,} vectors × {D} dims")
        t0 = time.time()

        # L2 normalize for cosine similarity via inner product
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-9
        embeddings_norm = (embeddings / norms).astype(np.float32)

        # Optionally project to lower dim
        index_dim = D
        if self.project_dim < D and N >= self.project_dim * 2:
            embeddings_norm, pca = self._fit_pca(embeddings_norm, self.project_dim)
            self._pca = pca
            index_dim = self.project_dim
            logger.info(f"[VectorSearch] PCA: {D}→{index_dim} dims")

        # Build FAISS index
        if self._has_faiss:
            self._index = self._build_faiss_index(embeddings_norm, index_dim, N)
        else:
            # Fallback: store normalized embeddings as tensor
            self._torch_embeddings = torch.from_numpy(embeddings_norm)
            self._trained = True

        self._ids  = list(ids)
        self._meta = list(metadata) if metadata else [{} for _ in ids]

        elapsed = time.time() - t0
        logger.info(
            f"[VectorSearch] Index built: {N:,} vectors | "
            f"{elapsed:.2f}s | type={self.index_type}"
        )

    def _build_faiss_index(
        self, embeddings: np.ndarray, dim: int, N: int
    ):
        import faiss

        if self.index_type == "flat" or N < 1000:
            # Exact inner product (= cosine for normalized vecs)
            index = faiss.IndexFlatIP(dim)
            index.add(embeddings)

        elif self.index_type == "hnsw":
            index = faiss.IndexHNSWFlat(dim, self.hnsw_m, faiss.METRIC_INNER_PRODUCT)
            index.hnsw.efConstruction = 200
            index.hnsw.efSearch = 64
            index.add(embeddings)

        elif self.index_type == "ivfpq":
            n_clusters = min(self.n_clusters, N // 10)
            quantizer  = faiss.IndexFlatIP(dim)
            index      = faiss.IndexIVFPQ(
                quantizer, dim, n_clusters, self.m_subvecs, 8,
                faiss.METRIC_INNER_PRODUCT
            )
            index.train(embeddings)
            index.add(embeddings)
            index.nprobe = self.n_probe

        else:  # ivfflat (default)
            n_clusters = min(self.n_clusters, max(1, N // 10))
            quantizer  = faiss.IndexFlatIP(dim)
            index      = faiss.IndexIVFFlat(quantizer, dim, n_clusters,
                                             faiss.METRIC_INNER_PRODUCT)
            index.train(embeddings)
            index.add(embeddings)
            index.nprobe = self.n_probe

        self._trained = True
        logger.info(
            f"[VectorSearch] FAISS {self.index_type} ready | "
            f"ntotal={index.ntotal:,}"
        )
        return index

    def _fit_pca(
        self, embeddings: np.ndarray, target_dim: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """PCA whitening for dimensionality reduction."""
        if self._has_faiss:
            import faiss
            pca = faiss.PCAMatrix(embeddings.shape[1], target_dim, -0.5)  # -0.5 = whitening
            pca.train(embeddings)
            projected = pca.apply_py(embeddings)
            # Re-normalize after PCA
            norms = np.linalg.norm(projected, axis=1, keepdims=True) + 1e-9
            return (projected / norms).astype(np.float32), pca
        else:
            # NumPy fallback PCA
            mean    = embeddings.mean(axis=0)
            centered = embeddings - mean
            U, S, Vt = np.linalg.svd(centered, full_matrices=False)
            components = Vt[:target_dim]
            projected  = centered @ components.T
            norms = np.linalg.norm(projected, axis=1, keepdims=True) + 1e-9
            pca_data = {"mean": mean, "components": components, "type": "numpy"}
            return (projected / norms).astype(np.float32), pca_data

    # ── Search ────────────────────────────────────────────────────────────────

    def search(
        self,
        query: Union[torch.Tensor, np.ndarray],
        k: int = 10,
        filter_fn: Optional[callable] = None,
        query_id: str = "query",
    ) -> SearchResponse:
        """
        Search for the k most similar retinal images.

        Args:
            query:     (D,) or (1,D) embedding tensor or numpy array
            k:         number of results to return
            filter_fn: optional function (metadata_dict → bool) for pre-filtering
            query_id:  identifier for this query (for logging/API response)

        Returns:
            SearchResponse with ranked results + clinical metadata
        """
        if not self._trained and self._index is None and not hasattr(self, "_torch_embeddings"):
            raise RuntimeError("[VectorSearch] Index not built. Call .build() first.")

        t0 = time.time()

        # Prepare query
        if isinstance(query, torch.Tensor):
            q = query.detach().cpu().float().numpy()
        else:
            q = np.array(query, dtype=np.float32)

        if q.ndim == 1:
            q = q[np.newaxis, :]

        # L2 normalize
        q = q / (np.linalg.norm(q, axis=1, keepdims=True) + 1e-9)

        # PCA projection if fitted
        if self._pca is not None:
            q = self._apply_pca(q)

        # Retrieve more if filtering
        retrieve_k = min(k * 4 if filter_fn else k, len(self._ids))

        # FAISS search
        if self._has_faiss and self._index is not None:
            scores, indices = self._index.search(q.astype(np.float32), retrieve_k)
            scores   = scores[0].tolist()
            indices  = indices[0].tolist()
        else:
            # PyTorch fallback
            q_t    = torch.from_numpy(q)
            sims   = (q_t @ self._torch_embeddings.T).squeeze(0)
            top_s, top_i = sims.topk(retrieve_k)
            scores   = top_s.tolist()
            indices  = top_i.tolist()

        # Build results with metadata
        results = []
        for rank, (score, idx) in enumerate(zip(scores, indices)):
            if idx < 0 or idx >= len(self._ids):
                continue

            meta = self._meta[idx]

            # Apply optional metadata filter
            if filter_fn is not None and not filter_fn(meta):
                continue

            dr_grade = meta.get("dr_grade") or meta.get("label")
            results.append(SearchResult(
                rank=len(results),
                image_id=self._ids[idx],
                score=round(float(score), 4),
                distance=round(float(1.0 - score), 4),
                dr_grade=int(dr_grade) if dr_grade is not None else None,
                dr_label=DR_GRADE_LABELS.get(int(dr_grade)) if dr_grade is not None else None,
                amd_stage=meta.get("amd_stage"),
                dataset=meta.get("dataset"),
                visit_date=meta.get("visit_date"),
                image_path=meta.get("image_path"),
                extra={k: v for k, v in meta.items()
                       if k not in ("dr_grade", "label", "amd_stage",
                                    "dataset", "visit_date", "image_path")},
            ))

            if len(results) >= k:
                break

        search_time_ms = (time.time() - t0) * 1000

        return SearchResponse(
            query_id=query_id,
            num_results=len(results),
            search_time_ms=search_time_ms,
            index_size=len(self._ids),
            results=results,
        )

    def search_by_filter(
        self,
        dr_grade: Optional[int] = None,
        dataset:  Optional[str] = None,
        k: int = 50,
    ) -> List[Dict]:
        """
        Retrieve cases matching a metadata filter (no embedding query).

        Useful for: "give me 50 Moderate DR cases from APTOS"
        """
        results = []
        for i, meta in enumerate(self._meta):
            match = True
            if dr_grade is not None and meta.get("dr_grade") != dr_grade:
                match = False
            if dataset is not None and meta.get("dataset") != dataset:
                match = False
            if match:
                results.append({"image_id": self._ids[i], "metadata": meta})
            if len(results) >= k:
                break
        return results

    def _apply_pca(self, vectors: np.ndarray) -> np.ndarray:
        if self._has_faiss and hasattr(self._pca, "apply_py"):
            projected = self._pca.apply_py(vectors)
        else:
            mean       = self._pca["mean"]
            components = self._pca["components"]
            projected  = (vectors - mean) @ components.T

        norms = np.linalg.norm(projected, axis=1, keepdims=True) + 1e-9
        return (projected / norms).astype(np.float32)

    # ── Index Building from Pipeline / Dataset ────────────────────────────────

    def build_from_pipeline(
        self,
        pipeline,                          # RetinaGPTPipeline
        image_dir: str,
        labels_csv: Optional[str] = None,
        dataset_name: str = "dataset",
        batch_size: int = 32,
        max_images: Optional[int] = None,
        save_path: Optional[str] = None,
    ) -> int:
        """
        Build index by encoding all images in a directory.

        Args:
            pipeline:     RetinaGPTPipeline with loaded model
            image_dir:    directory containing fundus images
            labels_csv:   optional CSV with image_id → label mapping
            dataset_name: name to store in metadata
            batch_size:   images per batch
            max_images:   limit total images (for testing)
            save_path:    auto-save index after building

        Returns:
            Number of images indexed
        """
        from pathlib import Path
        import cv2

        image_dir = Path(image_dir)
        extensions = {".png", ".jpg", ".jpeg", ".tif"}
        image_files = sorted([f for f in image_dir.rglob("*")
                               if f.suffix.lower() in extensions])

        if max_images:
            image_files = image_files[:max_images]

        # Load labels if provided
        labels = {}
        if labels_csv:
            try:
                import pandas as pd
                df = pd.read_csv(labels_csv)
                id_col  = next((c for c in df.columns
                                if any(k in c.lower() for k in ["id", "image"])), df.columns[0])
                lbl_col = next((c for c in df.columns
                                if any(k in c.lower() for k in ["label", "diagnosis", "grade", "level"])),
                               df.columns[-1])
                labels = {str(row[id_col]): int(row[lbl_col]) for _, row in df.iterrows()}
            except Exception as e:
                logger.warning(f"[VectorSearch] Could not load labels: {e}")

        logger.info(f"[VectorSearch] Encoding {len(image_files):,} images from {image_dir}")

        all_embeddings, all_ids, all_meta = [], [], []
        t0 = time.time()

        for i, img_path in enumerate(image_files):
            if (i + 1) % 100 == 0:
                elapsed = time.time() - t0
                logger.info(f"  {i+1}/{len(image_files)} | {(i+1)/elapsed:.1f} img/s")

            try:
                result = pipeline.analyze(str(img_path), explain=False)

                if result.embedding is None:
                    continue

                all_embeddings.append(result.embedding.astype(np.float32))
                all_ids.append(img_path.stem)

                meta = {
                    "image_path": str(img_path),
                    "dataset":    dataset_name,
                    "dr_grade":   result.dr_grade,
                    "amd_stage":  result.amd_stage,
                }
                if img_path.stem in labels:
                    meta["label"] = labels[img_path.stem]

                all_meta.append(meta)

            except Exception as e:
                logger.warning(f"  Failed on {img_path.name}: {e}")
                continue

        if not all_embeddings:
            logger.error("[VectorSearch] No embeddings generated!")
            return 0

        embeddings_array = np.stack(all_embeddings)
        self.build(embeddings_array, all_ids, all_meta)

        if save_path:
            self.save(save_path)

        total_time = time.time() - t0
        logger.info(
            f"[VectorSearch] ✅ Indexed {len(all_ids):,} images in "
            f"{total_time:.1f}s ({len(all_ids)/total_time:.1f} img/s)"
        )
        return len(all_ids)

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(self, path: str):
        """Save index + metadata to disk."""
        import pickle
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        # Save metadata
        meta_path = path.with_suffix(".meta.json")
        with open(meta_path, "w") as f:
            json.dump({
                "ids":          self._ids,
                "metadata":     self._meta,
                "config": {
                    "embed_dim":   self.embed_dim,
                    "index_type":  self.index_type,
                    "project_dim": self.project_dim,
                },
            }, f, indent=2)

        # Save FAISS index
        if self._has_faiss and self._index is not None:
            import faiss
            faiss.write_index(self._index, str(path))
        elif hasattr(self, "_torch_embeddings"):
            torch.save(self._torch_embeddings, str(path) + ".pt")

        # Save PCA if fitted
        if self._pca is not None:
            with open(str(path) + ".pca", "wb") as f:
                pickle.dump(self._pca, f)

        logger.info(f"[VectorSearch] Saved: {len(self._ids):,} vectors → {path}")

    @classmethod
    def load(cls, path: str) -> "VectorSearchEngine":
        """Load index from disk."""
        import pickle
        path = Path(path)

        meta_path = path.with_suffix(".meta.json")
        with open(meta_path) as f:
            data = json.load(f)

        config = data.get("config", {})
        engine = cls(
            embed_dim=config.get("embed_dim",   1024),
            index_type=config.get("index_type", "ivfflat"),
            project_dim=config.get("project_dim", 256),
        )
        engine._ids  = data["ids"]
        engine._meta = data["metadata"]

        # Load FAISS index
        if engine._has_faiss and path.exists():
            import faiss
            engine._index   = faiss.read_index(str(path))
            engine._trained = True
        elif (Path(str(path) + ".pt")).exists():
            engine._torch_embeddings = torch.load(str(path) + ".pt", map_location="cpu")
            engine._trained = True

        # Load PCA
        pca_path = Path(str(path) + ".pca")
        if pca_path.exists():
            with open(pca_path, "rb") as f:
                engine._pca = pickle.load(f)

        logger.info(f"[VectorSearch] Loaded: {len(engine._ids):,} vectors from {path}")
        return engine

    # ── Stats ─────────────────────────────────────────────────────────────────

    def stats(self) -> Dict:
        dr_dist = {}
        for meta in self._meta:
            grade = meta.get("dr_grade") or meta.get("label")
            if grade is not None:
                grade = int(grade)
                dr_dist[grade] = dr_dist.get(grade, 0) + 1

        return {
            "total_vectors":  len(self._ids),
            "embed_dim":      self.embed_dim,
            "index_dim":      self.project_dim,
            "index_type":     self.index_type,
            "faiss":          self._has_faiss,
            "trained":        self._trained,
            "dr_distribution": {
                DR_GRADE_LABELS.get(k, str(k)): v
                for k, v in sorted(dr_dist.items())
            },
        }

    def __len__(self) -> int:
        return len(self._ids)

    def __repr__(self) -> str:
        return (
            f"VectorSearchEngine("
            f"n={len(self._ids):,}, "
            f"type={self.index_type}, "
            f"dim={self.embed_dim}→{self.project_dim}, "
            f"faiss={'✓' if self._has_faiss else '✗'})"
        )
