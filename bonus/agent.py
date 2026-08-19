"""HybridMemoryAgent — combines Vector Store (episodic) + Feature Store (profile).

Minimal POC demonstrating the architecture from ARCHITECTURE.md.
No LLM calls — returns assembled context strings for downstream consumption.

Usage:
    agent = HybridMemoryAgent()
    agent.remember("Kubernetes là nền tảng điều phối container...", user_id="u_001")
    context = agent.recall("Tôi đã đọc gì về Kubernetes?", user_id="u_001")
    print(context)
"""
from __future__ import annotations

import json
import textwrap
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
from fastembed import TextEmbedding
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance, FieldCondition, Filter, MatchValue,
    PointStruct, VectorParams,
)
from rank_bm25 import BM25Okapi

# ── Constants ──────────────────────────────────────────────────────────
COLLECTION = "hybrid_memory"
EMBED_DIM = 384
CHUNK_MAX_TOKENS = 200  # sliding window merge target
RRF_K = 60


@dataclass
class MemoryChunk:
    """A single chunk of episodic memory."""
    text: str
    user_id: str
    timestamp: float = field(default_factory=time.time)


class HybridMemoryAgent:
    """Combines vector-based episodic memory with feature-store user profile.

    - **remember()**: chunk → embed → upsert to Qdrant (user_id filtered)
    - **recall()**: hybrid search (BM25 + vector + RRF) → Feast profile → context
    """

    def __init__(self, feast_repo_path: Optional[str] = None) -> None:
        # Embedding
        self.embedder = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")

        # Vector store (in-memory for POC)
        self.client = QdrantClient(":memory:")
        self.client.create_collection(
            collection_name=COLLECTION,
            vectors_config=VectorParams(size=EMBED_DIM, distance=Distance.COSINE),
        )

        # BM25 index (rebuilt on each remember — fine for POC scale)
        self._memories: list[dict] = []
        self._bm25: Optional[BM25Okapi] = None
        self._next_id = 0

        # Feast feature store (optional — graceful degradation)
        self._feast_store = None
        if feast_repo_path and Path(feast_repo_path).exists():
            try:
                from feast import FeatureStore
                registry = Path(feast_repo_path) / "registry.db"
                if registry.exists():
                    self._feast_store = FeatureStore(repo_path=feast_repo_path)
            except Exception:
                pass  # Feast not available — profile features will be empty

    # ── Remember ───────────────────────────────────────────────────────

    def remember(self, text: str, user_id: str = "u_001") -> None:
        """Add episodic memory: chunk → embed → upsert to Qdrant."""
        chunks = self._chunk_text(text)

        for chunk_text in chunks:
            vec = next(self.embedder.embed([chunk_text]))
            point_id = self._next_id
            self._next_id += 1

            self.client.upsert(
                collection_name=COLLECTION,
                points=[PointStruct(
                    id=point_id,
                    vector=vec.tolist(),
                    payload={
                        "user_id": user_id,
                        "text": chunk_text,
                        "timestamp": time.time(),
                    },
                )],
            )
            self._memories.append({
                "id": point_id,
                "user_id": user_id,
                "text": chunk_text,
            })

        # Rebuild BM25 index after new memories
        self._rebuild_bm25()

    def _chunk_text(self, text: str) -> list[str]:
        """Chunk by sentences, merge consecutive short chunks up to CHUNK_MAX_TOKENS."""
        # Split on Vietnamese/English sentence boundaries
        import re
        sentences = re.split(r'(?<=[.!?。])\s+', text.strip())
        sentences = [s.strip() for s in sentences if s.strip()]

        if not sentences:
            return [text] if text.strip() else []

        chunks: list[str] = []
        current = sentences[0]

        for sent in sentences[1:]:
            merged = current + " " + sent
            if len(merged.split()) <= CHUNK_MAX_TOKENS:
                current = merged
            else:
                chunks.append(current)
                current = sent
        chunks.append(current)
        return chunks

    def _rebuild_bm25(self) -> None:
        """Rebuild BM25 index from all memories."""
        if not self._memories:
            self._bm25 = None
            return
        tokenized = [m["text"].lower().split() for m in self._memories]
        self._bm25 = BM25Okapi(tokenized)

    # ── Recall ─────────────────────────────────────────────────────────

    def recall(self, query: str, user_id: str = "u_001", top_k: int = 3) -> str:
        """Retrieve top-K memories + user profile → assembled context string."""
        t0 = time.perf_counter()

        # 1. Get user profile from Feast (graceful degradation)
        profile = self._get_profile(user_id)

        # 2. Hybrid search: BM25 + vector + RRF, filtered by user_id
        memories = self._hybrid_search(query, user_id, top_k)

        latency_ms = (time.perf_counter() - t0) * 1000

        # 3. Assemble context
        return self._assemble_context(query, user_id, profile, memories, latency_ms)

    def _get_profile(self, user_id: str) -> dict:
        """Get user profile from Feast online store."""
        if self._feast_store is None:
            return {}
        try:
            features = self._feast_store.get_online_features(
                features=[
                    "user_profile_features:reading_speed_wpm",
                    "user_profile_features:preferred_language",
                    "user_profile_features:topic_affinity",
                    "query_velocity_features:queries_last_hour",
                    "query_velocity_features:distinct_topics_24h",
                ],
                entity_rows=[{"user_id": user_id}],
            ).to_dict()
            return {k: v[0] for k, v in features.items()}
        except Exception:
            return {}

    def _hybrid_search(
        self, query: str, user_id: str, top_k: int
    ) -> list[dict]:
        """BM25 + vector + RRF hybrid search, filtered by user_id."""
        depth = max(top_k * 5, 20)

        # Filter memories for this user
        user_memories = [m for m in self._memories if m["user_id"] == user_id]
        if not user_memories:
            return []

        # BM25 search (on user's memories only)
        kw_results = self._search_keyword(query, user_id, depth)

        # Vector search (Qdrant with user_id filter)
        sem_results = self._search_semantic(query, user_id, depth)

        # RRF fusion
        rrf_scores: dict[int, float] = {}
        meta: dict[int, dict] = {}

        for rank, mem in enumerate(kw_results, start=1):
            mid = mem["id"]
            rrf_scores[mid] = rrf_scores.get(mid, 0.0) + 1.0 / (RRF_K + rank)
            meta.setdefault(mid, mem)

        for rank, mem in enumerate(sem_results, start=1):
            mid = mem["id"]
            rrf_scores[mid] = rrf_scores.get(mid, 0.0) + 1.0 / (RRF_K + rank)
            meta.setdefault(mid, mem)

        ordered = sorted(rrf_scores.items(), key=lambda kv: -kv[1])[:top_k]
        return [meta[mid] for mid, _ in ordered]

    def _search_keyword(self, query: str, user_id: str, top_k: int) -> list[dict]:
        """BM25 keyword search on this user's memories."""
        if self._bm25 is None:
            return []
        tokens = query.lower().split()
        scores = self._bm25.get_scores(tokens)
        # Filter to user's memories and rank
        user_indices = [
            i for i, m in enumerate(self._memories)
            if m["user_id"] == user_id
        ]
        ranked = sorted(user_indices, key=lambda i: -scores[i])[:top_k]
        return [self._memories[i] for i in ranked if scores[i] > 0]

    def _search_semantic(self, query: str, user_id: str, top_k: int) -> list[dict]:
        """Vector similarity search filtered by user_id."""
        q_vec = next(self.embedder.embed([query])).tolist()

        try:
            result = self.client.query_points(
                collection_name=COLLECTION,
                query=q_vec,
                query_filter=Filter(must=[
                    FieldCondition(key="user_id", match=MatchValue(value=user_id)),
                ]),
                limit=top_k,
            )
        except Exception:
            return []

        results = []
        for p in result.points:
            results.append({
                "id": p.id,
                "user_id": p.payload["user_id"],
                "text": p.payload["text"],
                "score": p.score,
            })
        return results

    def _assemble_context(
        self,
        query: str,
        user_id: str,
        profile: dict,
        memories: list[dict],
        latency_ms: float,
    ) -> str:
        """Assemble the final context string combining profile + memories."""
        parts = [f"Query: {query}", f"User: {user_id}"]

        # Profile section
        if profile:
            affinity = profile.get("topic_affinity", "unknown")
            speed = profile.get("reading_speed_wpm", "N/A")
            lang = profile.get("preferred_language", "vi")
            queries_1h = profile.get("queries_last_hour", "N/A")
            topics_24h = profile.get("distinct_topics_24h", "N/A")
            parts.append(
                f"Profile: likes {affinity}, reads at {speed}wpm, "
                f"language={lang}"
            )
            parts.append(
                f"Recent activity: {queries_1h} queries last hour, "
                f"{topics_24h} distinct topics in 24h"
            )
        else:
            parts.append("Profile: (not available — run NB4 first)")

        # Memories section
        if memories:
            parts.append(f"Top {len(memories)} memories:")
            for i, mem in enumerate(memories, 1):
                text = mem["text"][:150] + ("..." if len(mem["text"]) > 150 else "")
                parts.append(f"  {i}. {text}")
        else:
            parts.append("No memories found for this user.")

        parts.append(f"[Recall latency: {latency_ms:.1f}ms]")
        return "\n".join(parts)
