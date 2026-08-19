# Hybrid Memory Architecture for Vietnamese AI Assistant

**Author:** Nguyễn Hùng Mạnh · Cohort 4 · Lab 19 Bonus Challenge

---

## 1. Architecture Overview

This system provides a personal AI assistant with **two complementary memory types**, each backed by the appropriate storage primitive from Lab 19:

```
┌─────────────────────────────────────────────────────────┐
│                    User Query                           │
│           "Tôi đã đọc gì về Kubernetes?"                │
└──────────────────────┬──────────────────────────────────┘
                       │
                       ▼
          ┌────────────────────────┐
          │    HybridMemoryAgent   │
          │  ┌──────────────────┐  │
          │  │  Query Analyzer  │  │
          │  │  (intent detect) │  │
          │  └───────┬──────────┘  │
          │          │             │
          │    ┌─────┴──────┐     │
          │    ▼            ▼     │
          │ ┌────────┐ ┌───────┐  │
          │ │ Recall  │ │Profile│  │
          │ │ Path    │ │ Path  │  │
          │ └───┬────┘ └───┬───┘  │
          └─────┼──────────┼──────┘
                │          │
        ┌───────┴──┐  ┌───┴─────────┐
        │  Qdrant  │  │    Feast     │
        │ (Vector) │  │(Feature Store│
        │          │  │   SQLite)    │
        ├──────────┤  ├─────────────┤
        │Episodic  │  │ user_profile │
        │Memory    │  │ query_veloc. │
        │- convos  │  │ item_popular │
        │- docs    │  │              │
        │- notes   │  │ TTL-aware    │
        └──────────┘  └─────────────┘
                │          │
                ▼          ▼
          ┌────────────────────────┐
          │   Context Assembler    │
          │                        │
          │ "User likes <topic>,   │
          │  reads at <speed>wpm.  │
          │  Recent: <N> queries.  │
          │  Top memories:         │
          │  1. <memory_1>         │
          │  2. <memory_2>         │
          │  3. <memory_3>"        │
          └────────────────────────┘
```

**Data flow:**
1. **Remember path:** User input → chunk text → embed with `fastembed` → upsert to Qdrant with `user_id` payload filter → update query velocity features in Feast.
2. **Recall path:** Query → (a) hybrid search Qdrant filtered by `user_id` for episodic memories, (b) `get_online_features()` from Feast for stable profile + recent activity → assemble context string combining both.

---

## 2. Three Architecture Decisions (with Explicit Tradeoffs)

### Decision 1: Chunking Strategy — Per-message with Sliding Window Merge

**Options considered:**

| Strategy | Retrieval Quality | Storage Cost | Context Window Fit |
|---|---|---|---|
| Per-message | High precision, low recall | Low (1 vector/msg) | Easy — each chunk is short |
| Per-conversation | Low precision, high recall | Very low | Hard — chunks are long |
| Semantic break | Best precision+recall | Medium | Medium |
| Fixed token window | Medium | Medium | Predictable |

**Decision:** Per-message chunking with **sliding window merge** (merge consecutive messages from same conversation into chunks of ~200 tokens when they share topic continuity, detected by cosine similarity > 0.8 between consecutive embeddings).

**Tradeoff reasoning:** Pure per-message loses conversational context ("what about the other option?" references previous message). Pure per-conversation produces chunks too long for the 384-dim `bge-small` embedding to represent well — the embedding "averages out" over the whole conversation and lands in a vague centroid (exactly the NB6 lesson about single-shot vs. agentic retrieval for compound queries). Sliding window merge preserves local context while keeping chunks short enough for precise retrieval. The cost is ~30% more vectors than per-conversation, but Qdrant in-memory handles 10K vectors trivially.

**Rejected alternative:** Storing episodic memory as embedding features inside Feast (one `user_embedding` feature per user, updated by averaging all memory embeddings). Rejected because: (a) re-index cycle for memories (hourly, as new conversations happen) is completely different from profile refresh (daily/weekly), and mixing them in one store causes either stale memories or unnecessary profile re-materialization; (b) a single averaged embedding per user destroys the ability to retrieve *specific* memories — you get "what topics this user is generally about" but lose "what did they say about Kubernetes specifically."

### Decision 2: Feature Schema — Tabular Features with Topic Affinity String

**Feature schema for Feast:**

| Feature View | Entity | Features | TTL | Source | Rationale |
|---|---|---|---|---|---|
| `user_profile` | `user_id` | `reading_speed_wpm` (Int64), `preferred_language` (String), `topic_affinity` (String) | 30 days | Daily batch from user settings | Slow-moving attributes — daily refresh sufficient |
| `query_velocity` | `user_id` | `queries_last_hour` (Int64), `distinct_topics_24h` (Int64) | 1 hour | Streaming / micro-batch | Fast-moving — must reflect recent behavior for "đang quan tâm gì?" queries |
| `item_popularity` | `doc_id` | `click_count_24h` (Int64), `ctr_7d` (Float32) | 24 hours | Hourly batch aggregation | Used for re-ranking — boost popular docs in recall results |

**Decision:** Use **tabular features** (explicit, interpretable) rather than **embedding features** (latent user preference vectors derived from interaction history).

**Tradeoff reasoning:** Embedding features (e.g., averaging the last 50 query embeddings into a `user_preference_embedding`) would capture latent interests better. But: (a) they are opaque — when the assistant says "based on your interests," you can't explain *which* interest; (b) they require a separate model/pipeline to compute and keep fresh; (c) for a 200-user POC, tabular features with explicit `topic_affinity` are sufficient and debuggable. The `topic_affinity` string ("cloud", "ai_ml", "security") is directly used to boost retrieval results via Qdrant payload filter — simple, auditable, no hidden state.

**When to upgrade:** If the user base grows past ~10K and explicit topic tags become too coarse, introduce a learned embedding feature alongside (not replacing) the tabular ones — exactly the "embedding as feature" pattern from NB8 §6.

### Decision 3: Freshness Strategy — Tiered by Use Case (Vietnamese-Context Aware)

**Three freshness tiers:**

| Use Case | Freshness Need | Strategy | Latency |
|---|---|---|---|
| "Tôi đã đọc gì về X?" (episodic recall) | Near-real-time | **Synchronous write-through**: `remember()` embeds + upserts to Qdrant immediately | < 1s (embed ~30ms + upsert ~5ms for bge-small) |
| "Tôi đang quan tâm gì?" (activity pattern) | Minutes | **Micro-batch**: `query_velocity` features updated every 5 min via `materialize-incremental` | 0–5 min stale |
| "Recommend đọc gì tiếp" (profile-based) | Hours/Days | **Daily batch**: `user_profile` features refreshed once per day | Up to 24h stale |

**Decision:** Episodic memory gets synchronous write-through (immediate availability). Activity features use 5-minute micro-batch. Profile uses daily batch.

**Tradeoff reasoning:** Making everything synchronous (sub-second) is possible but wasteful — `topic_affinity` changes maybe once a month, so computing it every request is pure overhead. Conversely, making episodic memory batch-only (5-min delay) means a user who just told the assistant "remember this" won't see it in their next query 10 seconds later — that's a broken UX. The tiered approach matches *business freshness requirements* to *engineering cost* (exactly the Feast TTL lesson from NB4: `user_profile` TTL=30d vs `query_velocity` TTL=1h).

---

## 3. Vietnamese-Context Awareness

### Tokenization for BM25 keyword search

Vietnamese is a **syllable-based** language where words are multi-syllable ("cơ sở dữ liệu" = "database" = 4 syllables). Whitespace-split tokenization (used in `app/search.py`) treats each syllable as a separate token, which:

- **Hurts BM25 precision:** "cơ sở" (foundation/base) matches "cơ sở dữ liệu" (database) and "cơ sở hạ tầng" (infrastructure) equally — a false positive.
- **But works for hybrid search:** The vector component (`bge-small` or `bge-m3`) captures semantic similarity regardless of tokenization. Hybrid search with RRF compensates for BM25's weakness on Vietnamese.

**Decision for POC:** Use whitespace tokenization (same as lab) + rely on hybrid mode to compensate. For production, integrate `underthesea` word segmenter (not `pyvi` — `underthesea` has better accuracy on modern Vietnamese text and active maintenance as of 2026).

**Tradeoff:** `underthesea` adds ~200ms to first load and ~2ms per query for segmentation. For sub-50ms P99 (NB3 rubric), this is acceptable but must be measured. `pyvi` is faster (~0.5ms) but less accurate on code-switching text (common among Vietnamese developers who mix "deploy lên cloud", "fix cái bug này").

### Code-switching (vi/en mix)

Vietnamese tech users frequently code-switch: "anh ơi check giúp em cái pipeline này" mixes Vietnamese and English. The embedding model choice matters:

- `bge-small-en` (lab default): Weak on Vietnamese paraphrases (NB2 lesson — ~24-32% recall).
- `bge-m3` (multilingual): Strong on both vi and en, handles code-switching well. **Recommended for production.**

The POC uses `bge-small` to match lab rubric thresholds, but the `Embedder` class already supports swapping via `EMBEDDING_BACKEND=bge-m3`.

---

## 4. Honest Limitations

This POC does **not** handle:

1. **Privacy isolation per user:** Qdrant payload filter (`user_id`) is "soft" isolation — a bug in filter construction leaks all users' memories (exactly the OWASP LLM08 pattern from NB7 §4). Production needs per-user collections or per-user encryption.
2. **Memory CRUD:** No update/delete on individual memories. Users cannot say "forget that conversation."
3. **Memory consolidation:** No periodic summarization of old memories. Over months, the vector store grows unbounded.
4. **Encryption at rest:** SQLite Feast registry and Qdrant in-memory store are unencrypted.
5. **Multi-device sync:** Single-process, no replication or sync protocol.
6. **Semantic cache:** Not integrated into the recall path. Adding it (NB7 pattern) would reduce embedding compute by ~80% for repeated queries but requires careful threshold tuning per corpus.

---

## 5. Vibe Coding Reflection

**Most effective prompt:** "Implement HybridMemoryAgent with remember() and recall(). remember() chunks by sentences, embeds with fastembed, upserts to Qdrant with user_id payload. recall() does hybrid search filtered by user_id, then gets Feast online features, then assembles a context string." — Giving the complete data flow in one prompt produced correct code in one shot.

**Prompt that failed:** "Make a good AI memory system" — Too vague, AI generated a bloated class with LLM calls, LangChain, and 15 dependencies. The fix: specify *exactly* what each method does and what it returns, then review the diff.
