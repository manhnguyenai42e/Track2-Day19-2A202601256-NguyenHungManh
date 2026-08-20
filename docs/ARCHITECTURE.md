# Kiến trúc Hệ thống — Day 19 Vector & Feature Store Lab

## 1. Tổng quan
Hệ thống kết hợp **Vector Database (Qdrant)** và **Feature Store (Feast)** cùng thuật toán **Hybrid Search (BM25 + Dense Vectors + RRF)** để xây dựng hệ thống truy xuất và cá nhân hóa ngữ cảnh cho AI Agent.

```
+-------------------------------------------------------------------------------+
|                                FastAPI /search                                |
|                                (app/main.py)                                  |
+---------------------------------------+---------------------------------------+
                                        |
        +-------------------------------+-------------------------------+
        |                                                               |
+-------v---------------+                                       +-------v---------------+
|  Hybrid Search Engine |                                       |  Feast Feature Store  |
|     (app/search.py)   |                                       |   (app/features.py)   |
+-------+---------------+                                       +-------+---------------+
        |                                                               |
        +---> BM25 (rank-bm25)                                          +---> SQLite Online Store (User profile)
        +---> FastEmbed (bge-small-en, ONNX)                            +---> Parquet Offline Store (PIT joins)
        +---> Qdrant (:memory: / server)                                +---> On-demand Feature View (Spike detection)
        +---> RRF Fusion: 1 / (k + rank)
```

## 2. Các Module cốt lõi
- `app/search.py`: Triển khai Sparse BM25 + Dense Qdrant + RRF fusion $1/(k + \text{rank})$.
- `app/embeddings.py`: Pluggable backend (FastEmbed, sentence-transformers, OpenAI).
- `app/features.py`: Feast feature store helper, PIT join, on-demand feature view.
- `app/metadata.py` & `app/filters.py`: Filtered-ANN vs Pre-filter vs Post-filter.
- `app/agent.py`: Agentic query decomposition & iterative retrieval.
- `app/cache.py`: Semantic cache với Cosine Similarity threshold và namespace isolation.
- `app/main.py`: REST API FastAPI exposing `/search` và `/healthz`.

## 3. Bonus Module: HybridMemoryAgent
- `bonus/agent.py`: Kết hợp Episodic Memory (Qdrant) và Semantic Profile (Feast) cho trợ lý cá nhân.
- `bonus/demo.py`: Kịch bản demo 5 tình huống truy xuất thực tế.
