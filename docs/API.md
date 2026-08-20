# API Documentation — Day 19 Search & Memory API

## 1. Healthcheck

### `GET /healthz`
Kiểm tra trạng thái sẵn sàng của Search Engine.

- **Response:**
```json
{
  "ready": true,
  "size": 1000,
  "embed_model": "BAAI/bge-small-en-v1.5",
  "embed_dim": 384,
  "qdrant_mode": "memory"
}
```

---

## 2. Search Endpoint

### `GET /search`
Thực hiện tìm kiếm tài liệu với 3 chế độ: `keyword`, `semantic`, hoặc `hybrid`.

- **Query Parameters:**
  - `q` (string, required): Câu truy vấn của người dùng.
  - `mode` (string, optional, default: `"hybrid"`): Chế độ tìm kiếm (`keyword` | `semantic` | `hybrid`).
  - `top_k` (integer, optional, default: `10`): Số lượng kết quả trả về.
  - `rrf_k` (integer, optional, default: `60`): Hằng số RRF làm mượt rank.

- **Response (`SearchResponse`):**
```json
{
  "query": "cloud computing tự động mở rộng",
  "mode": "hybrid",
  "hits": [
    {
      "doc_id": "doc_0102",
      "title": "Auto-scaling và cân bằng tải trên đám mây",
      "text": "Tự động điều chỉnh số lượng instance...",
      "score": 0.0328
    }
  ],
  "latency_ms": 12.4
}
```
