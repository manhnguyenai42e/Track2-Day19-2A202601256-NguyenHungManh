"""Generate compound (multi-intent) queries + brute-force ground truth for NB6.

Outputs data/agent_queries.jsonl -- 12 questions, each deliberately combining
TWO topic clusters, e.g.

    "tự động mở rộng theo lưu lượng và cân bằng tải giữa nhiều region"

Ground truth = the exact top-N nearest documents for EACH sub-question,
computed by brute-force cosine (no ANN, no filter). That makes the evaluation
in NB6 honest: both strategies are scored against the same correct answer, and
both are given the SAME retrieval budget. The only difference measured is
strategy -- one embedding of a compound question versus two of its parts.

Run:  python scripts/gen_agent_queries.py
"""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.filters import FilteredIndex          # noqa: E402
from app.search import Searcher                # noqa: E402

PER_SIDE = 8   # ground-truth docs per sub-question -> 16 total per question

# (topic_a, sub_query_a, topic_b, sub_query_b) -- concepts lifted from the
# same vocabulary seed_corpus.py used, so they are genuinely in the corpus.
PAIRS = [
    ("cloud", "tự động mở rộng theo lưu lượng",
     "networking", "cân bằng tải giữa nhiều region"),
    ("security", "bảo vệ dữ liệu nhạy cảm khi lưu trữ",
     "database", "đảm bảo tính nhất quán giữa nhiều bản sao"),
    ("ai_ml", "tinh chỉnh mô hình trên domain cụ thể",
     "data_eng", "xử lý sự kiện streaming với Kafka và Flink"),
    ("devops", "triển khai blue-green giảm rủi ro release",
     "backend", "circuit breaker tránh cascading failure"),
    ("frontend", "tối ưu LCP và FID cho Core Web Vitals",
     "mobile", "tối ưu kích thước APK cho ứng dụng di động"),
    ("database", "tối ưu truy vấn với chỉ mục B-tree",
     "cloud", "tối ưu chi phí với spot instance"),
    ("networking", "tối ưu độ trễ end-to-end",
     "devops", "rollback tự động khi error rate tăng"),
    ("ai_ml", "huấn luyện mô hình trên dataset lớn",
     "security", "xác thực hai yếu tố cho người dùng"),
    ("backend", "thiết kế API idempotent cho retry an toàn",
     "data_eng", "schema evolution không phá vỡ downstream"),
    ("mobile", "đồng bộ dữ liệu khi mất kết nối",
     "frontend", "tải lười component theo route"),
    ("cloud", "tách biệt môi trường dev và prod",
     "devops", "triển khai blue-green giảm rủi ro release"),
    ("data_eng", "xử lý sự kiện streaming với Kafka và Flink",
     "database", "tối ưu truy vấn với chỉ mục B-tree"),
]


def main() -> int:
    corpus = ROOT / "data" / "corpus_vn.jsonl"
    if not corpus.exists():
        print(f"missing {corpus} -- run `make seed` first")
        return 1

    print("building index (embeds 1000 docs, ~15 s)...", flush=True)
    index = FilteredIndex.from_searcher(Searcher.from_corpus(corpus))
    print("index built, computing ground truth...", flush=True)

    out = ROOT / "data" / "agent_queries.jsonl"
    everything = lambda d: True                       # noqa: E731
    with out.open("w", encoding="utf-8") as f:
        for i, (ta, qa, tb, qb) in enumerate(PAIRS):
            gold_a = index.exact_top_k(index.embed(qa), everything, PER_SIDE)
            gold_b = index.exact_top_k(index.embed(qb), everything, PER_SIDE)
            f.write(json.dumps({
                "query_id": f"mq_{i:03d}",
                "question": f"{qa} và {qb}",
                "sub_questions": [qa, qb],
                "topics": [ta, tb],
                "relevant_doc_ids": sorted(set(gold_a) | set(gold_b)),
                "gold_a": gold_a,
                "gold_b": gold_b,
            }, ensure_ascii=False) + "\n")
    print(f"wrote {out} ({len(PAIRS)} compound queries, {PER_SIDE} gold docs per side)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
