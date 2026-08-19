"""Demo script for the HybridMemoryAgent — 5 query scenarios.

Each scenario demonstrates a different aspect of the hybrid memory architecture:
  1. Pure episodic recall (vector search)
  2. Profile-aware recommendation (Feast features)
  3. Recent activity query (query velocity features)
  4. Paraphrase query (vector similarity wins)
  5. Mixed query (hybrid search + profile)

Usage:
    cd <repo_root>
    python bonus/demo.py

Exits 0 on success, prints context for each query.
"""
from __future__ import annotations

import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

# Add repo root to path so imports work
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from bonus.agent import HybridMemoryAgent  # noqa: E402


def main() -> None:
    # Initialize agent with optional Feast (will gracefully degrade if NB4 not run)
    feast_repo = ROOT / "app" / "feast_repo"
    agent = HybridMemoryAgent(
        feast_repo_path=str(feast_repo) if feast_repo.exists() else None,
    )

    # ── Seed episodic memories ─────────────────────────────────────────
    print("=" * 70)
    print("Seeding episodic memories...")
    print("=" * 70)

    memories = [
        ("Kubernetes là nền tảng điều phối container mã nguồn mở. "
         "Nó tự động hóa việc triển khai, mở rộng và quản lý các ứng dụng container. "
         "Pod là đơn vị nhỏ nhất trong Kubernetes, chứa một hoặc nhiều container."),

        ("Cloud computing cho phép truy cập tài nguyên máy tính qua internet. "
         "Ba mô hình chính là IaaS, PaaS, và SaaS. "
         "Auto-scaling tự động điều chỉnh tài nguyên theo nhu cầu sử dụng."),

        ("Bảo mật hệ thống yêu cầu nhiều lớp phòng thủ. "
         "Xác thực đa yếu tố (MFA) giúp bảo vệ tài khoản. "
         "Zero Trust Architecture kiểm tra mọi request, không tin tưởng mặc định."),

        ("Machine Learning pipeline bao gồm thu thập dữ liệu, tiền xử lý, "
         "huấn luyện mô hình, và triển khai. Feature engineering quyết định "
         "chất lượng model. Data leakage là lỗi nghiêm trọng nhất trong ML pipeline."),

        ("DevOps kết hợp development và operations. CI/CD pipeline tự động hóa "
         "quá trình build, test, và deploy. Infrastructure as Code (IaC) "
         "quản lý hạ tầng bằng code thay vì cấu hình thủ công."),
    ]

    for mem in memories:
        agent.remember(mem, user_id="u_001")
    print(f"  Seeded {len(memories)} memory chunks for user u_001\n")

    # ── 5 Query Scenarios ──────────────────────────────────────────────
    queries = [
        {
            "question": "Tôi đã đọc gì về Kubernetes?",
            "scenario": "1. Pure episodic recall (vector search)",
            "explanation": "Simple keyword+semantic match on episodic memories",
        },
        {
            "question": "Recommend tôi đọc gì tiếp theo",
            "scenario": "2. Profile-aware recommendation (needs topic_affinity)",
            "explanation": "Uses Feast profile features to personalize results",
        },
        {
            "question": "Tôi đang quan tâm gì gần đây?",
            "scenario": "3. Recent activity query (needs queries_last_hour)",
            "explanation": "Uses query_velocity features for recency-aware response",
        },
        {
            "question": "Tài liệu về tự động mở rộng hạ tầng theo nhu cầu",
            "scenario": "4. Paraphrase query (vector similarity wins)",
            "explanation": "No exact keywords match — relies on embedding similarity",
        },
        {
            "question": "Cho tôi summary về bảo mật cloud",
            "scenario": "5. Mixed query (hybrid search + profile)",
            "explanation": "Combines keyword 'bảo mật' + semantic 'cloud' + profile",
        },
    ]

    for q in queries:
        print("=" * 70)
        print(f"Scenario: {q['scenario']}")
        print(f"Intent:   {q['explanation']}")
        print("-" * 70)

        context = agent.recall(q["question"], user_id="u_001")
        print(context)
        print()

    print("=" * 70)
    print("All 5 queries completed successfully.")
    print("=" * 70)


if __name__ == "__main__":
    main()
