# Reflection — Lab 19

**Tên:** Nguyễn Hùng Mạnh
**Cohort:** Cohort 4
**Path đã chạy:** lite

---

## Câu hỏi (≤ 200 chữ)

> Trên golden set 50 queries, mode nào thắng ở loại query nào (`exact` /
> `paraphrase` / `mixed`), và tại sao? Khi nào bạn **không** dùng hybrid
> (i.e. khi nào pure BM25 hoặc pure vector là lựa chọn đúng)?

Trên golden set, **BM25 thắng trên `exact` queries** vì chứa từ kỹ thuật verbatim trong corpus — keyword matching đã đủ mạnh. **Semantic vector có ưu thế trên `paraphrase` queries** khi từ tiếng Việt diễn đạt lại không xuất hiện verbatim, nhưng bge-small-en yếu trên tiếng Việt nên recall chỉ ~24-32%. **Hybrid thắng rõ trên `mixed` queries** (~100% vs 97-98%) vì kết hợp cả keyword signal lẫn semantic understanding qua RRF.

Hybrid thắng trung bình tổng thể nhờ robust trên mọi kiểu query — đó là lý do production luôn default hybrid.

**Khi nào KHÔNG dùng hybrid:**
1. **Latency-critical systems** (< 5ms): hybrid gọi cả BM25 lẫn vector, tốn gấp đôi. Pure BM25 nhanh hơn đáng kể.
2. **Exact-match dominated corpus** (logs, mã lỗi): BM25 đủ tốt, vector search thêm noise.
3. **Multilingual semantic-only** (khi đổi sang bge-m3): trên corpus với paraphrase thuần, vector alone có thể đủ nếu embedding model đủ mạnh — hybrid chỉ thêm latency mà không cải thiện recall.

---

## Điều ngạc nhiên nhất khi làm lab này

Post-filter recall sập hoàn toàn ở filter ~4% corpus mà không có bất kỳ warning hay exception nào — hệ thống âm thầm trả lời sai. Và semantic cache 0.75 ngưỡng (con số AWS công bố) vẫn để lọt false hits đáng kể trên corpus này.

---

## Bonus challenge

- [x] Đã làm bonus (xem `bonus/`)
- [ ] Pair work với: _(làm một mình)_
