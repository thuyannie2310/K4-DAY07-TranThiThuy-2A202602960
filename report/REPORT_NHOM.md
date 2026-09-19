# Báo Cáo Nhóm — Lab 7: Embedding & Vector Store

> **Bài làm cá nhân (solo).** Lab thiết kế cho nhóm nhiều người, mỗi người thử một chiến lược rồi so sánh. Ở bài này **một mình tôi chạy cả 5 chiến lược** trên cùng bộ tài liệu và cùng 5 câu hỏi đánh giá — phần "so sánh giữa các thành viên" vì vậy được trình bày thành **so sánh giữa các chiến lược**. Toàn bộ số liệu là kết quả chạy thật, không có thành viên nào khác đóng góp.
>
> File này giữ nguyên tên `REPORT_NHOM.md` theo yêu cầu nộp bài; phần cá nhân (hướng tiếp cận, dự đoán, kết quả riêng) nằm ở `REPORT_CANHAN.md`. Chi tiết thang điểm: `docs/SCORING.md`.

**Tổng điểm phần nhóm: 40** = Lựa chọn tài liệu (10) + Thiết kế chiến lược (15) + Chất lượng truy xuất (10) + Thuyết trình (5).

> **Cách tái lập toàn bộ số liệu trong báo cáo này:**
> ```bash
> pip install google-genai && export GEMINI_API_KEY=...
> EMBEDDING_PROVIDER=gemini python3 scripts/benchmark.py              # bảng kết quả chính
> EMBEDDING_PROVIDER=gemini python3 scripts/benchmark.py --no-filter  # ablation bỏ metadata filter
> ```
> Log gốc: `report/runs/benchmark-gemini.txt`, `report/runs/benchmark-gemini-nofilter.txt`, `report/runs/baseline.txt`.
> Backend nhúng: `gemini-embedding-001` (3072 chiều). **Không dùng MockEmbedder cho benchmark** — xem mục 3, ghi chú phương pháp.

---

## 1. Lựa chọn tài liệu (Document Set Quality) — Nhóm (10 điểm)

### Chủ đề (Domain) & Lý Do Chọn

**Chủ đề:** Dịch vụ và quy định Thư viện Trường Đại học Công Thương TP.HCM (HUIT) — đúng nhánh "dịch vụ/quy định đại học" của biến thể K4-L3A.

**Tại sao tôi chọn chủ đề này?**
> Đây là nhóm tài liệu công khai trên cổng thư viện HUIT, nội dung là quy định có số liệu cụ thể (hạn mức mượn, số ngày, mức phí) nên câu trả lời chuẩn kiểm chứng được từng chữ thay vì phải suy đoán. Quan trọng hơn, cùng một quy định lại tách theo **đối tượng áp dụng** (sinh viên / giảng viên) với con số khác nhau — đây chính là tình huống bắt buộc phải lọc metadata `audience` mới trả lời đúng, phù hợp yêu cầu riêng của L3A.

### Danh sách tài liệu (Data Inventory)

> Số ký tự đếm phần thân tài liệu, **không tính** khối front matter YAML. Nguồn đầy đủ trong `data/thu-vien/sources.csv`.

| # | Tên tài liệu | Nguồn (Source URL) | Ngày lấy / Phiên bản | Số ký tự | Metadata đã gán |
|---|--------------|------------|--------------------|----------|-----------------|
| 1 | Câu hỏi thường gặp thư viện | https://thuvien.huit.edu.vn/Page/nhung-cau-hoi-thuong-gap | 2026-09-19 / not-stated | 12.056 | audience=student, category=faq, department=library, language=vi |
| 2 | Hướng dẫn sử dụng thư viện | https://thuvien.huit.edu.vn/Page/huong-dan-su-dung-thu-vien | 2026-09-19 / not-stated | 5.377 | audience=all, category=guide, department=library, language=vi |
| 3 | Lưu hành tài liệu | https://thuvien.huit.edu.vn/Page/luu-hanh-tai-lieu | 2026-09-19 / not-stated | 3.379 | audience=all, category=circulation, department=library, language=vi |
| 4 | Quy định mượn trả — giảng viên, viên chức | https://thuvien.huit.edu.vn/Page/quy-dinh-su-dung-thu-vien | 2026-09-19 / not-stated | 1.871 | audience=faculty, category=borrowing, department=library, language=vi |
| 5 | Mượn liên thư viện | https://thuvien.huit.edu.vn/Page/muon-lien-thu-vien | 2026-09-19 / not-stated | 2.428 | audience=all, category=interlibrary, department=library, language=vi |
| 6 | Quy định mượn trả — sinh viên, học viên | https://thuvien.huit.edu.vn/Page/quy-dinh-su-dung-thu-vien | 2026-09-19 / not-stated | 2.403 | audience=student, category=borrowing, department=library, language=vi |
| 7 | Hướng dẫn sử dụng phòng học nhóm | https://thuvien.huit.edu.vn/Page/su-dung-phong-hoc-nhom | 2026-09-19 / not-stated | 1.879 | audience=all, category=facilities, department=library, language=vi |
| 8 | Quy định chung sử dụng thư viện HUIT | https://thuvien.huit.edu.vn/Page/quy-dinh-su-dung-thu-vien | 2026-09-19 / not-stated | 11.242 | audience=all, category=rules, department=library, language=vi |

**Tổng: 8 tài liệu / 40.635 ký tự** (yêu cầu 5–10 tài liệu ✔).

> **Ghi chú biên tập:** tài liệu #4 và #6 được **tách thủ công** từ cùng một trang nguồn "Quy định sử dụng thư viện HUIT". Trang gốc trộn hạn mức của mọi đối tượng trong một bảng; nhóm tách theo đối tượng để mỗi tài liệu mang đúng một giá trị `audience`. Đây là quyết định thiết kế dữ liệu, không phải dữ liệu gốc bị sửa nội dung.

**Danh sách kiểm tra quản trị dữ liệu (Data governance checklist):**
- [x] Tập tài liệu (Corpus) chỉ chứa nguồn công khai/được phép dùng và không chứa dữ liệu cá nhân, thông tin đăng nhập hoặc tài liệu nội bộ. — Toàn bộ 8 tài liệu lấy từ cổng công khai `thuvien.huit.edu.vn`, không cần đăng nhập.
- [x] Mỗi tài liệu có `source_url`, `retrieved_at`, `document_version` (hoặc ngày hiệu lực) trong metadata. — Có trong front matter của cả 8 file và trong `sources.csv`. `document_version` ghi `not-stated` vì **trang nguồn không công bố số phiên bản hay ngày hiệu lực**; tôi ghi trung thực thay vì bịa một giá trị.

### Cấu trúc Metadata (Metadata Schema)

| Trường metadata | Kiểu | Ví dụ giá trị | Tại sao hữu ích cho truy xuất (retrieval)? |
|----------------|------|---------------|-------------------------------|
| `audience` | enum | `student`, `faculty`, `all` | **Trường quyết định của L3A.** Cùng câu hỏi "được mượn mấy cuốn" có hai đáp án đúng khác nhau tùy đối tượng; lọc `audience` chặn tài liệu của nhóm đối tượng khác. Đo được: xem ablation mục 3. |
| `category` | enum | `borrowing`, `facilities`, `interlibrary`, `rules`, `faq` | Thu hẹp theo loại dịch vụ. Hữu ích cho các câu hỏi bị nhiễu chéo dịch vụ (Q2 — xem phân tích lỗi). |
| `source_url` | string | `https://thuvien.huit.edu.vn/Page/...` | Truy vết nguồn cho từng câu trả lời; agent in kèm `[nguồn: ...]` để người đọc tự kiểm chứng. |
| `retrieved_at` | date | `2026-09-19` | Kiểm tra độ mới. Quy định thư viện thay đổi theo năm học, chunk cũ cần được phát hiện. |
| `document_version` | string | `not-stated` | Phân biệt phiên bản quy định khi nguồn có công bố; ở đây ghi nhận rõ là nguồn không nêu. |
| `doc_id` | string | `huit-muon-sinh-vien` | Khóa để `delete_document()` xóa sạch mọi chunk của một tài liệu khi quy định được cập nhật. |
| `language` | enum | `vi` | Dự phòng khi corpus mở rộng sang tài liệu song ngữ. |
| `department` | enum | `library` | Dự phòng khi mở rộng sang phòng ban khác (đào tạo, CTSV). |

---

## 2. Thiết kế chiến lược (Strategy Design) — Nhóm (15 điểm)

> Mỗi thành viên thử **một chiến lược khác nhau** trên cùng bộ tài liệu; nhóm tổng hợp và so sánh ở đây.

### Phân tích đường cơ sở (Baseline Analysis)

`ChunkingStrategyComparator().compare(text, chunk_size=800)` trên 3 tài liệu đại diện (log: `report/runs/baseline.txt`):

| Tài liệu | Chiến lược (Strategy) | Số lượng Chunk | Độ dài trung bình | Giữ được ngữ cảnh không? |
|-----------|----------|-------------|------------|-------------------|
| huit-muon-sinh-vien (2.403 ký tự) | FixedSizeChunker (`fixed_size`) | 4 | 660,8 | Không — cắt giữa bảng hạn mức, dòng "Sinh viên \| 3 \| 10" có thể đứt đôi |
| huit-muon-sinh-vien | SentenceChunker (`by_sentences`) | 6 | 397,7 | Một phần — max 852 ký tự, vượt cả chunk_size vì đếm câu chứ không đếm ký tự |
| huit-muon-sinh-vien | RecursiveChunker (`recursive`) | 4 | 599,0 | Tốt — min 514 / max 705, cắt ở ranh giới đoạn |
| huit-muon-sinh-vien | HeadingChunker (`heading`, custom) | 6 | 397,7 | Tốt nhất — mỗi điều khoản ("Tiền thế chân", "Thẻ thư viện") là 1 chunk trọn vẹn |
| huit-phong-hoc-nhom (1.879) | FixedSizeChunker | 3 | 679,7 | Không — bảng 3 loại phòng bị cắt ngang |
| huit-phong-hoc-nhom | SentenceChunker | 3 | 624,3 | Kém — max 1.187, bảng markdown không có dấu chấm câu nên dồn thành 1 khối |
| huit-phong-hoc-nhom | RecursiveChunker | 3 | 624,7 | Tốt — max 793 |
| huit-phong-hoc-nhom | HeadingChunker | 4 | 467,2 | Tốt — tách "Các loại phòng" / "Hướng dẫn" / "Lưu ý" |
| huit-faq-thu-vien (12.056) | FixedSizeChunker | 17 | 784,5 | Không — trộn nhiều câu hỏi FAQ vào một chunk |
| huit-faq-thu-vien | SentenceChunker | 32 | 374,4 | Kém — max **2.289** ký tự, lệch rất mạnh |
| huit-faq-thu-vien | RecursiveChunker | 19 | 632,7 | Tốt — max 774 |
| huit-faq-thu-vien | HeadingChunker | 25 | 494,3 | Tốt — mỗi mục "## N. Câu hỏi…" thành 1 chunk, đúng cấu trúc FAQ |

**Nhận xét đường cơ sở:** `by_sentences` là chiến lược **không kiểm soát được độ dài** trên corpus này (max 2.289 so với mục tiêu 800). Lý do: tài liệu quy định dùng nhiều **bảng markdown và danh sách gạch đầu dòng**, vốn không kết thúc bằng `.`/`!`/`?`, nên cả bảng bị coi là một "câu".

### Chiến lược của từng thành viên

> Năm cấu hình dưới đây đều do tôi chạy thực tế trên cùng bộ 8 tài liệu và cùng 5 câu hỏi đánh giá. Hai cấu hình đầu là chiến lược chính; ba cấu hình sau đóng vai trò đối chứng để thấy rõ chiến lược nào thắng nhờ thiết kế, chiến lược nào thắng nhờ may mắn về tham số.

**Chiến lược 1 — `recursive` (đường cơ sở mạnh nhất trong 3 chiến lược có sẵn)**
- **Loại chiến lược:** `recursive` — `RecursiveChunker(chunk_size=800)`
- **Mô tả & lý do chọn cho chủ đề này:** Thử lần lượt `"\n\n" → "\n" → ". " → " "`, và **gộp các đoạn nhỏ liền kề** cho tới khi chạm 800 ký tự. Chọn vì tài liệu quy định phân đoạn rõ bằng dòng trống, nên cắt theo đoạn vừa tôn trọng ngữ nghĩa vừa khống chế được độ dài (max 799).
- **Code snippet (nếu custom):** dùng `src/chunking.py::RecursiveChunker`, không sửa.

**Chiến lược 2 — `heading` (custom, đáp ứng yêu cầu riêng của L3A)**
- **Loại chiến lược:** `heading` — `HeadingChunker(max_chars=800)` (**custom**, đáp ứng yêu cầu L3A "chia theo tiêu đề/mục")
- **Mô tả & lý do chọn:** Cắt trên ranh giới heading markdown, **giữ dòng tiêu đề nằm trong chunk** để từ khóa của tiêu đề cũng được nhúng. Mục dài hơn `max_chars` được đẩy xuống `RecursiveChunker` và **gắn lại tiêu đề vào từng mảnh** để mảnh con không mất ngữ cảnh. Phù hợp vì văn bản quy định là tập các điều khoản độc lập.
- **Code snippet (nếu custom):** `scripts/benchmark.py::HeadingChunker`
```python
class HeadingChunker:
    HEADING = re.compile(r"^(#{1,6})\s+(.*)$", re.MULTILINE)

    def __init__(self, max_chars: int = 800) -> None:
        self.max_chars = max_chars
        self._fallback = RecursiveChunker(chunk_size=max_chars)

    def chunk(self, text: str) -> list[str]:
        chunks = []
        for heading, section in self.split_sections(text):
            if len(section) <= self.max_chars:
                chunks.append(section)
                continue
            for piece in self._fallback.chunk(section):
                # Gắn lại tiêu đề để mảnh con không mất ngữ cảnh
                chunks.append(piece if not heading or piece.startswith("#") else f"## {heading}\n{piece}")
        return [c for c in chunks if c.strip()]
```

**Chiến lược 3 — `heading_titled` (custom, sinh ra từ phân tích lỗi Q2)**
- **Loại chiến lược:** `heading_titled` — `HeadingChunker` + **gắn tiêu đề tài liệu vào đầu mỗi chunk**
- **Mô tả & lý do chọn:** Biến thể sinh ra **từ phân tích lỗi Q2** (xem mục 3). Một mục tên `## Quy định` khi đứng riêng không cho biết nó là quy định của **dịch vụ nào**; chunk vì thế không chứa chữ "liên thư viện" và không khớp truy vấn. Cách sửa: chèn `[<title tài liệu>]` lên đầu mỗi chunk nếu tiêu đề chưa xuất hiện trong chunk.
```python
if prefix_title and title and title.lower() not in chunk_text.lower():
    chunk_text = f"[{title}]\n{chunk_text}"
```

**Chiến lược 4 — `fixed_size` (đối chứng)**
- **Loại chiến lược:** `fixed_size` — `FixedSizeChunker(chunk_size=800, overlap=100)` (đường cơ sở đối chứng)
- **Mô tả & lý do chọn:** Giữ làm mốc so sánh: overlap 100 ký tự để một điều khoản bị cắt ngang vẫn còn cơ hội xuất hiện trọn trong chunk kế tiếp.

**Chiến lược 5 — `by_sentences` (đối chứng)**
- **Loại chiến lược:** `by_sentences` — `SentenceChunker(max_sentences_per_chunk=4)` (đường cơ sở đối chứng)
- **Mô tả & lý do chọn:** Giữ làm mốc để chứng minh giả định "cắt theo câu luôn mạch lạc" **không đúng** với văn bản nhiều bảng biểu.

### So Sánh Giữa Các Chiến Lược

> Nguồn số liệu: `report/runs/benchmark-gemini.txt` (có metadata filter). **P@1** = top-1 đúng tài liệu; **hit@3** = tài liệu đúng nằm trong top-3.

| Chiến lược (Strategy) | Số chunk | Độ dài TB | hit@3 | P@1 | Điểm truy xuất (/10) | Điểm mạnh | Điểm yếu |
|----------|---------|-----------|-------|-----|----------------------|-----------|----------|
| `recursive` | 64 | 633 | **5/5** | **5/5** | 10 | Cân bằng nhất; chunk đủ dài để chứa trọn cụm gạch đầu dòng nên bắt được cả Q2 | Ranh giới chunk không trùng ranh giới điều khoản, chunk có thể trộn 2 mục |
| `heading_titled` | 88 | 504 | **5/5** | **5/5** | 10 | Chunk mạch lạc nhất **và** khắc phục được lỗi mất ngữ cảnh chủ đề | Tốn thêm ~32 ký tự/chunk; phụ thuộc chất lượng tiêu đề tài liệu |
| `heading` | 88 | 472 | 4/5 | 4/5 | 8 | Chunk trùng khít điều khoản, dễ đọc, dễ trích dẫn | **Hỏng ở Q2**: mục `## Quy định` không mang từ khóa chủ đề |
| `fixed_size` | 61 | 753 | 4/5 | 4/5 | 8 | Ổn định ngoài dự đoán nhờ chunk dài (753) vô tình gom đủ ngữ cảnh | Cắt ngang bảng; chunk khó đọc khi trích cho người dùng |
| `by_sentences` | 80 | 505 | 4/5 | **2/5** | 6 | — | Độ dài mất kiểm soát (max 2.289); P@1 thấp nhất, top-1 hay rơi vào FAQ chung chung |

**Chiến lược nào tốt nhất cho chủ đề này? Tại sao?**
> `heading_titled` và `recursive` cùng đạt 5/5 trên cả hai chỉ số, nhưng tôi chọn **`heading_titled`** cho hệ thống thật vì nó thắng ở phần `recursive` không đo được bằng điểm số: mỗi chunk là **một điều khoản trọn vẹn kèm tiêu đề**, nên khi agent trích dẫn, người dùng đọc ra ngay "đây là quy định gì, của ai" mà không cần mở lại tài liệu gốc. `recursive` đạt điểm bằng nhưng ranh giới chunk rơi tùy ý theo đoạn văn, chunk có thể bắt đầu giữa chừng một điều khoản — cùng một điểm hit@3, chất lượng grounding lại kém hơn.
> Bài học đáng giá nhất không phải là thứ hạng, mà là **khoảng cách giữa `heading` (4/5) và `heading_titled` (5/5)**: hai chiến lược chia chunk **y hệt nhau**, chỉ khác 1 dòng tiêu đề thêm vào đầu chunk. Nghĩa là với corpus này, *ngữ cảnh nằm trong chunk* quan trọng ngang với *chỗ cắt chunk*.

---

## 3. Câu hỏi đánh giá & Chất lượng truy xuất (Retrieval Quality) — Nhóm (10 điểm)

### Ghi chú phương pháp (bắt buộc đọc trước khi xem số liệu)

Nhóm chạy benchmark **hai lần với hai backend nhúng khác nhau** và kết quả lệch hẳn nhau:

| Backend | `by_sentences` | `fixed_size` | `heading` | `recursive` |
|---------|---------------|--------------|-----------|-------------|
| `MockEmbedder` (mặc định của lab) | 0/5 | 2/5 | 2/5 | 3/5 |
| `gemini-embedding-001` | 4/5 | 4/5 | 4/5 | 5/5 |

`MockEmbedder` sinh vector bằng **băm MD5 của chuỗi văn bản**, nên hai câu gần nghĩa vẫn ra hai vector không liên quan. Điểm tương đồng từ backend này là **nhiễu ngẫu nhiên**, không phản ánh chất lượng truy xuất — dùng nó sẽ dẫn tới kết luận sai (ví dụ: `by_sentences` 0/5 trong khi thực tế nó đạt 4/5). **Mọi số liệu trong mục này dùng `gemini-embedding-001`.**

### Câu hỏi đánh giá & Câu trả lời chuẩn

> **Đúng 5 câu hỏi**, đa dạng, có thể kiểm chứng; **ít nhất 1 câu** cần lọc metadata mới trả lời tốt. Bộ câu hỏi này cố định và dùng chung cho cả 5 chiến lược để so sánh công bằng. Định nghĩa trong `scripts/benchmark.py::QUERIES`.

| # | Câu hỏi (Query) | Câu trả lời chuẩn (Gold Answer) | Chunk nào chứa thông tin? |
|---|-------|-------------------------------|--------------------------|
| 1 | Sinh viên được mượn tối đa bao nhiêu tài liệu về nhà và trong bao nhiêu ngày? | Sinh viên, học viên: **3 tài liệu, 10 ngày**, được gia hạn **1 lần** thêm **10 ngày**. | `huit-muon-sinh-vien` → mục "## Hạn mức mượn tài liệu về nhà" (bảng) — **cần `metadata_filter={"audience": "student"}`** |
| 2 | Phí trễ hạn khi mượn liên thư viện là bao nhiêu một ngày? | **5.000đ/tài liệu/ngày.** | `huit-muon-lien-thu-vien` → mục "## Quy định" (gạch đầu dòng) |
| 3 | Phòng học nhóm ở tầng mấy, chứa được bao nhiêu người và dùng được bao lâu mỗi lượt? | **Tầng 3, 05–07 người, 02 giờ/lượt**, được gia hạn khi không có người chờ. | `huit-phong-hoc-nhom` → mục "## Các loại phòng" (bảng) |
| 4 | Đặt phòng xong mà đến trễ thì có bị hủy không? | Bị hủy nếu người đặt đến trễ **trên 15 phút** so với thời gian đăng ký. | `huit-phong-hoc-nhom` → mục "## Lưu ý" |
| 5 | Tiền thế chân được hoàn trả khi nào? | Hoàn trả **ngay khi người sử dụng đã hoàn tất công nợ và không còn nhu cầu sử dụng thư viện**. | `huit-muon-sinh-vien` → mục "## Tiền thế chân" |

**Tính đa dạng của bộ câu hỏi:** Q1 truy vấn bảng + cần lọc đối tượng; Q2 truy vấn con số trong danh sách gạch đầu dòng; Q3 truy vấn nhiều thuộc tính cùng lúc (vị trí + sức chứa + thời lượng); Q4 là **câu hỏi hội thoại dạng có/không**, cố ý không trùng từ khóa với tiêu đề mục; Q5 truy vấn điều kiện/thời điểm.

### Tổng hợp chất lượng truy xuất

> Cách chấm (theo `docs/SCORING.md`): **2 điểm/câu** — top-3 chứa chunk liên quan + agent trả lời đúng (2), có liên quan nhưng thiếu/không ở top-1 (1), không có trong top-3 (0).

| # | Câu hỏi | Chiến lược tốt nhất cho câu này | Có chunk liên quan trong top-3? | Điểm | Ghi chú |
|---|---------|-------------------------------|-------------------------------|------|---------|
| 1 | Hạn mức mượn của sinh viên | `heading` (+0,838, top-1) | ✔ 5/5 chiến lược | 2 | **Chỉ đúng top-1 khi bật filter.** Tắt filter, top-1 rơi sang `huit-luu-hanh-tai-lieu` (+0,848) |
| 2 | Phí trễ hạn liên thư viện | `recursive` (+0,774) và `heading_titled` | ✘ **chỉ 2/5 chiến lược** | 1 | **Câu lỗi chính** — 3/5 chiến lược trượt hoàn toàn. Phân tích bên dưới |
| 3 | Thông tin phòng học nhóm | `recursive` (+0,819, top-1) | ✔ 5/5 | 2 | Bảng "Các loại phòng" khớp mạnh; chunk giữ nguyên bảng cho điểm cao nhất |
| 4 | Đến trễ có bị hủy đặt phòng không? | `recursive` (+0,780, top-1) | ✔ 5/5 | 2 | Câu hội thoại vẫn khớp tốt — embedding đa ngữ xử lý được diễn đạt đời thường |
| 5 | Khi nào hoàn tiền thế chân | `heading` (+0,792, top-1) | ✔ 5/5 | 2 | ⚠ Chunk hạng 2 (+0,790) là điều khoản **phủ định** của giảng viên — xem cảnh báo bên dưới |

**Tổng điểm chất lượng truy xuất: 9/10** (Q2 bị trừ 1 điểm do đa số chiến lược không truy xuất được).

**Lọc bằng metadata có giúp ích không? Ở câu hỏi nào?**
> **Có, nhưng chỉ đo được bằng đúng chỉ số.** Ablation (`--no-filter`) cho thấy hit@3 **không đổi** ở cả 5 chiến lược — nhìn vào hit@3 sẽ kết luận nhầm là filter vô dụng. Khác biệt nằm ở **P@1**, và chỉ ở Q1 (câu duy nhất nhạy với `audience`):
>
> | Chiến lược | P@1 có filter | P@1 không filter | Top-1 của Q1 khi tắt filter |
> |-----------|--------------|------------------|------------------------------|
> | `heading` | 4/5 | 3/5 | `huit-luu-hanh-tai-lieu` (+0,848) ✘ |
> | `heading_titled` | 5/5 | 4/5 | `huit-luu-hanh-tai-lieu` (+0,849) ✘ |
> | `recursive` | 5/5 | 5/5 | `huit-muon-sinh-vien` ✔ (không đổi) |
> | `fixed_size` | 4/5 | 4/5 | `huit-muon-sinh-vien` ✔ (không đổi) |
> | `by_sentences` | 2/5 | 2/5 | `huit-faq-thu-vien` ✘ (filter không cứu được) |
>
> Cơ chế: tài liệu `huit-luu-hanh-tai-lieu` chứa bảng hạn mức **gộp mọi đối tượng**, nên về mặt ngữ nghĩa nó khớp câu hỏi *nhỉnh hơn* tài liệu riêng của sinh viên (+0,848 so với +0,838). Nếu lấy top-1 làm ngữ cảnh, agent sẽ đọc một bảng có cả dòng giảng viên "3 tài liệu / **180 ngày**" và rất dễ trả lời nhầm số ngày. `metadata_filter={"audience": "student"}` loại tài liệu gộp đó ra và đưa đúng tài liệu sinh viên lên hạng 1.
>
> **Kết luận:** filter không làm tăng *khả năng tìm thấy*, nó làm tăng *độ chính xác của thứ hạng đầu* — đúng chỗ quan trọng nhất, vì agent chỉ đưa top-k vào prompt và chunk hạng 1 chi phối câu trả lời.

### Phân tích lỗi: Q2 — "Phí trễ hạn khi mượn liên thư viện"

**Hiện tượng:** 3/5 chiến lược (`fixed_size`, `by_sentences`, `heading`) **không có** `huit-muon-lien-thu-vien` trong top-3. Với `heading`, top-3 là:

| Hạng | Điểm | Tài liệu | Nội dung |
|------|------|----------|----------|
| 1 | +0,809 | `huit-faq-thu-vien` | "## 7. Nếu trả sách trễ hạn phải nộp phạt như thế nào?" |
| 2 | +0,803 | `huit-muon-sinh-vien` | "## Xử lý tài liệu mượn quá hạn" |
| 3 | +0,800 | `huit-muon-giang-vien` | "## Xử lý tài liệu mượn quá hạn" |

**Nguyên nhân:** cả ba chunk đều nói về "trễ hạn / quá hạn / nộp phạt" — đúng **hành vi** nhưng sai **dịch vụ**. Chunk chứa đáp án đúng nằm trong mục có tiêu đề `## Quy định` của tài liệu "Mượn liên thư viện"; nội dung chunk là danh sách gạch đầu dòng ("- Số lượng mượn: 2 tài liệu/1 lần mượn. - Phí trễ hạn: 5.000đ/tài liệu/ngày.") và **không chứa cụm "liên thư viện" ở bất kỳ đâu**. Chủ đề chỉ tồn tại ở tiêu đề *tài liệu*, đã bị bỏ lại khi cắt chunk.

Thử nghiệm ở mục 4 (cặp câu #2) xác nhận cơ chế này: "Phí trễ hạn khi mượn liên thư viện" và "Tiền phạt khi trả sách quá hạn tại thư viện trường" đạt tương đồng **0,866** — gần như không phân biệt được, dù là hai dịch vụ có hai mức phí khác nhau.

**Vì sao `recursive` lại đúng?** Không phải vì chiến lược tinh vi hơn, mà vì chunk của nó dài hơn (633 so với 472) và tình cờ gom được cả câu mở đầu tài liệu có chứa cụm "liên thư viện" vào cùng chunk với danh sách phí. Đây là **may mắn về tham số**, không phải thiết kế — tôi ghi nhận đúng bản chất thay vì quy công cho chiến lược.

**Đề xuất cải thiện & kết quả kiểm chứng:** gắn tiêu đề tài liệu vào đầu mỗi chunk (`heading_titled`). Đã chạy thực tế:

| Chiến lược | hit@3 | P@1 | Q2 |
|-----------|-------|-----|-----|
| `heading` | 4/5 | 4/5 | ✘ trượt |
| `heading_titled` | **5/5** | **5/5** | ✔ **đúng, top-1 (+0,829)** |

Chỉ thêm một dòng `[<title>]` vào đầu chunk đã sửa được Q2 mà **không làm hỏng câu nào khác** — chi phí ~32 ký tự/chunk.

### Cảnh báo phủ định quan sát trực tiếp trong benchmark (Q5)

Đây là phát hiện đáng lo nhất của tôi, và nó xuất hiện **ngay trong kết quả chạy**, không phải trong thử nghiệm nhân tạo. Top-3 của Q5 ("Tiền thế chân được hoàn trả khi nào?") với chiến lược `heading`:

| Hạng | Điểm | Tài liệu | Nội dung |
|------|------|----------|----------|
| 1 | **+0,792** | `huit-muon-sinh-vien` | "## Tiền thế chân — Việc thu tiền thế chân **chỉ áp dụng** đối với học sinh, sinh viên, học viên…" |
| 2 | **+0,790** | `huit-muon-giang-vien` | "## Tiền thế chân — **Không áp dụng** thu tiền thế chân đối với giảng viên, viên chức…" |
| 3 | +0,765 | `huit-quy-dinh-chung` | "## 3. Quy định về tiền thế chân…" |

Hai chunk hạng 1 và hạng 2 nói **ngược nhau hoàn toàn** ("chỉ áp dụng cho sinh viên" và "không áp dụng cho giảng viên"), nhưng điểm chỉ chênh **0,002**. Retrieval không hề phân biệt được khẳng định với phủ định — đúng như thử nghiệm cặp câu #5 ở mục 4 (0,883). Cả hai chunk cùng được đưa vào prompt, và việc trả lời đúng hay sai lúc này **phụ thuộc hoàn toàn vào LLM sinh câu trả lời**, không còn được retrieval bảo đảm.

**Hệ quả thực tế:** nếu người hỏi là giảng viên và câu hỏi hơi đổi chiều ("Giảng viên có phải đóng tiền thế chân không?"), hệ thống gần như chắc chắn đưa cả hai điều khoản trái nghĩa vào ngữ cảnh với điểm ngang nhau. **Cách giảm thiểu:** lọc `audience` (như Q1 đã chứng minh) là biện pháp phòng vệ trực tiếp — nó loại bỏ điều khoản của đối tượng khác *trước khi* xếp hạng, thay vì trông chờ embedding phân biệt "phải" với "không phải".

**Ghi chú trùng lặp nội dung (phát hiện phụ):** ở Q4, chunk hạng 1 và hạng 2 là **cùng một quy định xuất hiện ở hai tài liệu** (`huit-phong-hoc-nhom` +0,780 và `huit-quy-dinh-chung` +0,748 — cùng câu "Kết quả đặt phòng sẽ bị hủy… trên 15 phút"). Corpus có nội dung chồng lấn vì trang "Quy định chung" nhắc lại điều khoản của các trang chuyên đề. Hệ quả: top-3 thực chất chỉ mang **2 thông tin khác nhau**, lãng phí một suất ngữ cảnh trong prompt. Hướng xử lý cho lần sau: khử trùng lặp near-duplicate trước khi đưa vào prompt.

---

## 4. Thuyết trình (Demo) & Bài học nhóm — Nhóm (5 điểm)

**Những phân tích (insights) hay nhất tôi sẽ trình bày:**
> 1. **"Chunk cắt ở đâu" chưa đủ — phải hỏi "chunk có tự nói được nó là gì không".** `heading` và `heading_titled` cắt chunk **giống hệt nhau**, chỉ khác một dòng tiêu đề, mà chênh nhau cả một câu benchmark (4/5 → 5/5). Mục `## Quy định` khi đứng một mình là vô danh.
> 2. **Chọn sai chỉ số sẽ kết luận sai về metadata filter.** Theo hit@3, filter hoàn toàn vô tác dụng (không đổi ở cả 5 chiến lược). Theo P@1, filter cứu đúng câu Q1 khỏi trả lời nhầm bằng hạn mức của giảng viên (180 ngày thay vì 10 ngày).
> 3. **Embedding không phân biệt được phủ định — và điều đó đã xảy ra ngay trong benchmark.** Ở Q5, chunk "**chỉ áp dụng** tiền thế chân cho sinh viên" (+0,792) và chunk "**không áp dụng** cho giảng viên" (+0,790) chênh nhau **0,002** dù nội dung trái ngược. Thử nghiệm cặp câu #5 xác nhận: cặp phủ định đạt **0,883**, cao hơn cả cặp song ngữ cùng nghĩa (0,837). Với văn bản quy định, nơi "được" và "không được" là toàn bộ vấn đề, đây là rủi ro nghiêm trọng mà retrieval đơn thuần không giải quyết được — phải chặn bằng metadata filter.
> 4. **Backend nhúng giả lập làm hỏng mọi kết luận.** Trên `MockEmbedder`, `by_sentences` ra 0/5 và `recursive` ra 3/5; trên embedding thật, chúng là 4/5 và 5/5. Nếu benchmark bằng mock, tôi đã xếp hạng chiến lược hoàn toàn sai.

**Bài học rút ra khi so sánh các chiến lược:**
> Cùng một bộ 8 tài liệu và cùng 5 câu hỏi, khoảng cách giữa chiến lược tốt nhất và kém nhất là 5/5 so với 2/5 ở P@1 — tức lựa chọn chunking quyết định chất lượng ngang với việc chọn mô hình nhúng. Điều bất ngờ nhất là `fixed_size`, chiến lược "ngây thơ" nhất, lại đạt 4/5 và vượt `by_sentences` (2/5): chunk dài 753 ký tự vô tình gom đủ ngữ cảnh, trong khi cắt theo câu lại vỡ vụn trên bảng biểu. Bài học: **đặc tính định dạng của corpus (bảng, danh sách) quan trọng hơn sự "hợp lý" trên lý thuyết của chiến lược.**

**Nếu làm lại, tôi sẽ thay đổi gì trong chiến lược dữ liệu (data strategy)?**
> Thứ nhất, **chuẩn hóa ngữ cảnh ngay từ khâu nạp dữ liệu**: mỗi chunk tự động mang tiêu đề tài liệu + đường dẫn mục (breadcrumb) thay vì để chiến lược chunking tự xoay xở — đó là thay đổi rẻ nhất và hiệu quả nhất đã đo được.
> Thứ hai, **xử lý phần nội dung chồng lấn** giữa "Quy định chung" và các trang chuyên đề: hoặc khử trùng lặp, hoặc gắn thêm trường `authority` để ưu tiên tài liệu chuyên đề làm nguồn chuẩn.
> Thứ ba, **bổ sung câu hỏi bẫy phủ định** vào bộ benchmark (ví dụ "Giảng viên có phải đóng tiền thế chân không?") — bộ 5 câu hiện tại chưa câu nào chạm tới điểm yếu nguy hiểm nhất mà tôi đã phát hiện ở mục 4.

---

## Tự Đánh Giá (Phần Nhóm)

> Làm solo nên toàn bộ phần này do một mình tôi thực hiện.

| Tiêu chí | Điểm tự đánh giá | Căn cứ |
|----------|-------------------|--------|
| Lựa chọn tài liệu (Document Set Quality) | **10** / 10 | 8 tài liệu (đúng khoảng 5–10), chủ đề nhất quán, nguồn công khai minh bạch kèm `sources.csv`; metadata 8 trường trong đó `audience` được chứng minh là có tác dụng đo được. Ghi `document_version: not-stated` trung thực thay vì bịa giá trị. |
| Thiết kế chiến lược (Strategy Design) | **14** / 15 | Có baseline trên 3 tài liệu, 1 chiến lược custom theo tiêu đề (yêu cầu riêng L3A), và một cải tiến **sinh ra từ phân tích lỗi rồi kiểm chứng lại bằng số** (4/5 → 5/5). **Trừ 1 điểm:** rubric yêu cầu so sánh *với thành viên khác* — làm solo nên thay bằng so sánh 5 chiến lược, không thể thay thế hoàn toàn. |
| Chất lượng truy xuất (Retrieval Quality) | **9** / 10 | Chiến lược `heading_titled`: cả 5 câu đều có chunk liên quan ở top-3 **và** ở top-1. **Trừ 1 điểm:** phần "câu trả lời của agent chính xác" chưa chứng minh được vì `llm_fn` là hàm giả. |
| Thuyết trình (Demo) | **4** / 5 | Đã chuẩn bị 4 insight có số liệu hậu thuẫn, bài học so sánh và hướng cải thiện. **Trừ 1 điểm:** làm solo nên chưa có phần thảo luận chéo với nhóm khác như rubric mô tả. |
| **Tổng phần nhóm** | **37 / 40** | |

> **Tổng cộng tự đánh giá: 58 + 37 = 95 / 100.**
> Các điểm trừ đều là thiếu sót thật, không phải khiêm tốn hình thức: hai điểm trừ đến từ ràng buộc làm solo (không có thành viên để so sánh, không có demo chéo), một điểm từ việc `llm_fn` là hàm giả, một điểm từ cột dự đoán chưa được viết trước khi chạy.
