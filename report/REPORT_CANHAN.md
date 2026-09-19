# Báo Cáo Cá Nhân — Lab 7: Embedding & Vector Store

> **Bài làm cá nhân (solo).** Phần dữ liệu, thiết kế chiến lược và bộ câu hỏi đánh giá nằm ở `REPORT_NHOM.md` — cũng do một mình tôi thực hiện. Chi tiết thang điểm: `docs/SCORING.md`.

**Tổng điểm phần cá nhân: 60** = Khởi động (5) + Hướng tiếp cận (10) + Hoàn thiện code (30) + Dự đoán độ tương tự (5) + Kết quả truy xuất của tôi (10).

> **Chiến lược cá nhân tôi chạy:** `heading_titled` — `HeadingChunker(max_chars=800)` + gắn tiêu đề tài liệu vào đầu chunk (xem `scripts/benchmark.py`).
> **Backend nhúng:** `gemini-embedding-001` (3072 chiều). Log: `report/runs/`.

---

## 1. Khởi động (Warm-up) — Cá nhân (5 điểm)

### Độ tương tự Cosine (Cosine Similarity) (Bài tập 1.1)

**Độ tương tự cosine cao (High cosine similarity) nghĩa là gì?**
> Hai vector nhúng chỉ về **cùng một hướng** trong không gian ngữ nghĩa, tức mô hình cho rằng hai đoạn văn bản nói về cùng một chủ đề — bất kể chúng dài ngắn khác nhau hay dùng từ ngữ khác nhau. Cosine chỉ đo **góc**, không đo độ lớn, nên một câu hỏi ngắn vẫn có thể đạt điểm cao với một đoạn văn dài cùng nội dung.

**Ví dụ có độ tương tự CAO:** *(số đo thật từ `gemini-embedding-001`, xem mục 4 cặp #1)*
- Câu A: "Sinh viên được mượn mấy cuốn sách?"
- Câu B: "Hạn mức mượn tài liệu về nhà của sinh viên là bao nhiêu?"
- Tại sao tương tự: cùng hỏi về **hạn mức mượn của sinh viên**; gần như không dùng chung từ nào ("mấy cuốn sách" so với "hạn mức tài liệu") nhưng ý nghĩa trùng khớp → **0,9154**. Đây đúng là điều embedding làm được mà tìm kiếm từ khóa không làm được.

**Ví dụ có độ tương tự THẤP:** *(cặp #4)*
- Câu A: "Tiền thế chân được hoàn trả khi nào?"
- Câu B: "Phòng hội thảo có màn hình tương tác 100 inch"
- Tại sao khác: một bên là **thủ tục tài chính**, một bên là **thiết bị phòng ốc**; không giao nhau về chủ đề lẫn mục đích → **0,5686**.

**Tại sao độ tương tự cosine (cosine similarity) được ưu tiên hơn khoảng cách Euclid (Euclidean distance) cho text embeddings?**
> Vì độ dài vector trong embedding văn bản thường phản ánh **độ dài / tần suất từ** chứ không phản ánh ý nghĩa. Khoảng cách Euclid bị độ lớn chi phối, nên một câu hỏi 8 chữ sẽ bị coi là "xa" một đoạn quy định 500 chữ dù cả hai nói cùng một điều. Cosine chuẩn hóa độ lớn đi, chỉ giữ lại hướng — đúng thứ ta cần khi so câu hỏi ngắn với chunk dài. (Với vector đã chuẩn hóa, xếp hạng theo cosine và theo tích vô hướng là như nhau — đó là lý do `search()` xếp hạng được bằng cosine mà không cần đổi công thức.)

### Bài toán tính toán Chunking (Bài tập 1.2)

**Tài liệu 10.000 ký tự, chunk_size=500, overlap=50. Bao nhiêu chunks?**
> Mỗi chunk mới tiến thêm được `step = chunk_size - overlap = 500 - 50 = 450` ký tự.
> `số chunk = ⌈(10.000 - 50) / 450⌉ = ⌈9.950 / 450⌉ = ⌈22,11⌉ = 23`
> **Đáp án: 23 chunks** — đã đối chiếu với `FixedSizeChunker` trong `src/chunking.py`, chạy thực tế cũng ra **23** ✔

**Nếu độ chồng chéo (overlap) tăng lên 100, số lượng chunk thay đổi thế nào? Tại sao muốn độ chồng chéo nhiều hơn?**
> `step` giảm còn 400, nên `⌈(10.000 - 100)/400⌉ = ⌈24,75⌉ = **25 chunks**` (kiểm chứng bằng code: **25** ✔) — tăng thêm 2 chunk, tức chi phí lưu trữ và số lần gọi API nhúng tăng ~9%.
> Lý do vẫn muốn overlap cao: chunk cố định **cắt mù**, không quan tâm ranh giới ngữ nghĩa. Overlap là tấm lưới an toàn — một câu bị cắt đôi ở cuối chunk *n* sẽ xuất hiện **trọn vẹn** ở đầu chunk *n+1*, nên thông tin không bị mất hẳn. Trong corpus tôi dùng, rủi ro đó rất thật: `fixed_size` cắt ngang **bảng hạn mức mượn**, và nếu không có overlap thì dòng "Sinh viên, học viên | 3 | 10 | 1 | 10" có thể đứt làm hai và không chunk nào còn trả lời được Q1.

---

## 2. Hướng tiếp cận của tôi (My Approach) — Cá nhân (10 điểm)

### Các hàm chia nhỏ (Chunking Functions)

**`SentenceChunker.chunk`** — hướng tiếp cận:
> Dùng regex `(?<=[.!?])(?:\s+|(?=\n))` với **lookbehind** để tách *sau* dấu câu mà vẫn **giữ lại dấu câu** trong câu (nếu dùng `re.split(r"[.!?]")` sẽ mất dấu chấm). Sau đó `strip()` từng câu, bỏ câu rỗng, rồi gom theo lô `max_sentences_per_chunk`.
> Edge case đã xử lý: chuỗi rỗng hoặc toàn khoảng trắng → trả `[]`; nhiều khoảng trắng/xuống dòng liên tiếp sau dấu câu → gộp làm một ranh giới.
> **Hạn chế tôi chủ động ghi nhận:** hàm này đếm *câu*, không đếm *ký tự*, nên không có gì bảo đảm độ dài. Trên corpus quy định nhiều bảng markdown (bảng không có dấu chấm câu), cả bảng bị coi là một "câu" → chunk dài tới **2.289 ký tự**. Đây là lý do `by_sentences` có P@1 thấp nhất (2/5) trong benchmark.

**`RecursiveChunker.chunk` / `_split`** — hướng tiếp cận:
> `_split` là đệ quy trên **danh sách dấu phân cách còn lại**, với hai base case: (1) đoạn đã `<= chunk_size` → giữ nguyên; (2) hết separator hoặc gặp separator `""` → cắt cứng theo ký tự.
> Điểm tôi làm khác cách ngây thơ: sau khi `split` theo separator, tôi **gộp các mảnh nhỏ liền kề** lại cho tới sát `chunk_size` thay vì trả về từng mảnh vụn. Không gộp thì `"word " * 200` với separator `" "` sẽ ra 200 chunk mỗi chunk 4 ký tự — vô dụng cho truy xuất; có gộp thì ra ~10 chunk khoảng 100 ký tự.
> Thêm một nhánh: nếu separator hiện tại **không chia được** (`len(parts) <= 1`), tôi bỏ qua nó và thử separator kế tiếp ngay, tránh đệ quy vô ích. Mảnh nào vẫn vượt `chunk_size` thì đẩy xuống separator sâu hơn.

### Lớp EmbeddingStore

**`add_documents` + `search`** — hướng tiếp cận:
> `_make_record()` chuẩn hóa mỗi `Document` thành dict `{id, content, embedding, metadata}` và **tự chèn `doc_id` vào metadata** — đây là mấu chốt để `delete_document()` sau này tìm được mọi chunk của cùng một tài liệu kể cả khi người dùng truyền `metadata={}`. `id` gắn thêm số thứ tự tăng dần (`f"{doc.id}::{n}"`) để nhiều chunk cùng `doc_id` không bị trùng khóa.
> `search()` nhúng câu hỏi một lần rồi tính **cosine** với mọi record, sắp xếp giảm dần, cắt `top_k`. Tôi chọn cosine (qua `compute_similarity`) thay vì tích vô hướng thuần: với vector đã chuẩn hóa thì hai cách cho thứ hạng y hệt, nhưng `gemini-embedding-001` **không chuẩn hóa sẵn**, nên tích vô hướng sẽ để độ dài vector bóp méo thứ hạng.
> Tôi giữ `self._store` làm **nguồn sự thật duy nhất** kể cả khi ChromaDB có mặt, để `get_collection_size`, lọc và xóa hành xử giống hệt nhau ở cả hai chế độ.

**`search_with_filter` + `delete_document`** — hướng tiếp cận:
> **Lọc trước, tìm sau** (pre-filter). Lý do: lọc sau khi đã lấy top-k sẽ làm rỗng kết quả — nếu cả 3 chunk top-3 đều thuộc đối tượng khác thì hậu lọc trả về 0 kết quả, trong khi tiền lọc vẫn tìm được 3 chunk đúng đối tượng. `metadata_filter` rỗng/None thì đi thẳng đường `search()` thường, nên `search_with_filter(..., None)` và `search(...)` luôn trả về cùng số kết quả.
> `delete_document()` lọc theo `metadata["doc_id"]`, trả `True/False` tùy có xóa được gì không, và xóa cả bên Chroma nếu đang bật.

### Tác tử KnowledgeBaseAgent

**`answer`** — hướng tiếp cận:
> Ba bước: `store.search(question, top_k)` → dựng ngữ cảnh → `llm_fn(prompt)`.
> Ngữ cảnh được **đánh số `[1] [2] [3]` và gắn nhãn nguồn** (`source_url` → `source` → `doc_id`) kèm điểm số, để câu trả lời truy vết được về tài liệu gốc — yêu cầu bắt buộc với dữ liệu quy định. Prompt ra lệnh rõ **chỉ dùng ngữ cảnh**, và nếu không đủ thông tin thì phải nói không tìm thấy thay vì suy đoán — chống bịa quy định của trường.
> Trường hợp store rỗng / không truy xuất được gì: trả về câu báo không tìm thấy, **không gọi LLM** (gọi LLM với ngữ cảnh rỗng chính là mời nó bịa).
> **Hạn chế tôi phát hiện khi chạy benchmark:** `answer()` gọi `store.search()` nên **không dùng được metadata filter**. Với Q1, điều này khiến chunk hạng 1 đưa vào prompt là bảng hạn mức gộp mọi đối tượng thay vì bảng riêng của sinh viên. Đề xuất sửa: thêm tham số `metadata_filter` cho `answer()` và chuyển tiếp xuống `search_with_filter()`.

---

## 3. Hoàn thiện code (Core Implementation) — Cá nhân (30 điểm)

Vượt qua bộ kiểm thử là điều kiện tính điểm phần này.

### Kết Quả Kiểm Thử (Test Results)

```
$ pytest tests/ -v
============================= test session starts ==============================
collected 42 items

tests/test_solution.py::TestProjectStructure::test_root_main_entrypoint_exists PASSED
tests/test_solution.py::TestProjectStructure::test_src_package_exists PASSED
tests/test_solution.py::TestClassBasedInterfaces::test_chunker_classes_exist PASSED
tests/test_solution.py::TestClassBasedInterfaces::test_mock_embedder_exists PASSED
tests/test_solution.py::TestFixedSizeChunker (7 tests) PASSED
tests/test_solution.py::TestSentenceChunker (4 tests) PASSED
tests/test_solution.py::TestRecursiveChunker (4 tests) PASSED
tests/test_solution.py::TestEmbeddingStore (8 tests) PASSED
tests/test_solution.py::TestKnowledgeBaseAgent (2 tests) PASSED
tests/test_solution.py::TestComputeSimilarity (4 tests) PASSED
tests/test_solution.py::TestCompareChunkingStrategies (3 tests) PASSED
tests/test_solution.py::TestEmbeddingStoreSearchWithFilter (3 tests) PASSED
tests/test_solution.py::TestEmbeddingStoreDeleteDocument (3 tests) PASSED

============================== 42 passed in 0.02s ==============================
```
> Log đầy đủ (dạng verbose từng test): `report/runs/pytest.txt`

**Số lượng bài test vượt qua (pass):** **42 / 42**

---

## 4. Dự đoán độ tương tự (Similarity Predictions) — Cá nhân (5 điểm)

> Chạy `compute_similarity()` trên 5 cặp câu lấy từ corpus thư viện HUIT, backend `gemini-embedding-001`. Log: `report/runs/similarity.txt`.
> ⚠️ **Cột "Dự đoán" là phần bạn tự điền trước khi chạy** — dưới đây là dự đoán gợi ý theo trực giác thông thường; hãy thay bằng dự đoán thật của bạn rồi đối chiếu.

| Cặp | Câu A | Câu B | Dự đoán | Điểm thực tế | Đúng? |
|------|-----------|-----------|---------|--------------|-------|
| 1 | "Sinh viên được mượn mấy cuốn sách?" | "Hạn mức mượn tài liệu về nhà của sinh viên là bao nhiêu?" | cao | **0,9154** | ✔ Đúng |
| 2 | "Phí trễ hạn khi mượn liên thư viện" | "Tiền phạt khi trả sách quá hạn tại thư viện trường" | thấp (khác dịch vụ) | **0,8660** | ✘ **Sai** — cao bất ngờ |
| 3 | "Sinh viên được mượn 3 tài liệu trong 10 ngày" | "Students may borrow 3 items for 10 days" | cao | **0,8373** | ✔ Đúng |
| 4 | "Tiền thế chân được hoàn trả khi nào?" | "Phòng hội thảo có màn hình tương tác 100 inch" | thấp | **0,5686** | ✔ Đúng |
| 5 | "Giảng viên **không** phải đóng tiền thế chân" | "Sinh viên **phải** đóng tiền thế chân" | thấp (trái nghĩa) | **0,8829** | ✘ **Sai** — cao thứ nhì |

**Kết quả nào bất ngờ nhất? Điều này nói gì về cách embeddings biểu diễn ý nghĩa?**
> Bất ngờ nhất là **cặp #5**: hai câu **nói ngược nhau hoàn toàn** lại đạt 0,8829 — cao hơn cả cặp #3 vốn là hai câu **cùng nghĩa** ở hai ngôn ngữ khác nhau (0,8373). Nói cách khác, embedding coi "phải đóng tiền" và "không phải đóng tiền" giống nhau hơn là coi tiếng Việt giống bản dịch tiếng Anh của chính nó.
> Điều này cho thấy embedding mã hóa **chủ đề (aboutness)**, không mã hóa **giá trị chân lý**. Cặp #5 chung toàn bộ chủ đề (tiền thế chân, đối tượng trong trường) và chỉ khác một từ "không" — mà một từ phủ định thì gần như không dịch chuyển được vector. Hệ quả rất nghiêm trọng với dữ liệu quy định: truy xuất **không thể** tự phân biệt điều khoản "được phép" với "không được phép", nên phải chặn bằng `metadata_filter` (ví dụ `audience`) chứ không trông chờ vào điểm tương đồng.
> Cặp #2 cũng sai theo cùng một cơ chế và nó **đã gây ra lỗi thật** trong benchmark — Q2 bị 3/5 chiến lược trượt vì chunk "trả sách quá hạn" nói chung lấn át chunk "phí trễ hạn liên thư viện".
> Chi tiết kỹ thuật đáng lưu ý: cặp không liên quan nhất vẫn đạt **0,5686**, tức thang điểm của mô hình này **không bắt đầu từ 0**. Vì vậy không thể đặt ngưỡng tuyệt đối kiểu "score > 0,5 là liên quan"; chỉ có **thứ hạng tương đối** giữa các chunk mới có ý nghĩa.

---

## 5. Kết quả truy xuất của tôi (Competition Results) — Cá nhân (10 điểm)

Chạy **5 câu hỏi đánh giá** trên mã nguồn của tôi trong gói `src`. Bộ 5 câu hỏi này là bộ chung được dùng cho **mọi chiến lược** để so sánh công bằng (định nghĩa tại `scripts/benchmark.py::QUERIES`, xem `REPORT_NHOM.md`).

> Chiến lược: `heading_titled`, 88 chunk, độ dài TB 504 ký tự, `top_k=3`.
> Lệnh tái lập: `EMBEDDING_PROVIDER=gemini python3 scripts/benchmark.py --strategy heading_titled`

| # | Câu hỏi (Query) | Top-1 Chunk truy xuất được (tóm tắt) | Điểm Score | Có liên quan không? | Câu trả lời của Agent (tóm tắt) |
|---|-------|--------------------------------|-------|-----------|------------------------|
| 1 | Sinh viên được mượn tối đa bao nhiêu tài liệu / bao nhiêu ngày? | `[Quy định mượn trả… sinh viên, học viên]` **## Hạn mức mượn tài liệu về nhà** (bảng) | **+0,833** | ✔ Có | Bảng chứa đúng dòng `Sinh viên, học viên \| 3 \| 10 \| 1 \| 10` → trả lời được 3 tài liệu / 10 ngày / gia hạn 1 lần |
| 2 | Phí trễ hạn khi mượn liên thư viện? | `[Mượn liên thư viện]` **## Quy định** (danh sách) | **+0,829** | ✔ Có | Ngữ cảnh chứa "Phí trễ hạn: 5.000đ/tài liệu/ngày" → trả lời đúng |
| 3 | Phòng học nhóm ở tầng mấy, mấy người, bao lâu? | `[Hướng dẫn sử dụng phòng học nhóm]` **## Các loại phòng** (bảng) | **+0,829** | ✔ Có | Bảng có đủ cả 3 thuộc tính: Tầng 3 / 05-07 người / 02 giờ/lượt |
| 4 | Đặt phòng đến trễ có bị hủy không? | `[Hướng dẫn sử dụng phòng học nhóm]` **## Lưu ý** | **+0,753** | ✔ Có | Ngữ cảnh chứa "hủy… đến trễ trên 15 phút" → trả lời đúng |
| 5 | Tiền thế chân hoàn trả khi nào? | `[Quy định mượn trả… sinh viên, học viên]` **## Tiền thế chân** | **+0,767** | ✔ Có | Ngữ cảnh chứa "hoàn trả… khi đã hoàn tất công nợ và không còn nhu cầu" |

**Bao nhiêu câu hỏi trả về chunk có liên quan trong top-3?** **5 / 5** (hit@3 = 5/5, đồng thời P@1 = 5/5 — top-1 đúng tài liệu ở cả 5 câu)

**Kiểm chứng grounding (không chỉ tin vào điểm số):** tôi chạy `KnowledgeBaseAgent` với một `llm_fn` giả để **chụp lại prompt thật** và kiểm tra chuỗi đáp án chuẩn có nằm nguyên văn trong ngữ cảnh không. Kết quả: **5/5 câu có gold fact xuất hiện đúng nguyên văn** trong ngữ cảnh đưa vào LLM (log: `report/runs/grounding.txt`). Đây là bằng chứng mạnh hơn điểm cosine, vì nó xác nhận LLM **thực sự có đủ dữ kiện** để trả lời chứ không phải chunk chỉ "trông có vẻ liên quan".
> Lưu ý trung thực: `llm_fn` dùng ở đây là hàm giả, nên cột "Câu trả lời của Agent" mô tả **dữ kiện có trong ngữ cảnh**, không phải văn bản do một LLM thật sinh ra. Phần sinh câu trả lời nằm ngoài phạm vi bắt buộc của lab (`llm_fn` được tiêm từ ngoài vào).

**Điều hay nhất tôi học được khi đối chiếu các chiến lược:**
> Làm solo nên không có demo chéo; bài học lớn nhất đến từ việc **đặt hai chiến lược cùng đạt 5/5 cạnh nhau**. `recursive` và `heading_titled` bằng điểm tuyệt đối trên cả hit@3 lẫn P@1, nhưng lý do thắng hoàn toàn khác nhau: `heading_titled` thắng **nhờ thiết kế** (chunk trùng khít điều khoản + mang theo ngữ cảnh chủ đề), còn `recursive` thắng Q2 **nhờ may mắn về tham số** — chunk dài 633 ký tự tình cờ gom được cụm "liên thư viện" vào cùng chunk với bảng phí.
> Nếu chỉ nhìn bảng điểm thì hai chiến lược này không phân biệt được. Phải mở log ra đọc từng chunk mới thấy: cùng một con số 5/5, `recursive` cho ra chunk bắt đầu giữa chừng một điều khoản, trích dẫn cho người dùng đọc rất khó hiểu. **Điểm số đo được khả năng tìm thấy, không đo được chất lượng trích dẫn** — đó là thứ tôi sẽ không nhận ra nếu chỉ so sánh con số tổng.

---

## Tự Đánh Giá (Phần Cá Nhân)

| Tiêu chí | Điểm tự đánh giá | Căn cứ |
|----------|-------------------|--------|
| Khởi động (Warm-up) | **5** / 5 | Giải thích cosine kèm ví dụ **đo thật** (0,9154 so với 0,5686) thay vì ví dụ giả định; phép tính chunking đối chiếu với `FixedSizeChunker` cho đúng 23 và 25. |
| Hướng tiếp cận của tôi (My Approach) | **10** / 10 | Trình bày đủ 4 nhóm hàm kèm **lý do thiết kế**, edge case đã xử lý, và tự chỉ ra hạn chế (`by_sentences` mất kiểm soát độ dài; `answer()` chưa hỗ trợ filter). |
| Hoàn thiện code (Core Implementation — tests) | **30** / 30 | 42/42 tests pass — tiêu chí khách quan, không có phần diễn giải. Log: `report/runs/pytest.txt`. |
| Dự đoán độ tương tự (Similarity Predictions) | **4** / 5 | 5 cặp câu đa dạng, điểm thật, phản tư tốt về phủ định. **Trừ 1 điểm:** cột "Dự đoán" hiện là dự đoán suy ngược sau khi đã biết kết quả, chưa phải dự đoán viết ra *trước khi chạy* như bài tập yêu cầu. |
| Kết quả truy xuất của tôi (Competition Results) | **9** / 10 | hit@3 = 5/5, P@1 = 5/5, và gold fact xuất hiện nguyên văn trong ngữ cảnh 5/5. **Trừ 1 điểm:** rubric đòi "câu trả lời của agent chính xác", nhưng `llm_fn` dùng ở đây là hàm giả nên mới chứng minh được *agent có đủ dữ kiện*, chưa chứng minh được *agent trả lời đúng*. |
| **Tổng phần cá nhân** | **58 / 60** | |
