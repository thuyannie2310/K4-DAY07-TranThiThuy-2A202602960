from typing import Callable

from .store import EmbeddingStore

PROMPT_TEMPLATE = """Bạn là trợ lý tra cứu quy định/dịch vụ đại học.
Chỉ trả lời dựa trên NGỮ CẢNH bên dưới. Nếu ngữ cảnh không đủ thông tin,
hãy nói rõ là không tìm thấy trong tài liệu thay vì suy đoán.

NGỮ CẢNH:
{context}

CÂU HỎI: {question}

TRẢ LỜI (kèm nguồn trích dẫn [nguồn: ...]):"""

NO_CONTEXT_MESSAGE = "Không tìm thấy tài liệu liên quan trong cơ sở tri thức để trả lời câu hỏi này."


class KnowledgeBaseAgent:
    """
    An agent that answers questions using a vector knowledge base.

    Retrieval-augmented generation (RAG) pattern:
        1. Retrieve top-k relevant chunks from the store.
        2. Build a prompt with the chunks as context.
        3. Call the LLM to generate an answer.
    """

    def __init__(self, store: EmbeddingStore, llm_fn: Callable[[str], str]) -> None:
        self.store = store
        self.llm_fn = llm_fn

    def _format_context(self, results: list[dict]) -> str:
        """Number each chunk and label its source so answers can be traced back."""
        blocks = []
        for index, result in enumerate(results, start=1):
            metadata = result.get("metadata") or {}
            source = metadata.get("source_url") or metadata.get("source") or metadata.get("doc_id", "unknown")
            blocks.append(
                f"[{index}] (nguồn: {source} | score={result.get('score', 0.0):.3f})\n{result.get('content', '')}"
            )
        return "\n\n".join(blocks)

    def answer(self, question: str, top_k: int = 3) -> str:
        results = self.store.search(question, top_k=top_k)
        if not results:
            return NO_CONTEXT_MESSAGE

        prompt = PROMPT_TEMPLATE.format(context=self._format_context(results), question=question)
        return self.llm_fn(prompt)
