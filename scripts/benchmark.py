"""
Phase 2 benchmark harness — K4-L3A (HUIT library services corpus).

Ingests data/thu-vien/*.md (metadata from YAML front matter), chunks the corpus
with several strategies, and runs the 5 agreed benchmark queries against each.

Usage:
    python3 scripts/benchmark.py                 # all strategies
    python3 scripts/benchmark.py --strategy heading
    EMBEDDING_PROVIDER=local python3 scripts/benchmark.py

Embedding backend follows EMBEDDING_PROVIDER (mock | local | openai | gemini),
same rule as main.py. NOTE: the default `mock` backend is a hash of the text and
carries no semantic meaning — scores from it say nothing about retrieval quality.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv

from src import (
    EMBEDDING_PROVIDER_ENV,
    GEMINI_EMBEDDING_MODEL,
    LOCAL_EMBEDDING_MODEL,
    OPENAI_EMBEDDING_MODEL,
    Document,
    EmbeddingStore,
    FixedSizeChunker,
    GeminiEmbedder,
    LocalEmbedder,
    OpenAIEmbedder,
    RecursiveChunker,
    SentenceChunker,
    _mock_embed,
)

CORPUS_DIR = Path(__file__).resolve().parent.parent / "data" / "thu-vien"
CACHE_DIR = Path(__file__).resolve().parent.parent / ".cache"
FRONT_MATTER = re.compile(r"\A---\s*\n(.*?)\n---\s*\n", re.DOTALL)


# --------------------------------------------------------------------------
# Custom strategy (K4-L3A requirement: chunk by heading/section)
# --------------------------------------------------------------------------
class HeadingChunker:
    """Split markdown on headings, keeping each section's heading with its body.

    Design rationale: the HUIT corpus is regulation text where each `##` section
    is one self-contained rule ("Hạn mức mượn", "Tiền thế chân", "Phí sử dụng
    dịch vụ"). Cutting on headings keeps a rule whole and carries its title into
    the chunk, so the heading's keywords are themselves searchable. Sections
    longer than max_chars fall back to RecursiveChunker, with the heading
    re-prefixed onto every piece so no fragment loses its context.
    """

    HEADING = re.compile(r"^(#{1,6})\s+(.*)$", re.MULTILINE)

    def __init__(self, max_chars: int = 800) -> None:
        self.max_chars = max_chars
        self._fallback = RecursiveChunker(chunk_size=max_chars)

    def split_sections(self, text: str) -> list[tuple[str, str]]:
        """Return [(heading, section_text), ...] in document order."""
        matches = list(self.HEADING.finditer(text))
        if not matches:
            return [("", text.strip())] if text.strip() else []

        sections: list[tuple[str, str]] = []
        preamble = text[: matches[0].start()].strip()
        if preamble:
            sections.append(("", preamble))

        for index, match in enumerate(matches):
            end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
            heading = match.group(2).strip()
            body = text[match.end() : end].strip()
            if heading or body:
                sections.append((heading, f"{match.group(0).strip()}\n{body}".strip()))
        return sections

    def chunk(self, text: str) -> list[str]:
        chunks: list[str] = []
        for heading, section in self.split_sections(text):
            if len(section) <= self.max_chars:
                chunks.append(section)
                continue
            for piece in self._fallback.chunk(section):
                # Re-prefix the heading so split pieces keep their context.
                chunks.append(piece if not heading or piece.startswith("#") else f"## {heading}\n{piece}")
        return [c for c in chunks if c.strip()]


# (chunker factory, prefix each chunk with the document title?)
STRATEGIES = {
    "fixed_size": (lambda: FixedSizeChunker(chunk_size=800, overlap=100), False),
    "by_sentences": (lambda: SentenceChunker(max_sentences_per_chunk=4), False),
    "recursive": (lambda: RecursiveChunker(chunk_size=800), False),
    "heading": (lambda: HeadingChunker(max_chars=800), False),
    # Improvement proposed by the Q2 failure analysis: a section heading like
    # "## Quy định" is topic-less on its own, so the chunk never mentions which
    # service it regulates. Prefixing the document title restores that context.
    "heading_titled": (lambda: HeadingChunker(max_chars=800), True),
}


# --------------------------------------------------------------------------
# Corpus loading
# --------------------------------------------------------------------------
def parse_front_matter(raw: str) -> tuple[dict, str]:
    """Extract the simple `key: value` YAML front matter used by this corpus."""
    match = FRONT_MATTER.match(raw)
    if not match:
        return {}, raw

    metadata = {}
    for line in match.group(1).splitlines():
        if ":" in line:
            key, _, value = line.partition(":")
            metadata[key.strip()] = value.strip()
    return metadata, raw[match.end() :]


def load_corpus(corpus_dir: Path = CORPUS_DIR) -> list[tuple[dict, str]]:
    docs = []
    for path in sorted(corpus_dir.glob("*.md")):
        metadata, body = parse_front_matter(path.read_text(encoding="utf-8"))
        metadata.setdefault("doc_id", path.stem)
        metadata["file_path"] = str(path.relative_to(corpus_dir.parent.parent))
        docs.append((metadata, body))
    return docs


def build_store(strategy_name: str, embedder) -> tuple[EmbeddingStore, int]:
    """Chunk the whole corpus with one strategy and load it into a store."""
    factory, prefix_title = STRATEGIES[strategy_name]
    chunker = factory()
    store = EmbeddingStore(collection_name=f"huit_{strategy_name}", embedding_fn=embedder)

    documents: list[Document] = []
    for metadata, body in load_corpus():
        title = metadata.get("title", "")
        for index, chunk_text in enumerate(chunker.chunk(body)):
            if prefix_title and title and title.lower() not in chunk_text.lower():
                chunk_text = f"[{title}]\n{chunk_text}"
            chunk_metadata = dict(metadata)
            chunk_metadata["chunk_index"] = index
            chunk_metadata["strategy"] = strategy_name
            documents.append(
                Document(id=metadata["doc_id"], content=chunk_text, metadata=chunk_metadata)
            )

    store.add_documents(documents)
    return store, len(documents)


# --------------------------------------------------------------------------
# The 5 agreed benchmark queries (gold answers quoted from the corpus)
# --------------------------------------------------------------------------
@dataclass
class BenchmarkQuery:
    number: int
    question: str
    gold_answer: str
    expected_doc_id: str
    metadata_filter: dict | None = None
    note: str = ""


QUERIES = [
    BenchmarkQuery(
        number=1,
        question="Sinh viên được mượn tối đa bao nhiêu tài liệu về nhà và trong bao nhiêu ngày?",
        gold_answer="Sinh viên, học viên: 3 tài liệu, 10 ngày, được gia hạn 1 lần thêm 10 ngày.",
        expected_doc_id="huit-muon-sinh-vien",
        metadata_filter={"audience": "student"},
        note=(
            "Câu bắt buộc dùng metadata filter (K4-L3A). Không lọc, chunk hạn mức của "
            "giảng viên (3 tài liệu / 180 ngày) cạnh tranh trực tiếp và dễ cho câu trả lời sai."
        ),
    ),
    BenchmarkQuery(
        number=2,
        question="Phí trễ hạn khi mượn liên thư viện là bao nhiêu một ngày?",
        gold_answer="5.000đ/tài liệu/ngày.",
        expected_doc_id="huit-muon-lien-thu-vien",
    ),
    BenchmarkQuery(
        number=3,
        question="Phòng học nhóm ở tầng mấy, chứa được bao nhiêu người và dùng được bao lâu mỗi lượt?",
        gold_answer="Tầng 3, 05-07 người, 02 giờ/lượt, được gia hạn khi không có người chờ.",
        expected_doc_id="huit-phong-hoc-nhom",
    ),
    BenchmarkQuery(
        number=4,
        question="Đặt phòng xong mà đến trễ thì có bị hủy không?",
        gold_answer="Bị hủy nếu người đặt đến trễ trên 15 phút so với thời gian đăng ký.",
        expected_doc_id="huit-phong-hoc-nhom",
        note="Câu hỏi dạng hội thoại, không trùng từ khóa với tiêu đề mục.",
    ),
    BenchmarkQuery(
        number=5,
        question="Tiền thế chân được hoàn trả khi nào?",
        gold_answer=(
            "Thư viện hoàn trả tiền thế chân ngay khi người sử dụng đã hoàn tất công nợ "
            "và không còn nhu cầu sử dụng thư viện."
        ),
        expected_doc_id="huit-muon-sinh-vien",
    ),
]


# --------------------------------------------------------------------------
# Runner
# --------------------------------------------------------------------------
class CachedEmbedder:
    """Wrap an embedder with a disk cache + retry, so repeat runs cost no quota.

    The Gemini free tier is rate-limited and one full sweep embeds every chunk of
    every strategy; caching by text hash keeps re-runs instant and free.
    """

    def __init__(self, inner, cache_path: Path, backend_name: str) -> None:
        self._inner = inner
        self._cache_path = cache_path
        self._backend_name = backend_name
        self._cache: dict[str, list[float]] = {}
        self._misses = 0
        if cache_path.exists():
            try:
                self._cache = json.loads(cache_path.read_text(encoding="utf-8"))
            except Exception:
                self._cache = {}

    def __call__(self, text: str) -> list[float]:
        key = hashlib.sha256(text.encode("utf-8")).hexdigest()
        if key in self._cache:
            return self._cache[key]

        delay = 2.0
        for attempt in range(6):
            try:
                vector = self._inner(text)
                break
            except Exception as exc:
                message = str(exc).lower()
                retryable = "429" in message or "rate" in message or "quota" in message or "503" in message
                if not retryable or attempt == 5:
                    raise
                print(f"   [rate-limit] retry in {delay:.0f}s ...")
                time.sleep(delay)
                delay *= 2

        self._cache[key] = vector
        self._misses += 1
        if self._misses % 25 == 0:
            self.flush()
        return vector

    def flush(self) -> None:
        self._cache_path.parent.mkdir(parents=True, exist_ok=True)
        self._cache_path.write_text(json.dumps(self._cache), encoding="utf-8")


def select_embedder():
    load_dotenv(override=False)
    provider = os.getenv(EMBEDDING_PROVIDER_ENV, "mock").strip().lower()
    builders = {
        "local": lambda: LocalEmbedder(model_name=os.getenv("LOCAL_EMBEDDING_MODEL", LOCAL_EMBEDDING_MODEL)),
        "openai": lambda: OpenAIEmbedder(model_name=os.getenv("OPENAI_EMBEDDING_MODEL", OPENAI_EMBEDDING_MODEL)),
        "gemini": lambda: GeminiEmbedder(model_name=os.getenv("GEMINI_EMBEDDING_MODEL", GEMINI_EMBEDDING_MODEL)),
    }
    if provider in builders:
        try:
            inner = builders[provider]()
            name = getattr(inner, "_backend_name", provider)
            cache = CACHE_DIR / f"embeddings-{provider}-{name.replace('/', '_')}.json"
            return CachedEmbedder(inner, cache, name), provider
        except Exception as exc:
            print(f"[warn] provider '{provider}' unavailable ({exc}); falling back to mock")
    return _mock_embed, "mock"


def run_strategy(strategy_name: str, embedder, top_k: int = 3, use_filter: bool = True) -> dict:
    store, chunk_count = build_store(strategy_name, embedder)
    lengths = [len(record["content"]) for record in store._store]
    avg_len = sum(lengths) / len(lengths) if lengths else 0

    print(f"\n{'=' * 78}")
    print(f"STRATEGY: {strategy_name}   chunks={chunk_count}   avg_len={avg_len:.0f}   max_len={max(lengths)}")
    print("=" * 78)

    hits = 0
    precision_at_1 = 0
    for query in QUERIES:
        metadata_filter = query.metadata_filter if use_filter else None
        results = (
            store.search_with_filter(query.question, top_k=top_k, metadata_filter=metadata_filter)
            if metadata_filter
            else store.search(query.question, top_k=top_k)
        )

        retrieved_ids = [r["metadata"].get("doc_id") for r in results]
        hit = query.expected_doc_id in retrieved_ids
        top1 = bool(retrieved_ids) and retrieved_ids[0] == query.expected_doc_id
        hits += hit
        precision_at_1 += top1

        print(f"\nQ{query.number}: {query.question}")
        print(f"   gold : {query.gold_answer}")
        print(f"   filter: {metadata_filter or '-'}")
        for rank, result in enumerate(results, start=1):
            marker = "<<<" if result["metadata"].get("doc_id") == query.expected_doc_id else "   "
            preview = " ".join(result["content"].split())[:90]
            print(f"   {rank}. {result['score']:+.3f} [{result['metadata'].get('doc_id')}] {preview}... {marker}")
        print(f"   -> hit@{top_k}: {'YES' if hit else 'NO'}   P@1: {'YES' if top1 else 'NO'}")

    print(f"\n   SUMMARY {strategy_name}: hit@{top_k} = {hits}/{len(QUERIES)}   P@1 = {precision_at_1}/{len(QUERIES)}")
    return {
        "strategy": strategy_name,
        "chunks": chunk_count,
        "avg_len": avg_len,
        "hits": hits,
        "p_at_1": precision_at_1,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the K4-L3A Phase 2 retrieval benchmark.")
    parser.add_argument("--strategy", choices=sorted(STRATEGIES), help="run a single strategy (default: all)")
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--no-filter", action="store_true", help="ignore metadata filters (ablation)")
    args = parser.parse_args()

    embedder, provider = select_embedder()
    print(f"Embedding backend: {getattr(embedder, '_backend_name', type(embedder).__name__)} (provider={provider})")
    if provider == "mock":
        print(
            "[warn] MockEmbedder hashes text, so similarity scores are effectively random.\n"
            "       Use EMBEDDING_PROVIDER=local (or gemini/openai) for meaningful numbers."
        )

    names = [args.strategy] if args.strategy else sorted(STRATEGIES)
    summaries = [run_strategy(n, embedder, top_k=args.top_k, use_filter=not args.no_filter) for n in names]

    print(f"\n{'=' * 78}\nOVERALL\n{'=' * 78}")
    print(f"{'strategy':16s} {'chunks':>7s} {'avg_len':>8s} {'hit@' + str(args.top_k):>8s} {'P@1':>6s}")
    for s in summaries:
        total = len(QUERIES)
        print(
            f"{s['strategy']:16s} {s['chunks']:7d} {s['avg_len']:8.0f} "
            f"{str(s['hits']) + '/' + str(total):>8s} {str(s['p_at_1']) + '/' + str(total):>6s}"
        )

    if isinstance(embedder, CachedEmbedder):
        embedder.flush()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
