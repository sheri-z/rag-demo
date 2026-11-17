"""Chunk cleaned pages, embed them, and build one FAISS index per company.

Following the original approach:
    - chunks never cross page boundaries, and each chunk records its page number,
      so retrieval can later expand a hit to its full parent page
    - 300-token chunks with 50-token overlap, split recursively on Markdown structure
    - one IndexFlatIP per report: exact search, and inner product equals cosine
      similarity because OpenAI embeddings are unit length

Output in data/erc2/indexes/:
    <sha1>.faiss         vectors, row i belongs to chunk i
    <sha1>.chunks.json   [{"id", "page", "text"}]

Usage:
    uv run python -m rag_demo.ingest
"""

import json

import faiss
import numpy as np
import tiktoken
from langchain_text_splitters import Language, RecursiveCharacterTextSplitter

from rag_demo.config import get_openai, get_settings
from rag_demo.paths import CLEAN_DIR, INDEX_DIR

CHUNK_TOKENS = 300
OVERLAP_TOKENS = 50
EMBED_BATCH = 256  # inputs per request, well below the API limits
EMBED_PRICE_PER_M = 0.02  # USD per 1M tokens, text-embedding-3-small

ENCODING = tiktoken.get_encoding("cl100k_base")  # tokenizer of the embedding models

SPLITTER = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    encoding_name="cl100k_base",
    chunk_size=CHUNK_TOKENS,
    chunk_overlap=OVERLAP_TOKENS,
    separators=RecursiveCharacterTextSplitter.get_separators_for_language(Language.MARKDOWN),
)


def chunk_pages(pages: list[dict]) -> list[dict]:
    chunks = []
    for page in pages:
        for text in SPLITTER.split_text(page["text"]):
            chunks.append({"id": len(chunks), "page": page["page"], "text": text})
    return chunks


def embed(texts: list[str]) -> np.ndarray:
    client, model = get_openai(), get_settings().embedding_model
    vectors = []
    for start in range(0, len(texts), EMBED_BATCH):
        response = client.embeddings.create(model=model, input=texts[start : start + EMBED_BATCH])
        vectors.extend(item.embedding for item in response.data)
    return np.array(vectors, dtype="float32")


def build_index(vectors: np.ndarray) -> faiss.Index:
    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)
    return index


def main() -> None:
    paths = sorted(CLEAN_DIR.glob("*.json"))
    if not paths:
        raise SystemExit(f"nothing in {CLEAN_DIR}; run rag_demo.clean first")
    INDEX_DIR.mkdir(parents=True, exist_ok=True)

    total_tokens = 0
    for path in paths:
        sha1 = path.stem
        pages = json.loads(path.read_text())["pages"]
        chunks = chunk_pages(pages)
        tokens = [len(ENCODING.encode(c["text"])) for c in chunks]
        total_tokens += sum(tokens)

        vectors = embed([c["text"] for c in chunks])
        faiss.write_index(build_index(vectors), str(INDEX_DIR / f"{sha1}.faiss"))
        (INDEX_DIR / f"{sha1}.chunks.json").write_text(
            json.dumps(chunks, ensure_ascii=False, indent=2)
        )
        print(
            f"{sha1[:8]}  {len(pages):>4} pages  {len(chunks):>5} chunks  "
            f"tokens/chunk avg={np.mean(tokens):.0f} max={max(tokens)}"
        )

    print(f"embedded {total_tokens:,} tokens, about ${total_tokens / 1e6 * EMBED_PRICE_PER_M:.3f}")


if __name__ == "__main__":
    main()
