from __future__ import annotations
import math
import re
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
from backend.core.config import settings
from io import BytesIO

@dataclass
class Chunk:
    doc_name: str
    chunk_id: int
    text: str
    page_num: int | None  

_INDEX_CHUNKS: List[Chunk] = []
_INDEX_DF: Dict[str, int] = {}
_INDEX_NDOCS: int = 0

_INDEX_EMB: List[List[float]] = []      
_SEMANTIC_READY: bool = False

def _tokenize(text: str) -> List[str]:
    tokens = re.findall(r"[A-Za-z0-9]+", (text or "").lower())
    return [t for t in tokens if len(t) >= 2]

def _chunk_text(text: str, chunk_size: int = 900, overlap: int = 150) -> List[str]:
    text = (text or "").strip()
    if not text:
        return []
    chunks = []
    start = 0
    n = len(text)
    while start < n:
        end = min(start + chunk_size, n)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end == n:
            break
        start = max(0, end - overlap)
    return chunks

def extract_text_from_txt(data: bytes) -> str:
    try:
        return data.decode("utf-8", errors="ignore")
    except Exception:
        return data.decode("latin-1", errors="ignore")

def extract_text_from_pdf_pages(data: bytes) -> List[Tuple[int, str]]:
    """
    Returns list of (page_num, page_text), page_num starts from 1.
    """
    try:
        from PyPDF2 import PdfReader  
    except Exception as e:
        raise RuntimeError(f"PyPDF2 is not installed or cannot be imported: {e}")

    reader = PdfReader(BytesIO(data))
    pages: List[Tuple[int, str]] = []

    for i, p in enumerate(reader.pages, start=1):
        try:
            pages.append((i, (p.extract_text() or "").strip()))
        except Exception:
            pages.append((i, ""))

    return pages

def _l2_normalize(vec: List[float]) -> List[float]:
    s = 0.0
    for v in vec:
        s += v * v
    norm = math.sqrt(s) if s > 0 else 1.0
    return [v / norm for v in vec]

def _dot(a: List[float], b: List[float]) -> float:
    return sum(x * y for x, y in zip(a, b))

def _embed_texts_openai(texts: List[str]) -> List[List[float]]:
    """
    Returns normalized embeddings for each text (aligned order).
    """
    try:
        from openai import OpenAI
    except Exception as e:
        raise RuntimeError(f"OpenAI SDK not available: {e}")

    api_key = (settings.openai_api_key or "").strip()
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY missing")

    model = (getattr(settings, "ir_embedding_model", "") or "text-embedding-3-small").strip()

    client = OpenAI(api_key=api_key)

    resp = client.embeddings.create(model=model, input=texts)
    vectors = [item.embedding for item in resp.data]
    return [_l2_normalize(v) for v in vectors]

def clear_index() -> None:
    global _INDEX_CHUNKS, _INDEX_DF, _INDEX_NDOCS, _INDEX_EMB, _SEMANTIC_READY
    _INDEX_CHUNKS = []
    _INDEX_DF = {}
    _INDEX_NDOCS = 0
    _INDEX_EMB = []
    _SEMANTIC_READY = False

def build_index(files: List[Tuple[str, bytes]], chunk_size: int = 900, overlap: int = 150) -> Dict:
    global _INDEX_CHUNKS, _INDEX_DF, _INDEX_NDOCS, _INDEX_EMB, _SEMANTIC_READY

    clear_index()

    all_chunks: List[Chunk] = []

    for filename, data in files:
        lower = filename.lower().strip()

        if lower.endswith(".txt"):
            text = extract_text_from_txt(data)
            chunks = _chunk_text(text, chunk_size=chunk_size, overlap=overlap)
            for i, c in enumerate(chunks):
                all_chunks.append(Chunk(doc_name=filename, chunk_id=i, text=c, page_num=None))
            continue
        elif lower.endswith(".pdf"):
            pages = extract_text_from_pdf_pages(data)
            for page_num, page_text in pages:
                chunks = _chunk_text(page_text, chunk_size=chunk_size, overlap=overlap)
                for i, c in enumerate(chunks):
                    all_chunks.append(Chunk(doc_name=filename, chunk_id=i, text=c, page_num=page_num))
            continue
        else:
            continue

        chunks = _chunk_text(text, chunk_size=chunk_size, overlap=overlap)
        for i, c in enumerate(chunks):
            all_chunks.append(Chunk(doc_name=filename, chunk_id=i, text=c))

    _INDEX_CHUNKS = all_chunks
    _INDEX_NDOCS = len(_INDEX_CHUNKS)

    df: Dict[str, int] = {}
    for ch in _INDEX_CHUNKS:
        seen = set(_tokenize(ch.text))
        for term in seen:
            df[term] = df.get(term, 0) + 1
    _INDEX_DF = df

    want_semantic = bool(getattr(settings, "use_semantic_ir", False))
    if want_semantic and _INDEX_NDOCS > 0:
        texts = [c.text for c in _INDEX_CHUNKS]

        batch_size = 64
        embeddings: List[List[float]] = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            embeddings.extend(_embed_texts_openai(batch))

        _INDEX_EMB = embeddings
        _SEMANTIC_READY = True
    else:
        _INDEX_EMB = []
        _SEMANTIC_READY = False

    return {
        "num_files": len(files),
        "num_chunks": _INDEX_NDOCS,
        "vocab_size": len(_INDEX_DF),
        "semantic_ready": _SEMANTIC_READY,
    }

def _search_keyword(query: str, top_k: int) -> Dict:
    query = (query or "").strip()
    if not query:
        return {"results": [], "note": "IR: empty query (keyword)."}

    if _INDEX_NDOCS == 0:
        return {"results": [], "note": "IR: index is empty. Build index first."}

    q_tokens = _tokenize(query)
    if not q_tokens:
        return {"results": [], "note": "IR: query has no valid tokens."}

    idf: Dict[str, float] = {}
    for t in set(q_tokens):
        df = _INDEX_DF.get(t, 0)
        idf[t] = math.log((_INDEX_NDOCS + 1) / (df + 1)) + 1.0

    scored = []
    for idx, ch in enumerate(_INDEX_CHUNKS):
        tokens = _tokenize(ch.text)
        if not tokens:
            continue

        tf: Dict[str, int] = {}
        for tok in tokens:
            if tok in idf:
                tf[tok] = tf.get(tok, 0) + 1

        if not tf:
            continue

        score = sum(tf[t] * idf[t] for t in tf)
        scored.append((score, idx))

    scored.sort(key=lambda x: x[0], reverse=True)
    top = scored[: max(1, int(top_k))]

    results = []
    for score, idx in top:
        ch = _INDEX_CHUNKS[idx]
        results.append(
            {"doc_name": ch.doc_name, "chunk_id": ch.chunk_id, "score": float(score), "text": ch.text,"page_num": ch.page_num}
        )

    return {"results": results, "note": f"IR (keyword): returned top {len(results)} chunks."}

def _search_semantic(query: str, top_k: int) -> Dict:
    query = (query or "").strip()
    if not query:
        return {"results": [], "note": "IR: empty query (semantic)."}

    if _INDEX_NDOCS == 0:
        return {"results": [], "note": "IR: index is empty. Build index first."}

    if not _SEMANTIC_READY or not _INDEX_EMB:
        return {"results": [], "note": "IR: semantic index not ready. Enable USE_SEMANTIC_IR and rebuild index."}

    q_vec = _embed_texts_openai([query])[0]  

    scored = []
    for idx, vec in enumerate(_INDEX_EMB):
        score = _dot(q_vec, vec)  
        scored.append((score, idx))

    scored.sort(key=lambda x: x[0], reverse=True)
    top = scored[: max(1, int(top_k))]

    results = []
    for score, idx in top:
        ch = _INDEX_CHUNKS[idx]
        results.append(
            {"doc_name": ch.doc_name, "chunk_id": ch.chunk_id, "score": float(score), "page_num": ch.page_num, "text": ch.text}
        )

    return {"results": results, "note": f"IR (semantic): returned top {len(results)} chunks (cosine similarity)."}
  
def search(query: str, top_k: int = 5, mode: str = "auto") -> Dict:
    """
    mode:
      - "keyword"
      - "semantic"
      - "auto" (semantic if ready, else keyword)
    """
    mode = (mode or "auto").strip().lower()

    if mode == "semantic":
        return _search_semantic(query, top_k)
    if mode == "keyword":
        return _search_keyword(query, top_k)

    if _SEMANTIC_READY:
        return _search_semantic(query, top_k)
    return _search_keyword(query, top_k)
