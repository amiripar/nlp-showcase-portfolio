from __future__ import annotations
from typing import List, Dict, Tuple, Optional, Any
import re
from backend.core.config import settings
from backend.nlp.chunking import chunk_text
from backend.nlp.openai_client import embed_texts, rag_answer_with_llm
from backend.vectorstore.faiss_store import FaissStore
from backend.nlp.pdf_loader import extract_text_from_pdf_bytes
from backend.services.rag_persistence import (
    ensure_storage,
    new_corpus_id,
    corpus_dir,
    init_meta,
    write_meta,
    read_meta,
    list_corpora,
    delete_corpus,
)

_ACTIVE_CORPUS_ID: Optional[str] = None

def _embedding_model_name() -> str:
    return getattr(settings, "openai_embedding_model", "embedding-model")

def rag_list_corpora() -> List[Dict[str, Any]]:
    ensure_storage()
    return list_corpora()

def rag_set_active(corpus_id: Optional[str]) -> None:
    global _ACTIVE_CORPUS_ID
    _ACTIVE_CORPUS_ID = corpus_id

def rag_get_active() -> Optional[str]:
    return _ACTIVE_CORPUS_ID

def _save_corpus(store: FaissStore, meta: Dict[str, Any]) -> None:
    cdir = corpus_dir(meta["corpus_id"])
    store.save(cdir)
    write_meta(meta["corpus_id"], meta)

def _load_store(corpus_id: str) -> Tuple[Optional[FaissStore], str]:
    meta = read_meta(corpus_id)
    if not meta:
        return None, "Corpus meta not found."

    store = FaissStore()
    info = store.load(corpus_dir(corpus_id))
    if not info.get("loaded"):
        return None, f"Failed to load corpus store: {info.get('reason', 'unknown')}"

    return store, "ok"

def _choose_corpus_id(corpus_id: Optional[str]) -> Optional[str]:
    if corpus_id:
        return corpus_id
    return rag_get_active()

def rag_ingest_text(text: str, name: Optional[str] = None) -> Tuple[str, Dict, str]:
    """
    Create a new corpus from provided text and persist it.
    Returns: (corpus_id, stats, note)
    """
    ensure_storage()

    corpus_id = new_corpus_id()
    corpus_name = name or f"Text corpus {corpus_id[:8]}"

    clean = (text or "").strip()
    chunks = chunk_text(clean, settings.rag_chunk_size, settings.rag_chunk_overlap) if clean else []
    if not chunks:
        meta = init_meta(
            corpus_id=corpus_id,
            name=corpus_name,
            source_type="text",
            source_name="Text box",
            chunk_size=settings.rag_chunk_size,
            overlap=settings.rag_chunk_overlap,
            embedding_model=_embedding_model_name(),
        )
        meta["chunk_count"] = 0
        write_meta(corpus_id, meta)
        rag_set_active(corpus_id)
        return corpus_id, {"chunks_indexed": 0, "corpus_id": corpus_id}, "No chunks created from text."

    embeddings = embed_texts(chunks)

    metas = [{"source": "Text box", "page_num": None, "chunk_id": i} for i in range(len(chunks))]

    store = FaissStore()
    store.add(embeddings, chunks, metas)

    meta = init_meta(
        corpus_id=corpus_id,
        name=corpus_name,
        source_type="text",
        source_name="Text box",
        chunk_size=settings.rag_chunk_size,
        overlap=settings.rag_chunk_overlap,
        embedding_model=_embedding_model_name(),
    )
    meta["chunk_count"] = len(chunks)

    _save_corpus(store, meta)
    rag_set_active(corpus_id)

    return corpus_id, {"chunks_indexed": len(chunks), "corpus_id": corpus_id}, "RAG corpus created and saved."

def rag_ingest_pdf(pdf_bytes: bytes, filename: str = "uploaded.pdf", name: Optional[str] = None) -> Tuple[str, Dict, str]:
    """
    Create a new corpus from a PDF (page-aware chunks) and persist it.
    Returns: (corpus_id, stats, note)
    """
    ensure_storage()

    corpus_id = new_corpus_id()
    corpus_name = name or f"PDF corpus {corpus_id[:8]}"

    pages, note = extract_text_from_pdf_bytes(pdf_bytes)
    if not pages:
        meta = init_meta(
            corpus_id=corpus_id,
            name=corpus_name,
            source_type="pdf",
            source_name=filename,
            chunk_size=settings.rag_chunk_size,
            overlap=settings.rag_chunk_overlap,
            embedding_model=_embedding_model_name(),
        )
        meta["chunk_count"] = 0
        meta["page_count"] = 0
        write_meta(corpus_id, meta)
        rag_set_active(corpus_id)
        return corpus_id, {"chunks_indexed": 0, "pages": 0, "corpus_id": corpus_id}, f"No text extracted. {note}"

    all_chunks: List[str] = []
    metas: List[Dict[str, Any]] = []

    for p in pages:
        page_text = (p.get("text") or "").strip()
        if not page_text:
            continue

        page_chunks = chunk_text(page_text, settings.rag_chunk_size, settings.rag_chunk_overlap)
        for j, ch in enumerate(page_chunks):
            all_chunks.append(ch)
            metas.append(
                {
                    "source": filename,
                    "page_num": p.get("page_num"),
                    "chunk_id": j,
                }
            )

    if not all_chunks:
        meta = init_meta(
            corpus_id=corpus_id,
            name=corpus_name,
            source_type="pdf",
            source_name=filename,
            chunk_size=settings.rag_chunk_size,
            overlap=settings.rag_chunk_overlap,
            embedding_model=_embedding_model_name(),
        )
        meta["chunk_count"] = 0
        meta["page_count"] = len(pages)
        write_meta(corpus_id, meta)
        rag_set_active(corpus_id)
        return corpus_id, {"chunks_indexed": 0, "pages": len(pages), "corpus_id": corpus_id}, "No chunks created from extracted PDF text."

    embeddings = embed_texts(all_chunks)

    store = FaissStore()
    store.add(embeddings, all_chunks, metas)

    meta = init_meta(
        corpus_id=corpus_id,
        name=corpus_name,
        source_type="pdf",
        source_name=filename,
        chunk_size=settings.rag_chunk_size,
        overlap=settings.rag_chunk_overlap,
        embedding_model=_embedding_model_name(),
    )
    meta["chunk_count"] = len(all_chunks)
    meta["page_count"] = len(pages)

    _save_corpus(store, meta)
    rag_set_active(corpus_id)

    return (
        corpus_id,
        {"chunks_indexed": len(all_chunks), "pages": len(pages), "corpus_id": corpus_id},
        f"Indexed PDF with page-aware chunks. {note}",
    )

def rag_ask(question: str, corpus_id: Optional[str] = None) -> Tuple[str, List[Dict], str]:
    q = (question or "").strip()
    if not q:
        return "", [], "Invalid question (empty)."

    cid = _choose_corpus_id(corpus_id)
    if not cid:
        return "No indexed data available.", [], "RAG: no active corpus selected."

    store, status = _load_store(cid)
    if store is None:
        return "No indexed data available.", [], f"RAG: {status}"

    try:
        q_emb = embed_texts([q])[0]
    except Exception as e:
        return (
            "I couldn't reach the AI service (network issue). Please try again.",
            [],
            f"RAG: embedding failed: {type(e).__name__}",
        )

    top_k = max(int(getattr(settings, "rag_top_k", 4)), 12)
    retrieved = store.search(q_emb, top_k=top_k)
    if not retrieved:
        return "Not enough information in the context.", [], "RAG: no matches."

    def _lexical_score(text: str, query: str) -> float:
        t = (text or "").lower()
        ql = (query or "").lower()
        s = 0.0

        if "table of contents" in t or re.search(r"\bcontents\b", t):
            s += 5.0
        if re.search(r"\bchapter\s+\d+\b", t):
            s += 1.5
        if re.search(r"\.\s*\.\s*\.\s*\d+\b", t):
            s += 2.5

        for tok in re.findall(r"[a-z0-9]+", ql):
            if len(tok) >= 4 and tok in t:
                s += 0.15
        return s

    for r in retrieved:
        r["_lex"] = _lexical_score(r.get("text", ""), q)

    retrieved.sort(key=lambda x: (x.get("_lex", 0.0), x.get("score", 0.0)), reverse=True)

    contexts = [r.get("text", "") for r in retrieved if r.get("text")]
    answer = rag_answer_with_llm(q, contexts)

    return answer, retrieved, f"RAG: corpus={cid} retrieval + lexical rerank + grounded answer."

def rag_clear() -> str:
    """
    Clears only the ACTIVE corpus pointer (does not delete saved corpora).
    """
    rag_set_active(None)
    return "Active corpus cleared (saved corpora remain)."

def rag_delete(corpus_id: str) -> str:
    ok, note = delete_corpus(corpus_id)
    if rag_get_active() == corpus_id:
        rag_set_active(None)
    return note

def rag_index(documents: List[str]) -> Tuple[Dict, str]:
    """
    Backward-compatible: indexes the concatenation of provided documents into a NEW saved corpus.
    Returns stats + note.
    """
    text = "\n\n".join([d for d in documents if (d or "").strip()])
    cid, stats, note = rag_ingest_text(text, name=None)
    return stats, note

def rag_index_pdf(pdf_bytes: bytes, filename: str = "uploaded.pdf") -> Tuple[Dict, str]:
    """
    Backward-compatible: builds a NEW saved corpus from a PDF.
    """
    cid, stats, note = rag_ingest_pdf(pdf_bytes, filename=filename, name=None)
    return stats, note
