from __future__ import annotations
import os
import json
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

CORPORA_ROOT = os.path.join("storage", "rag", "corpora")
META_FILENAME = "meta.json"

def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()

def ensure_storage() -> None:
    os.makedirs(CORPORA_ROOT, exist_ok=True)

def new_corpus_id() -> str:
    return uuid.uuid4().hex

def corpus_dir(corpus_id: str) -> str:
    return os.path.join(CORPORA_ROOT, corpus_id)

def meta_path(corpus_id: str) -> str:
    return os.path.join(corpus_dir(corpus_id), META_FILENAME)

def write_meta(corpus_id: str, meta: Dict[str, Any]) -> None:
    ensure_storage()
    os.makedirs(corpus_dir(corpus_id), exist_ok=True)
    with open(meta_path(corpus_id), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

def read_meta(corpus_id: str) -> Optional[Dict[str, Any]]:
    path = meta_path(corpus_id)
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def list_corpora() -> List[Dict[str, Any]]:
    ensure_storage()
    items: List[Dict[str, Any]] = []
    for cid in os.listdir(CORPORA_ROOT):
        cdir = corpus_dir(cid)
        if not os.path.isdir(cdir):
            continue
        meta = read_meta(cid)
        if meta:
            items.append(meta)

    items.sort(key=lambda x: x.get("created_at", ""), reverse=True)
    return items

def init_meta(
    corpus_id: str,
    name: str,
    source_type: str,
    source_name: str,
    chunk_size: int,
    overlap: int,
    embedding_model: str,
) -> Dict[str, Any]:
    return {
        "corpus_id": corpus_id,
        "name": name,
        "source_type": source_type,     
        "source_name": source_name,     
        "created_at": _now_iso(),
        "chunk_size": chunk_size,
        "overlap": overlap,
        "embedding_model": embedding_model,
        "doc_count": 1,
        "chunk_count": 0,
    }

def delete_corpus(corpus_id: str) -> Tuple[bool, str]:
    """
    Deletes a corpus directory. No recycle bin.
    """
    cdir = corpus_dir(corpus_id)
    if not os.path.isdir(cdir):
        return False, "Corpus not found."

    for root, dirs, files in os.walk(cdir, topdown=False):
        for fn in files:
            os.remove(os.path.join(root, fn))
        for d in dirs:
            os.rmdir(os.path.join(root, d))
    os.rmdir(cdir)
    return True, "Corpus deleted."
