from __future__ import annotations
from typing import List, Dict, Any
from pathlib import Path
import json
import numpy as np
import faiss

class FaissStore:
    """
    Simple FAISS vector store with persistence:
      - index.faiss
      - texts.jsonl
      - metas.jsonl
    """
    def __init__(self):
        self.index: faiss.Index | None = None
        self.texts: List[str] = []
        self.metas: List[Dict[str, Any]] = []

    def reset(self) -> None:
        self.index = None
        self.texts = []
        self.metas = []

    def add(self, embeddings: List[List[float]], texts: List[str], metas: List[Dict[str, Any]]) -> None:
        if not embeddings or not texts:
            return
        if len(texts) != len(embeddings):
            raise ValueError(f"texts and embeddings length mismatch: {len(texts)} != {len(embeddings)}")
        if metas is None:
            metas = [{} for _ in texts]
        if len(metas) != len(texts):
            raise ValueError(f"metas and texts length mismatch: {len(metas)} != {len(texts)}")

        vecs = np.array(embeddings, dtype="float32")
        if vecs.ndim != 2:
            raise ValueError("embeddings must be a 2D array-like")

        if self.index is None:
            dim = int(vecs.shape[1])
            self.index = faiss.IndexFlatIP(dim)

        self.index.add(vecs)
        self.texts.extend(texts)
        self.metas.extend(metas)

    def search(self, query_embedding: List[float], top_k: int = 10) -> List[Dict[str, Any]]:
        if self.index is None or not self.texts:
            return []

        q = np.array([query_embedding], dtype="float32")
        scores, ids = self.index.search(q, int(top_k))

        results: List[Dict[str, Any]] = []
        for score, idx in zip(scores[0], ids[0]):
            if int(idx) == -1:
                continue
            idx = int(idx)
            meta = self.metas[idx] if idx < len(self.metas) else {}
            results.append(
                {
                    "score": float(score),
                    "text": self.texts[idx],
                    **(meta or {}),
                }
            )
        return results

    def save(self, dir_path: str | Path) -> Dict[str, Any]:
        """
        Save store to disk. Returns status dict.
        """
        d = Path(dir_path)
        d.mkdir(parents=True, exist_ok=True)

        if self.index is None or not self.texts:
            (d / "EMPTY").write_text("empty\n", encoding="utf-8")
            return {"saved": False, "reason": "empty_store", "path": str(d)}

        faiss.write_index(self.index, str(d / "index.faiss"))

        texts_path = d / "texts.jsonl"
        metas_path = d / "metas.jsonl"

        with texts_path.open("w", encoding="utf-8") as f:
            for t in self.texts:
                f.write(json.dumps({"text": t}, ensure_ascii=False) + "\n")

        with metas_path.open("w", encoding="utf-8") as f:
            for m in self.metas:
                f.write(json.dumps(m or {}, ensure_ascii=False) + "\n")

        return {"saved": True, "path": str(d), "count": len(self.texts)}

    def load(self, dir_path: str | Path) -> Dict[str, Any]:
        """
        Load store from disk. Returns status dict.
        """
        d = Path(dir_path)
        index_path = d / "index.faiss"
        texts_path = d / "texts.jsonl"
        metas_path = d / "metas.jsonl"

        if not index_path.exists() or not texts_path.exists() or not metas_path.exists():
            self.reset()
            return {"loaded": False, "reason": "missing_files", "path": str(d)}

        self.index = faiss.read_index(str(index_path))

        self.texts = []
        with texts_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                self.texts.append(obj.get("text", ""))

        self.metas = []
        with metas_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                self.metas.append(json.loads(line))

        if len(self.metas) != len(self.texts):
            if len(self.metas) < len(self.texts):
                self.metas.extend([{} for _ in range(len(self.texts) - len(self.metas))])
            else:
                self.metas = self.metas[: len(self.texts)]

        return {"loaded": True, "path": str(d), "count": len(self.texts)}
