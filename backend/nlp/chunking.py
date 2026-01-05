from typing import List


def chunk_text(text: str, chunk_size: int = 800, overlap: int = 120) -> List[str]:
    t = text.strip()
    if not t:
        return []

    chunks = []
    start = 0
    n = len(t)

    while start < n:
        end = min(start + chunk_size, n)
        chunk = t[start:end].strip()
        if chunk:
            chunks.append(chunk)

        if end == n:
            break

        start = max(end - overlap, start + 1)

    return chunks
