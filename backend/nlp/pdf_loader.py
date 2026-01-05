from typing import List, Dict, Tuple
import io
from PyPDF2 import PdfReader

def extract_text_from_pdf_bytes(pdf_bytes: bytes) -> Tuple[List[Dict], str]:
    """
    Returns:
      pages: [
        { "page_num": 1, "text": "..." },
        ...
      ]
      note: str
    """
    try:
        reader = PdfReader(io.BytesIO(pdf_bytes))
    except Exception as e:
        return [], f"Failed to read PDF: {e}"

    pages: List[Dict] = []

    for i, page in enumerate(reader.pages):
        try:
            text = page.extract_text() or ""
        except Exception:
            text = ""

        text = text.strip()
        if text:
            pages.append(
                {
                    "page_num": i + 1,
                    "text": text,
                }
            )

    if not pages:
        return [], "No extractable text found (possibly scanned PDF)."

    return pages, f"Extracted {len(pages)} pages."
