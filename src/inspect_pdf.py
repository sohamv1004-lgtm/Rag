import fitz  # PyMuPDF
from pathlib import Path

for pdf in sorted(Path("data/raw").glob("*.pdf")):
    doc = fitz.open(pdf)
    empty = [i + 1 for i, p in enumerate(doc) if len(p.get_text().strip()) < 50]
    total_chars = sum(len(p.get_text()) for p in doc)
    print(f"{pdf.name}: {len(doc)} pages, {total_chars} chars, "
          f"{len(empty)} near-empty pages {empty[:10]}")