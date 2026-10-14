import fitz
from pathlib import Path

out = Path("data/processed")
out.mkdir(parents=True, exist_ok=True)

for pdf in sorted(Path("data/raw").glob("*.pdf")):
    doc = fitz.open(pdf)
    with open(out / f"{pdf.stem}.txt", "w", encoding="utf-8") as f:
        for i, page in enumerate(doc, start=1):
            f.write(f"\n\n=== PAGE {i} ===\n")
            f.write(page.get_text())
    print("wrote", pdf.stem)
    