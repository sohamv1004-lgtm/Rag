import json
import re
from pathlib import Path

CHUNK_WORDS = 300
OVERLAP = 50

src = Path("data/processed")
out = src / "chunks.jsonl"

chunks = []
for txt in sorted(src.glob("*.txt")):
    text = txt.read_text(encoding="utf-8")
    parts = re.split(r"=== PAGE (\d+) ===", text)
    words = []
    for i in range(1, len(parts), 2):
        page = int(parts[i])
        for w in parts[i + 1].split():
            words.append((w, page))
    step = CHUNK_WORDS - OVERLAP
    for start in range(0, len(words), step):
        window = words[start:start + CHUNK_WORDS]
        if len(window) < 30:
            continue
        chunks.append({
            "doc": txt.stem,
            "page": window[0][1],
            "text": " ".join(w for w, _ in window),
        })

with open(out, "w", encoding="utf-8") as f:
    for c in chunks:
        f.write(json.dumps(c, ensure_ascii=False) + "\n")

print(len(chunks), "chunks written to", out)