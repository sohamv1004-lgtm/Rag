import json
import re
import sys
from rank_bm25 import BM25Okapi

chunks = [json.loads(l) for l in open("data/processed/chunks.jsonl", encoding="utf-8")]
tok = lambda s: re.findall(r"\w+", s.lower())
bm25 = BM25Okapi([tok(c["text"]) for c in chunks])

query = " ".join(sys.argv[1:])
scores = bm25.get_scores(tok(query))
top = sorted(range(len(chunks)), key=lambda i: -scores[i])[:5]

for i in top:
    c = chunks[i]
    print(f"[{c['doc']} | PDF page {c['page']}] score={scores[i]:.1f}")
    print(c["text"][:300].replace("\n", " "), "...\n")