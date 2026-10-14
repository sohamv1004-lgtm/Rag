import csv
import json
import re
from collections import defaultdict
from rank_bm25 import BM25Okapi

K = 5
tok = lambda s: re.findall(r"\w+", s.lower())
norm = lambda s: re.sub(r"\s+", " ", s.lower().replace(",", ""))

chunks = [json.loads(l) for l in open("data/processed/chunks.jsonl", encoding="utf-8")]
bm25 = BM25Okapi([tok(c["text"]) for c in chunks])
docs_available = {c["doc"] for c in chunks}

with open("eval/questions.csv", encoding="utf-8") as f:
    rows = [r for r in csv.DictReader(f) if r["split"] == "dev"]

stats = defaultdict(lambda: [0, 0, 0])  # type -> [total, doc_hit, fact_hit]
misses = []
for r in rows:
    if r["type"] == "refuse":
        continue  # refusals are scored later, once the LLM step exists
    if r["source_doc"] not in docs_available:
        print("skipped (no such source_doc):", r["id"], r["source_doc"])
        continue
    scores = bm25.get_scores(tok(r["question"]))
    top = sorted(range(len(chunks)), key=lambda i: -scores[i])[:K]
    got = [chunks[i] for i in top]
    doc_hit = any(c["doc"] == r["source_doc"] for c in got)
    fact_hit = any(
        c["doc"] == r["source_doc"] and norm(r["must_contain"]) in norm(c["text"])
        for c in got
    )
    s = stats[r["type"]]
    s[0] += 1
    s[1] += doc_hit
    s[2] += fact_hit
    if not fact_hit:
        misses.append((r["id"], r["question"], [f'{c["doc"]} p{c["page"]}' for c in got[:3]]))

print(f"\nTop-{K} retrieval on DEV questions (holdout not used)")
print(f'{"type":10} {"n":>3} {"right doc":>10} {"fact found":>11}')
tot = [0, 0, 0]
for t, (n, d, f_) in sorted(stats.items()):
    print(f"{t:10} {n:3} {d / n:10.0%} {f_ / n:11.0%}")
    tot = [tot[0] + n, tot[1] + d, tot[2] + f_]
if tot[0]:
    print(f'{"ALL":10} {tot[0]:3} {tot[1] / tot[0]:10.0%} {tot[2] / tot[0]:11.0%}')
print("\nMissed (fact not in top 5):")
for i, q, g in misses:
    print(" ", i, "|", q, "|", g)
    