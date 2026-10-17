import csv
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

K = 5
MODEL = "BAAI/bge-small-en-v1.5"
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

tok = lambda s: re.findall(r"\w+", s.lower())
norm = lambda s: re.sub(r"\s+", " ", s.lower().replace(",", ""))

chunks = [json.loads(l) for l in open("data/processed/chunks.jsonl", encoding="utf-8")]
bm25 = BM25Okapi([tok(c["text"]) for c in chunks])

model = SentenceTransformer(MODEL)
cache = Path("data/processed/embeddings.npy")
emb = np.load(cache) if cache.exists() else None
if emb is None or emb.shape[0] != len(chunks):
    emb = model.encode([c["text"] for c in chunks], normalize_embeddings=True, show_progress_bar=True)
    np.save(cache, emb)


def rank_bm25(q):
    return list(np.argsort(-bm25.get_scores(tok(q))))


def rank_dense(q):
    v = model.encode([QUERY_PREFIX + q], normalize_embeddings=True)[0]
    return list(np.argsort(-(emb @ v)))


def rank_hybrid(q):
    # reciprocal rank fusion of the two rankings
    scores = defaultdict(float)
    for ranking in (rank_bm25(q), rank_dense(q)):
        for pos, i in enumerate(ranking[:50]):
            scores[int(i)] += 1 / (60 + pos + 1)
    return sorted(scores, key=lambda i: -scores[i])


with open("eval/questions.csv", encoding="utf-8") as f:
    rows = [r for r in csv.DictReader(f) if r["split"] == "dev" and r["type"] != "refuse"]

docs = {c["doc"] for c in chunks}
rows = [r for r in rows if r["source_doc"] in docs]

for name, ranker in [("bm25", rank_bm25), ("dense", rank_dense), ("hybrid", rank_hybrid)]:
    stats = defaultdict(lambda: [0, 0, 0])
    misses = []
    for r in rows:
        got = [chunks[int(i)] for i in ranker(r["question"])[:K]]
        doc_hit = any(c["doc"] == r["source_doc"] for c in got)
        fact_hit = any(c["doc"] == r["source_doc"] and norm(r["must_contain"]) in norm(c["text"]) for c in got)
        s = stats[r["type"]]
        s[0] += 1
        s[1] += doc_hit
        s[2] += fact_hit
        if not fact_hit:
            misses.append((r["id"], r["question"], [f'{c["doc"]} p{c["page"]}' for c in got[:3]]))
    print(f"\n=== {name.upper()} (top-{K}, dev questions only) ===")
    print(f'{"type":10} {"n":>3} {"right doc":>10} {"fact found":>11}')
    tot = [0, 0, 0]
    for t, (n, d, f_) in sorted(stats.items()):
        print(f"{t:10} {n:3} {d / n:10.0%} {f_ / n:11.0%}")
        tot = [tot[0] + n, tot[1] + d, tot[2] + f_]
    print(f'{"ALL":10} {tot[0]:3} {tot[1] / tot[0]:10.0%} {tot[2] / tot[0]:11.0%}')
    if name == "hybrid":
        print("\nMissed by hybrid:")
        for i, q, g in misses:
            print(" ", i, "|", q, "|", g)
            
