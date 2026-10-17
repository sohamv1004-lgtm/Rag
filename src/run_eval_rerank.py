import csv
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder, SentenceTransformer

K = 8               # chunks we judge (and would send to the LLM)
CANDIDATES = 50     # chunks the reranker gets to look at
EMBED_MODEL = "BAAI/bge-small-en-v1.5"
RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

tok = lambda s: re.findall(r"\w+", s.lower())
norm = lambda s: re.sub(r"\s+", " ", s.lower().replace(",", ""))

chunks = [json.loads(l) for l in open("data/processed/chunks.jsonl", encoding="utf-8")]
bm25 = BM25Okapi([tok(c["text"]) for c in chunks])

embedder = SentenceTransformer(EMBED_MODEL)
cache = Path("data/processed/embeddings.npy")
emb = np.load(cache) if cache.exists() else None
if emb is None or emb.shape[0] != len(chunks):
    emb = embedder.encode([c["text"] for c in chunks], normalize_embeddings=True, show_progress_bar=True)
    np.save(cache, emb)

reranker = CrossEncoder(RERANK_MODEL, max_length=512)


def rank_hybrid(q):
    bm = list(np.argsort(-bm25.get_scores(tok(q))))
    v = embedder.encode([QUERY_PREFIX + q], normalize_embeddings=True)[0]
    de = list(np.argsort(-(emb @ v)))
    scores = defaultdict(float)
    for ranking in (bm, de):
        for pos, i in enumerate(ranking[:100]):
            scores[int(i)] += 1 / (60 + pos + 1)
    return sorted(scores, key=lambda i: -scores[i])


def rank_rerank(q):
    cand = rank_hybrid(q)[:CANDIDATES]
    ce = reranker.predict([(q, chunks[i]["text"]) for i in cand])
    order = np.argsort(-ce)
    return [cand[int(j)] for j in order]


with open("eval/questions.csv", encoding="utf-8") as f:
    rows = [r for r in csv.DictReader(f) if r["split"] == "dev" and r["type"] != "refuse"]
docs = {c["doc"] for c in chunks}
rows = [r for r in rows if r["source_doc"] in docs]

for name, ranker in [("hybrid", rank_hybrid), (f"hybrid + rerank (top {CANDIDATES})", rank_rerank)]:
    stats = defaultdict(lambda: [0, 0, 0])
    misses = []
    for r in rows:
        full = ranker(r["question"])
        got = [chunks[int(i)] for i in full[:K]]
        doc_hit = any(c["doc"] == r["source_doc"] for c in got)
        fact_hit = any(c["doc"] == r["source_doc"] and norm(r["must_contain"]) in norm(c["text"]) for c in got)
        s = stats[r["type"]]
        s[0] += 1
        s[1] += doc_hit
        s[2] += fact_hit
        if not fact_hit:
            rank = next((n + 1 for n, i in enumerate(full)
                         if chunks[int(i)]["doc"] == r["source_doc"] and norm(r["must_contain"]) in norm(chunks[int(i)]["text"])), None)
            misses.append((r["id"], r["question"], f"correct chunk ranked: {rank}"))
    print(f"\n=== {name.upper()} (top-{K}, dev questions only) ===")
    print(f'{"type":10} {"n":>3} {"right doc":>10} {"fact found":>11}')
    tot = [0, 0, 0]
    for t, (n, d, f_) in sorted(stats.items()):
        print(f"{t:10} {n:3} {d / n:10.0%} {f_ / n:11.0%}")
        tot = [tot[0] + n, tot[1] + d, tot[2] + f_]
    print(f'{"ALL":10} {tot[0]:3} {tot[1] / tot[0]:10.0%} {tot[2] / tot[0]:11.0%}')
    print("Missed:")
    for i, q, g in misses:
        print(" ", i, "|", q, "|", g)
        
