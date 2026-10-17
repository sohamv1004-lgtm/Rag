import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder, SentenceTransformer

EMBED_MODEL = "BAAI/bge-small-en-v1.5"
RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
CANDIDATES = 50

tok = lambda s: re.findall(r"\w+", s.lower())


class Retriever:
    """Hybrid (BM25 + embeddings) search followed by a cross-encoder reranker."""

    def __init__(self):
        self.chunks = [json.loads(l) for l in open("data/processed/chunks.jsonl", encoding="utf-8")]
        self.bm25 = BM25Okapi([tok(c["text"]) for c in self.chunks])
        self.embedder = SentenceTransformer(EMBED_MODEL)
        cache = Path("data/processed/embeddings.npy")
        emb = np.load(cache) if cache.exists() else None
        if emb is None or emb.shape[0] != len(self.chunks):
            emb = self.embedder.encode([c["text"] for c in self.chunks], normalize_embeddings=True)
            np.save(cache, emb)
        self.emb = emb
        self.reranker = CrossEncoder(RERANK_MODEL, max_length=512)

    def _hybrid(self, q):
        bm = list(np.argsort(-self.bm25.get_scores(tok(q))))
        v = self.embedder.encode([QUERY_PREFIX + q], normalize_embeddings=True)[0]
        de = list(np.argsort(-(self.emb @ v)))
        scores = defaultdict(float)
        for ranking in (bm, de):
            for pos, i in enumerate(ranking[:100]):
                scores[int(i)] += 1 / (60 + pos + 1)
        return sorted(scores, key=lambda i: -scores[i])

    def retrieve(self, question, k=8):
        cand = self._hybrid(question)[:CANDIDATES]
        ce = self.reranker.predict([(question, self.chunks[i]["text"]) for i in cand])
        order = np.argsort(-ce)
        return [self.chunks[cand[int(j)]] for j in order[:k]]
        