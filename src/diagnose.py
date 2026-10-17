import csv
import json
import re
import sys

norm = lambda s: re.sub(r"\s+", " ", s.lower().replace(",", ""))
chunks = [json.loads(l) for l in open("data/processed/chunks.jsonl", encoding="utf-8")]
rows = {r["id"]: r for r in csv.DictReader(open("eval/questions.csv", encoding="utf-8"))}

for qid in sys.argv[1:]:
    r = rows[qid]
    phrase = norm(r["must_contain"])
    in_doc = [c for c in chunks if c["doc"] == r["source_doc"]]
    hits = [c["page"] for c in in_doc if phrase in norm(c["text"])]
    print(f'{qid} | {r["question"]}')
    print(f'   phrase: "{r["must_contain"]}"  in doc: {r["source_doc"]} ({len(in_doc)} chunks)')
    if hits:
        print(f"   phrase IS in {len(hits)} chunk(s), starting on PDF pages {hits}")
        print("   -> search can find it but ranked it outside the top 5 (try reranker or more chunks)")
    else:
        parts = [w for w in phrase.split()[:3]]
        near = [c["page"] for c in in_doc if " ".join(parts) in norm(c["text"])]
        print("   phrase is in NO chunk.")
        if near:
            print(f"   first words appear on pages {near}: likely split across a chunk boundary")
        else:
            print("   -> table text may be scrambled, or the phrase is worded differently. Search the .txt file for it.")