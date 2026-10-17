"""Run the LLM on the eval questions and save every answer to eval/answers_<split>.csv.
Usage (from the Rag folder):  python src/run_answers.py [dev|holdout]
Safe to stop (Ctrl+C) and rerun: finished questions are skipped."""
import csv, os, re, sys, time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from answer import ask, make_client, build_prompt, StopRun, REFUSAL, K, MODEL, PROVIDER
from retriever import Retriever

BASE = Path(__file__).resolve().parent.parent
SPLIT = sys.argv[1] if len(sys.argv) > 1 else "dev"
IN_FILE = BASE / "eval" / "questions.csv"
OUT_FILE = BASE / "eval" / f"answers_{SPLIT}.csv"
FIELDS = ["id", "type", "question", "expected_answer", "model", "answer", "verdict"]
PAUSE = 4  # seconds between calls


def norm(s):
    return re.sub(r"\s+", " ", (s or "").lower().replace(",", "")).strip()


def verdict(q, answer):
    refused = REFUSAL.lower() in answer.lower()
    if q["type"] == "refuse":
        return "pass" if refused else "FAIL"
    if refused:
        return "FAIL (refused)"
    if norm(q["must_contain"]) and norm(q["must_contain"]) in norm(answer):
        return "pass"
    return "check by hand"


def read_done():
    if not OUT_FILE.exists():
        return []
    with open(OUT_FILE, newline="", encoding="utf-8-sig") as f:
        return [r for r in csv.DictReader(f)
                if r.get("answer") and not r["answer"].startswith("ERROR")]


def write_all(rows):
    tmp = OUT_FILE.with_suffix(".tmp")
    with open(tmp, "w", newline="", encoding="utf-8-sig") as f:   # utf-8-sig so Excel opens it correctly
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)
    os.replace(tmp, OUT_FILE)


def summary(rows):
    counts = defaultdict(lambda: defaultdict(int))
    for r in rows:
        counts[r["type"]][r["verdict"]] += 1
    print("\nSummary by type:")
    for t, c in counts.items():
        print(f"  {t}: {dict(c)}")


def main():
    with open(IN_FILE, newline="", encoding="utf-8-sig") as f:
        questions = [r for r in csv.DictReader(f) if r["split"].strip() == SPLIT]
    rows = read_done()
    write_all(rows)                      # file exists from the very start
    done_ids = {r["id"] for r in rows}
    todo = [q for q in questions if q["id"] not in done_ids]
    print(f"Model: {PROVIDER}/{MODEL}")
    print(f"Saving to: {OUT_FILE}")
    print(f"{len(rows)} already done, {len(todo)} to do ({SPLIT}, {len(questions)} questions)\n")

    retriever = Retriever()
    client = make_client()
    failed = 0
    for q in todo:
        chunks = retriever.retrieve(q["question"], k=K)
        try:
            answer = ask(client, build_prompt(q["question"], chunks))
        except StopRun as e:
            print(f"\nSTOPPED: {str(e)[:300]}")
            print(f"Progress is saved ({len(rows)} rows). Fix the problem and run the same command again.")
            summary(rows)
            return
        if answer.startswith("ERROR"):
            failed += 1
            print(f"{q['id']}: {answer[:200]}  (not saved, will retry next run)")
            continue
        rows.append({"id": q["id"], "type": q["type"], "question": q["question"],
                     "expected_answer": q["expected_answer"], "model": f"{PROVIDER}/{MODEL}",
                     "answer": " ".join(answer.split()), "verdict": verdict(q, answer)})
        write_all(rows)
        print(f"{q['id']:5} {q['type']:9} {rows[-1]['verdict']}\n      {rows[-1]['answer'][:250]}\n")
        time.sleep(PAUSE)

    if failed:
        print(f"{failed} question(s) failed temporarily; run again to finish them.")
    summary(rows)


if __name__ == "__main__":
    main()
    