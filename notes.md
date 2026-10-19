Query ‘category certificate mandatory’: 2023 FAQ ranked 1st (15.9), 2026 FAQ 2nd (15.3). Retrieval finds both, but can’t tell which is current.
Baseline-
type         n  right doc  fact found
conflict     4        75%         75%
lookup      32        94%         69%
two_docs     1       100%        100%
ALL         37        92%         70%
The initial BM-25 fails and retrieves wrong documents for the missed cases.
Missed (fact not in top 5):
  Q1b | How many IITs are there in India? | ['jee_main_26_faq p5', 'jee_main_2023_faq p9', 'jee_main_2023_faq p5']
  === HYBRID + RERANK (TOP 50) (top-5, dev questions only) ===
type         n  right doc  fact found
conflict     4       100%         25%
lookup      32       100%         88%
two_docs     1       100%        100%
ALL         37       100%         81%

=== HYBRID (top-5, dev questions only) ===
type         n  right doc  fact found
conflict     4        75%         75%
lookup      32        97%         78%
two_docs     1       100%        100%
ALL         37        95%         78%
Update-Hybrid outperformed the rerank one initially.

Hybrid+rerank+labels makes it 86% and fetching top 8 makes it 89%
After evaluation using k=8:
Majority of them passed using Nvidia Api and a few were manually checked and 2 failed whose reasons are given in the csv file.
How the llm performed on the holdout set
Summary by type:
  two_docs: {'pass': 1, 'check by hand': 1}
  lookup: {'check by hand': 2, 'FAIL (refused)': 1, 'pass': 2}
  conflict: {'pass': 1}
  refuse: {'pass': 2}
### Failure: Q26 (retrieval)
search can find it but ranked it outside the top 8 (confirmed with diagnose.py).
Ideas not yet tested:
- Larger candidate pool or a stronger reranker model
- LLM query rewriting before retrieval
- HyDE (embed a model-written hypothetical answer instead of the raw question)
Holdout (8 retrieval-scored questions, top-8):
 hybrid        6/8 fact found, 8/8 right doc
 hybrid+rerank 5/8 fact found, 7/8 right doc
Small sample; one question = 12.5 points. Reranker gain from dev did not replicate on holdout.
Table heavy chunks were extracted poorly(Q22 AND Q17)
=== HYBRID + RERANK (TOP 50) (top-8, holdout questions only) ===
type         n      right doc     fact found
conflict     1    1/1   100%    1/1   100%
lookup       5    4/5    80%    3/5    60%
two_docs     2    2/2   100%    1/2    50%
ALL          8    7/8    88%    5/8    62%





