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
  