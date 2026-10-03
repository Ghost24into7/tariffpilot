# Evals
Datasets: data/golden/dev.jsonl (12) and the frozen test.jsonl (10): **synthetic cases over a 17-line sample tariff**.
These prove the harness and gate work; they are NOT evidence of real-world accuracy. Replace with a date-split of real
CBP CROSS rulings (`tariffpilot split`) and rerun before quoting any number.

### dev set

| system | accuracy_2 | accuracy_4 | accuracy_6 | accuracy_8 | auto_accept_precision | coverage | hallucination_rate | citation_faithfulness | abstention_recall | mean_steps |
|---|---|---|---|---|---|---|---|---|---|---|
| retrieval-only (no LLM) | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 0.0 | 1.0 | 0.5 | 0.0 |
| agent (offline) | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 0.9 | 0.0 | 1.0 | 1.0 | 1.83 |

### test set

| system | accuracy_2 | accuracy_4 | accuracy_6 | accuracy_8 | auto_accept_precision | coverage | hallucination_rate | citation_faithfulness | abstention_recall | mean_steps |
|---|---|---|---|---|---|---|---|---|---|---|
| retrieval-only (no LLM) | 0.8889 | 0.8889 | 0.7778 | 0.7778 | 0.7778 | 1.0 | 0.0 | 1.0 | 1.0 | 0.0 |
| agent (offline) | 0.8889 | 0.8889 | 0.7778 | 0.7778 | 0.875 | 0.8889 | 0.0 | 1.0 | 1.0 | 1.9 |

"agent (offline)" is the deterministic heuristic policy running through the full agent/guardrail/audit stack.
Run `TP_MODE=gemini make eval` for real-LLM numbers (responses are cached, so later runs use `TP_MODE=replay` at zero quota).
