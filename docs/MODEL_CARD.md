# Model card
Intended use: decision support for HTS classification by small importers/exporters, with human review.
Not for: legal advice, customs filing, sanctions screening, or handling private customer data.
System: Gemini Flash/Flash-Lite (free tier) in a bounded ReAct loop over hybrid retrieval; all outputs schema-validated and grounded.
Known limits: only the loaded tariff subset is searchable; rates cover the general column only; classification depends
on the quality of the description; offline mode uses a deterministic heuristic, not an LLM.
Evaluation: see docs/EVALS.md (numbers there are on a tiny synthetic set until real CROSS data is ingested).
