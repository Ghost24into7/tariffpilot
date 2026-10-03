# Risk register (NIST AI RMF mapping)
| Risk | Function | Control in this repo |
|---|---|---|
| Hallucinated tariff code | Measure/Manage | output guardrail (leaf + citation + quote grounding), eval metric hallucination_rate = 0 gate |
| Prompt injection | Manage | input heuristics, fenced untrusted input, tool allowlist, no network/file tools |
| PII leakage | Govern/Manage | redaction before agent/log, audit stores input hash only |
| Over-reliance | Govern | confidence floor, needs_review queue, disclaimer on every result |
| Feedback poisoning | Manage | authorized reviewers, quarantine -> active gate, dedup, daily caps |
| Quota exhaustion | Manage | gateway limiter, budgets, typed degradation |
| Audit tampering | Govern | hash chain + verify-audit CLI |
| Free-tier data use | Govern | public-data-only policy, documented in README |
