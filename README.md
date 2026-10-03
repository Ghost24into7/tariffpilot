# TariffPilot

Agentic decision support for **customs (HTS) classification**, aimed at small importers/exporters who can't afford enterprise
trade-compliance tools. Built to run on free tiers (Gemini API free tier, GitHub Student/Free, GitHub Actions).

```
request -> input guardrails (injection, PII) -> ReAct agent (tools, skills, precedents)
        -> output guardrails (leaf code, citation + quote grounding, confidence floor, restrictions)
        -> hash-chained audit log -> human review queue -> quarantined precedents -> promotion gate
LLM calls: Gateway (rate limit, cache/replay, retry, breaker, budgets, Flash/Flash-Lite routing)
```

## Quick start (no API key needed)
```bash
pip install -e ".[dev]"
make test                 # 69 offline tests
make demo                 # classify with the offline heuristic policy
make gate                 # eval regression gate used in CI
```
`TP_MODE=offline` uses a deterministic stand-in policy (not an LLM) so everything runs with zero quota.

## Go live (Gemini free tier)
1. `cp .env.example .env`, set `GEMINI_API_KEY`, confirm model IDs and your live RPM/RPD in AI Studio, update `policy/policy.yaml` `allowed_models`.
2. `pip install -e ".[gemini,api,ui]"` then `python scripts/live_smoke.py` (3 calls).
3. `make eval` with `TP_MODE=gemini`; later runs with `TP_MODE=replay` cost nothing.
4. Real data: put the USITC HTS JSON export at `data/raw/hts.json` and CBP CROSS rulings as JSONL (`id,date,hts_code,text`) at
   `data/raw/rulings.jsonl`; `python -m tariffpilot split data/raw/rulings.jsonl data/golden/` builds a chronological split.
5. `make run` (API, docs at /docs) or `make ui`. Optional tracing: `pip install -e ".[otel]"` and `docker compose --profile obs up`.

## What is implemented
Gateway · hybrid BM25+vector RAG with RRF · ReAct loop with duplicate/loop detection, clarification interrupt and bounded steps ·
SKILL.md progressive loading · versioned prompt registry · structured outputs · input/tool/output guardrails (40-case red-team set) ·
policy-as-code · hash-chained audit with retention/erasure · review queue · precedent quarantine/promotion gate · reflection ·
evals with frozen test hash and CI gate · tracing (JSONL + optional OTel) · FastAPI (auth, rate limit, SSE) · Streamlit · Docker · CI.

## Honest limitations
- Bundled data is a **17-line sample tariff and synthetic rulings**; eval numbers are for harness validation only. Sample duty rates are fixtures: verify against the current USITC schedule.
- The Gemini client, Gemini embedder, FastAPI app, Streamlit UI and OTel bridge were written but **not executed** in the build environment (no network or packages there). Expect small fixes on first run.
- The agent loop is hand-written (not LangGraph) by design; see docs/adr. No LLM-as-judge yet.
- Free-tier Gemini prompts may be used by Google for training: public data only. Decision support, not legal advice.
