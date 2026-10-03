# Progress
Done: core, gateway, hybrid RAG, ReAct agent, skills, prompts, guardrails, governance (audit/review/policy), precedents
with promotion gate, evals + CI gate, tracing, API, UI, Docker, CI. 69 offline tests pass.
Next (needs your machine/accounts, see README "Go live"): ingest real USITC HTS + CBP CROSS, run scripts/live_smoke.py,
record a real eval run in replay cache, add LLM-judge for reasoning quality, deploy to a free host.
Known: sample data is a 17-line HTS subset + synthetic rulings; Gemini/FastAPI/Streamlit/OTel code paths are unexecuted.
