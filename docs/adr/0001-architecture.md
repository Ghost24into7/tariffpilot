# ADR 1: Zero-dependency core, thin optional adapters
Decision: the agent, retrieval, gateway, guardrails, audit and evals use only the stdlib + PyYAML. Gemini, FastAPI,
Streamlit and OpenTelemetry are optional adapters.
Why: the whole core is unit-testable offline in milliseconds, CI uses zero LLM quota, and free-tier limits cannot break builds.
Trade-off: a hand-written ReAct state machine instead of LangGraph. The loop is ~100 lines and fully tested;
swapping in LangGraph later only touches agent/loop.py.
