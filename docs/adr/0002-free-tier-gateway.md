# ADR 2: Quota-aware LLM gateway
All model calls pass through Gateway: sliding-window RPM + daily cap, disk cache (doubles as record/replay), retry with
jitter and retry-after, circuit breaker, per-request call budget, Flash-Lite/Flash routing, schema repair (1 retry).
Quota exhaustion degrades to a typed abstention, never a crash. Limits come from env/AI Studio, not code.
