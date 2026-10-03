# Security
Threat model (STRIDE summary): spoofing (API keys, reviewer allowlist), tampering (hash-chained audit), repudiation
(trace_id on every decision), information disclosure (PII redaction, hash-only audit input), DoS (per-key rate limit,
input size caps, per-request LLM budget), elevation (tool allowlist; no shell/network tools).
Never commit .env. Report vulnerabilities privately via GitHub security advisories.
