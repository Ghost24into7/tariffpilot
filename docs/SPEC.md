# TariffPilot spec
User describes a product. System returns a ClassificationResult (schemas.py): status (classified | needs_review |
needs_info | abstained | blocked), hts_code, confidence, GRI path, citations, alternatives, duty estimate,
restrictions, trace_id, disclaimer.
Policy: abstain or escalate on low confidence, conflicting evidence, missing decisive attributes, dual-use/hazard words.
Never emit a code that is not a leaf in the loaded tariff. Every citation must resolve and every quote must appear in its source.
Non-goals: legal advice, filing entries, handling real customer data (free-tier Gemini prompts may be used for training).
