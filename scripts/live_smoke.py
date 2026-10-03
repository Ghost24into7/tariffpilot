"""Budget-capped live check against Gemini (3 calls). Run: TP_MODE=gemini GEMINI_API_KEY=... python scripts/live_smoke.py"""
import os

os.environ.setdefault("TP_MODE", "gemini")
from tariffpilot.service import build_service  # noqa: E402

svc = build_service()
for text in ["Men's cotton knitted t-shirt, short sleeve", "Wireless bluetooth earbuds with charging case",
             "Stainless steel insulated water bottle 750 ml"]:
    r = svc.classify(text)
    print(r.status, r.hts_code, r.confidence, r.abstain_reason)
print(svc.gateway.stats, "quota left today:", svc.gateway.limiter.remaining_today)
