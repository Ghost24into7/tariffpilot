"""FastAPI surface. Requires: pip install '.[api]'. API-key auth, per-key rate limit, SSE streaming."""
from __future__ import annotations

import json
import queue
import threading
import time
from collections import defaultdict, deque

from ..service import Service, build_service


def create_app(service: Service | None = None):
    from fastapi import Depends, FastAPI, Header, HTTPException
    from fastapi.responses import StreamingResponse
    from pydantic import BaseModel, Field

    svc = service or build_service()
    app = FastAPI(title="TariffPilot", version="0.1.0",
                  description="Decision support for customs classification. Not legal advice.")
    hits: dict[str, deque] = defaultdict(deque)

    def auth(x_api_key: str = Header(default="")) -> str:
        keys = svc.settings.api_keys
        if keys and x_api_key not in keys:
            raise HTTPException(401, "invalid API key")
        who = x_api_key or "anonymous"
        now, q = time.time(), hits[who]
        while q and now - q[0] > 60:
            q.popleft()
        if len(q) >= svc.settings.api_rate_per_min:
            raise HTTPException(429, "rate limit exceeded")
        q.append(now)
        return who

    def reviewer(x_reviewer: str = Header(default="")) -> str:
        if x_reviewer not in svc.policy.authorized_reviewers:
            raise HTTPException(403, "not an authorized reviewer")
        return x_reviewer

    class ClassifyIn(BaseModel):
        text: str = Field(min_length=1, max_length=2000)
        clarifications: str = Field(default="", max_length=1000)

    class ResolveIn(BaseModel):
        action: str
        correct_code: str | None = None
        rationale: str = Field(default="", max_length=500)

    @app.get("/healthz")
    def healthz():
        ok, _ = svc.audit.verify_chain()
        return {"status": "ok", "audit_chain": ok, "mode": svc.settings.mode,
                "llm_quota_remaining_today": svc.gateway.limiter.remaining_today}

    @app.get("/metrics")
    def metrics():
        return {**svc.gateway.stats, "llm_quota_remaining_today": svc.gateway.limiter.remaining_today,
                "review_pending": len(svc.review.pending())}

    @app.post("/classify")
    def classify(body: ClassifyIn, _: str = Depends(auth)):
        return svc.classify(body.text, body.clarifications).to_dict()

    @app.post("/classify/stream")
    def classify_stream(body: ClassifyIn, _: str = Depends(auth)):
        q: queue.Queue = queue.Queue()

        def work():
            try:
                res = svc.classify(body.text, body.clarifications, on_event=lambda e: q.put(("step", e)))
                q.put(("result", res.to_dict()))
            except Exception as e:  # surface failures to the stream instead of hanging it
                q.put(("error", {"error": type(e).__name__}))
            q.put(None)

        threading.Thread(target=work, daemon=True).start()

        def gen():
            while (item := q.get()) is not None:
                yield f"event: {item[0]}\ndata: {json.dumps(item[1])}\n\n"
        return StreamingResponse(gen(), media_type="text/event-stream")

    @app.get("/review-queue")
    def review_queue(_: str = Depends(reviewer)):
        return svc.review.pending()

    @app.post("/review/{item_id}")
    def resolve(item_id: int, body: ResolveIn, who: str = Depends(reviewer)):
        try:
            return {"precedent_id": svc.review.resolve(item_id, who, body.action, body.correct_code, body.rationale)}
        except (KeyError, ValueError) as e:
            raise HTTPException(400, str(e))

    @app.get("/trace/{trace_id}")
    def trace(trace_id: str, _: str = Depends(reviewer)):
        return svc.audit.get(trace_id)

    return app


app = None  # create lazily: `uvicorn tariffpilot.api.app:create_app --factory`
