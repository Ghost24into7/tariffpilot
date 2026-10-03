"""Gemini client (google-genai). Imported lazily; not exercised by offline tests."""
from __future__ import annotations

from typing import Any

from .base import LLMError, TransientLLMError, Usage


class GeminiClient:
    def __init__(self, api_key: str):
        if not api_key:
            raise LLMError("GEMINI_API_KEY is not set")
        from google import genai  # pip install '.[gemini]'
        self._genai = genai
        self._client = genai.Client(api_key=api_key)

    def complete(self, *, model: str, system: str, user: str, schema: dict | None,
                 meta: dict[str, Any]) -> tuple[str, Usage]:
        types = self._genai.types
        cfg = types.GenerateContentConfig(
            system_instruction=system, response_mime_type="application/json",
            response_json_schema=schema, temperature=0.1)
        try:
            r = self._client.models.generate_content(model=model, contents=user, config=cfg)
        except Exception as e:  # map SDK errors onto our retry taxonomy
            msg = str(e)
            if any(t in msg for t in ("429", "RESOURCE_EXHAUSTED", "500", "503", "UNAVAILABLE", "DEADLINE")):
                raise TransientLLMError(msg[:200]) from e
            raise LLMError(msg[:200]) from e
        um = getattr(r, "usage_metadata", None)
        return (r.text or ""), Usage(getattr(um, "prompt_token_count", 0) or 0,
                                     getattr(um, "candidates_token_count", 0) or 0)
