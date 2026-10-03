from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol


class LLMError(Exception): ...
class QuotaExhausted(LLMError): ...
class ReplayMiss(LLMError): ...
class CircuitOpen(LLMError): ...


class TransientLLMError(LLMError):
    """429 / 5xx style errors that are worth retrying."""
    def __init__(self, msg: str, retry_after: float | None = None):
        super().__init__(msg)
        self.retry_after = retry_after


@dataclass
class Usage:
    tokens_in: int = 0
    tokens_out: int = 0


class LLMClient(Protocol):
    def complete(self, *, model: str, system: str, user: str, schema: dict | None,
                 meta: dict[str, Any]) -> tuple[str, Usage]: ...
