"""Minimal tracer (zero-dependency) with an optional OpenTelemetry bridge."""
from __future__ import annotations

import contextvars
import json
import threading
import time
import uuid
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterator

_current: contextvars.ContextVar["Span | None"] = contextvars.ContextVar("tp_span", default=None)


@dataclass
class Span:
    name: str
    trace_id: str
    span_id: str
    parent_id: str | None
    start: float
    end: float = 0.0
    status: str = "ok"
    attrs: dict[str, Any] = field(default_factory=dict)

    @property
    def duration_ms(self) -> float:
        return round((self.end - self.start) * 1000, 3)


class InMemoryExporter:
    def __init__(self) -> None:
        self.spans: list[Span] = []

    def export(self, span: Span) -> None:
        self.spans.append(span)


class JsonlExporter:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def export(self, span: Span) -> None:
        rec = asdict(span)
        rec["duration_ms"] = span.duration_ms
        with self._lock, self.path.open("a") as f:
            f.write(json.dumps(rec, default=str) + "\n")


class OTelBridge:
    """Re-emits finished spans to OpenTelemetry if the SDK is installed (pip install .[otel])."""
    def __init__(self, service: str = "tariffpilot"):
        from opentelemetry import trace  # optional dependency
        self._tracer = trace.get_tracer(service)

    def export(self, span: Span) -> None:
        s = self._tracer.start_span(span.name, start_time=int(span.start * 1e9))
        for k, v in span.attrs.items():
            s.set_attribute(k, v if isinstance(v, (str, int, float, bool)) else str(v))
        s.set_attribute("tp.trace_id", span.trace_id)
        s.end(end_time=int(span.end * 1e9))


class Tracer:
    def __init__(self, exporters: list | None = None, clock=time.time):
        self.exporters = exporters if exporters is not None else []
        self.clock = clock

    @contextmanager
    def span(self, name: str, **attrs: Any) -> Iterator[Span]:
        parent = _current.get()
        s = Span(name, parent.trace_id if parent else uuid.uuid4().hex, uuid.uuid4().hex[:16],
                 parent.span_id if parent else None, self.clock(), attrs=dict(attrs))
        token = _current.set(s)
        try:
            yield s
        except Exception as e:
            s.status = "error"
            s.attrs["error"] = repr(e)[:300]
            raise
        finally:
            s.end = self.clock()
            _current.reset(token)
            for ex in self.exporters:
                ex.export(s)

    @staticmethod
    def current_trace_id() -> str:
        s = _current.get()
        return s.trace_id if s else ""
