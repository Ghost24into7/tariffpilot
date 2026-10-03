"""Append-only, hash-chained audit log. Retention purges payloads but keeps the chain verifiable."""
from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
import time
from pathlib import Path

GENESIS = "0" * 64


def _h(*parts: str) -> str:
    return hashlib.sha256("|".join(parts).encode()).hexdigest()


class AuditLog:
    def __init__(self, path: Path | str):
        if str(path) != ":memory:":
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(str(path), check_same_thread=False)
        self._lock = threading.Lock()
        self.db.execute("""CREATE TABLE IF NOT EXISTS audit(
            seq INTEGER PRIMARY KEY AUTOINCREMENT, ts REAL, trace_id TEXT, payload TEXT,
            payload_hash TEXT, prev_hash TEXT, hash TEXT)""")
        self.db.commit()

    def append(self, trace_id: str, payload: dict, ts: float | None = None) -> str:
        with self._lock:
            ts = time.time() if ts is None else ts
            body = json.dumps(payload, sort_keys=True, default=str)
            ph = hashlib.sha256(body.encode()).hexdigest()
            row = self.db.execute("SELECT hash FROM audit ORDER BY seq DESC LIMIT 1").fetchone()
            prev = row[0] if row else GENESIS
            h = _h(prev, repr(ts), trace_id, ph)
            self.db.execute("INSERT INTO audit(ts,trace_id,payload,payload_hash,prev_hash,hash) VALUES(?,?,?,?,?,?)",
                            (ts, trace_id, body, ph, prev, h))
            self.db.commit()
            return h

    def verify_chain(self) -> tuple[bool, int | None]:
        """Returns (ok, first_bad_seq)."""
        prev = GENESIS
        for seq, ts, tid, payload, ph, pv, h in self.db.execute(
                "SELECT seq,ts,trace_id,payload,payload_hash,prev_hash,hash FROM audit ORDER BY seq"):
            if pv != prev or h != _h(prev, repr(ts), tid, ph):
                return False, seq
            if payload is not None and hashlib.sha256(payload.encode()).hexdigest() != ph:
                return False, seq
            prev = h
        return True, None

    def get(self, trace_id: str) -> list[dict]:
        rows = self.db.execute("SELECT payload FROM audit WHERE trace_id=? ORDER BY seq", (trace_id,)).fetchall()
        return [json.loads(r[0]) for r in rows if r[0] is not None]

    def purge_older_than(self, days: int, now: float | None = None) -> int:
        cutoff = (time.time() if now is None else now) - days * 86400
        with self._lock:
            cur = self.db.execute("UPDATE audit SET payload=NULL WHERE ts<? AND payload IS NOT NULL", (cutoff,))
            self.db.commit()
            return cur.rowcount

    def delete_trace_payload(self, trace_id: str) -> int:
        """Erasure request: drop the payload for one trace, keep the hash chain intact."""
        with self._lock:
            cur = self.db.execute("UPDATE audit SET payload=NULL WHERE trace_id=?", (trace_id,))
            self.db.commit()
            return cur.rowcount
