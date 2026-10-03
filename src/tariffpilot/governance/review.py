"""Human review queue. Approvals/corrections become quarantined precedents (never auto-active)."""
from __future__ import annotations

import json
import sqlite3
import threading
import time
from pathlib import Path

from ..learning.precedents import PrecedentStore
from .policy import Policy


class ReviewQueue:
    def __init__(self, path: Path | str, policy: Policy, precedents: PrecedentStore):
        if str(path) != ":memory:":
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(str(path), check_same_thread=False)
        self.policy, self.precedents, self._lock = policy, precedents, threading.Lock()
        self.db.execute("""CREATE TABLE IF NOT EXISTS review(
            id INTEGER PRIMARY KEY AUTOINCREMENT, trace_id TEXT, product TEXT, result TEXT, status TEXT,
            reviewer TEXT, correct_code TEXT, rationale TEXT, created REAL, resolved REAL)""")
        self.db.commit()

    def enqueue(self, trace_id: str, product: str, result: dict) -> int:
        with self._lock:
            cur = self.db.execute("INSERT INTO review(trace_id,product,result,status,created) VALUES(?,?,?,?,?)",
                                  (trace_id, product, json.dumps(result), "pending", time.time()))
            self.db.commit()
            return int(cur.lastrowid)

    def pending(self) -> list[dict]:
        return [{"id": r[0], "trace_id": r[1], "product": r[2], "result": json.loads(r[3])}
                for r in self.db.execute("SELECT id,trace_id,product,result FROM review WHERE status='pending'")]

    def resolve(self, item_id: int, reviewer: str, action: str, correct_code: str | None = None,
                rationale: str = "") -> str | None:
        if reviewer not in self.policy.authorized_reviewers:
            raise PermissionError(f"reviewer '{reviewer}' is not authorized")
        if action not in ("approve", "correct", "reject"):
            raise ValueError("action must be approve|correct|reject")
        row = self.db.execute("SELECT product,result,status FROM review WHERE id=?", (item_id,)).fetchone()
        if not row or row[2] != "pending":
            raise KeyError(f"review item {item_id} not pending")
        product, result = row[0], json.loads(row[1])
        code = correct_code if action == "correct" else result.get("hts_code")
        if action == "correct" and not code:
            raise ValueError("correct_code required for action=correct")
        pid = None
        if action != "reject" and code:
            pid = self.precedents.add(product, code, rationale, reviewer)
        with self._lock:
            self.db.execute("UPDATE review SET status=?,reviewer=?,correct_code=?,rationale=?,resolved=? WHERE id=?",
                            (action, reviewer, code, rationale, time.time(), item_id))
            self.db.commit()
        return pid
