"""CLI: classify, eval, split, verify-audit, review."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import asdict

from .config import Settings
from .evals.metrics import check_gate, evaluate, load_dataset, retrieval_baseline, to_markdown
from .rag.corpus import load_jsonl, split_by_date
from .service import build_service


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="tariffpilot")
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("classify"); c.add_argument("text")
    e = sub.add_parser("eval")
    e.add_argument("--system", choices=["baseline", "agent", "both"], default="both")
    e.add_argument("--split", choices=["dev", "test"], default="dev")
    e.add_argument("--gate", action="store_true", help="exit 1 if the policy eval gate fails")
    e.add_argument("--report", help="write markdown report to this path")
    s = sub.add_parser("split"); s.add_argument("rulings"); s.add_argument("outdir")
    sub.add_parser("verify-audit")
    sub.add_parser("review-list")
    a = ap.parse_args(argv)

    if a.cmd == "split":
        parts = split_by_date(load_jsonl(a.rulings))
        for name, rows in parts.items():
            with open(f"{a.outdir}/rulings_{name}.jsonl", "w") as f:
                f.writelines(json.dumps(r) + "\n" for r in rows)
        print({k: len(v) for k, v in parts.items()})
        return 0

    svc = build_service(Settings.from_env())
    if a.cmd == "classify":
        print(json.dumps(svc.classify(a.text).to_dict(), indent=2))
    elif a.cmd == "verify-audit":
        ok, bad = svc.audit.verify_chain()
        print("audit chain OK" if ok else f"audit chain BROKEN at seq {bad}")
        return 0 if ok else 1
    elif a.cmd == "review-list":
        print(json.dumps(svc.review.pending(), indent=2))
    elif a.cmd == "eval":
        path = svc.settings.data_dir / "golden" / f"{a.split}.jsonl"
        if a.split == "test":
            want = (svc.settings.data_dir / "golden" / "test.jsonl.sha256").read_text().strip()
            if hashlib.sha256(path.read_bytes()).hexdigest() != want:
                print("FROZEN test set was modified; refusing to run", file=sys.stderr)
                return 2
        ds = load_dataset(path)
        results = {}
        if a.system in ("baseline", "both"):
            results["retrieval-only (no LLM)"] = evaluate(ds, retrieval_baseline(svc.corpus), svc.corpus)[0]
        if a.system in ("agent", "both"):
            label = f"agent ({svc.settings.mode})"
            results[label] = evaluate(ds, lambda t: svc.classify(t), svc.corpus)[0]
        md = to_markdown(f"{a.split} set", results)
        print(md)
        if a.report:
            open(a.report, "w").write(md)
        if a.gate:
            fails = [f for m in results.values() for f in check_gate(m, svc.policy.eval_gate)]
            if fails:
                print("GATE FAILED:", fails, file=sys.stderr)
                return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
