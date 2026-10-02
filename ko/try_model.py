"""Run a Needle archive on a Korean report dataset, print each call, and optionally save the scores.

Usage: python ko/try_model.py [weights.cact | -] [data.jsonl] [--out ko/results/name.json]
"-" (or no weights) runs the published base model.
"""
import argparse
import datetime
import json
import os
import platform
import socket
import statistics
import sys
import time

os.environ.setdefault("NEEDLE_TELEMETRY", "0")
sys.path.insert(0, os.path.dirname(__file__))

import needle
from schema import TOOLS


def first_args(calls):
    return (calls[0].get("arguments") or {}) if isinstance(calls, list) and calls else {}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("weights", nargs="?", default="-")
    ap.add_argument("data", nargs="?", default="ko/data/smoke.jsonl")
    ap.add_argument("--out", help="write per-row results and scores to this JSON file")
    ap.add_argument("--auto-date", action=argparse.BooleanOptionalAction, default=None,
                    help="SDK date fact; default on for the published model, off for a weights file")
    args = ap.parse_args()
    weights = None if args.weights == "-" else args.weights
    # Tuned weights saw no system turn in training, so skip the auto date fact
    # (as `needle finetune` advises); base-model archives pass --auto-date.
    auto_date = weights is None if args.auto_date is None else args.auto_date

    def make_agent():
        return needle.Needle(tools=TOOLS, weights=weights, auto_date=auto_date)

    agent = make_agent()
    rows = [json.loads(line) for line in open(args.data, encoding="utf-8")]
    results = []
    for row in rows:
        want = row["answers"]
        try:
            agent.reset()
        except Exception:
            agent = make_agent()
        t0 = time.perf_counter()
        try:
            r = agent.complete(row["query"])
            got, conf, error = r.get("function_calls") or [], r.get("confidence"), r.get("error")
            thought = r.get("reasoning")
        except Exception as exc:  # e.g. the engine emitting invalid UTF-8 kills the worker
            got, conf, error, thought = [], None, f"{type(exc).__name__}: {exc}", None
            agent = make_agent()
        ms = (time.perf_counter() - t0) * 1000
        w, g = first_args(want), first_args(got)
        result = {
            "query": row["query"], "want": want, "got": got, "reasoning": thought,
            "confidence": conf, "error": error,
            "latency_ms": round(ms, 1),
            "exact": got == want,
            "decision": bool(got) == bool(want),  # reported vs refused, right call either way
            "incident_type": bool(want) and g.get("incident_type") == w.get("incident_type"),
            "location": bool(want) and g.get("location") == w.get("location"),
        }
        results.append(result)
        mark = "O" if result["exact"] else "X"
        shown = error if error else json.dumps(got, ensure_ascii=False)
        print(f"{mark} conf={conf} | {row['query']} -> {shown}")
    agent.close()

    positives = [r for r in results if r["want"]]
    scores = {
        "n": len(results),
        "exact": sum(r["exact"] for r in results),
        "decision": sum(r["decision"] for r in results),
        "positives": len(positives),
        "incident_type": sum(r["incident_type"] for r in positives),
        "location": sum(r["location"] for r in positives),
        "errors": sum(bool(r["error"]) for r in results),
        "latency_p50_ms": round(statistics.median(r["latency_ms"] for r in results), 1),
    }
    print(f"exact {scores['exact']}/{scores['n']} | decision {scores['decision']}/{scores['n']} | "
          f"incident_type {scores['incident_type']}/{scores['positives']} | "
          f"location {scores['location']}/{scores['positives']} | errors {scores['errors']}")

    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump({
                "weights": args.weights, "data": args.data, "auto_date": auto_date, "scores": scores,
                "host": socket.gethostname(), "platform": platform.platform(),
                "needle_version": needle.__version__,
                "time": datetime.datetime.now().isoformat(timespec="seconds"),
                "rows": results,
            }, f, ensure_ascii=False, indent=1)
        print(f"saved {args.out}")


if __name__ == "__main__":
    main()
