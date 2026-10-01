"""Run a Needle archive on the Korean report queries and print what it calls.

Usage: python ko/try_model.py [weights.cact] [data.jsonl]
"""
import json
import os
import sys

os.environ.setdefault("NEEDLE_TELEMETRY", "0")
sys.path.insert(0, os.path.dirname(__file__))

import needle
from schema import TOOLS

weights = sys.argv[1] if len(sys.argv) > 1 and sys.argv[1] != "-" else None
data = sys.argv[2] if len(sys.argv) > 2 else "ko/data/smoke.jsonl"
agent = needle.Needle(tools=TOOLS, weights=weights)
ok = 0
rows = [json.loads(line) for line in open(data)]
for row in rows:
    agent.reset()
    try:
        r = agent.complete(row["query"])
        got, conf = r.get("function_calls") or [], r.get("confidence")
    except Exception as exc:
        got, conf = f"ERROR {type(exc).__name__}", None
    hit = got == row["answers"]
    ok += hit
    print(f"{'O' if hit else 'X'} conf={conf} | {row['query']} -> {json.dumps(got, ensure_ascii=False)}")
print(f"exact {ok}/{len(rows)}")
agent.close()
