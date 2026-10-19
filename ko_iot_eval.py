"""Score a Needle archive on the Korean IoT command set (testsets_ko.py) with bench.py's scoring.

Usage: python ko_iot_eval.py --weights models/x.cact --label x [--repeats 3]
       (--weights - runs the published base model with the SDK's date fact)
Writes results/ko_iot_<label>.json.
"""
import argparse
import json
import os
import platform
import socket
import statistics

os.environ.setdefault("NEEDLE_TELEMETRY", "0")

import needle

import bench
import testsets as T
from testsets_ko import KO_CASES


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="-")
    ap.add_argument("--label", required=True)
    ap.add_argument("--repeats", type=int, default=3)
    args = ap.parse_args()
    weights = None if args.weights == "-" else args.weights
    # tuned archives were trained without a system turn, so no date fact (see CLAUDE.md)
    agent = bench.Resilient(lambda: needle.Needle(tools=T.TOOLS, weights=weights, auto_date=weights is None))
    agent.complete("warm up")
    pid = bench.worker_pid(agent)
    rows = bench.run_toolcalls(agent, KO_CASES, repeats=args.repeats)
    mem = bench.proc_status(pid)
    agent.close()
    scores = bench.score(rows)
    a = scores["ALL"]
    print(f"[{args.label}] exact raw={a['raw_exact_acc']:.3f} tool={a['raw_tool_acc']:.3f} "
          f"p50={a['latency']['p50']:.0f}ms errors={a['errors']}")
    for s, v in scores.items():
        if s != "ALL":
            print(f"  {s:<8} n={v['n']:>2} exact={v['raw_exact_acc']:.2f} tool={v['raw_tool_acc']:.2f}")
    os.makedirs("results", exist_ok=True)
    with open(f"results/ko_iot_{args.label}.json", "w", encoding="utf-8") as f:
        json.dump({"label": args.label, "weights": args.weights, "host": socket.gethostname(),
                   "platform": platform.platform(), "needle_version": needle.__version__,
                   "worker_mem_mb": mem, "scores": scores, "rows": rows,
                   "prefill_tps_median": statistics.median([r["prefill_tps"] for r in rows if r["prefill_tps"]] or [0]),
                   "decode_tps_median": statistics.median([r["decode_tps"] for r in rows if r["decode_tps"]] or [0])},
                  f, ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
