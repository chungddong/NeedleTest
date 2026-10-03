"""Latency, memory and accuracy of report_incident archives on the Pi's C runtime.

Starts bin/linux-arm64/needle --serve with the single report_incident tool (the
deployment shape), replays a dataset, and scores it the way try_model.py does.
Tuned archives run with no system prompt, as they were trained; the published base
gets the date fact the SDK would add (pass --date).

Usage (repo root, on the Pi):
  python ko/pi_bench.py models/needle3-ko-v2-20L.cact [...] --data ko/data/test_human.jsonl
Writes ko/results/pi-c-<model name>_<data name>.json.
"""
import argparse
import datetime
import json
import os
import statistics
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import bench
import cbench
from schema import TOOLS_JSON


def first_args(calls):
    return (calls[0].get("arguments") or {}) if calls else {}


def run(model, rows, repeats, threads, date):
    tools_path = os.path.join("ko", "out", "pi_tools.json")
    os.makedirs(os.path.dirname(tools_path), exist_ok=True)
    with open(tools_path, "w", encoding="utf-8") as f:
        json.dump(TOOLS_JSON, f, ensure_ascii=False)
    system_path = None
    if date:
        system_path = os.path.join("ko", "out", "pi_system.txt")
        with open(system_path, "w") as f:
            f.write(datetime.datetime.now().strftime("date: %Y-%m-%d %a %H:%M"))

    cbench.cool()
    temp_start = bench.vcgencmd("measure_temp")
    server = cbench.Server(None, threads, tools_path, system_path, model=model)
    t1 = time.perf_counter()
    server.complete("warm up")
    first_s = time.perf_counter() - t1

    results = []
    for row in rows:
        want, lat, outs = row["answers"], [], set()
        for _ in range(repeats):
            server.reset()
            t0 = time.perf_counter()
            r = server.complete(row["query"])
            lat.append((time.perf_counter() - t0) * 1000)
            got = r.get("function_calls") or []
            outs.add(json.dumps(got, sort_keys=True, ensure_ascii=False))
        w, g = first_args(want), first_args(got)
        results.append({
            "query": row["query"], "want": want, "got": got, "error": r.get("error"),
            "latency_ms": round(statistics.median(lat), 1), "distinct_outputs": len(outs),
            "prefill_tps": r.get("prefill_tps"), "decode_tps": r.get("decode_tps"),
            "exact": got == want, "decision": bool(got) == bool(want),
            "incident_type": bool(want) and g.get("incident_type") == w.get("incident_type"),
            "location": bool(want) and g.get("location") == w.get("location"),
        })
    mem = bench.proc_status(server.proc.pid)
    server.close()

    lat = [r["latency_ms"] for r in results]
    positives = [r for r in results if r["want"]]
    scores = {
        "n": len(results), "exact": sum(r["exact"] for r in results),
        "decision": sum(r["decision"] for r in results), "positives": len(positives),
        "incident_type": sum(r["incident_type"] for r in positives),
        "location": sum(r["location"] for r in positives),
        "errors": sum(bool(r["error"]) for r in results),
        "unstable": sum(r["distinct_outputs"] > 1 for r in results),
        "latency_p50_ms": round(statistics.median(lat), 1),
        "latency_p90_ms": round(bench.pct(lat, 90), 1), "latency_max_ms": round(max(lat), 1),
        "prefill_tps_median": statistics.median(r["prefill_tps"] for r in results if r["prefill_tps"]),
        "decode_tps_median": statistics.median(r["decode_tps"] for r in results if r["decode_tps"]),
        "peak_rss_mb": round(mem.get("VmHWM", 0), 1),
        "ready_s": round(server.ready_s, 2), "first_query_s": round(first_s, 2),
    }
    return {"model": model, "model_mb": round(os.path.getsize(model) / 1e6, 1), "runtime": "c",
            "threads": threads, "repeats": repeats, "date_fact": date, "scores": scores,
            "temp_start": temp_start, "temp_end": bench.vcgencmd("measure_temp"),
            "throttled_end": bench.vcgencmd("get_throttled"),
            "time": datetime.datetime.now().isoformat(timespec="seconds"), "rows": results}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("models", nargs="+")
    ap.add_argument("--data", default="ko/data/test_human.jsonl")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--date", action="store_true", help="add the SDK's date fact (published base)")
    args = ap.parse_args()
    rows = [json.loads(line) for line in open(args.data, encoding="utf-8")]
    data_name = os.path.splitext(os.path.basename(args.data))[0]
    for model in args.models:
        out = run(model, rows, args.repeats, args.threads, args.date)
        name = os.path.splitext(os.path.basename(model))[0]
        path = os.path.join("ko", "results", f"pi-c-{name}_{data_name}.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=1)
        s = out["scores"]
        print(f"{name}: exact {s['exact']}/{s['n']} type {s['incident_type']}/{s['positives']} "
              f"loc {s['location']}/{s['positives']} | p50 {s['latency_p50_ms']} ms "
              f"p90 {s['latency_p90_ms']} ms | rss {s['peak_rss_mb']} MB | {path}", flush=True)


if __name__ == "__main__":
    main()
