"""Collect results/*.json into one compact dataset and render report.html from report_template.html."""
import glob
import json
import platform

import needle
from needle.agent import fetch


def load(name):
    with open(f"results/{name}.json") as f:
        return json.load(f)


def suite_scores(r):
    return {s: {k: round(v[k], 4) for k in ("raw_exact_acc", "conf_exact_acc", "gated_exact_acc",
                                             "raw_tool_acc", "conf_tool_acc")}
            | {"n": v["n"], "p50": round(v["latency"]["p50"]), "p90": round(v["latency"]["p90"]),
               "max": round(v["latency"]["max"])}
            for s, v in r["scores"].items()}


def main():
    off = load("20L-w2-official")
    c_runs = [json.load(open(p)) for p in glob.glob("results/c-*.json")]
    depth = sorted((r for r in c_runs if r["threads"] == 4), key=lambda r: r["depth"])
    threads = sorted((r for r in c_runs if r["depth"] == 20), key=lambda r: r["threads"])

    def c_row(r):
        a = r["scores"]["ALL"]
        return {"depth": r["depth"], "threads": r["threads"],
                "conf": a["conf_exact_acc"], "raw": a["raw_exact_acc"], "strict": a["gated_exact_acc"],
                "p50": a["latency"]["p50"], "p90": a["latency"]["p90"],
                "prefill": r["prefill_tps_median"], "decode": r["decode_tps_median"],
                "rss": r["server_mem_mb"]["VmHWM"], "ready": r["cold_start"]["ready_s"],
                "errors": a["errors"],
                "suites": {s: v["conf_exact_acc"] for s, v in r["scores"].items() if s != "ALL"}}

    builds = []
    for label, layers in [("4L-w4", 4), ("8L-w4", 8), ("12L-w4", 12), ("16L-w4", 16), ("20L-w4", 20),
                          ("20L-w2-official", 20)]:
        r = load(label)
        a = r["scores"]["ALL"]
        builds.append({"label": label, "layers": layers, "bits": 2 if "w2" in label else 4,
                       "mb": r["weights_mb"], "conf": a["conf_exact_acc"], "raw": a["raw_exact_acc"],
                       "p50": a["latency"]["p50"], "rss": r["worker_mem_mb"].get("VmHWM"),
                       "extract": r["extraction"]["field_acc"], "embed": r["embedding"]["centroid_acc"],
                       "errors": a["errors"], "restarts": r.get("worker_restarts") or 0})

    failures = [{"suite": x["suite"], "query": x["query"], "want": x["want"], "got": x["got"],
                 "conf": x["confidence"], "error": x["error"], "ok": x["conf_exact_ok"],
                 "strict_ok": x["gated_exact_ok"], "ungrounded": (x["validation"] or {}).get("ungrounded"),
                 "latency": round(x["latency_ms"])}
                for x in off["toolcalls"] if x["conf_exact_ok"] < 1 or x["gated_exact_ok"] < 1]

    all_lat = sorted(x["latency_ms"] for x in off["toolcalls"])
    data = {
        "meta": {"device": open("/proc/device-tree/model").read().strip("\x00 "),
                 "kernel": platform.release(), "python": platform.python_version(),
                 "sdk": needle.__version__, "engine": fetch.engine_version(3),
                 "cases": len(off["toolcalls"]), "repeats": len(off["toolcalls"][0]["latency_all_ms"]),
                 "model_mb": off["weights_mb"]},
        "official": {"suites": suite_scores(off), "prefill": off["prefill_tps_median"],
                     "decode": off["decode_tps_median"], "rss": off["worker_mem_mb"]["VmHWM"],
                     "cold": off["cold_start"], "latencies": [round(x) for x in all_lat],
                     "extraction": {k: off["extraction"][k] for k in ("field_acc", "record_acc")}
                                   | {"p50": off["extraction"]["latency"]["p50"],
                                      "fails": [{"text": x["text"], "got": x["got"]}
                                                for x in off["extraction"]["rows"] if not x["all_ok"]]},
                     "embedding": {k: off["embedding"][k] for k in ("dim", "nn_acc", "centroid_acc")}
                                  | {"p50": off["embedding"]["latency"]["p50"]},
                     "scaling": [{"n": s["n_tools"], "conf": s["conf_exact_acc"], "p50": s["latency"]["p50"],
                                  "rss": s["worker_mem_mb"]["VmHWM"], "init": s["init_s"]}
                                 for s in off["scaling"]]},
        "depth": [c_row(r) for r in depth],
        "threads": [c_row(r) for r in threads],
        "concurrency": json.load(open("results/concurrency.json")),
        "builds": builds,
        "failures": failures,
    }
    with open("results/report_data.json", "w") as f:
        json.dump(data, f, ensure_ascii=False, indent=1)
    html = open("report_template.html").read().replace("/*DATA*/null", json.dumps(data, ensure_ascii=False))
    with open("report.html", "w") as f:
        f.write(html)
    print("wrote report.html", len(html), "bytes;", len(failures), "failure rows")


if __name__ == "__main__":
    main()
