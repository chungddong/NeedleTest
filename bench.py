"""Benchmark one Needle 3 archive: accuracy, latency and memory on the test sets.

Usage: python bench.py --weights models/x.cact --label x [--extract] [--embed] [--scaling]
Writes results/<label>.json.
"""
import argparse
import json
import math
import os
import re
import statistics
import subprocess
import sys
import time

os.environ.setdefault("NEEDLE_TELEMETRY", "0")

import psutil
import needle

import testsets as T


# --------------------------------------------------------------------------- system probes

def vcgencmd(*args):
    try:
        return subprocess.run(["vcgencmd", *args], capture_output=True, text=True, timeout=5).stdout.strip()
    except Exception:
        return None


def proc_status(pid):
    """VmRSS / VmHWM (peak RSS) in MB for a pid."""
    out = {}
    try:
        with open(f"/proc/{pid}/status") as f:
            for line in f:
                if line.startswith(("VmRSS", "VmHWM")):
                    key, value = line.split(":")
                    out[key] = int(value.split()[0]) / 1024
    except OSError:
        pass
    return out


def worker_pid(agent):
    worker = getattr(agent, "_worker", None)
    process = getattr(worker, "_process", None)
    return process.pid if process else os.getpid()


# --------------------------------------------------------------------------- scoring

_ARTICLES = re.compile(r"^(the|a|an|my|some)\s+")


def norm(value):
    if isinstance(value, str):
        s = re.sub(r"\s+", " ", value.casefold().strip())
        s = s.strip(" .!?,'\"")
        return _ARTICLES.sub("", s)
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return value


def value_ok(want, got):
    if isinstance(want, T.OneOf):
        return any(value_ok(w, got) for w in want.values)
    if isinstance(want, T.Text):
        if not isinstance(got, str):
            return False
        w, g = norm(want.value), norm(got)
        return bool(g) and (w == g or w in g or g in w)
    if isinstance(want, (int, float)) and isinstance(got, (int, float)):
        return math.isclose(float(want), float(got), abs_tol=1e-6)
    return norm(want) == norm(got)


def args_ok(want, got):
    got = {k: v for k, v in (got or {}).items() if v is not None}
    return set(want) == set(got) and all(value_ok(want[k], got[k]) for k in want)


def match_calls(want, got):
    """Returns (names_ok, exact_ok) for an unordered multiset of calls."""
    names_ok = sorted(c["name"] for c in want) == sorted(c.get("name") for c in got)
    if not names_ok:
        return False, False
    remaining = list(got)
    for w in want:
        hit = next((g for g in remaining
                    if g.get("name") == w["name"] and args_ok(w["arguments"], g.get("arguments"))), None)
        if hit is None:
            return True, False
        remaining.remove(hit)
    return True, True


def gated(response, min_conf, strict):
    """Calls an app would act on. Confidence gate only, or with strict=True the full production
    contract from needle.environments._harness that also drops ungrounded/negated calls."""
    calls = response.get("function_calls") or []
    validation = response.get("validation") or {}
    if strict and calls and (validation.get("ungrounded") or validation.get("negation")):
        return []
    conf = response.get("confidence")
    if calls and conf is not None and conf < min_conf:
        return []
    return calls


def jsonable(x):
    if isinstance(x, T.OneOf):
        return {"one_of": list(x.values)}
    if isinstance(x, T.Text):
        return {"text~": x.value}
    if isinstance(x, dict):
        return {k: jsonable(v) for k, v in x.items()}
    if isinstance(x, list):
        return [jsonable(v) for v in x]
    return x


def pct(xs, p):
    xs = sorted(xs)
    if not xs:
        return None
    k = (len(xs) - 1) * p / 100
    lo, hi = math.floor(k), math.ceil(k)
    return xs[lo] + (xs[hi] - xs[lo]) * (k - lo)


def summarize_latency(ms):
    return {"n": len(ms), "mean": statistics.fmean(ms), "p50": pct(ms, 50), "p90": pct(ms, 90),
            "p99": pct(ms, 99), "min": min(ms), "max": max(ms)}


# --------------------------------------------------------------------------- suites

class Resilient:
    """An agent that rebuilds its engine worker after a crash, counting restarts.
    The failing call still raises, so the case is scored as an error."""

    def __init__(self, factory):
        self.factory, self.agent, self.restarts = factory, factory(), 0

    def _restart(self):
        try:
            self.agent.close()
        except Exception:
            pass
        self.agent = self.factory()
        self.restarts += 1

    def complete(self, text):
        try:
            return self.agent.complete(text)
        except Exception:
            self._restart()
            raise

    def reset(self):
        try:
            self.agent.reset()
        except Exception:
            self._restart()

    def embed(self, text):
        return self.agent.embed(text)

    def close(self):
        self.agent.close()

    @property
    def _worker(self):
        return self.agent._worker

GATE = 0.4


def run_toolcalls(agent, cases_by_suite, repeats=1):
    rows = []
    for suite, cases in cases_by_suite.items():
        for query, want in cases:
            lat, flags, outputs = [], [], []
            for _ in range(repeats):
                agent.reset()
                t0 = time.perf_counter()
                try:
                    r = agent.complete(query)
                except Exception as exc:  # e.g. the engine emitting invalid UTF-8
                    r = {"error": f"{type(exc).__name__}: {exc}"}
                lat.append((time.perf_counter() - t0) * 1000)
                raw = r.get("function_calls") or []
                outputs.append(json.dumps(raw, sort_keys=True, ensure_ascii=False))
                flags.append(match_calls(want, raw)
                             + match_calls(want, gated(r, GATE, strict=False))
                             + match_calls(want, gated(r, GATE, strict=True)))
            # each flag is the fraction of repeats that passed, so sums give mean accuracy
            frac = [sum(f[i] for f in flags) / len(flags) for i in range(6)]
            rows.append({
                "suite": suite, "query": query, "want": jsonable(want), "got": raw,
                "error": r.get("error"), "distinct_outputs": len(set(outputs)),
                "latency_ms": statistics.median(lat), "latency_all_ms": lat,
                "prefill_tps": r.get("prefill_tps"), "decode_tps": r.get("decode_tps"),
                "peak_ram_mb": r.get("peak_ram_mb"), "confidence": r.get("confidence"),
                "type": r.get("type"), "validation": r.get("validation"), "reasoning": r.get("reasoning"),
                "raw_tool_ok": frac[0], "raw_exact_ok": frac[1],
                "conf_tool_ok": frac[2], "conf_exact_ok": frac[3],
                "gated_tool_ok": frac[4], "gated_exact_ok": frac[5],
            })
    return rows


def score(rows):
    out = {}
    suites = sorted({r["suite"] for r in rows}, key=list(T.CASES).index)
    for s in suites + ["ALL"]:
        rs = [r for r in rows if s == "ALL" or r["suite"] == s]
        n = len(rs)
        out[s] = {
            "n": n,
            "raw_tool_acc": sum(r["raw_tool_ok"] for r in rs) / n,
            "raw_exact_acc": sum(r["raw_exact_ok"] for r in rs) / n,
            "conf_tool_acc": sum(r["conf_tool_ok"] for r in rs) / n,
            "conf_exact_acc": sum(r["conf_exact_ok"] for r in rs) / n,
            "gated_tool_acc": sum(r["gated_tool_ok"] for r in rs) / n,
            "gated_exact_acc": sum(r["gated_exact_ok"] for r in rs) / n,
            "unstable": sum(r["distinct_outputs"] > 1 for r in rs),
            "errors": sum(bool(r["error"]) for r in rs),
            "latency": summarize_latency([r["latency_ms"] for r in rs]),
        }
    return out


def run_extraction(weights):
    rows = []
    agents = {}
    try:
        for schema, text, want in T.EXTRACTION:
            name = schema._needle_tool["name"]
            if name not in agents:
                agents[name] = Resilient(lambda: needle.Needle(tools=[schema], weights=weights))
                agents[name].complete("warm up")
            agent = agents[name]
            agent.reset()
            t0 = time.perf_counter()
            try:
                r = agent.complete(text)
            except Exception as exc:
                r = {"error": f"{type(exc).__name__}: {exc}"}
            ms = (time.perf_counter() - t0) * 1000
            calls = r.get("function_calls") or r.get("suppressed_calls") or []
            got = calls[0].get("arguments") if calls else {}
            fields = {k: value_ok(v, (got or {}).get(k)) for k, v in want.items()}
            rows.append({"schema": name, "text": text, "want": jsonable(want), "got": got,
                         "fields_ok": fields, "all_ok": all(fields.values()), "latency_ms": ms})
    finally:
        for a in agents.values():
            a.close()
    nf = sum(len(r["fields_ok"]) for r in rows)
    return {
        "rows": rows,
        "field_acc": sum(sum(r["fields_ok"].values()) for r in rows) / nf,
        "record_acc": sum(r["all_ok"] for r in rows) / len(rows),
        "latency": summarize_latency([r["latency_ms"] for r in rows]),
    }


def cos(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    return dot / (math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b)) + 1e-12)


def run_embedding(agent):
    lat = []

    def emb(text):
        t0 = time.perf_counter()
        v = agent.embed(text)
        lat.append((time.perf_counter() - t0) * 1000)
        return v

    index = [(route, emb(ex)) for route, exs in T.ROUTES.items() for ex in exs]
    dim = len(index[0][1])
    centroids = {}
    for route in T.ROUTES:
        vs = [v for r, v in index if r == route]
        centroids[route] = [sum(col) / len(vs) for col in zip(*vs)]
    rows = []
    for q, want in T.ROUTE_QUERIES:
        v = emb(q)
        nn = max(index, key=lambda rv: cos(v, rv[1]))[0]
        cen = max(centroids, key=lambda r: cos(v, centroids[r]))
        rows.append({"query": q, "want": want, "nn": nn, "centroid": cen})
    return {
        "dim": dim, "rows": rows,
        "nn_acc": sum(r["nn"] == r["want"] for r in rows) / len(rows),
        "centroid_acc": sum(r["centroid"] == r["want"] for r in rows) / len(rows),
        "latency": summarize_latency(lat),
    }


def run_scaling(weights, counts):
    base_schemas = [t._needle_tool for t in T.TOOLS]
    subset = {k: T.CASES[k] for k in ("basic", "args", "refusal")}
    out = []
    for n in counts:
        tools = json.dumps(base_schemas + T.filler_tools(n))
        t0 = time.perf_counter()
        try:
            agent = needle.Needle(tools=tools, weights=weights)
            agent.complete("warm up")
        except Exception as exc:
            out.append({"n_tools": len(T.TOOLS) + n, "error": str(exc)})
            continue
        init_s = time.perf_counter() - t0
        pid = worker_pid(agent)
        rows = run_toolcalls(agent, subset)
        mem = proc_status(pid)
        agent.close()
        s = score(rows)["ALL"]
        out.append({"n_tools": len(T.TOOLS) + n, "tools_json_bytes": len(tools), "init_s": init_s,
                    "gated_exact_acc": s["gated_exact_acc"], "conf_exact_acc": s["conf_exact_acc"],
                    "raw_exact_acc": s["raw_exact_acc"],
                    "latency": s["latency"], "worker_mem_mb": mem,
                    "prefill_tps_median": statistics.median([r["prefill_tps"] for r in rows if r["prefill_tps"]]),
                    "failures": [{"q": r["query"], "got": r["got"], "conf": r["confidence"]}
                                 for r in rows if r["conf_exact_ok"] < 1]})
        print(f"  scaling n_tools={len(T.TOOLS) + n}: exact conf={s['conf_exact_acc']:.3f} "
              f"p50={s['latency']['p50']:.0f}ms", flush=True)
    return out


# --------------------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--extract", action="store_true")
    ap.add_argument("--embed", action="store_true")
    ap.add_argument("--scaling", action="store_true")
    args = ap.parse_args()

    result = {"label": args.label, "weights": args.weights,
              "weights_mb": os.path.getsize(args.weights) / 1e6,
              "needle_version": needle.__version__,
              "temp_start": vcgencmd("measure_temp"), "throttled_start": vcgencmd("get_throttled")}

    # cold start: spawn engine worker, load archive, bind tools, first inference
    t0 = time.perf_counter()
    agent = Resilient(lambda: needle.Needle(tools=T.TOOLS, weights=args.weights))
    t_init = time.perf_counter() - t0
    pid = worker_pid(agent)
    mem_loaded = proc_status(pid)
    t1 = time.perf_counter()
    agent.complete("warm up")
    t_first = time.perf_counter() - t1
    result["cold_start"] = {"init_s": t_init, "first_query_s": t_first}
    result["mem_after_load_mb"] = mem_loaded
    result["threads"] = psutil.Process(pid).num_threads()

    t0 = time.perf_counter()
    rows = run_toolcalls(agent, T.CASES, repeats=args.repeats)
    result["toolcall_wall_s"] = time.perf_counter() - t0
    result["toolcalls"] = rows
    result["scores"] = score(rows)
    result["prefill_tps_median"] = statistics.median([r["prefill_tps"] for r in rows if r["prefill_tps"]])
    result["decode_tps_median"] = statistics.median([r["decode_tps"] for r in rows if r["decode_tps"]])
    result["engine_peak_ram_mb"] = max(r["peak_ram_mb"] or 0 for r in rows)
    s = result["scores"]["ALL"]
    print(f"[{args.label}] toolcalls exact raw={s['raw_exact_acc']:.3f} conf={s['conf_exact_acc']:.3f} "
          f"strict={s['gated_exact_acc']:.3f} "
          f"p50={s['latency']['p50']:.0f}ms", flush=True)

    if args.embed:
        result["embedding"] = run_embedding(agent)
        print(f"[{args.label}] embed nn={result['embedding']['nn_acc']:.3f} "
              f"centroid={result['embedding']['centroid_acc']:.3f}", flush=True)
    result["worker_mem_mb"] = proc_status(worker_pid(agent))
    result["worker_restarts"] = agent.restarts
    agent.close()

    if args.extract:
        result["extraction"] = run_extraction(args.weights)
        print(f"[{args.label}] extract field={result['extraction']['field_acc']:.3f}", flush=True)
    if args.scaling:
        result["scaling"] = run_scaling(args.weights, [0, 10, 20, 40])

    result["temp_end"] = vcgencmd("measure_temp")
    result["throttled_end"] = vcgencmd("get_throttled")
    os.makedirs("results", exist_ok=True)
    with open(f"results/{args.label}.json", "w") as f:
        json.dump(result, f, ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
