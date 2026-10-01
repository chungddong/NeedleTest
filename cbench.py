"""Benchmark the standalone C runtime (bin/linux-arm64/needle --serve) on the tool-call test set.

Sweeps runtime depth (--depth, same 2-bit weights) and thread count (--threads), measuring
accuracy, latency and the server process's memory with no Python in the engine process.

Usage: python cbench.py --depths 2 4 ... 20 --threads 4 [--thread-sweep 1 2 3 4]
Writes results/c-d<depth>-t<threads>.json.
"""
import argparse
import datetime
import json
import os
import socket
import statistics
import subprocess
import time
import urllib.request

import bench
import testsets as T

BIN = "bin/linux-arm64/needle"
MODEL = "bin/linux-arm64/needle3.cact"


def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def post(port, path, body=None):
    req = urllib.request.Request(f"http://127.0.0.1:{port}{path}",
                                 data=json.dumps(body or {}).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as resp:
        raw = resp.read().decode()
    return json.loads(raw) if raw.strip() else {}


class Server:
    def __init__(self, depth, threads, tools_path, system_path):
        self.port = free_port()
        cmd = [BIN, "--model", MODEL, "--tools", tools_path, "--system", system_path,
               "--serve", "--port", str(self.port), "--threads", str(threads)]
        if depth:
            cmd += ["--depth", str(depth)]
        self.t0 = time.perf_counter()
        self.proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        while True:
            if self.proc.poll() is not None:
                raise RuntimeError(self.proc.stderr.read().decode())
            try:
                with socket.create_connection(("127.0.0.1", self.port), timeout=0.2):
                    break
            except OSError:
                time.sleep(0.02)
        self.ready_s = time.perf_counter() - self.t0

    def complete(self, text):
        return post(self.port, "/complete", {"input": text})

    def reset(self):
        post(self.port, "/reset")

    def close(self):
        self.proc.terminate()
        self.proc.wait()


def run(depth, threads, repeats):
    os.makedirs("results", exist_ok=True)
    tools_path, system_path = "results/tools.json", "results/system.txt"
    with open(tools_path, "w") as f:
        json.dump([t._needle_tool for t in T.TOOLS], f)
    with open(system_path, "w") as f:
        f.write(datetime.datetime.now().strftime("date: %Y-%m-%d %a %H:%M"))

    result = {"label": f"c-d{depth}-t{threads}", "runtime": "c", "depth": depth, "threads": threads,
              "temp_start": bench.vcgencmd("measure_temp")}
    server = Server(depth, threads, tools_path, system_path)
    t1 = time.perf_counter()
    server.complete("warm up")
    result["cold_start"] = {"ready_s": server.ready_s, "first_query_s": time.perf_counter() - t1}
    result["mem_after_load_mb"] = bench.proc_status(server.proc.pid)

    t0 = time.perf_counter()
    rows = bench.run_toolcalls(server, T.CASES, repeats=repeats)
    result["toolcall_wall_s"] = time.perf_counter() - t0
    result["server_mem_mb"] = bench.proc_status(server.proc.pid)
    server.close()

    result["toolcalls"] = rows
    result["scores"] = bench.score(rows)
    result["prefill_tps_median"] = statistics.median([r["prefill_tps"] for r in rows if r["prefill_tps"]])
    result["decode_tps_median"] = statistics.median([r["decode_tps"] for r in rows if r["decode_tps"]])
    result["engine_peak_ram_mb"] = max(r["peak_ram_mb"] or 0 for r in rows)
    result["temp_end"] = bench.vcgencmd("measure_temp")
    result["throttled_end"] = bench.vcgencmd("get_throttled")
    with open(f"results/{result['label']}.json", "w") as f:
        json.dump(result, f, ensure_ascii=False, indent=1)
    s = result["scores"]["ALL"]
    print(f"[{result['label']}] exact raw={s['raw_exact_acc']:.3f} conf={s['conf_exact_acc']:.3f} "
          f"strict={s['gated_exact_acc']:.3f} p50={s['latency']['p50']:.0f}ms "
          f"rss={result['server_mem_mb'].get('VmHWM', 0):.1f}MB", flush=True)


def cool(limit=60):
    while True:
        t = bench.vcgencmd("measure_temp") or ""
        try:
            if float(t.split("=")[1].split("'")[0]) < limit:
                return
        except (IndexError, ValueError):
            return
        time.sleep(10)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--depths", type=int, nargs="*", default=[])
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--thread-sweep", type=int, nargs="*", default=[])
    ap.add_argument("--repeats", type=int, default=3)
    args = ap.parse_args()
    for d in args.depths:
        cool()
        run(d, args.threads, args.repeats)
    for t in args.thread_sweep:
        cool()
        run(20, t, args.repeats)


if __name__ == "__main__":
    main()
