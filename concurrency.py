"""Aggregate throughput: K independent C servers sharing the Pi's 4 cores.

Each server gets its own client thread replaying the same query list, so this measures
how many independent queries per second the board sustains in each core split.
Writes results/concurrency.json.
"""
import json
import statistics
import threading
import time

import bench
import cbench
import testsets as T

QUERIES = [q for s in ("basic", "args", "casual") for q, _ in T.CASES[s]]
CONFIGS = [(1, 4), (2, 2), (4, 1), (1, 1)]  # (servers, threads per server)


def drive(server, latencies):
    for q in QUERIES:
        server.reset()
        t0 = time.perf_counter()
        server.complete(q)
        latencies.append((time.perf_counter() - t0) * 1000)


def main():
    out = []
    for n_servers, threads in CONFIGS:
        cbench.cool()
        servers = [cbench.Server(20, threads, "results/tools.json", "results/system.txt")
                   for _ in range(n_servers)]
        for s in servers:
            s.complete("warm up")
        lats = [[] for _ in servers]
        workers = [threading.Thread(target=drive, args=(s, l)) for s, l in zip(servers, lats)]
        t0 = time.perf_counter()
        for w in workers:
            w.start()
        for w in workers:
            w.join()
        wall = time.perf_counter() - t0
        mem = sum(bench.proc_status(s.proc.pid).get("VmHWM", 0) for s in servers)
        for s in servers:
            s.close()
        flat = [x for l in lats for x in l]
        row = {"servers": n_servers, "threads": threads, "queries": len(flat), "wall_s": wall,
               "qps": len(flat) / wall, "latency_p50": statistics.median(flat),
               "latency_p90": bench.pct(flat, 90), "total_peak_rss_mb": mem,
               "temp_end": bench.vcgencmd("measure_temp"), "throttled": bench.vcgencmd("get_throttled")}
        out.append(row)
        print(f"{n_servers} server(s) x {threads} thread(s): {row['qps']:.2f} q/s, "
              f"p50 {row['latency_p50']:.0f} ms, rss {mem:.0f} MB", flush=True)
    with open("results/concurrency.json", "w") as f:
        json.dump(out, f, indent=1)


if __name__ == "__main__":
    main()
