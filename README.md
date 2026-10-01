# Needle 3 on Raspberry Pi 5

Accuracy, latency and memory benchmarks for [Cactus Compute's Needle 3](https://github.com/cactus-compute/needle)
tool-calling model on a Raspberry Pi 5 (16 GB), using hand-written test sets.

**Full results, findings and run log (Korean): [RESULTS.md](RESULTS.md)**. The charted report is
`report.html`; download it and open it in a browser, since GitHub shows HTML as source.

## Key results

Official 2-bit, 20-layer `needle3.cact` (35.3 MB), measured 2026-10-01:

| | |
|---|---|
| English tool-call accuracy | 90% (80 queries, confidence >= 0.4) |
| Korean tool-call accuracy | 17% (12 queries) |
| Latency | p50 361 ms, p90 627 ms, worst 4.7 s |
| Peak memory | 74 MB (C runtime), 112 MB (Python SDK) |
| Structured extraction | 98% of fields |

- The advertised ~4,000 tok/s on a Pi 5 matches prefill at `--depth 2`, where accuracy is near zero. Depth 18-20 is the usable range.
- `--depth` does not reduce memory; the whole archive stays loaded.
- Accuracy drops from 91% to about 65% once more than 30 tools are registered.
- The SDK's grounding check rejects correct conversions such as "2 hours" -> `minutes=120`.

## Test sets (`testsets.py`)

- 11 tools (weather, timer, alarm, message, call, music, navigation, calendar, lights, thermostat, volume)
- 92 tool-call queries: basic 20, args 20, casual 15, multi 10, refusal 15, korean 12
- 16 structured-extraction records, 24 embedding routing queries, tool-count scaling (11 to 51 tools)

## Setup

```bash
uv venv .venv
uv pip install --python .venv/bin/python -r requirements.txt
export NEEDLE_TELEMETRY=0

# standalone C runtime and the official 2-bit weights
.venv/bin/needle download linux-arm64 --out bin
mkdir -p models && cp bin/linux-arm64/needle3.cact models/needle3-20L-w2-official.cact

# 4-bit layer rungs from the public exporter (needs JAX, about 30-70 s each)
.venv/bin/python build_rungs.py 4 8 12 16 20
```

## Run

```bash
./run_all.sh                                   # Python SDK: official model + W4 rungs
.venv/bin/python cbench.py --depths 2 4 6 8 10 12 14 16 18 20 --thread-sweep 1 2 3
.venv/bin/python concurrency.py                # aggregate throughput, 1-4 servers
.venv/bin/python gen_report.py                 # results/*.json -> report.html
```

| Script | What it measures |
|---|---|
| `bench.py` | One `.cact` archive through the Python SDK: tool calls, extraction, embeddings, tool-count scaling |
| `cbench.py` | The C runtime (`needle --serve`) across `--depth` and `--threads` |
| `concurrency.py` | Queries per second with several servers sharing the 4 cores |
| `smoke.py` | Quick check that the engine loads and answers |

Scoring has three views: **raw** model output, **gated** (confidence >= 0.4), and **strict**
(gated plus dropping calls the SDK marks ungrounded or negated, as in the official environment harness).
