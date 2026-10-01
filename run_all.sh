#!/usr/bin/env bash
# Run the benchmark on every archive in models/, cooling the SoC below 60C between runs.
set -uo pipefail
cd "$(dirname "$0")"
cool() { while (( $(vcgencmd measure_temp | grep -oE '[0-9]+' | head -1) >= 60 )); do sleep 10; done; }
cool; .venv/bin/python bench.py --weights models/needle3-20L-w2-official.cact --label 20L-w2-official --extract --embed --scaling
for L in 4 8 12 16 20; do
  cool; .venv/bin/python bench.py --weights models/needle3-${L}L-w4.cact --label ${L}L-w4 --extract --embed
done
