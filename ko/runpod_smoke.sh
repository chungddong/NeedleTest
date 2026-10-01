#!/usr/bin/env bash
# First run on the GPU server: set up, then prove train -> build -> engine inference
# works end to end on Korean before spending time on real data.
# Run from the repository root:  bash ko/runpod_smoke.sh
set -euo pipefail
cd "$(dirname "$0")/.."
export NEEDLE_TELEMETRY=0

if [ ! -x .venv/bin/python ]; then
  python3 -m venv .venv
  .venv/bin/pip install -q --upgrade pip
  .venv/bin/pip install -q "cactus-needle[train,gpu]==3.0.6" psutil
fi
.venv/bin/python -c "import jax; print('jax devices:', jax.devices())"

.venv/bin/python ko/make_smoke_data.py
.venv/bin/python ko/make_test_data.py

echo "== base model on the smoke set (expect Korean rows to fail)"
.venv/bin/python ko/try_model.py - ko/data/smoke.jsonl | tail -1

echo "== overfit the 24 smoke rows; loss should fall well below 0.1"
mkdir -p ko/out
.venv/bin/needle finetune ko/data/smoke.jsonl --epochs 10 --batch-size 4 --lr 1e-3 \
  --max-len 512 --val-split 0 --out ko/out/smoke_lora.safetensors

echo "== merge the adapter and export a .cact"
.venv/bin/needle build checkpoints/needle3.safetensors --lora ko/out/smoke_lora.safetensors \
  --out ko/out/smoke-20L.cact

echo "== tuned model on the rows it trained on (should mostly pass), then on the human test set"
.venv/bin/python ko/try_model.py ko/out/smoke-20L.cact ko/data/smoke.jsonl | tail -1
.venv/bin/python ko/try_model.py ko/out/smoke-20L.cact ko/data/test_human.jsonl | tail -1
