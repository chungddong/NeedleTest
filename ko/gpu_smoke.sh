#!/usr/bin/env bash
# First run on a GPU machine (Linux or WSL2 with an NVIDIA GPU): set up, then prove
# train -> build -> engine inference works end to end on Korean before real training.
# Run from the repository root:  bash ko/gpu_smoke.sh
# Scores are saved to ko/results/smoke_*.json for committing.
set -euo pipefail
cd "$(dirname "$0")/.."
export NEEDLE_TELEMETRY=0
# Let JAX grow GPU memory as needed instead of grabbing 75% up front (8 GB cards, WSL2).
export XLA_PYTHON_CLIENT_PREALLOCATE=false

if [ ! -x .venv/bin/python ]; then
  python3 -m venv .venv || { echo "python3-venv missing: sudo apt-get install -y python3-venv"; exit 1; }
  .venv/bin/pip install -q --upgrade pip
  .venv/bin/pip install -q "cactus-needle[train,gpu]==3.0.6" psutil
fi
.venv/bin/python -c "import jax; print('jax devices:', jax.devices())"

.venv/bin/python ko/make_smoke_data.py
.venv/bin/python ko/make_test_data.py
mkdir -p ko/out ko/results

# 10 epochs left incident_type unlearned (loss 0.07 is mostly easy JSON and copied bytes); 50 memorizes.
echo "== overfit the 24 smoke rows; loss should fall below 0.01"
.venv/bin/needle finetune ko/data/smoke.jsonl --epochs 50 --batch-size 4 --lr 1e-3 \
  --max-len 512 --val-split 0 --out ko/out/smoke_lora.safetensors 2>&1 | tee ko/results/smoke_train.log

echo "== adapter in JAX with the engine's <think> prefill (should mostly pass; if it does and the engine does not, suspect export/engine)"
.venv/bin/python ko/jax_score.py ko/out/smoke_lora.safetensors ko/data/smoke.jsonl 2>&1 \
  | grep -v 'unauthenticated requests' | tee ko/results/smoke_jax.log | tail -1

echo "== merge the adapter and export a .cact"
.venv/bin/needle build checkpoints/needle3.safetensors --lora ko/out/smoke_lora.safetensors \
  --out ko/out/smoke-20L.cact

echo "== engine on the rows it trained on (should mostly pass), then on the human test set"
.venv/bin/python ko/try_model.py ko/out/smoke-20L.cact ko/data/smoke.jsonl \
  --out ko/results/smoke-20L_smoke.json | tail -2
.venv/bin/python ko/try_model.py ko/out/smoke-20L.cact ko/data/test_human.jsonl \
  --out ko/results/smoke-20L_test_human.json | tail -2
