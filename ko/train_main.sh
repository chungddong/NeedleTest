#!/usr/bin/env bash
# Main Korean fine-tune on a GPU machine (WSL2), after ko/data/train.jsonl exists.
#   1. split train.jsonl 90/10 (seed 0) into fit/val (val is held out from training data,
#      never ko/data/test_human.jsonl)
#   2. learning-rate sweep on fit, each scored through the engine on val
#   3. retrain the best rate on all of train.jsonl, export 12/16/20 layers
#   4. score each depth on the human test set; English regression with bench.py
# Run from the repository root:  bash ko/train_main.sh
# Options (environment): EPOCHS (3), BATCH (8), LRS ("1e-4 3e-4 1e-3"); LR=<rate> skips the
# split and sweep and trains on all rows directly; TAG=<name> writes ko-<name>-<L>L results
# and ko/out/ko_<name>_lora.safetensors instead of overwriting the untagged run.
set -euo pipefail
cd "$(dirname "$0")/.."
export NEEDLE_TELEMETRY=0 XLA_PYTHON_CLIENT_PREALLOCATE=false
PY=.venv/bin/python
NEEDLE=.venv/bin/needle
EPOCHS=${EPOCHS:-3}
BATCH=${BATCH:-8}
LRS=${LRS:-"1e-4 3e-4 1e-3"}
LR=${LR:-}
TAG=${TAG:-}
NAME=ko${TAG:+-$TAG}          # ko or ko-<tag>, used in result names
ADAPTER=ko/out/ko${TAG:+_$TAG}_lora.safetensors
TRAINLOG=ko/results/train${TAG:+_$TAG}.log
mkdir -p ko/out/split ko/results

# 512 fits batch 8 on an 8 GB card (1024 ran out of memory); rows longer than the cap are
# truncated at the end, so the count is printed (evidence-format data on 2026-10-02: 1 of 3,000
# rows, 514 tokens; label-only data: 0).
MAXCAP=${MAXCAP:-512}
MAXLEN=$($PY - "$MAXCAP" <<'EOF'
import sys
from needle.model.finetune import fit_max_len, read_examples, render_example
from needle.model.tokenizer import get_tokenizer
cap, tok = int(sys.argv[1]), get_tokenizer()
over = sum(len(tok.encode(p)) + len(tok.encode(t)) + 2 > cap
           for p, t in (render_example(ex) for ex in read_examples("ko/data/train.jsonl")))
print(f"rows longer than {cap}: {over}", file=sys.stderr)
print(fit_max_len("ko/data/train.jsonl", tok, cap))
EOF
)
MAXLEN=$(echo "$MAXLEN" | tail -1)
echo "== max-len $MAXLEN"

if [ -n "$LR" ]; then
  BEST=$LR
  echo "== lr $LR given, skipping the sweep"
else
$PY - <<'EOF'
import random
rows = open("ko/data/train.jsonl", encoding="utf-8").read().splitlines()
idx = list(range(len(rows)))
random.Random(0).shuffle(idx)
n_val = len(rows) // 10
val = set(idx[:n_val])
with open("ko/out/split/fit.jsonl", "w", encoding="utf-8") as fit, open("ko/out/split/val.jsonl", "w", encoding="utf-8") as v:
    for i, row in enumerate(rows):
        (v if i in val else fit).write(row + "\n")
print(f"== split: {len(rows) - n_val} fit / {n_val} val")
EOF

for LR in $LRS; do
  echo "== sweep lr $LR"
  $NEEDLE finetune ko/out/split/fit.jsonl --epochs "$EPOCHS" --batch-size "$BATCH" --lr "$LR" \
    --max-len "$MAXLEN" --val-split 0 --out "ko/out/sweep_lr${LR}.safetensors" 2>&1 \
    | grep --line-buffered -v 'unauthenticated requests' | tee "ko/results/sweep_lr${LR}_train.log"
  $NEEDLE build checkpoints/needle3.safetensors --lora "ko/out/sweep_lr${LR}.safetensors" \
    --out "ko/out/sweep_lr${LR}-20L.cact"
  $PY ko/try_model.py "ko/out/sweep_lr${LR}-20L.cact" ko/out/split/val.jsonl \
    --out "ko/results/sweep_lr${LR}_val.json" | tail -2
done

BEST=$($PY - "$LRS" <<'EOF'
import json, sys
best = None
for lr in sys.argv[1].split():
    s = json.load(open(f"ko/results/sweep_lr{lr}_val.json", encoding="utf-8"))["scores"]
    key = (s["exact"], s["decision"], s["incident_type"], s["location"], -s["errors"])
    print(f"   lr {lr}: exact {s['exact']}/{s['n']} decision {s['decision']} type {s['incident_type']} "
          f"location {s['location']} errors {s['errors']}", file=sys.stderr)
    if best is None or key > best[0]:
        best = (key, lr)
print(best[1])
EOF
)
echo "== best lr on val: $BEST"
fi

echo "== final: lr $BEST, $EPOCHS epochs on all of train.jsonl -> $ADAPTER"
$NEEDLE finetune ko/data/train.jsonl --epochs "$EPOCHS" --batch-size "$BATCH" --lr "$BEST" \
  --max-len "$MAXLEN" --val-split 0 --out "$ADAPTER" 2>&1 \
  | grep --line-buffered -v 'unauthenticated requests' | tee "$TRAINLOG"

for L in 12 16 20; do
  echo "== build and score ${L}L"
  $NEEDLE build checkpoints/needle3.safetensors --lora "$ADAPTER" \
    --layers "$L" --out "ko/out/needle3-${NAME}-${L}L.cact"
  $PY ko/try_model.py "ko/out/needle3-${NAME}-${L}L.cact" ko/data/test_human.jsonl \
    --out "ko/results/${NAME}-${L}L_test_human.json" | tail -2
done

echo "== English regression (results/${NAME}-20L.json)"
$PY bench.py --weights "ko/out/needle3-${NAME}-20L.cact" --label "${NAME}-20L" --repeats 1 | tail -15

if [ -f ko/results/base-20L-w4_test_human.json ] && [ -f results/pc-base-20L-w4.json ]; then
  echo "== base model reference already measured, skipping"
else
  echo "== base model, same machine and 4-bit export, for a like-for-like comparison"
  mkdir -p models
  [ -f models/needle3-20L-w4.cact ] || $PY build_rungs.py 20
  $PY ko/try_model.py models/needle3-20L-w4.cact ko/data/test_human.jsonl --auto-date \
    --out ko/results/base-20L-w4_test_human.json | tail -2
  $PY bench.py --weights models/needle3-20L-w4.cact --label pc-base-20L-w4 --repeats 1 | tail -15
fi
echo "== done"
