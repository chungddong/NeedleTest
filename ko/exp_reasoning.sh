#!/usr/bin/env bash
# Follow-up experiments on the same fit/val split, after ko/train_main.sh finishes:
#   label_e3  label-only reasoning (no evidence phrase), 3 epochs, lr 1e-3
#   evid_e8   evidence + label reasoning (the split's own, made before this change), 8 epochs, lr 1e-3
#   label_e8  label-only reasoning, 8 epochs, lr 1e-3
# Each is scored through the engine on the held-out 300 rows (never the human test set).
# Needs ko/out/split/{fit,val}.jsonl from ko/train_main.sh. Run on 2026-10-02 (results:
# ko/results/exp_*); label_e8 won, so train.jsonl now carries label-only reasoning.
cd "$(dirname "$0")/.."
export NEEDLE_TELEMETRY=0 XLA_PYTHON_CLIENT_PREALLOCATE=false
while pgrep -f 'ko/train_main.sh' > /dev/null; do sleep 30; done
echo "== train_main finished, starting experiments $(date +%T)"
mkdir -p ko/out/exp
.venv/bin/python - <<'EOF'
import json
def label_only(answers):
    if not answers:
        return "no report"
    return " | ".join(f"{a['arguments'].get('incident_type')}; households {a['arguments'].get('households', 'none')}; "
                      f"hazard {a['arguments'].get('hazard', 'none')}" for a in answers)
for name in ("fit", "val"):
    with open(f"ko/out/split/{name}.jsonl", encoding="utf-8") as f, \
         open(f"ko/out/exp/{name}_label.jsonl", "w", encoding="utf-8") as out:
        for line in f:
            r = json.loads(line)
            r["reasoning"] = label_only(r["answers"])
            out.write(json.dumps(r, ensure_ascii=False) + "\n")
print("wrote label-only fit/val")
EOF
run() {  # name data epochs lr
  echo "== exp $1 (epochs $3, lr $4) $(date +%T)"
  .venv/bin/needle finetune "$2" --epochs "$3" --batch-size 8 --lr "$4" --max-len 512 --val-split 0 \
    --out "ko/out/exp/$1.safetensors" 2>&1 | grep --line-buffered -v 'unauthenticated requests' \
    > "ko/results/exp_$1_train.log"
  grep -a 'epoch' "ko/results/exp_$1_train.log" | tail -1
  .venv/bin/needle build checkpoints/needle3.safetensors --lora "ko/out/exp/$1.safetensors" \
    --out "ko/out/exp/$1-20L.cact" > /dev/null 2>&1
  .venv/bin/python ko/try_model.py "ko/out/exp/$1-20L.cact" ko/out/split/val.jsonl \
    --out "ko/results/exp_$1_val.json" | tail -2 | head -1
}
run label_e3 ko/out/exp/fit_label.jsonl 3 1e-3
run evid_e8 ko/out/split/fit.jsonl 8 1e-3
run label_e8 ko/out/exp/fit_label.jsonl 8 1e-3
echo "== experiments done $(date +%T)"
