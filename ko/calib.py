"""Confidence-binned accuracy (calibration) of a LoRA adapter, decoded in JAX.

Tuned archives carry no confidence head (the engine reports None), so confidence is
taken from the model's own token probabilities while decoding greedily with the
engine's "<think>" prefill (as ko/jax_score.py):
  seq    probability of the whole generated output (product of chosen-token probabilities)
  first  probability of the first generated token, the decision (incident type or "no")
Rows are binned by confidence at 0.50, 0.55, ..., 0.95 and each bin gets its count,
exact-call accuracy and mean confidence; ECE summarizes the gap. An overfit model piles
up at 0.95-1.00 and is much less accurate than its confidence in the 0.50-0.95 bins.
The selective table shows coverage and accuracy when only outputs at or above each
threshold are accepted (the role the dropped confidence head used to play).

Usage: python ko/calib.py ADAPTER DATA --name NAME   (writes ko/results/calib_NAME.json)
"""
import argparse
import json
import os

os.environ.setdefault("NEEDLE_TELEMETRY", "0")

import jax
import jax.numpy as jnp
import numpy as np

from needle.model.architecture import SimpleAttentionNetwork
from needle.model.checkpoints import read_adapter
from needle.model.finetune import (DEFAULT_BASE, TOOL_CALL_END, TOOL_CALL_START, _calls_of,
                                   merge_lora, read_examples, render_example)
from needle.model.quantize import WEIGHT_BITS, configure_deploy, cq_ste_params
from needle.model.run import load_checkpoint
from needle.model.tokenizer import BOS_ID, EOS_ID, PAD_ID, THINK_START, get_tokenizer

MAX_NEW = 192
BUF_LEN = 768
EDGES = [0.0, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0000001]
THRESHOLDS = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]


def bins(rows, key):
    out = []
    for lo, hi in zip(EDGES, EDGES[1:]):
        group = [r for r in rows if lo <= r[key] < hi]
        n = len(group)
        out.append({"bin": f"{lo:.2f}-{min(hi, 1.0):.2f}", "n": n,
                    "acc": sum(r["exact"] for r in group) / n if n else None,
                    "conf": sum(r[key] for r in group) / n if n else None})
    total = len(rows)
    ece = sum(b["n"] / total * abs(b["acc"] - b["conf"]) for b in out if b["n"])
    selective = []
    for t in THRESHOLDS:
        kept = [r for r in rows if r[key] >= t]
        selective.append({"threshold": t, "coverage": len(kept) / total,
                          "acc": sum(r["exact"] for r in kept) / len(kept) if kept else None})
    mid = [r for r in rows if 0.5 <= r[key] < 0.95]
    return {"bins": out, "ece": ece, "share_ge_095": sum(r[key] >= 0.95 for r in rows) / total,
            "acc_ge_095": (lambda g: sum(r["exact"] for r in g) / len(g) if g else None)(
                [r for r in rows if r[key] >= 0.95]),
            "acc_050_095": sum(r["exact"] for r in mid) / len(mid) if mid else None,
            "n_050_095": len(mid), "mean_conf": sum(r[key] for r in rows) / total,
            "selective": selective}


def show(name, rep, acc):
    print(f"-- confidence = {name}   accuracy {acc:.3f}  mean conf {rep['mean_conf']:.3f}  ECE {rep['ece']:.3f}  "
          f"share >=0.95 {rep['share_ge_095']:.2f} (acc {rep['acc_ge_095'] if rep['acc_ge_095'] is not None else float('nan'):.3f})  "
          f"0.50-0.95: n {rep['n_050_095']} acc {rep['acc_050_095'] if rep['acc_050_095'] is not None else float('nan'):.3f}")
    print("   bin          n    acc   conf")
    for b in rep["bins"]:
        if b["n"]:
            print(f"   {b['bin']}  {b['n']:4d}  {b['acc']:.3f}  {b['conf']:.3f}")
    print("   accept if conf >= t:  " + "  ".join(
        f"{s['threshold']:.2f}: {s['coverage'] * 100:.0f}% kept, acc {s['acc']:.3f}" if s["acc"] is not None
        else f"{s['threshold']:.2f}: none" for s in rep["selective"][::3]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("adapter")
    ap.add_argument("data")
    ap.add_argument("--name", required=True)
    args = ap.parse_args()
    prefill = THINK_START + "\n"

    params, config = load_checkpoint(DEFAULT_BASE)
    config.dtype = "float32"
    params = jax.device_put(jax.tree.map(lambda a: np.asarray(a).astype(np.float32), params))
    model = SimpleAttentionNetwork(config)
    configure_deploy(act_bits=getattr(config, "act_bits", 8), kv_bits=getattr(config, "kv_bits", 8))
    tokenizer = get_tokenizer(config.vocab_size)
    adapter = read_adapter(args.adapter)
    lora = {tuple(k.split("/")): {"A": jnp.asarray(v["A"]), "B": jnp.asarray(v["B"])}
            for k, v in adapter["lora"].items()}
    merged = cq_ste_params(merge_lora(params, lora, adapter["scale"]), WEIGHT_BITS)

    @jax.jit
    def step(p, tokens, pos):
        logits = model.apply({"params": p}, tokens, quant=True)[0, pos].astype(jnp.float32)
        probs = jax.nn.softmax(logits)
        nxt = jnp.argmax(logits)
        return nxt, probs[nxt]

    def generate(prompt):
        ids = [BOS_ID] + tokenizer.encode(prompt)
        buf = jnp.full((1, BUF_LEN), PAD_ID, jnp.int32).at[0, :len(ids)].set(jnp.array(ids, jnp.int32))
        out, probs = [], []
        for pos in range(len(ids) - 1, min(BUF_LEN - 1, len(ids) - 1 + MAX_NEW)):
            nxt, p = step(merged, buf, pos)
            nxt, p = int(nxt), float(p)
            probs.append(p)
            if nxt == EOS_ID:
                break
            out.append(nxt)
            buf = buf.at[0, pos + 1].set(nxt)
        return tokenizer.decode(out), probs

    rows = []
    for ex in read_examples(args.data):
        prompt, _ = render_example({**ex, "answers": []})
        text, probs = generate(prompt + prefill)
        text = prefill + text
        want = _calls_of(TOOL_CALL_START + json.dumps(ex.get("answers", []), separators=(",", ":"))
                         + TOOL_CALL_END)
        seq = float(np.exp(np.sum(np.log(np.maximum(probs, 1e-12)))))
        rows.append({"query": ex["query"], "exact": int(_calls_of(text) == want),
                     "seq": seq, "first": probs[0] if probs else 0.0, "tokens": len(probs)})
    acc = sum(r["exact"] for r in rows) / len(rows)
    print(f"== {args.name}: {len(rows)} rows, exact {sum(r['exact'] for r in rows)}/{len(rows)} "
          f"| {args.adapter} | {args.data}")
    report = {"name": args.name, "adapter": args.adapter, "data": args.data, "n": len(rows),
              "exact": acc, "seq": bins(rows, "seq"), "first": bins(rows, "first"), "rows": rows}
    show("seq (whole output)", report["seq"], acc)
    show("first (decision token)", report["first"], acc)
    os.makedirs("ko/results", exist_ok=True)
    with open(f"ko/results/calib_{args.name}.json", "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=1)
    print(f"saved ko/results/calib_{args.name}.json")


if __name__ == "__main__":
    main()
