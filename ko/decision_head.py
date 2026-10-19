"""Can a (fine-tuned) Needle 3 act as a lightweight decision model?

Instead of generating a tool call, read Needle's hidden states once and answer typed
questions with small linear heads (like Laya / Jev "System 1" decision models).

  extract  run the model once per sentence (JAX, GPU if available) and save features
  fit      train one softmax head per question on train features, calibrate it with
           temperature scaling on a held-out split, and score the test sets

Features per sentence (layers 10, 15, 20 = last):
  query  - mean hidden state over the sentence alone (BOS + query), like an encoder
  prompt - hidden state at the last prompt token (tools + query + "assistant\\n"), i.e.
           the state right before the model starts writing its answer

Questions: incident_type (6 types + "none" for refusals) and hazard (none/electrocution/fire).

Usage (repo root):
  python ko/decision_head.py extract --adapter ko/out/ko_v2_lora.safetensors \\
      --data ko/data/train.jsonl ko/data/test_human.jsonl ko/data/test_extra.jsonl --name v2
  python ko/decision_head.py fit --name v2
  (--adapter none uses the published base checkpoint)
"""
import argparse
import json
import os
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
FEAT_DIR = os.path.join(HERE, "out", "features")
RES_DIR = os.path.join(HERE, "results")
LAYERS = (10, 15, 20)
TYPES = ["outage", "pole_down", "line_down", "transformer_noise", "spark", "fire", "none"]
HAZARDS = ["none", "electrocution", "fire"]


def labels(row):
    a = row["answers"][0]["arguments"] if row["answers"] else {}
    return {"incident_type": a.get("incident_type", "none"), "hazard": a.get("hazard", "none")}


# ----------------------------------------------------------------------------- extract

def load_model(adapter, checkpoint):
    import jax
    import jax.numpy as jnp
    from needle.model.run import load_checkpoint
    from needle.model.architecture import SimpleAttentionNetwork
    from needle.model.checkpoints import read_adapter
    from needle.model.finetune import merge_lora
    from needle.model.quantize import configure_deploy, cq_ste_params, WEIGHT_BITS

    params, config = load_checkpoint(checkpoint)
    config.dtype = "float32"
    params = jax.tree.map(lambda a: np.asarray(a).astype(np.float32), params)
    if adapter and adapter != "none":
        ad = read_adapter(adapter)
        lora = {tuple(k.split("/")): {"A": jnp.asarray(v["A"]), "B": jnp.asarray(v["B"])}
                for k, v in ad["lora"].items()}
        params = merge_lora(params, lora, ad["scale"])
    configure_deploy(act_bits=getattr(config, "act_bits", 8), kv_bits=getattr(config, "kv_bits", 8))
    params = cq_ste_params(params, WEIGHT_BITS)  # score under the 4-bit numerics the .cact ships
    model = SimpleAttentionNetwork(config)

    @jax.jit
    def cells(tokens):
        return model.apply({"params": params}, tokens, quant=True,
                           method=SimpleAttentionNetwork.hidden_cells)

    return cells, config, jax.default_backend()


def encode_batches(texts, tok, pad_to, bos=True):
    from needle.model.tokenizer import BOS_ID, PAD_ID
    ids = [([BOS_ID] if bos else []) + tok.encode(t) for t in texts]
    longest = max(len(i) for i in ids)
    if longest > pad_to:
        raise SystemExit(f"a sequence has {longest} tokens > {pad_to}; raise --{'prompt' if pad_to > 200 else 'query'}-len")
    arr = np.full((len(ids), pad_to), PAD_ID, np.int32)
    for k, i in enumerate(ids):
        arr[k, :len(i)] = i
    return arr, np.array([len(i) for i in ids])


def extract(args):
    import jax.numpy as jnp
    from needle.model.tokenizer import get_tokenizer
    from needle.model.finetune import render_example

    cells, config, backend = load_model(args.adapter, args.checkpoint)
    tok = get_tokenizer(config.vocab_size)
    print(f"backend {backend}, {config.num_layers} layers, d_model {config.d_model}")
    os.makedirs(FEAT_DIR, exist_ok=True)
    for path in args.data:
        rows = [json.loads(line) for line in open(path, encoding="utf-8")]
        if args.limit:
            rows = rows[:args.limit]
        queries = [r["query"] for r in rows]
        prompts = [render_example({**r, "answers": []})[0] for r in rows]
        out = {"query": [], "prompt": []}
        t0 = time.perf_counter()
        for start in range(0, len(rows), args.batch):
            for kind, texts, pad in (("query", queries, args.query_len), ("prompt", prompts, args.prompt_len)):
                arr, lens = encode_batches(texts[start:start + args.batch], tok, pad)
                c = np.asarray(cells(jnp.asarray(arr)))          # [B, T, L+1, D]
                c = c[:, :, list(LAYERS), :]                       # [B, T, 3, D]
                if kind == "query":
                    mask = (np.arange(pad)[None, :] < lens[:, None])[..., None, None]
                    feat = (c * mask).sum(1) / lens[:, None, None]  # mean over real tokens
                else:
                    feat = c[np.arange(len(lens)), lens - 1]          # last prompt token
                out[kind].append(feat.reshape(len(lens), -1).astype(np.float32))  # float16 overflows: hidden states exceed 65504
        secs = time.perf_counter() - t0
        name = os.path.splitext(os.path.basename(path))[0]
        dst = os.path.join(FEAT_DIR, f"{args.name}_{name}.npz")
        lab = [labels(r) for r in rows]
        np.savez_compressed(dst, query=np.concatenate(out["query"]), prompt=np.concatenate(out["prompt"]),
                            incident_type=np.array([l["incident_type"] for l in lab]),
                            hazard=np.array([l["hazard"] for l in lab]), text=np.array(queries))
        print(f"{name}: {len(rows)} rows in {secs:.1f}s ({1000 * secs / len(rows):.1f} ms/row incl. both inputs) -> {dst}")


# ----------------------------------------------------------------------------- fit

def softmax(z):
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


def train_softmax(X, y, k, l2=1e-3, epochs=300, lr=0.5):
    """Multinomial logistic regression, full-batch gradient descent (features standardised)."""
    n, d = X.shape
    W, b = np.zeros((d, k)), np.zeros(k)
    Y = np.eye(k)[y]
    for _ in range(epochs):
        P = softmax(X @ W + b)
        G = P - Y
        W -= lr * (X.T @ G / n + l2 * W)
        b -= lr * G.mean(0)
    return W, b


def fit_temperature(logits, y):
    best = (1.0, 1e9)
    for T in np.exp(np.linspace(np.log(0.05), np.log(10), 200)):
        p = softmax(logits / T)
        nll = -np.log(p[np.arange(len(y)), y] + 1e-12).mean()
        if nll < best[1]:
            best = (T, nll)
    return best[0]


def ece(conf, correct, bins=10):
    edges = np.linspace(0, 1, bins + 1)
    total = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            total += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return float(total)


def fit(args):
    def load(split):
        return np.load(os.path.join(FEAT_DIR, f"{args.name}_{split}.npz"), allow_pickle=False)

    train = load("train")
    tests = {s: load(s) for s in args.tests if os.path.exists(os.path.join(FEAT_DIR, f"{args.name}_{s}.npz"))}
    rng = np.random.default_rng(0)
    order = rng.permutation(len(train["text"]))
    cal, fit_idx = order[:args.cal], order[args.cal:]
    report = {"name": args.name, "layers": LAYERS, "results": []}
    for kind in ("query", "prompt"):
        Xall = train[kind].astype(np.float32)
        d_layer = Xall.shape[1] // len(LAYERS)
        for li, layer in list(enumerate(LAYERS)) + [(None, "all")]:
            sl = slice(None) if li is None else slice(li * d_layer, (li + 1) * d_layer)
            X = Xall[:, sl]
            mu, sd = X[fit_idx].mean(0), X[fit_idx].std(0) + 1e-6
            # z-score per dimension, clipped: the last layer has a few dimensions near 1e7
            z = lambda A: np.clip((A - mu) / sd, -10, 10)
            for question, classes in (("incident_type", TYPES), ("hazard", HAZARDS)):
                y = np.array([classes.index(v) for v in train[question]])
                W, b = train_softmax(z(X[fit_idx]), y[fit_idx], len(classes))
                T = fit_temperature(z(X[cal]) @ W + b, y[cal])
                row = {"features": kind, "layer": layer, "question": question, "temperature": round(float(T), 3)}
                cal_p = softmax((z(X[cal]) @ W + b) / T)
                row["heldout_acc"] = round(float((cal_p.argmax(1) == y[cal]).mean()), 4)
                for split, data in tests.items():
                    Xt = data[kind].astype(np.float32)[:, sl]
                    yt = np.array([classes.index(v) for v in data[question]])
                    p = softmax((z(Xt) @ W + b) / T)
                    pred, conf = p.argmax(1), p.max(1)
                    ok = pred == yt
                    row[split] = {"acc": round(float(ok.mean()), 4), "n": int(len(yt)),
                                  "ece": round(ece(conf, ok), 4),
                                  "acc_conf_ge_0.9": round(float(ok[conf >= 0.9].mean()), 4) if (conf >= 0.9).any() else None,
                                  "share_conf_ge_0.9": round(float((conf >= 0.9).mean()), 4)}
                report["results"].append(row)
                tl = "  ".join(f"{s} {row[s]['acc']:.3f} (ece {row[s]['ece']:.3f})" for s in tests)
                print(f"{kind:<6} layer {str(layer):<3} {question:<13} heldout {row['heldout_acc']:.3f}  {tl}")
    os.makedirs(RES_DIR, exist_ok=True)
    dst = os.path.join(RES_DIR, f"dhead_{args.name}.json")
    with open(dst, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=1)
    print("saved", dst)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("extract")
    e.add_argument("--adapter", required=True, help="LoRA adapter .safetensors, or 'none' for the base model")
    e.add_argument("--checkpoint", default="checkpoints/needle3.safetensors")
    e.add_argument("--data", nargs="+", required=True)
    e.add_argument("--name", required=True)
    e.add_argument("--batch", type=int, default=32)
    e.add_argument("--query-len", type=int, default=192)
    e.add_argument("--prompt-len", type=int, default=512)
    e.add_argument("--limit", type=int, default=0, help="only the first N rows per file (smoke test)")
    f = sub.add_parser("fit")
    f.add_argument("--name", required=True)
    f.add_argument("--tests", nargs="+", default=["test_human", "test_extra"])
    f.add_argument("--cal", type=int, default=300, help="training rows held out for temperature scaling")
    args = ap.parse_args()
    os.environ.setdefault("NEEDLE_TELEMETRY", "0")
    extract(args) if args.cmd == "extract" else fit(args)


if __name__ == "__main__":
    main()
