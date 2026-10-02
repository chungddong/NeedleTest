"""Score a LoRA adapter in JAX on a dataset, without building a .cact or running the engine.

Separates training problems from export/engine problems: it renders each row with the
same prompt `needle finetune` trains on, merges the adapter under the export's W4 + A8
numerics, and decodes greedily. By default it prefills "<think>\\n" after the assistant
tag, because the engine always opens a think block before the tool call; --no-think lets
the model start on its own.

Usage: python ko/jax_score.py ko/out/smoke_lora.safetensors ko/data/smoke.jsonl [--no-think]
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
from needle.model.tokenizer import BOS_ID, EOS_ID, PAD_ID, THINK_END, THINK_START, get_tokenizer

MAX_NEW = 192
BUF_LEN = 768  # fixed decode buffer, so the step function compiles once


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("adapter")
    ap.add_argument("data")
    ap.add_argument("--think", action=argparse.BooleanOptionalAction, default=True,
                    help="prefill <think> like the engine does (default on)")
    args = ap.parse_args()
    prefill = THINK_START + "\n" if args.think else ""

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
        return model.apply({"params": p}, tokens, quant=True)[0, pos]

    def logits_at(tokens, pos):
        return step(merged, tokens, pos)

    def generate(prompt):
        ids = [BOS_ID] + tokenizer.encode(prompt)
        buf = jnp.full((1, BUF_LEN), PAD_ID, jnp.int32).at[0, :len(ids)].set(jnp.array(ids, jnp.int32))
        out = []
        for pos in range(len(ids) - 1, min(BUF_LEN - 1, len(ids) - 1 + MAX_NEW)):
            nxt = int(jnp.argmax(logits_at(buf, pos)))
            if nxt == EOS_ID:
                break
            out.append(nxt)
            buf = buf.at[0, pos + 1].set(nxt)
        return tokenizer.decode(out)

    n = exact = 0
    for ex in read_examples(args.data):
        prompt, _ = render_example({**ex, "answers": []})
        text = prefill + generate(prompt + prefill)
        want = _calls_of(TOOL_CALL_START + json.dumps(ex.get("answers", []), separators=(",", ":"))
                         + TOOL_CALL_END)
        ok = _calls_of(text) == want
        n += 1
        exact += ok
        call = (text.split(TOOL_CALL_START, 1)[1].split(TOOL_CALL_END)[0]
                if TOOL_CALL_START in text else "(no call) " + text[:100])
        think = text.split(THINK_END)[0].replace(THINK_START, "").strip() if THINK_START in text else ""
        print(f"{'O' if ok else 'X'} {ex['query']} -> {call}" + (f"   [think: {think[:100]}]" if think else ""),
              flush=True)
    print(f"jax exact {exact}/{n} | {args.adapter} | W{WEIGHT_BITS}+A8 | prefill {prefill!r}")


if __name__ == "__main__":
    main()
