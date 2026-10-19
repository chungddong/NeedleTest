"""Check generated Korean IoT command rows and assemble the fine-tune files.

Input: one or more raw JSONL files, each line {"query", "answers", "style"?}, where
answers is a list of {"name", "arguments"} calls for testsets.TOOLS (empty = refuse).
Output: iot_ko/data/train.jsonl and iot_ko/data/val.jsonl (300 rows held out), in the
`needle finetune` format {"query", "tools", "reasoning", "answers", "style"}.

Rows are dropped when a call breaks the schema (unknown tool or key, missing required
key, enum or range violation, wrong type), when a copied string (city, name, destination,
music query) is not in the query, when the query is a near-duplicate of an evaluation
sentence (testsets_ko.py or testsets.py, char-bigram Jaccard >= 0.45) or of another row,
or when it uses a relative date. `reasoning` is the label alone (tool names or
"no call"): the engine always writes a think block first, and label-only reasoning
scored best in the earlier fine-tune (CLAUDE.md).

Usage (repo root): python iot_ko/assemble.py iot_ko/data/raw/*.jsonl
"""
import json
import os
import random
import re
import sys
from collections import Counter

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import testsets as T          # noqa: E402
from testsets_ko import KO_CASES  # noqa: E402

TOOLS = [t._needle_tool for t in T.TOOLS]
SCHEMA = {t["name"]: t["parameters"] for t in TOOLS}
COPIED = {("get_weather", "city"), ("call_contact", "name"), ("navigate_to", "destination"),
          ("play_music", "query"), ("send_message", "recipient")}
RELATIVE = re.compile(r"내일|모레|어제|오늘|다음 주|이번 주|tomorrow|today|next week", re.I)
OUT = os.environ.get("IOT_KO_OUT") or os.path.join(ROOT, "iot_ko", "data")


def bigrams(s):
    s = re.sub(r"\s", "", s.lower())
    return {s[i:i + 2] for i in range(len(s) - 1)}


def jaccard(a, b):
    return len(a & b) / max(1, len(a | b))


def problem(row):
    q, answers = row.get("query"), row.get("answers")
    if not isinstance(q, str) or not q.strip() or not isinstance(answers, list):
        return "bad row"
    for call in answers:
        name, args = call.get("name"), call.get("arguments") or {}
        if name not in SCHEMA:
            return "unknown tool"
        props, required = SCHEMA[name]["properties"], SCHEMA[name].get("required", [])
        if any(k not in props for k in args) or any(k not in args for k in required):
            return "keys"
        for k, v in args.items():
            p = props[k]
            if "enum" in p and v not in p["enum"]:
                return "enum"
            if p.get("type") == "integer":
                if not isinstance(v, int) or isinstance(v, bool):
                    return "type"
                if v < p.get("minimum", v) or v > p.get("maximum", v):
                    return "range"
            if p.get("type") == "string" and not isinstance(v, str):
                return "type"
            if k == "date" and not re.fullmatch(r"\d{4}-\d{2}-\d{2}", v):
                return "date format"
            if (name, k) in COPIED and v.lower() not in q.lower():
                return "not copied"
    if any(c["name"] == "create_event" for c in answers) and RELATIVE.search(q):
        return "relative date"
    return None


def reasoning(answers):
    return ", ".join(c["name"] for c in answers) if answers else "no call"


def main():
    raw = []
    for path in sys.argv[1:]:
        with open(path, encoding="utf-8") as f:
            raw += [json.loads(line) for line in f if line.strip()]
    evals = [q for cases in list(KO_CASES.values()) + list(T.CASES.values()) for q, _ in cases]
    eval_bg = [bigrams(q) for q in evals]
    kept, seen, dropped = [], [], Counter()
    for row in raw:
        why = problem(row)
        if why is None:
            bg = bigrams(row["query"])
            if any(jaccard(bg, e) >= 0.45 for e in eval_bg):
                why = "near eval"
            elif row["query"].strip() in {r["query"].strip() for r in kept}:
                why = "duplicate"
        if why:
            dropped[why] += 1
            continue
        kept.append({"query": row["query"].strip(), "tools": TOOLS, "reasoning": reasoning(row["answers"]),
                     "answers": row["answers"], "style": row.get("style")})
    random.Random(0).shuffle(kept)
    val, train = kept[:300], kept[300:]
    os.makedirs(OUT, exist_ok=True)
    for name, rows in (("train", train), ("val", val)):
        with open(os.path.join(OUT, f"{name}.jsonl"), "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    tools = Counter(c["name"] for r in kept for c in r["answers"]) + Counter(
        {"(no call)": sum(1 for r in kept if not r["answers"])})
    print(f"raw {len(raw)} -> kept {len(kept)} (train {len(train)}, val {len(val)}); dropped {dict(dropped)}")
    print("calls per tool:", dict(tools.most_common()))
    print("multi-call rows:", sum(1 for r in kept if len(r["answers"]) > 1),
          "| english rows:", sum(1 for r in kept if not re.search(r"[가-힣]", r["query"])))


if __name__ == "__main__":
    main()
