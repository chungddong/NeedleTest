"""Collect the Korean fine-tune results into ko/report.html (from ko/report_template.html).

Run from the repository root: python ko/make_report.py
"""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")

# test_human.jsonl row ranges, in the order ko/make_test_data.py writes them
CATEGORIES = [("표준어", 0, 10), ("사투리", 10, 14), ("오타·짧은 문자", 14, 19),
              ("현장 보고", 19, 22), ("외국인·영어", 22, 26), ("거절 대상", 26, 32)]


def load(name):
    with open(os.path.join(RES, name), encoding="utf-8") as f:
        return json.load(f)


def korean_reports(rows):
    return [r for r in rows if r["want"] and any("가" <= c <= "힣" for c in r["query"])]


def summary(result):
    rows, s = result["rows"], result["scores"]
    ko = korean_reports(rows)
    return {"exact": s["exact"], "n": s["n"], "decision": s["decision"],
            "type": s["incident_type"], "loc": s["location"], "positives": s["positives"],
            "ko_n": len(ko), "ko_exact": sum(r["exact"] for r in ko),
            "ko_type": sum(r["incident_type"] for r in ko), "ko_loc": sum(r["location"] for r in ko),
            "refusals": sum(r["exact"] for r in rows if not r["want"]),
            "refusal_n": sum(1 for r in rows if not r["want"])}


def main():
    base = load("pi-c-needle3-20L-w2-official_test_human.json")
    v2 = load("pi-c-needle3-ko-v2-20L_test_human.json")

    def category(i):
        return next(name for name, a, b in CATEGORIES if a <= i < b)

    sentences = []
    for i, (b, t) in enumerate(zip(base["rows"], v2["rows"])):
        assert b["query"] == t["query"]
        sentences.append({"i": i, "cat": category(i), "query": t["query"], "want": t["want"],
                          "base": {"got": b["got"], "exact": b["exact"], "type": b["incident_type"],
                                   "loc": b["location"], "decision": b["decision"], "error": b["error"],
                                   "ms": b["latency_ms"]},
                          "v2": {"got": t["got"], "exact": t["exact"], "type": t["incident_type"],
                                 "loc": t["location"], "decision": t["decision"], "error": t["error"],
                                 "ms": t["latency_ms"]}})

    stages = [
        {"name": "원본", "note": "공식 2비트, 학습 없음", **summary(load("base-20L-w2_test_human.json"))},
        {"name": "스모크", "note": "24문장만 학습", **summary(load("smoke-20L_test_human.json"))},
        {"name": "v1", "note": "3,000행 · 3에폭 · 근거+라벨", **summary(load("ko-20L_test_human.json"))},
        {"name": "v2", "note": "3,000행 · 8에폭 · 라벨만", **summary(load("ko-v2-20L_test_human.json"))},
    ]

    experiments = []
    for label, group, f in [("1e-4", "학습률 (3에폭)", "sweep_lr1e-4_val.json"),
                            ("3e-4", "학습률 (3에폭)", "sweep_lr3e-4_val.json"),
                            ("1e-3", "학습률 (3에폭)", "sweep_lr1e-3_val.json"),
                            ("라벨만 · 3에폭", "추론 형식 (1e-3)", "exp_label_e3_val.json"),
                            ("근거+라벨 · 8에폭", "추론 형식 (1e-3)", "exp_evid_e8_val.json"),
                            ("라벨만 · 8에폭", "추론 형식 (1e-3)", "exp_label_e8_val.json")]:
        s = load(f)["scores"]
        experiments.append({"label": label, "group": group, "exact": s["exact"], "n": s["n"],
                            "type": s["incident_type"], "loc": s["location"], "positives": s["positives"]})

    depth = []
    for name, label, size_note in [("needle3-20L-w2-official", "원본 2비트 20L", "공식"),
                                   ("needle3-ko-v2-12L", "v2 12L", "4비트"),
                                   ("needle3-ko-v2-16L", "v2 16L", "4비트"),
                                   ("needle3-ko-v2-20L", "v2 20L", "4비트")]:
        r = load(f"pi-c-{name}_test_human.json")
        s = r["scores"]
        depth.append({"label": label, "mb": r["model_mb"], "exact": s["exact"], "n": s["n"],
                      "p50": s["latency_p50_ms"], "p90": s["latency_p90_ms"], "max": s["latency_max_ms"],
                      "rss": s["peak_rss_mb"], "ko_exact": summary(r)["ko_exact"]})

    def english(path):
        with open(path, encoding="utf-8") as f:
            d = json.load(f)
        return {k: v["conf_exact_acc"] for k, v in d["scores"].items()} | \
               {"p50": d["scores"]["ALL"]["latency"]["p50"]}

    root = os.path.dirname(HERE)
    eng = {"base": english(os.path.join(root, "results", "20L-w4.json")),
           "v2": english(os.path.join(root, "results", "pi-ko-v2-20L.json"))}

    seen, train_examples = set(), []
    with open(os.path.join(HERE, "data", "train.jsonl"), encoding="utf-8") as f:
        train = [json.loads(line) for line in f]
    for row in train:
        if row["style"] not in seen and row["answers"]:
            seen.add(row["style"])
            train_examples.append({"style": row["style"], "query": row["query"], "answers": row["answers"]})
    label_counts = {}
    for row in train:
        key = row["answers"][0]["arguments"]["incident_type"] if row["answers"] else "none"
        label_counts[key] = label_counts.get(key, 0) + 1

    calib = load("calib_v2_test_human.json")["seq"]
    data = {"sentences": sentences, "categories": [c[0] for c in CATEGORIES], "stages": stages,
            "experiments": experiments, "depth": depth, "english": eng,
            "train_examples": train_examples, "train_n": len(train), "label_counts": label_counts,
            "calib": {"mean_conf": calib["mean_conf"], "ece": calib["ece"],
                      "share_ge_095": calib["share_ge_095"], "acc_ge_095": calib["acc_ge_095"],
                      "acc_050_095": calib["acc_050_095"], "exact": load("calib_v2_test_human.json")["exact"]},
            "v2_scores": v2["scores"], "base_scores": base["scores"]}

    with open(os.path.join(HERE, "report_template.html"), encoding="utf-8") as f:
        html = f.read().replace("/*DATA*/null", json.dumps(data, ensure_ascii=False))
    out = os.path.join(HERE, "report.html")
    with open(out, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"wrote {out} ({len(html):,} bytes)")


if __name__ == "__main__":
    main()
