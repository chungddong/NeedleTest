"""Break a saved result file down by category and error kind.

Usage: python ko/analyze.py ko/results/pi-c-needle3-ko-v2-20L_test_extra.json [--data ko/data/test_extra.jsonl]
Categories come from the dataset's `category` field (row order must match).
"""
import argparse
import collections
import json

FIELDS = ("incident_type", "location", "households", "hazard")


def args_of(calls):
    return (calls[0].get("arguments") or {}) if calls else {}


def location_error(want, got):
    if got is None:
        return "missing"
    if got == want:
        return None
    if want in got:
        return "too long (contains answer)"
    if got in want:
        return "too short (inside answer)"
    return "different span"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("result")
    ap.add_argument("--data", default="ko/data/test_extra.jsonl")
    a = ap.parse_args()
    res = json.load(open(a.result, encoding="utf-8"))
    data = [json.loads(line) for line in open(a.data, encoding="utf-8")]
    rows = res["rows"]
    assert [r["query"] for r in rows] == [d["query"] for d in data], "row order differs"

    by_cat = collections.OrderedDict()
    for r, d in zip(rows, data):
        by_cat.setdefault(d.get("category", "-"), []).append(r)
    print(f"{res.get('model', res.get('weights'))}  exact {res['scores']['exact']}/{res['scores']['n']}")
    print(f"{'category':<12} {'n':>3} {'exact':>6} {'decision':>8} {'type':>6} {'loc':>6}")
    for cat, rs in by_cat.items():
        pos = [r for r in rs if r["want"]]
        print(f"{cat:<12} {len(rs):>3} {sum(r['exact'] for r in rs):>6} {sum(r['decision'] for r in rs):>8} "
              f"{sum(r['incident_type'] for r in pos):>4}/{len(pos):<2} {sum(r['location'] for r in pos):>3}/{len(pos):<2}")

    kinds = collections.Counter()
    confusion = collections.Counter()
    examples = collections.defaultdict(list)
    for r in rows:
        w, g = args_of(r["want"]), args_of(r["got"])
        if r["exact"]:
            continue
        if r["want"] and not r["got"]:
            kinds["missed report"] += 1
            examples["missed report"].append(r["query"])
            continue
        if not r["want"] and r["got"]:
            kinds["false report"] += 1
            examples["false report"].append(f"{r['query']} -> {g.get('incident_type')}")
            continue
        if not r["want"]:
            continue
        if g.get("incident_type") != w.get("incident_type"):
            kinds["wrong type"] += 1
            confusion[(w.get("incident_type"), g.get("incident_type"))] += 1
            examples["wrong type"].append(f"{r['query']} -> {g.get('incident_type')} (want {w.get('incident_type')})")
        le = location_error(w.get("location"), g.get("location"))
        if le:
            kinds[f"location: {le}"] += 1
            examples[f"location: {le}"].append(f"{r['query']} -> \"{g.get('location')}\" (want \"{w.get('location')}\")")
        if g.get("households") != w.get("households"):
            kinds["households"] += 1
            examples["households"].append(f"{r['query']} -> {g.get('households')} (want {w.get('households')})")
        if g.get("hazard") != w.get("hazard"):
            kinds["hazard"] += 1
            examples["hazard"].append(f"{r['query']} -> {g.get('hazard')} (want {w.get('hazard')})")
    print("\nerror kinds (a row can have several):")
    for k, n in kinds.most_common():
        print(f"  {k:<32} {n}")
        for e in examples[k][:4]:
            print(f"      {e}")
    if confusion:
        print("\ntype confusion (want -> got):", ", ".join(f"{w}->{g} x{n}" for (w, g), n in confusion.most_common()))


if __name__ == "__main__":
    main()
