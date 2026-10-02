"""Re-score saved try_model.py results against the current dataset labels, without rerunning.

For label fixes: the model outputs (`got`) do not depend on the labels, so each row's
`want` is replaced with the dataset's current answers and the scores are recomputed the
same way try_model.py computes them. The file records what was rescored and when.

Usage: python ko/rescore.py ko/data/test_human.jsonl ko/results/*_test_human.json
"""
import datetime
import json
import sys


def first_args(calls):
    return (calls[0].get("arguments") or {}) if isinstance(calls, list) and calls else {}


def score_row(r):
    w, g = first_args(r["want"]), first_args(r["got"])
    r["exact"] = r["got"] == r["want"]
    r["decision"] = bool(r["got"]) == bool(r["want"])
    r["incident_type"] = bool(r["want"]) and g.get("incident_type") == w.get("incident_type")
    r["location"] = bool(r["want"]) and g.get("location") == w.get("location")


def main():
    data_path, result_paths = sys.argv[1], sys.argv[2:]
    labels = {}
    for line in open(data_path, encoding="utf-8"):
        row = json.loads(line)
        labels[row["query"]] = row["answers"]
    for path in result_paths:
        result = json.load(open(path, encoding="utf-8"))
        changed = [r["query"] for r in result["rows"] if r["query"] in labels and r["want"] != labels[r["query"]]]
        missing = [r["query"] for r in result["rows"] if r["query"] not in labels]
        before = dict(result["scores"])
        for r in result["rows"]:
            if r["query"] in labels:
                r["want"] = labels[r["query"]]
            score_row(r)
        rows, positives = result["rows"], [r for r in result["rows"] if r["want"]]
        result["scores"].update({
            "n": len(rows),
            "exact": sum(r["exact"] for r in rows),
            "decision": sum(r["decision"] for r in rows),
            "positives": len(positives),
            "incident_type": sum(r["incident_type"] for r in positives),
            "location": sum(r["location"] for r in positives),
        })
        if changed:
            result.setdefault("rescored", []).append({
                "time": datetime.datetime.now().isoformat(timespec="seconds"),
                "data": data_path, "relabeled": changed})
            with open(path, "w", encoding="utf-8") as f:
                json.dump(result, f, ensure_ascii=False, indent=1)
        diff = {k: f"{before.get(k)}->{v}" for k, v in result["scores"].items() if before.get(k) != v}
        print(f"{path}: relabeled {len(changed)} row(s){', missing ' + str(len(missing)) if missing else ''}"
              f" | {diff or 'scores unchanged'}")


if __name__ == "__main__":
    main()
