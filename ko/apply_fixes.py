"""Apply label-audit fixes to raw generated rows, before ko/gen_data.py --from assembles them.

Input is the JSON the generation workflow returns: {"results": [{"chunk", "files",
"label": {"fixes": [...]}, "span": {"fixes": [...]}}]}. Each fix names a raw file, a
1-based line, the row's query, and "fix" (corrected answers and evidence) or "drop".
Per row: any drop wins; fixes from both auditors that disagree on the answers drop the
row; otherwise the fix is applied. A fix whose query is not on the named line is matched
by query elsewhere in the same file, or skipped and reported.

Usage: python ko/apply_fixes.py <workflow_result.json> <raw_dir> <out_dir>
"""
import json
import os
import sys
from collections import Counter, defaultdict


def canon(answers):
    return json.dumps(answers, ensure_ascii=False, sort_keys=True)


def main():
    result_path, raw_dir, out_dir = sys.argv[1:4]
    result = json.load(open(result_path, encoding="utf-8"))
    os.makedirs(out_dir, exist_ok=True)
    stats = Counter()
    for chunk in result["results"]:
        names = [os.path.basename(f.replace("\\", "/")) for f in chunk["files"]]
        for name in names:  # rerunnable: rows are appended per file below
            if os.path.exists(os.path.join(out_dir, name)):
                os.remove(os.path.join(out_dir, name))
        rows = {}
        for name in names:
            path = os.path.join(raw_dir, name)
            if not os.path.exists(path):
                stats["missing file"] += 1
                print(f"missing {path}")
                continue
            for i, line in enumerate(open(path, encoding="utf-8"), 1):
                if line.strip():
                    rows[(name, i)] = line
        by_query = defaultdict(list)
        for key, line in rows.items():
            try:
                by_query[(key[0], json.loads(line)["query"])].append(key)
            except (json.JSONDecodeError, KeyError, TypeError):
                pass
        actions = defaultdict(list)
        for lens in ("label", "span"):
            for fix in ((chunk.get(lens) or {}).get("fixes") or []):
                name = os.path.basename(fix["file"].replace("\\", "/"))
                key = (name, fix["line"])
                try:
                    ok = key in rows and json.loads(rows[key])["query"] == fix["query"]
                except (json.JSONDecodeError, KeyError, TypeError):
                    ok = False
                if not ok:
                    found = by_query.get((name, fix["query"])) or []
                    if len(found) != 1:
                        stats["unmatched fix"] += 1
                        print(f"unmatched {lens} fix {name}:{fix['line']} {fix['query'][:40]}")
                        continue
                    key = found[0]
                actions[key].append((lens, fix))
        for (name, line_no), line in sorted(rows.items()):
            fixes = actions.get((name, line_no), [])
            if not fixes:
                out_line = line
            elif any(f["action"] == "drop" for _, f in fixes):
                stats["dropped by " + "+".join(sorted({l for l, f in fixes if f["action"] == "drop"}))] += 1
                continue
            elif len({canon(f.get("answers", [])) for _, f in fixes}) > 1:
                stats["dropped, auditors disagree"] += 1
                continue
            else:
                row = json.loads(line)
                lens, fix = sorted(fixes, key=lambda x: x[0] != "label")[0]
                row["answers"] = fix.get("answers", [])
                if fix.get("evidence"):
                    row["evidence"] = fix["evidence"]
                out_line = json.dumps(row, ensure_ascii=False) + "\n"
                stats["fixed by " + "+".join(sorted({l for l, _ in fixes}))] += 1
            with open(os.path.join(out_dir, name), "a", encoding="utf-8") as f:
                f.write(out_line if out_line.endswith("\n") else out_line + "\n")
            stats["rows out"] += 1
        stats["rows in"] += len(rows)
    print(json.dumps(dict(stats), ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
