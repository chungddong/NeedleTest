"""Generate Korean training data for report_incident, through OpenRouter or from raw rows.

The package's own `needle generate-data` prompt is English, so it yields English
queries. This prompt asks for the styles real outage reports come in: standard
Korean, dialects, typos and speech-to-text noise, short texts, field-worker shorthand,
non-native Korean, and some English, plus refusals.

Two ways to get rows:
- OpenRouter: this script sends PROMPT per batch (needs OPENROUTER_API_KEY, billed).
- Another generator (e.g. Claude subagents given `--print-prompt`) writes raw JSONL rows
  {"query", "evidence", "answers", "style"}; `--from` checks and assembles them.

Each row also gets a `reasoning` line (see schema.reasoning): the engine always
reasons in a <think> block before calling, so training rows must teach that block.
The generator writes only a short English evidence phrase; the label part comes from
the answers.

Rows are dropped when the answers break the schema (unknown keys, enum values,
households outside 1..100000), when a location is not copied verbatim from the query,
when the evidence phrase is missing or Korean, or when the query is, or nearly is, a
sentence of the evaluation set. Drop counts and the label mix are printed at the end.

Usage (repo root, WSL):
  export OPENROUTER_API_KEY=...        # your own key; generation is billed by OpenRouter
  python ko/gen_data.py --num 3000 --out ko/data/train.jsonl
  python ko/gen_data.py --print-prompt 0 --batch 10       # prompt for STYLES[0]
  python ko/gen_data.py --from ko/out/raw/*.jsonl --num 3000 --out ko/data/train.jsonl
"""
import argparse
import json
import os
import random
import re
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, os.path.dirname(__file__))

from needle.model.finetune import _openrouter, _parse_array
from schema import TOOLS_JSON, reasoning

# Pinned instead of needle's "deepseek/deepseek-flash-latest", which OpenRouter lists only as
# "~deepseek/deepseek-flash-latest" (an alias that currently points at this model) and whose
# target can change between runs.
DEFAULT_MODEL = "deepseek/deepseek-v4.1-flash"
HANGUL = re.compile("[ㄱ-ㆎ가-힣]")
SCHEMA = TOOLS_JSON[0]
PROPS = SCHEMA["parameters"]["properties"]
NEAR_TEST = 0.45  # character-bigram Jaccard at or above which a query counts as an evaluation sentence


def _bigrams(text):
    text = re.sub(r"\s+", "", text.lower())
    return {text[i:i + 2] for i in range(len(text) - 1)}


TEST_QUERIES = [json.loads(line)["query"].strip()
                for line in open(os.path.join(os.path.dirname(__file__), "data", "test_human.jsonl"),
                                 encoding="utf-8") if line.strip()]
TEST_BIGRAMS = [_bigrams(q) for q in TEST_QUERIES]


def near_test(query):
    grams = _bigrams(query)
    return any(len(grams & t) / max(1, len(grams | t)) >= NEAR_TEST for t in TEST_BIGRAMS)


def problem(row):
    """Why a generated row is unusable, or None when it is fine."""
    query = row.get("query")
    if not isinstance(query, str) or not query.strip():
        return "no query"
    if query.strip() in TEST_QUERIES or near_test(query):
        return "evaluation sentence"
    answers = row.get("answers")
    if not isinstance(answers, list):
        return "answers not a list"
    for call in answers:
        args = call.get("arguments") if isinstance(call, dict) else None
        if not isinstance(call, dict) or call.get("name") != SCHEMA["name"] or not isinstance(args, dict):
            return "bad call"
        if set(args) - set(PROPS) or not set(SCHEMA["parameters"]["required"]) <= set(args):
            return "bad keys"
        if args["incident_type"] not in PROPS["incident_type"]["enum"]:
            return "bad incident_type"
        if "hazard" in args and args["hazard"] not in PROPS["hazard"]["enum"]:
            return "bad hazard"
        households = args.get("households")
        if households is not None and (type(households) is not int or not 1 <= households <= 100000):
            return "bad households"
        if not isinstance(args["location"], str) or not args["location"].strip() or args["location"] not in query:
            return "location not copied from query"
    evidence = row.get("evidence")
    if not isinstance(evidence, str) or not evidence.strip() or HANGUL.search(evidence):
        return "missing or Korean evidence"
    return None


STYLES = [
    "표준어로 차분하게 신고하는 주민",
    "경상도 사투리를 쓰는 어르신",
    "전라도 사투리를 쓰는 어르신",
    "충청도 사투리를 쓰는 주민",
    "급하고 당황해서 문장이 끊기는 주민",
    "맞춤법과 띄어쓰기가 틀린 문자 메시지",
    "음성 인식(STT) 오류가 섞인 문장",
    "현장 작업자의 짧은 업무용 보고 (약어, 설비 번호 포함)",
    "한국어가 서툰 외국인 주민",
    "영어로 신고하는 외국인 주민",
]

PROMPT = """아래 도구 스키마로 학습 데이터를 만듭니다.

{tools}

태풍·폭우·산불 같은 재난 상황에서 정전이나 전력설비 피해를 신고하는 문장을 {n}개 만드세요.
이번 묶음의 화자: {style}

각 원소는 다음 형식의 JSON 객체이고, 전체를 JSON 배열로만 출력하세요.
{{"query": "<신고 문장>", "evidence": "<영어 요약>", "answers": [{{"name": "report_incident", "arguments": {{...}}}}]}}

규칙:
- evidence는 신고 문장이 말하는 내용을 2~8단어 영어로 요약한 구절입니다. 한글을 쓰지 않습니다.
  incident_type의 근거를 담고, households나 hazard를 넣었다면 그 근거도 담습니다.
  예: "power is out", "pole fell, says it is dangerous", "power out, 30 households stated".
  answers가 []이면 무엇에 대한 문장인지 씁니다. 예: "asks about the bill", "says the power is fine".
- arguments의 키와 incident_type, hazard 값은 스키마의 영어 값만 씁니다.
- incident_type: outage(전기가 안 들어옴), pole_down(전봇대·전주가 쓰러지거나 기울어짐),
  line_down(전선이 끊어지거나 늘어지거나 떨어짐), transformer_noise(변압기 소음),
  spark(불꽃·스파크, 불은 나지 않음), fire(설비에 불이 남).
- 신고 하나에 호출 하나만 만듭니다. 여러 현상이 함께 나오면 가장 심각한 설비 피해 하나로 정합니다:
  fire > pole_down > line_down > spark > transformer_noise > outage.
  예: 전봇대가 쓰러지고 전선이 바닥에 있으면 pole_down, 전선에서 불꽃만 튀면 spark.
- location은 query 안에 글자 그대로 들어 있는 장소 표현입니다(번역하거나 다듬지 않음).
  앞·옆·뒤편·근처·주변 같은 위치 말은 포함하고, 끝의 조사(에, 에서, 의 등)와
  설비 이름(전봇대, 전주, 변압기, 전선 등)은 뺍니다(예: "공원 입구 전주에서 불꽃이" → "공원 입구").
  영어는 near, by, on 같은 전치사를 빼고 명사구만 씁니다(예: "the school").
- households는 가구·세대 수를 직접 말한 경우에만 넣습니다. 아파트 동·호수 같은 숫자는 가구 수가 아닙니다.
- hazard는 감전이나 화재 위험을 직접 말한 경우에만 넣습니다. 불꽃이 튄다는 말만으로는 넣지 않고,
  불이 날 것 같다고 하면 fire, 위험하다거나 감전될 것 같다고 하면 electrocution입니다.
  실제로 불이 났으면 incident_type과 hazard 모두 fire입니다.
- 유형을 고루 섞습니다. outage가 가장 많고, 나머지 유형도 각각 여러 개 넣습니다.
  설비 피해 중 일부는 위험을 말하지 않는 문장으로 만들어 hazard가 없는 예도 충분히 둡니다.
- 지명은 실제와 가상을 섞고 매번 다르게 합니다 (OO리, OO동, 아파트 동호수, 시장, 학교, 다리 등).
  같은 지명을 되풀이하지 않고, 흔한 이름(행복, 햇살, 햇빛, 한빛, 푸른, 새마을 등)은 피합니다.
  문장 길이와 구성도 두세 단어 문자부터 두세 문장 설명까지 다양하게 합니다.
- {refusals}개 정도는 정전·설비 피해와 무관하거나(요금 문의, 명의 변경, 날씨), 정전이 아니라는 내용으로 만들고 answers를 []로 둡니다.
- JSON 배열만 출력합니다."""


def prompt_for(style, n):
    return PROMPT.format(tools=json.dumps(TOOLS_JSON, ensure_ascii=False, indent=1), n=n, style=style,
                         refusals=max(1, n // 8))


def assemble(raw_rows, style=None):
    """Check raw generated rows and turn the usable ones into training rows."""
    kept, dropped = [], Counter()
    for row in raw_rows:
        why = problem(row)
        if why:
            dropped[why] += 1
            continue
        kept.append({"query": row["query"], "tools": TOOLS_JSON,
                     "reasoning": reasoning(row["evidence"].strip(), row["answers"]),
                     "answers": row["answers"], "style": row.get("style") or style})
    return kept, dropped


def batch(style, n, model, api_key):
    text = _openrouter([{"role": "user", "content": prompt_for(style, n)}], model, api_key)
    return assemble(_parse_array(text), style)


def read_raw(paths):
    rows, bad = [], 0
    for path in paths:
        for line in open(path, encoding="utf-8"):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                bad += 1
                continue
            if isinstance(row, dict):
                rows.append(row)
            else:
                bad += 1
    return rows, bad


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--num", type=int, default=3000)
    ap.add_argument("--batch", type=int, default=25)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--out", default="ko/data/train.jsonl")
    ap.add_argument("--from", dest="sources", nargs="+", metavar="RAW_JSONL",
                    help="assemble raw rows written by another generator instead of calling OpenRouter")
    ap.add_argument("--print-prompt", type=int, metavar="STYLE_INDEX",
                    help="print the generation prompt for STYLES[i] with --batch rows and exit")
    args = ap.parse_args()

    if args.print_prompt is not None:
        print(prompt_for(STYLES[args.print_prompt], args.batch))
        return

    seen, rows, dropped, failed = set(), [], Counter(), 0

    def add(kept, why):
        dropped.update(why)
        for row in kept:
            key = row["query"].strip()
            if key in seen:
                dropped["duplicate"] += 1
            else:
                seen.add(key)
                rows.append(row)

    if args.sources:
        raw, bad = read_raw(args.sources)
        dropped["unparseable line"] += bad
        print(f"{len(raw)} raw rows from {len(args.sources)} file(s)", flush=True)
        add(*assemble(raw))
        random.Random(0).shuffle(rows)  # files come grouped by style; cap --num evenly
    else:
        api_key = os.environ.get("OPENROUTER_API_KEY")
        if not api_key:
            raise SystemExit("set OPENROUTER_API_KEY")
        jobs = [random.choice(STYLES) for _ in range(-(-args.num * 13 // 10 // args.batch))]
        print(f"model {args.model}, {len(jobs)} batches of {args.batch}", flush=True)
        with ThreadPoolExecutor(args.workers) as pool:
            futures = [pool.submit(batch, s, args.batch, args.model, api_key) for s in jobs]
            for fut in as_completed(futures):
                try:
                    add(*fut.result())
                except Exception as exc:
                    failed += 1
                    print("failed:", exc, flush=True)
                print(f"{len(rows)} examples", flush=True)
    dropped = +dropped
    print(f"dropped {sum(dropped.values())}: {dict(dropped.most_common())}; failed batches {failed}")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    rows = rows[:args.num]
    with open(args.out, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"wrote {len(rows)} examples to {args.out}")
    calls = [c["arguments"] for r in rows for c in r["answers"]]
    print("refusals", sum(not r["answers"] for r in rows),
          "| types", dict(Counter(a["incident_type"] for a in calls).most_common()),
          "| hazard", dict(Counter(a["hazard"] for a in calls if "hazard" in a)),
          "| households", sum("households" in a for a in calls),
          "| multi-call", sum(len(r["answers"]) > 1 for r in rows),
          "| styles", dict(Counter(r["style"] for r in rows).most_common()))


if __name__ == "__main__":
    main()
