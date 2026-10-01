"""Generate Korean training data for report_incident through OpenRouter.

The package's own `needle generate-data` prompt is English, so it yields English
queries. This prompt asks for the styles real outage reports come in: standard
Korean, dialects, typos and speech-to-text noise, short texts, field-worker shorthand,
non-native Korean, and some English, plus refusals.

Usage (on the GPU server):
  export OPENROUTER_API_KEY=...        # your own key; generation is billed by OpenRouter
  python ko/gen_data.py --num 3000 --out ko/data/train.jsonl
"""
import argparse
import json
import os
import random
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, os.path.dirname(__file__))

from needle.model.finetune import _openrouter, _parse_array, DEFAULT_MODEL
from schema import TOOLS_JSON

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
{{"query": "<신고 문장>", "answers": [{{"name": "report_incident", "arguments": {{...}}}}]}}

규칙:
- arguments의 키와 incident_type, hazard 값은 스키마의 영어 값만 씁니다.
- location은 신고자가 말한 표현을 그대로 옮깁니다 (번역하거나 다듬지 않음).
- households는 가구·세대 수를 직접 말한 경우에만 넣습니다.
- hazard는 감전이나 화재 위험을 직접 언급한 경우에만 넣습니다.
- 지명은 실제와 가상을 섞고 매번 다르게 합니다 (OO리, OO동, 아파트 동호수, 시장, 학교, 다리 등).
- {refusals}개 정도는 정전·설비 피해와 무관하거나(요금 문의, 명의 변경, 날씨), 정전이 아니라는 내용으로 만들고 answers를 []로 둡니다.
- JSON 배열만 출력합니다."""


def batch(style, n, model, api_key):
    text = _openrouter([{"role": "user", "content": PROMPT.format(
        tools=json.dumps(TOOLS_JSON, ensure_ascii=False, indent=1), n=n, style=style,
        refusals=max(1, n // 8))}], model, api_key)
    rows = _parse_array(text)
    for row in rows:
        row["tools"] = TOOLS_JSON
        row["style"] = style
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--num", type=int, default=3000)
    ap.add_argument("--batch", type=int, default=25)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--out", default="ko/data/train.jsonl")
    args = ap.parse_args()
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise SystemExit("set OPENROUTER_API_KEY")

    jobs = [random.choice(STYLES) for _ in range(-(-args.num * 13 // 10 // args.batch))]
    seen, rows = set(), []
    with ThreadPoolExecutor(args.workers) as pool:
        futures = [pool.submit(batch, s, args.batch, args.model, api_key) for s in jobs]
        for fut in as_completed(futures):
            try:
                for row in fut.result():
                    key = row["query"].strip()
                    if key not in seen:
                        seen.add(key)
                        rows.append(row)
            except Exception as exc:
                print("failed:", exc, flush=True)
            print(f"{len(rows)} examples", flush=True)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        for row in rows[:args.num]:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"wrote {min(len(rows), args.num)} examples to {args.out}")


if __name__ == "__main__":
    main()
