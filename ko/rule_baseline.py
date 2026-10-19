"""Keyword/regex baseline for report_incident, scored like try_model.py.

A hand-written rule system: keyword lists pick the incident type (with the label
precedence fire > pole_down > line_down > spark > transformer_noise > outage),
question and negation cues refuse, a token heuristic cuts the location, and regexes
read household counts (digits and Korean number words) and hazard phrases.

These rules were written after reading the evaluation sets, which favours the
baseline; the comparison is conservative for the fine-tuned model.

Usage: python ko/rule_baseline.py ko/data/test_human.jsonl --out ko/results/rule_test_human.json
"""
import argparse
import json
import os
import re
import statistics
import time

TYPE_RULES = [
    ("fire", [r"불이\s*(났|붙)", r"불났", r"불\s*나요", r"불붙", r"불남", r"화재", r"불길", r"타고\s*있",
              r"caught fire", r"on fire"]),
    ("pole_down", [r"(전봇대|전주|전신주|pole).{0,12}(쓰러|넘어|기울|자빠|도괴|fell|leaning|down)",
                   r"(쓰러|넘어)\S*\s*(전봇대|전주)"]),
    ("line_down", [r"(전선|전깃줄|선로|고압선|power line|wire|line).{0,12}"
                   r"(끊|끈|늘어|처|떨어|내려앉|단선|잠겼|널려|down|snapped|hanging)",
                   r"(downed|fallen) (power )?line"]),
    ("spark", [r"불꽃", r"스파크", r"spark"]),
    ("transformer_noise", [r"(변압|transformer).{0,15}(소리|소음|울|윙|웅|웽|지지직|이상음|buzz|noise|hum)"]),
    ("outage", [r"정\s*전", r"전기\S*\s*(가|이)?\s*(다\s*|싹\s*다\s*|하나도\s*)?(안\s*(들어|와)|나가|나갔|없|끊)",
                r"전기\s*안", r"불\s*다\s*꺼", r"깜깜", r"꺼졌", r"power (is |went )?out", r"no electricity",
                r"outage", r"power out"]),
]
NOT_FIRE = re.compile(r"불은?\s*안\s*났")
REFUSE = re.compile(r"\?|어떻게|얼마|알려\s*주|언제|되나요|있나요|있어요\?|뭐예요|하고\s*싶|복구됐|괜찮|아니에요|아니고|"
                    r"아닌|돌아왔|들어왔|사칭|상담|신청|버텨|문자 왔|scheduled|back on", re.I)
STRONG_REPORT = re.compile(r"쓰러졌|넘어졌|끊어졌|불이 났|불났")

HAZARD_ELEC = re.compile(r"감전|위험|electrocut|shock|dangerous", re.I)
HAZARD_ELEC_NEG = re.compile(r"위험해\s*보이진\s*않|안\s*위험|위험하지\s*않")
HAZARD_FIRE = re.compile(r"불\s*날\s*것|불날\s*것|불이\s*날|화재\s*위험|start a fire", re.I)

SINO = {"일": 1, "이": 2, "삼": 3, "사": 4, "오": 5, "육": 6, "칠": 7, "팔": 8, "구": 9}
NATIVE_TENS = {"열": 10, "스물": 20, "스무": 20, "서른": 30, "마흔": 40, "쉰": 50, "예순": 60, "일흔": 70,
               "여든": 80, "아흔": 90}
NATIVE_ONES = {"한": 1, "하나": 1, "두": 2, "둘": 2, "세": 3, "셋": 3, "네": 4, "넷": 4, "다섯": 5,
               "여섯": 6, "일곱": 7, "여덟": 8, "아홉": 9}


def sino_number(s):
    s = s.replace(" ", "")
    if not s or any(c not in "일이삼사오육칠팔구십백천" for c in s):
        return None
    total, cur = 0, 0
    for c in s:
        if c in SINO:
            cur = SINO[c]
        else:
            unit = {"십": 10, "백": 100, "천": 1000}[c]
            total += (cur or 1) * unit
            cur = 0
    return total + cur


def native_number(s):
    s = s.replace(" ", "")
    for tens, tv in sorted(NATIVE_TENS.items(), key=lambda kv: -len(kv[0])):
        if s.startswith(tens):
            rest = s[len(tens):]
            if not rest:
                return tv
            return tv + NATIVE_ONES[rest] if rest in NATIVE_ONES else None
    return NATIVE_ONES.get(s)


HOUSE = re.compile(r"(\d+|[일이삼사오육칠팔구십백천 ]+?|[가-힣]+?)\s*여?\s*(가구|세대|집|households)")


def households(text):
    for m in HOUSE.finditer(text):
        raw = m.group(1).strip()
        if raw.isdigit():
            return int(raw)
        for word in (raw, raw.split()[-1] if raw.split() else raw):
            n = sino_number(word) or native_number(word)
            if n:
                return n
    return None


EVENT_WORDS = re.compile(r"전봇대|전신주|전주|전선|전깃줄|변압\s*기|변압기|선로|고압선|배전함|정\s*전|전기|불꽃|스파크|"
                         r"쓰러|넘어|기울|끊|끈|늘어|처져|소리|소음|불|화재|감전|위험|확인|발생|접수|신고|"
                         r"\d+\s*(가구|세대)|가구|세대|집|요청|필요|통제|영향|빨리|진압|도괴|단선|이상음|처짐|민원|점검")
FILLER = {"지금", "저희", "여기", "현장보고", "현장", "결과", "사장님", "선생님", "안녕하세요", "우리", "아까부터", "제발"}
PARTICLE = re.compile(r"(에서|에|의|은|는|이|가|을|를|로|으로|쪽에|까지)$")
EN_LOC = re.compile(r"\b(?:at|on|in|near|by|behind|in front of)\s+((?:the\s+)?[A-Z][\w-]*(?:\s+[A-Z][\w-]*)*|the [a-z]+(?: [a-z]+)?)")


def location(text):
    m = EN_LOC.search(text)
    if m and re.search(r"[A-Za-z]{4,}", text) and not re.search(r"[가-힣]", text):
        return m.group(1).rstrip(",. ")
    tokens = re.findall(r"\S+", re.split(r"[.,]", text)[0] if text.count(".") > 1 else text.replace(",", " "))
    runs, cur = [], []
    for tok in tokens:
        if EVENT_WORDS.search(tok) or tok in FILLER:
            if cur:
                runs.append(cur)
            cur = []
        else:
            cur.append(tok)
    if cur:
        runs.append(cur)
    if not runs:
        return None
    best = runs[0]
    words = list(best)
    words[-1] = PARTICLE.sub("", words[-1]) or words[-1]
    return " ".join(words)


def predict(text):
    found = None
    for name, pats in TYPE_RULES:
        if any(re.search(p, text, re.I) for p in pats):
            if name == "fire" and NOT_FIRE.search(text):
                continue
            found = name
            break
    if found is None or (REFUSE.search(text) and not STRONG_REPORT.search(text)):
        return []
    args = {"incident_type": found}
    loc = location(text)
    if loc:
        args["location"] = loc
    n = households(text)
    if n:
        args["households"] = n
    if found == "fire" or HAZARD_FIRE.search(text):
        args["hazard"] = "fire"
    elif HAZARD_ELEC.search(text) and not HAZARD_ELEC_NEG.search(text):
        args["hazard"] = "electrocution"
    return [{"name": "report_incident", "arguments": args}]


def first_args(calls):
    return (calls[0].get("arguments") or {}) if calls else {}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("data")
    ap.add_argument("--out")
    a = ap.parse_args()
    rows = [json.loads(line) for line in open(a.data, encoding="utf-8")]
    results = []
    for row in rows:
        t0 = time.perf_counter()
        got = predict(row["query"])
        ms = (time.perf_counter() - t0) * 1000
        want = row["answers"]
        w, g = first_args(want), first_args(got)
        results.append({"query": row["query"], "want": want, "got": got, "error": None,
                        "latency_ms": round(ms, 3), "exact": got == want, "decision": bool(got) == bool(want),
                        "incident_type": bool(want) and g.get("incident_type") == w.get("incident_type"),
                        "location": bool(want) and g.get("location") == w.get("location")})
    pos = [r for r in results if r["want"]]
    scores = {"n": len(results), "exact": sum(r["exact"] for r in results),
              "decision": sum(r["decision"] for r in results), "positives": len(pos),
              "incident_type": sum(r["incident_type"] for r in pos), "location": sum(r["location"] for r in pos),
              "errors": 0, "latency_p50_ms": round(statistics.median(r["latency_ms"] for r in results), 3)}
    print(f"rule: exact {scores['exact']}/{scores['n']} decision {scores['decision']} "
          f"type {scores['incident_type']}/{scores['positives']} loc {scores['location']}/{scores['positives']}")
    if a.out:
        with open(a.out, "w", encoding="utf-8") as f:
            json.dump({"model": "rule_baseline", "data": a.data, "scores": scores, "rows": results},
                      f, ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
