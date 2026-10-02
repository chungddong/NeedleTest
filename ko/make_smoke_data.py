"""A tiny hand-written Korean dataset that only proves the fine-tune pipeline runs.

It is far too small to teach the model anything; the real dataset is built on the
GPU server (see FINETUNE.md). Writes ko/data/smoke.jsonl in the
`needle finetune` format: {"query", "tools", "reasoning", "answers"}.
Each row has a short English evidence phrase that schema.reasoning() turns into
the row's reasoning line.
"""
import json
import os

from schema import TOOLS_JSON, reasoning

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")


def call(incident_type, location, households=None, hazard=None):
    args = {"incident_type": incident_type, "location": location}
    if households is not None:
        args["households"] = households
    if hazard is not None:
        args["hazard"] = hazard
    return [{"name": "report_incident", "arguments": args}]


ROWS = [
    ("행복동 일대 정전됐어요", "power is out", call("outage", "행복동")),
    ("햇살아파트 102동 전기가 다 나갔어요", "power is all out", call("outage", "햇살아파트 102동")),
    ("신촌리 마을 30가구 정전입니다", "power out, 30 households stated",
     call("outage", "신촌리 마을", households=30)),
    ("OO리 전봇대가 쓰러졌어요 전선이 바닥에 있어요 위험해요",
     "pole fell, wire on the ground, says it is dangerous",
     call("pole_down", "OO리", hazard="electrocution")),
    ("중앙시장 앞 전주 넘어졌습니다", "utility pole fell over", call("pole_down", "중앙시장 앞")),
    ("학교 뒤편 전선이 끊어져서 늘어져 있어요", "power line snapped and hanging",
     call("line_down", "학교 뒤편")),
    ("다리 옆 전선 끊김", "power line cut", call("line_down", "다리 옆")),
    ("우리 집 앞 변압기에서 윙윙 소리가 엄청 나요", "transformer humming loudly",
     call("transformer_noise", "우리 집 앞")),
    ("버스정류장 위 변압기 소음 신고합니다", "reports transformer noise",
     call("transformer_noise", "버스정류장 위")),
    ("공원 입구 전주에서 불꽃이 튀어요", "sparks from a pole", call("spark", "공원 입구")),
    ("주유소 옆 전선에서 스파크가 나요 불날 것 같아요", "sparks from a line, says a fire may start",
     call("spark", "주유소 옆", hazard="fire")),
    ("산 밑 변압기에 불이 붙었어요", "transformer caught fire", call("fire", "산 밑", hazard="fire")),
    ("윗마을 전기가 다 나가삔다 아이가", "power is all out", call("outage", "윗마을")),
    ("아랫동네 전기 안 들어온 지 두 시간 됐슈", "no power for two hours", call("outage", "아랫동네")),
    ("동사무소 근처 50세대 정전", "power out, 50 households stated",
     call("outage", "동사무소 근처", households=50)),
    ("Power is out on Main Street", "power is out", call("outage", "Main Street")),
    ("A utility pole fell near the river bridge", "utility pole fell",
     call("pole_down", "the river bridge")),
    ("Sparks coming from the transformer by the school", "sparks from a transformer",
     call("spark", "the school")),
    ("오늘 날씨 어때요?", "asks about the weather", []),
    ("전기요금 얼마 나왔는지 알려줘", "asks about the electricity bill", []),
    ("정전 아니에요 잘 들어와요", "says the power is fine", []),
    ("이사 가는데 전기 명의 변경하고 싶어요", "wants to change the account holder", []),
    ("편의점 앞 전선이 축 늘어졌어요", "power line sagging", call("line_down", "편의점 앞")),
    ("새마을회관 주변 정전 20가구 정도", "power out, about 20 households stated",
     call("outage", "새마을회관 주변", households=20)),
]


def main():
    os.makedirs(DATA_DIR, exist_ok=True)
    path = os.path.join(DATA_DIR, "smoke.jsonl")
    with open(path, "w") as f:
        for query, evidence, answers in ROWS:
            f.write(json.dumps({"query": query, "tools": TOOLS_JSON,
                                "reasoning": reasoning(evidence, answers), "answers": answers},
                               ensure_ascii=False) + "\n")
    print(f"wrote {len(ROWS)} examples to {path}")


if __name__ == "__main__":
    main()
