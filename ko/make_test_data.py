"""Hand-written Korean evaluation set for report_incident. Never train on it.

Written separately from the generated training data so the score reflects
generalisation. Extend it with phrasing collected from news and community posts.
Writes ko/data/test_human.jsonl.
"""
import json
import os

from schema import TOOLS_JSON
from make_smoke_data import DATA_DIR, call

ROWS = [
    # standard Korean
    ("남산동 3통 지금 정전이에요", call("outage", "남산동 3통")),
    ("푸른마을 아파트 전체가 정전됐습니다 300세대쯤 돼요", call("outage", "푸른마을 아파트", households=300)),
    ("시청 사거리 신호등이랑 상가 전부 전기가 나갔어요", call("outage", "시청 사거리")),
    ("교회 옆 전봇대가 바람에 쓰러졌어요", call("pole_down", "교회 옆")),
    ("저수지 가는 길에 전주가 기울어서 넘어갔어요 전선이 길에 떨어져 있어요",
     call("pole_down", "저수지 가는 길", hazard="electrocution")),
    ("농협 창고 앞 전선이 끊어졌어요", call("line_down", "농협 창고 앞")),
    ("마을회관 뒤 변압기에서 이상한 소리가 계속 나요", call("transformer_noise", "마을회관 뒤")),
    ("초등학교 정문 앞 전선에서 파란 불꽃이 튀어요", call("spark", "초등학교 정문 앞")),
    ("야산 쪽 전주에 불이 났어요 연기가 많이 나요", call("fire", "야산 쪽", hazard="fire")),
    ("한빛빌라 가동 나동 정전 12가구", call("outage", "한빛빌라 가동 나동", households=12)),
    # dialect
    ("우리 동네 전기가 싹 다 나가부렀어라", call("outage", "우리 동네")),
    ("저 아래 전봇대가 넘어져뿟다 퍼뜩 와보이소", call("pole_down", "저 아래")),
    ("밭 옆에 전깃줄이 끊어져가 늘어져 있어유 위험혀유", call("line_down", "밭 옆", hazard="electrocution")),
    ("회관 앞 변압기가 웽웽 울어쌓는다", call("transformer_noise", "회관 앞")),
    # typos, short texts, speech-to-text noise
    ("정전요 행복빌라", call("outage", "행복빌라")),
    ("새싹유치원 앞 전봇데 쓰러짐", call("pole_down", "새싹유치원 앞")),
    ("전선 끈어짐 은행나무길", call("line_down", "은행나무길")),
    ("시장 입구에 스파크 나요 빨리요", call("spark", "시장 입구")),
    ("햇빛 마을 정전 이 십 가구", call("outage", "햇빛 마을", households=20)),
    # field worker shorthand
    ("현장보고 송정리 2번 전주 도괴 확인", call("pole_down", "송정리 2번 전주")),
    ("대촌동 변압기 소음 민원 현장 확인 완료", call("transformer_noise", "대촌동")),
    ("배전선로 단선 금곡교 인근 감전 주의 필요", call("line_down", "금곡교 인근", hazard="electrocution")),
    # non-native Korean and English
    ("여기 공장 전기 없어요 아파요 일 못해요 산업단지 3로", call("outage", "산업단지 3로")),
    ("기숙사 전기 안 와요 불 다 꺼졌어요", call("outage", "기숙사")),
    ("The power went out in Green Valley apartments", call("outage", "Green Valley apartments")),
    ("A power line is down on Harbor Road, someone could get shocked",
     call("line_down", "Harbor Road", hazard="electrocution")),
    # refusals and negatives
    ("이번 달 전기요금 왜 이렇게 많이 나왔어요?", []),
    ("태풍 언제 지나가요?", []),
    ("아까 정전됐다가 지금은 다시 들어왔어요 괜찮아요", []),
    ("전기차 충전소 어디 있어요?", []),
    ("한전 고객센터 운영시간 알려주세요", []),
    ("정전인 줄 알았는데 우리 집 차단기가 내려간 거였어요", []),
]


def main():
    os.makedirs(DATA_DIR, exist_ok=True)
    path = os.path.join(DATA_DIR, "test_human.jsonl")
    with open(path, "w") as f:
        for query, answers in ROWS:
            f.write(json.dumps({"query": query, "tools": TOOLS_JSON, "answers": answers},
                               ensure_ascii=False) + "\n")
    print(f"wrote {len(ROWS)} examples to {path}")


if __name__ == "__main__":
    main()
