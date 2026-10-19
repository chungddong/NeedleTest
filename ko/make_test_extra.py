"""A second, harder evaluation set for report_incident (2026-10-03). Never train on it.

Written by Claude (not by a person) without looking at train.jsonl, following the
generator's label rules (ko/gen_data.py PROMPT): one call per report with
fire > pole_down > line_down > spark > transformer_noise > outage; location copied
verbatim, keeping 앞/옆/근처-style words and dropping trailing particles and equipment
names (English drops prepositions); households only when a count is stated (not
apartment unit numbers); hazard only when danger is stated (sparks alone are not).

It leans on cases test_human.jsonl covers thinly: number words, hazard edge cases,
several phenomena in one report, long multi-sentence reports, Jeju and Gangwon dialect,
and refusals that contain outage keywords. Each row carries a `category`.
Writes ko/data/test_extra.jsonl.
"""
import json
import os

from make_smoke_data import DATA_DIR, call
from schema import TOOLS_JSON

R = []


def add(category, query, answers):
    R.append((category, query, answers))


# standard Korean
for q, a in [
    ("덕천동 주공아파트 쪽 지금 전기가 하나도 안 들어와요", call("outage", "덕천동 주공아파트 쪽")),
    ("연산로 버스터미널 앞 가로등이랑 가게들 다 꺼졌습니다", call("outage", "연산로 버스터미널 앞")),
    ("삼거리 약국 옆 전봇대가 도로 쪽으로 기울어 있어요", call("pole_down", "삼거리 약국 옆")),
    ("구름다리 아래 전선이 끊어져서 물웅덩이에 닿아 있어요 감전될까 봐 무서워요",
     call("line_down", "구름다리 아래", hazard="electrocution")),
    ("성당 주차장 변압기에서 지지직 소리가 계속 나요", call("transformer_noise", "성당 주차장")),
    ("체육공원 산책로 전선에서 불꽃이 번쩍번쩍해요", call("spark", "체육공원 산책로")),
    ("물류창고 뒤 전신주에 불이 붙어서 타고 있어요", call("fire", "물류창고 뒤", hazard="fire")),
    ("도서관 앞 사거리 정전이에요", call("outage", "도서관 앞 사거리")),
    ("강변 캠핑장 쪽 전깃줄이 축 처져서 땅에 거의 닿아요", call("line_down", "강변 캠핑장 쪽")),
    ("제2공단 입구 전주 두 개가 넘어졌어요", call("pole_down", "제2공단 입구")),
    ("중학교 운동장 옆 변압기에서 연기가 나고 불이 붙었어요", call("fire", "중학교 운동장 옆", hazard="fire")),
    ("목련아파트 108동 엘리베이터 멈추고 전기 나갔어요", call("outage", "목련아파트 108동")),
    ("시외버스 정류장 근처에서 스파크가 튀어요 불날 것 같아요", call("spark", "시외버스 정류장 근처", hazard="fire")),
    ("은행 건너편 전선이 바람에 끊어져 나풀거려요", call("line_down", "은행 건너편")),
    ("우체국 주변 정전입니다", call("outage", "우체국 주변")),
    ("논길 옆 변압기에서 웅 하는 소리가 밤새 났어요", call("transformer_noise", "논길 옆")),
]:
    add("표준어", q, a)

# dialects (Gyeongsang, Jeolla, Chungcheong, Jeju, Gangwon)
for q, a in [
    ("골목 안쪽 집들 전기가 마 다 나가뿟다", call("outage", "골목 안쪽")),
    ("다리 건너 전봇대가 자빠졌다 아이가", call("pole_down", "다리 건너")),
    ("시장통 전깃줄이 끊기가 길바닥에 널려 있다 위험하다", call("line_down", "시장통", hazard="electrocution")),
    ("공장 옆 변압기에 불났다 퍼뜩 온나", call("fire", "공장 옆", hazard="fire")),
    ("정자나무 옆 변압기가 징하게 울어부러", call("transformer_noise", "정자나무 옆")),
    ("방죽 너머 전봇대에서 불꽃이 튀어부러야", call("spark", "방죽 너머")),
    ("우리 마을 스무 가구 전기가 다 나가부렀당께", call("outage", "우리 마을", households=20)),
    ("학교 앞 전선이 끊어져갖고 불꽃이 튀어부러", call("line_down", "학교 앞")),
    ("저수지 둑 전깃줄이 늘어졌슈 애들 감전될까 겁나유", call("line_down", "저수지 둑", hazard="electrocution")),
    ("회관 뒷길 전봇대가 넘어갔슈", call("pole_down", "회관 뒷길")),
    ("읍내 장터 전기가 다 나갔어유", call("outage", "읍내 장터")),
    ("마을 입구 정전됐는디 서른 집쯤 될 거유", call("outage", "마을 입구", households=30)),
    ("올레길 입구 전깃줄 끊어젼 이수다", call("line_down", "올레길 입구")),
    ("알동네 전기 다 나가수다", call("outage", "알동네")),
    ("산 밑 펜션 쪽 전기가 안 들어와요 마카 깜깜하드래요", call("outage", "산 밑 펜션 쪽")),
]:
    add("사투리", q, a)

# typos, missing spaces, speech-to-text splits
for q, a in [
    ("중앙동 정전됫어요", call("outage", "중앙동")),
    ("전봇대 쓰러짐 감나무집앞", call("pole_down", "감나무집앞")),
    ("변압기 소리남 3단지", call("transformer_noise", "3단지")),
    ("전선끈김 위험 소방서뒤", call("line_down", "소방서뒤", hazard="electrocution")),
    ("해오름 아파트 정 전 됐어요", call("outage", "해오름 아파트")),
    ("전 봇대가 넘어 졌어요 철길 옆에", call("pole_down", "철길 옆")),
    ("스파크남 빨리와주세요 편의점앞", call("spark", "편의점앞")),
    ("불남 변압기 불남 공원 정문", call("fire", "공원 정문", hazard="fire")),
    ("정전 15가구 신평리", call("outage", "신평리", households=15)),
    ("변압 기에서 이상한 소리 나요 상가 뒤편", call("transformer_noise", "상가 뒤편")),
    ("전기안들어옴 매화빌라 2층", call("outage", "매화빌라 2층")),
    ("전선 내려앉음 하천변", call("line_down", "하천변")),
]:
    add("오타·STT", q, a)

# household counts in different forms
for q, a in [
    ("가구 수로는 마흔 가구 정도 정전이에요 대명리 쪽", call("outage", "대명리 쪽", households=40)),
    ("솔밭마을 50여 세대 전기가 끊겼습니다", call("outage", "솔밭마을", households=50)),
    ("벚꽃아파트 3동 전체 정전 백이십 세대예요", call("outage", "벚꽃아파트 3동", households=120)),
    ("7가구가 정전입니다 은하빌라", call("outage", "은하빌라", households=7)),
    ("삼백 가구 넘게 전기가 나갔어요 신도시 상업지구", call("outage", "신도시 상업지구", households=300)),
    ("무지개맨션 302호 전기 안 들어와요", call("outage", "무지개맨션 302호")),
    ("해안도로 펜션 단지 열두 집 정전", call("outage", "해안도로 펜션 단지", households=12)),
    ("장미아파트 2동 정전 80세대", call("outage", "장미아파트 2동", households=80)),
]:
    add("숫자 표현", q, a)

# hazard edge cases
for q, a in [
    ("전선에서 불꽃이 튀는데 옆에 마른 풀이 많아서 불날 것 같아요 등산로 입구",
     call("spark", "등산로 입구", hazard="fire")),
    ("배수로 옆 전선이 끊어졌는데 위험해 보이진 않아요", call("line_down", "배수로 옆")),
    ("놀이터 옆 전봇대 쓰러짐 애들 감전될까 봐 걱정이에요", call("pole_down", "놀이터 옆", hazard="electrocution")),
    ("주택가 골목 변압기에서 탁탁 소리 나면서 불꽃 튀어요", call("spark", "주택가 골목")),
    ("비닐하우스 옆 전선이 물에 잠겼어요 감전 위험 있어요", call("line_down", "비닐하우스 옆", hazard="electrocution")),
    ("축사 근처 전봇대에 불이 났어요", call("fire", "축사 근처", hazard="fire")),
    ("주민센터 앞 불꽃 튐 불은 안 났어요", call("spark", "주민센터 앞")),
    ("아파트 정문 전선 끊어져서 위험해요", call("line_down", "아파트 정문", hazard="electrocution")),
]:
    add("위험 경계", q, a)

# several phenomena in one report (precedence)
for q, a in [
    ("전봇대가 쓰러지면서 동네 전체가 정전됐어요 감리 마을", call("pole_down", "감리 마을")),
    ("변압기 소리가 나더니 정전됐어요 청솔빌라", call("transformer_noise", "청솔빌라")),
    ("전선이 끊어지면서 스파크가 튀었어요 고가도로 밑", call("line_down", "고가도로 밑")),
    ("정전되고 나서 전봇대에서 불이 났어요 미곡처리장 앞", call("fire", "미곡처리장 앞", hazard="fire")),
    ("공원 화장실 쪽 전선에서 불꽃이 튀다가 결국 정전됐어요", call("spark", "공원 화장실 쪽")),
    ("전봇대가 기울고 전선도 늘어졌어요 과수원 길", call("pole_down", "과수원 길")),
    ("아파트 단지 정전인데 변압기에서 불길이 보여요", call("fire", "아파트 단지", hazard="fire")),
    ("바람에 전주가 넘어가고 끊어진 전선에서 불꽃이 튀어요 위험해요 터널 입구",
     call("pole_down", "터널 입구", hazard="electrocution")),
]:
    add("복합 현상", q, a)

# long, multi-sentence reports
for q, a in [
    ("안녕하세요 지금 태풍 때문에 그런지 저희 동네 나루터 마을이 한 시간째 전기가 안 들어와요. 냉장고도 다 꺼졌어요.",
     call("outage", "나루터 마을")),
    ("밤새 바람이 엄청 불었는데 아침에 나가 보니 버스 차고지 앞에 전봇대가 쓰러져 있어요. 차들이 피해서 다니고 있어요.",
     call("pole_down", "버스 차고지 앞")),
    ("어르신 혼자 사시는 집인데요, 골목 끝 집 옆 전선이 끊어져서 마당에 떨어져 있대요. 감전될까 봐 나가지도 못하세요.",
     call("line_down", "골목 끝 집 옆", hazard="electrocution")),
    ("지금 퇴근하는 길인데 기차역 광장 쪽 변압기에서 펑 소리 나고 불꽃이 튀었어요. 다친 사람은 없어요.",
     call("spark", "기차역 광장 쪽")),
    ("산불이 번지면서 능선 아래 마을 입구 전선에 불이 붙었어요. 빨리 와주세요.",
     call("fire", "능선 아래 마을 입구", hazard="fire")),
    ("저희 가게가 시장 안쪽에 있는데요, 아까부터 시장 안쪽 가게들 전기가 다 나갔어요. 스물다섯 집 정도 돼요.",
     call("outage", "시장 안쪽", households=25)),
]:
    add("긴 설명", q, a)

# field-worker shorthand
for q, a in [
    ("현장 확인 결과 동림리 마을회관 앞 전주 도괴 1기", call("pole_down", "동림리 마을회관 앞")),
    ("고압선 단선 확인 서창교 북단 감전 위험 통제 중", call("line_down", "서창교 북단", hazard="electrocution")),
    ("변압기 이상음 접수 건 현장 도착 운암동 아파트 후문", call("transformer_noise", "운암동 아파트 후문")),
    ("배전함 스파크 발생 신가동 공구상가", call("spark", "신가동 공구상가")),
    ("현장보고 첨단산단 3로 전주 화재 진압 요청", call("fire", "첨단산단 3로", hazard="fire")),
    ("용봉동 일대 정전 확인 60가구 영향", call("outage", "용봉동 일대", households=60)),
    ("선로 처짐 확인 월곡천 산책로 구간", call("line_down", "월곡천 산책로 구간")),
    ("소음 민원 변압기 점검 필요 풍암 저수지 입구", call("transformer_noise", "풍암 저수지 입구")),
]:
    add("현장 보고", q, a)

# non-native Korean, English, mixed
for q, a in [
    ("사장님 공장 전기 없어요 하남산단 7번로", call("outage", "하남산단 7번로")),
    ("전봇대 넘어짐 위험해 원룸 앞", call("pole_down", "원룸 앞", hazard="electrocution")),
    ("불 나요 전선 불 나요 마트 뒤", call("fire", "마트 뒤", hazard="fire")),
    ("선생님 변압기 소리 커요 무서워요 기숙사 옆", call("transformer_noise", "기숙사 옆")),
    ("No electricity at Sunrise Hostel since this morning", call("outage", "Sunrise Hostel")),
    ("There's a pole leaning over the road near the gas station", call("pole_down", "the gas station")),
    ("Sparks are flying from a wire on Maple Street, it might start a fire",
     call("spark", "Maple Street", hazard="fire")),
    ("The transformer behind the community center is making a loud buzzing noise",
     call("transformer_noise", "the community center")),
    ("Downed power line in front of the elementary school, kids could get electrocuted",
     call("line_down", "the elementary school", hazard="electrocution")),
    ("Power outage affecting about 40 households in Riverside Village",
     call("outage", "Riverside Village", households=40)),
    ("아파트 101동 power out 했어요", call("outage", "아파트 101동")),
]:
    add("외국인·영어", q, a)

# refusals: unrelated questions, resolved or negated reports, keyword traps
for q in [
    "정전되면 냉장고 음식은 얼마나 버텨요?",
    "지난주 정전 피해 보상 받을 수 있나요?",
    "다음 주 화요일 계획 정전 시간 알려주세요",
    "전기요금 자동이체 변경하고 싶어요",
    "아까 깜빡했는데 지금은 괜찮아요",
    "전봇대 괜찮아요 쓰러진 거 아니에요",
    "누진세 구간이 어떻게 돼요?",
    "태양광 설치 상담 받고 싶어요",
    "뉴스 보니까 강원도에 큰 정전 났다던데 우리 동네는 괜찮나요?",
    "우리 집 두꺼비집이 자꾸 내려가요 어떻게 해요?",
    "전기차 충전 요금 얼마예요?",
    "정전 문자 왔는데 이거 진짜예요?",
    "비 오는데 전봇대 근처 가도 돼요?",
    "한전 직원 사칭 전화 받았어요",
    "이사 갈 집 전기 개통 신청은 어떻게 해요?",
    "불꽃놀이 축제 언제 해요?",
    "변압기가 뭐예요?",
    "정전 대비 손전등 어디서 사요?",
    "어제 정전됐던 거 복구됐어요 감사합니다",
    "Is there a scheduled power cut in my area tomorrow?",
]:
    add("거절 대상", q, [])


def check():
    enums = {p: set(TOOLS_JSON[0]["parameters"]["properties"][p].get("enum", []))
             for p in ("incident_type", "hazard")}
    seen = set()
    for cat, q, answers in R:
        assert q not in seen, f"duplicate: {q}"
        seen.add(q)
        for a in answers:
            args = a["arguments"]
            assert args["location"] in q, f"location not verbatim: {q} / {args['location']}"
            assert args["incident_type"] in enums["incident_type"], q
            assert args.get("hazard") in (None, *enums["hazard"]), q
            if args["incident_type"] == "fire":
                assert args.get("hazard") == "fire", f"fire without hazard fire: {q}"


def main():
    check()
    path = os.path.join(DATA_DIR, "test_extra.jsonl")
    with open(path, "w", encoding="utf-8") as f:
        for cat, q, answers in R:
            f.write(json.dumps({"query": q, "tools": TOOLS_JSON, "answers": answers, "category": cat},
                               ensure_ascii=False) + "\n")
    print(f"wrote {len(R)} examples to {path}")


if __name__ == "__main__":
    main()
