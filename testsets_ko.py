"""Korean IoT command evaluation set (same 11 tools as testsets.py). Never train on it.

Written by Claude on 2026-10-19 before any Korean IoT training data existed. Label
conventions (also the training-data rules in IOT_KO.md):
- rooms: 주방/부엌 -> kitchen, 거실 -> living_room, 침실/안방 -> bedroom, 화장실/욕실 -> bathroom
- names, cities, places, song queries and message bodies are copied as spoken (Korean);
  OneOf also accepts the English form, Text matches by containment
- times are 24-hour HH:MM, durations are converted to minutes, dates are absolute (with year)
- no relative dates: fine-tuned archives run without the SDK's date fact
"""
from testsets import C, OneOf, Text

KO_CASES = {
    "basic": [
        ("서울 날씨 어때?", [C("get_weather", city=OneOf("서울", "Seoul"))]),
        ("부산 지금 비 와?", [C("get_weather", city=OneOf("부산", "Busan"))]),
        ("15분 타이머 맞춰 줘", [C("set_timer", minutes=15)]),
        ("아침 6시 반에 깨워 줘", [C("set_alarm", time="06:30")]),
        ("오후 7시에 알람 맞춰", [C("set_alarm", time="19:00")]),
        ("엄마한테 늦는다고 문자 보내", [C("send_message", recipient=OneOf("엄마", "Mom"), message=Text("늦"))]),
        ("지수한테 전화해", [C("call_contact", name=OneOf("지수", "Jisu"))]),
        ("재즈 틀어 줘", [C("play_music", query=OneOf("재즈", "jazz"))]),
        ("공항으로 길 안내해 줘", [C("navigate_to", destination=OneOf("공항", "airport"))]),
        ("주방 불 켜", [C("control_lights", room="kitchen", action="on")]),
        ("침실 불 꺼 줘", [C("control_lights", room="bedroom", action="off")]),
        ("거실 불 좀 켜 줄래", [C("control_lights", room="living_room", action="on")]),
        ("온도 22도로 맞춰", [C("set_thermostat", temperature=22)]),
        ("볼륨 40으로 해", [C("set_volume", level=40)]),
        ("2026년 10월 14일에 치과 예약 일정 추가해 줘", [C("create_event", title=Text("치과"), date="2026-10-14")]),
    ],
    "args": [
        ("대구 날씨 섭씨로 알려 줘", [C("get_weather", city=OneOf("대구", "Daegu"), unit="celsius")]),
        ("2시간 타이머 맞춰", [C("set_timer", minutes=120)]),
        ("1시간 반 타이머 맞춰 줘", [C("set_timer", minutes=90)]),
        ("거실 조명 30%로 어둡게 해", [C("control_lights", room="living_room", action="dim", brightness_percent=30)]),
        ("침실 불 밝기 10퍼센트로 줄여", [C("control_lights", room="bedroom", action="dim", brightness_percent=10)]),
        ("밤 11시 15분에 알람", [C("set_alarm", time="23:15")]),
        ("기차역까지 걸어서 가는 길 알려 줘", [C("navigate_to", destination=Text("역"), mode="walking")]),
        ("버스로 박물관 가는 길 알려 줘", [C("navigate_to", destination=OneOf("박물관", "museum"), mode="transit")]),
        ("자전거로 해변 가는 길", [C("navigate_to", destination=OneOf("해변", "beach"), mode="cycling")]),
        ("볼륨 0으로 해", [C("set_volume", level=0)]),
        ("2026년 11월 3일 오전 9시에 팀 회의 일정 넣어 줘",
         [C("create_event", title=Text("팀 회의"), date="2026-11-03", time="09:00")]),
        ("온도 19도로 낮춰 줘", [C("set_thermostat", temperature=19)]),
    ],
    "casual": [
        ("부엌 불 꺼라", [C("control_lights", room="kitchen", action="off")]),
        ("화장실 불 좀 꺼줘요", [C("control_lights", room="bathroom", action="off")]),
        ("볼륨 80까지 올려", [C("set_volume", level=80)]),
        ("타이머 20분 ㄱㄱ", [C("set_timer", minutes=20)]),
        ("내일 아침 5시에 깨워줘", [C("set_alarm", time="05:00")]),
        ("날씨 어때 제주", [C("get_weather", city=OneOf("제주", "Jeju"))]),
        ("민수한테 전화 걸어", [C("call_contact", name=OneOf("민수", "Minsu"))]),
        ("아이유 노래 틀어", [C("play_music", query=Text("아이유"))]),
        ("너무 추워 난방 24도로", [C("set_thermostat", temperature=24)]),
        ("주방 조명 켜줭", [C("control_lights", room="kitchen", action="on")]),
    ],
    "multi": [
        ("주방 불 끄고 온도 20도로 맞춰",
         [C("control_lights", room="kitchen", action="off"), C("set_thermostat", temperature=20)]),
        ("10분 타이머 맞추고 클래식 틀어 줘",
         [C("set_timer", minutes=10), C("play_music", query=OneOf("클래식", "classical"))]),
        ("서울이랑 부산 날씨 알려 줘",
         [C("get_weather", city=OneOf("서울", "Seoul")), C("get_weather", city=OneOf("부산", "Busan"))]),
        ("침실이랑 화장실 불 켜",
         [C("control_lights", room="bedroom", action="on"), C("control_lights", room="bathroom", action="on")]),
        ("볼륨 20으로 하고 엄마한테 전화해",
         [C("set_volume", level=20), C("call_contact", name=OneOf("엄마", "Mom"))]),
        ("6시에 알람 맞추고 30분 타이머도", [C("set_alarm", time="06:00"), C("set_timer", minutes=30)]),
    ],
    "refusal": [
        ("선풍기 켜 줘", []),
        ("피자 주문해 줘", []),
        ("현관문 잠가", []),
        ("프랑스 수도가 어디야?", []),
        ("TV 켜", []),
        ("세탁기 돌려 줘", []),
        ("15 곱하기 23은?", []),
        ("농담 하나 해 줘", []),
        ("주방 불 켜지 마", []),
        ("로봇청소기 돌려", []),
        ("사진 찍어 줘", []),
        ("차고 문 열어", []),
    ],
}
