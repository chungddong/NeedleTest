# 한국어 IoT 명령 파인튜닝 (RTX 5080, 2026-10-19)

논문을 "Needle 모델을 써 본 결과"로 다시 쓴다: 툴콜링 모델 동향 → 여러 장비에서 IoT 명령 성능 → 한국어 파인튜닝 전후 비교.
이 문서는 그중 **한국어 파인튜닝**을 5080에서 하는 순서다. 영어 IoT 명령(`testsets.py`, 도구 11개, 92문항)과 **같은 도구 11개**로 한국어를 학습한다.

## 기준점 (원본 Needle 3, Pi 5, Python SDK)

한국어 IoT 평가셋(`testsets_ko.py`, 55문항): 정확 일치 **21.8%** (`results/ko_iot_base-pi5.json`)

| 분류 | 문항 | 정확 일치 |
|---|---|---|
| 기본 | 15 | 0.07 |
| 인자 변환 | 12 | 0.17 |
| 구어체·오타 | 10 | 0.10 |
| 다중 호출 | 6 | 0.00 |
| 거절 (선풍기, TV, 현관문 등) | 12 | 0.67 |

명령 43개 중 4개만 맞혔다. 거절 점수가 높은 것은 한국어를 못 알아들어 대부분 거절했기 때문이다.

## 지켜야 할 것

- **`testsets_ko.py`(평가셋)를 학습 데이터 생성에 쓰지 않는다.** 생성 에이전트에게 이 파일이나 그 문장을 보여주지 않는다. 영어 평가셋(`testsets.py`의 문장)도 마찬가지다.
- 학습 문장 하나가 도구 11개 정의 때문에 약 **900토큰**이다. `--max-len 1024`로 학습한다(512면 잘린다).
- 파인튜닝 모델은 날짜 정보 없이 실행된다(`auto_date=False`). 학습 데이터에 "내일", "다음 주" 같은 상대 날짜로 일정을 만드는 문장을 넣지 않는다.

## 준비

[DECISION_HEAD.md](DECISION_HEAD.md)의 "준비"와 같다 (WSL2, 저장소 clone, `.venv`, `jax[cuda12]` 최신, `jax.devices()`에 CudaDevice 확인).

## 1단계. 학습 데이터 생성 (Claude Code 에이전트)

**분량**: 원시 문장 약 2,400개 → 검사 후 약 2,000개 (검증 300 포함). 오늘 제출 일정에 맞춘 양이다. 에폭은 줄이지 않는다(정전 신고 실험에서 3에폭 108 대 8에폭 213).

**구성**
- 한국어 약 80%, 영어 약 20% (영어 능력 유지용, 영어 평가셋 문장과 다르게)
- 도구 11개(`testsets.TOOLS`)를 고르게. 조명·온도·볼륨·타이머·알람은 조금 더 많이
- 다중 호출(한 문장에 두 동작) 약 10%
- 거절 약 15%: 없는 기기(선풍기, 에어컨, TV, 세탁기, 로봇청소기, 도어락, 커튼), 할 수 없는 일(주문, 송금, 사진), 일반 질문·잡담, 부정문("주방 불 켜지 마")
- 화자 스타일: 존댓말, 반말, 짧은 명령, 오타·초성(ㄱㄱ), 음성 인식 오류, 사투리 조금, 영어 섞기

**라벨 규칙** (`testsets_ko.py` 머리말과 같다)
- 방: 주방·부엌 → `kitchen`, 거실 → `living_room`, 침실·안방 → `bedroom`, 화장실·욕실 → `bathroom` (그 외 방은 거절)
- 도시, 이름, 목적지, 노래 검색어, 메시지 받는 사람은 **말한 그대로** 옮긴다(번역하지 않음). 메시지 본문도 말한 내용 그대로
- 시간은 24시간 `HH:MM`, 기간은 분으로 변환(1시간 반 → 90), 온도 10~30, 볼륨·밝기 0~100
- 일정 날짜는 연도까지 말한 경우만 `YYYY-MM-DD`
- 단위·옵션은 말했을 때만 넣는다(`unit`, `mode`, `brightness_percent`)

**형식**: 원시 파일 `iot_ko/data/raw/*.jsonl`, 한 줄에 `{"query": "...", "answers": [{"name": "...", "arguments": {...}}], "style": "..."}`

**조립·검사**

```bash
.venv/bin/python iot_ko/assemble.py iot_ko/data/raw/*.jsonl
```

스키마 위반, 원문에 없는 이름·도시, 평가셋과 비슷한 문장(바이그램 유사도 0.45 이상), 중복, 상대 날짜를 걸러 내고 `iot_ko/data/train.jsonl`, `val.jsonl`(300)을 만든다. 추론 한 줄(`control_lights, set_thermostat` 또는 `no call`)은 자동으로 붙는다. 출력의 버린 이유와 도구별 개수를 보고, 많이 버려졌으면 생성 프롬프트를 고쳐 다시 만든다.

## 2단계. 학습

```bash
mkdir -p iot_ko/results
export XLA_PYTHON_CLIENT_PREALLOCATE=false
.venv/bin/needle finetune iot_ko/data/train.jsonl --epochs 8 --batch-size 8 --lr 1e-3 \
  --max-len 1024 --val-split 0 --out ko/out/iot_ko_lora.safetensors 2>&1 | tee iot_ko/results/train.log
```

설정은 정전 신고 실험에서 가장 좋았던 값(학습률 1e-3, 8에폭, 라벨만 쓰는 추론)이다. 메모리가 부족하면 `--batch-size 4`.

## 3단계. 내보내기 (20층만, 16층은 시간이 남으면)

```bash
.venv/bin/needle build checkpoints/needle3.safetensors --lora ko/out/iot_ko_lora.safetensors \
  --layers 20 --out ko/out/needle3-iot-ko-20L.cact
```

## 4단계. 5080에서 평가

```bash
.venv/bin/python ko_iot_eval.py --weights ko/out/needle3-iot-ko-20L.cact --label iot-ko-20L-pc5080
.venv/bin/python ko_iot_eval.py --weights - --label base-pc5080
.venv/bin/python bench.py --weights ko/out/needle3-iot-ko-20L.cact --label pc5080-iot-ko-20L   # 영어 유지 확인
```

Needle 엔진은 CPU에서 돈다. 그래서 5080 PC의 결과는 "x86 데스크톱 CPU" 환경의 지연·메모리로 쓴다.

**장비 비교용 영어 IoT 벤치마크(원본 모델)**도 5080 PC에서 한 번 돌린다:

```bash
.venv/bin/needle download needle3 --out models
.venv/bin/python bench.py --weights models/needle3.cact --label pc5080-base-20L-w2
```

## 5단계. Pi로 옮겨 측정

`.cact`를 Pi로 복사한다(`scp ko/out/needle3-iot-ko-*.cact chungman@<Pi 주소>:~/Develop/NeedleTest/models/`). Pi에서는 같은 `ko_iot_eval.py`와 `bench.py`로 지연·메모리를 잰다.

## 커밋할 것

`iot_ko/data/raw/`, `iot_ko/data/train.jsonl`, `val.jsonl`, `iot_ko/results/train.log`, `results/ko_iot_*.json`, `results/pc5080-*.json`. 모델 파일(`ko/out/`)은 커밋하지 않는다.

## 5080 Claude Code에 줄 요청 예

```
NeedleTest 저장소의 CLAUDE.md와 IOT_KO.md를 먼저 읽어줘.
IOT_KO.md 순서대로 한국어 IoT 명령 파인튜닝을 진행해줘.
1단계 학습 데이터는 Claude 에이전트로 생성하되 testsets_ko.py와 testsets.py의 문장은 절대 보여주지 마.
단계마다 결과를 요약해서 보여주고, 다음 단계로 넘어가기 전에 나한테 확인받아줘.
끝나면 결과 파일을 커밋하고 푸시해줘.
```
