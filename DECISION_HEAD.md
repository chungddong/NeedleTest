# Needle 판단 헤드 실험 (RTX 5080, 2026-10-19)

## 질문

한국어로 파인튜닝한 Needle 3(v2)를 **Laya·Jev 같은 가벼운 의사결정 모델로 쓸 수 있는가?**
도구 호출을 생성하지 않고, 모델을 한 번만 통과시켜 얻은 내부 상태 위에 작은 분류기(판단 헤드)를 붙여 질문에 답하게 한다.

## 왜 이 방식인가 (Pi에서 확인한 사실)

- Needle 3의 내장 임베딩(`agent.embed`)은 **한국어 문장을 구분하지 못한다.** 정전 신고와 요금 문의의 코사인 유사도가 0.988로, 정전 신고끼리(0.987)와 같다.
- 파인튜닝해서 내보낸 모델(`.cact`)에는 **임베딩 헤드가 없다** (`needle_embed failed (code -1)`). 공개 학습 체크포인트에 확신도 헤드만 있고 임베딩·라우터 헤드는 없기 때문이다.
- 그래서 엔진의 임베딩 대신 **학습 환경(JAX)에서 은닉 상태를 직접 꺼내** 판단 헤드를 학습한다. 결과가 좋으면 그 헤드를 엔진에서 쓰게 만드는 작업이 다음 단계다.
- 마지막 층(20층)의 은닉 상태에는 값이 1e7에 달하는 차원이 몇 개 있다. 스크립트는 차원별로 표준화한 뒤 ±10으로 자른다.

비교 대상 Laya(다국어판, Pi 5 CPU 실측): 메모리 2.1GB, 명령당 1.4~2.6초, 학습 없이 7/14 (`~/Develop/edge-ko-command/results/laya_smoke_pi5.json`).

## 판단 헤드가 넘어야 할 기준 (v2 생성 방식, Pi C 런타임)

| | 기본 평가셋 32 | 확장 평가셋 112 |
|---|---|---|
| 사건 유형 7가지 (6유형 + 거절) | 0.906 | 0.812 |
| 위험 3가지 (없음/감전/화재) | 0.906 | 0.929 |
| 참고: 원본 모델 사건 유형 | 0.250 | 0.241 |

위험은 대부분 "없음"이라 다수 클래스만 찍어도 높게 나온다. 사건 유형을 주 지표로 본다.

## 준비 (5080 PC, 한 번만)

5080이 Windows라면 3070 때와 같이 **WSL2 Ubuntu**에서 한다 ([FINETUNE.md](FINETUNE.md) "학습 장비 준비").

```bash
cd ~ && git clone https://github.com/chungddong/NeedleTest.git && cd NeedleTest
python3 -m venv .venv && .venv/bin/pip install -q -U pip
.venv/bin/pip install -q "cactus-needle[train,gpu]==3.0.6" psutil
.venv/bin/pip install -q -U "jax[cuda12]"
XLA_PYTHON_CLIENT_PREALLOCATE=false .venv/bin/python -c "import jax; print(jax.devices())"
```

- `jax.devices()`에 `CudaDevice`가 보여야 한다. RTX 5080(Blackwell)은 최신 JAX가 필요해서 마지막 줄에서 JAX를 올린다. CPU만 보이면 `nvidia-smi`로 드라이버를 확인한다.
- 체크포인트(`checkpoints/needle3.safetensors`, 242MB)는 처음 실행할 때 자동으로 받는다.

## 1단계. v2 어댑터 확보 (둘 중 하나)

**A. 3070에서 복사** (빠름). 3070 WSL의 `~/NeedleTest/ko/out/ko_v2_lora.safetensors`를 5080의 같은 위치(`~/NeedleTest/ko/out/`)로 옮긴다(USB, 공유 폴더, scp 등).

**B. 5080에서 다시 학습** (재현 확인 겸, 3070보다 빠를 것으로 예상):

```bash
export XLA_PYTHON_CLIENT_PREALLOCATE=false
TAG=v2 EPOCHS=8 LR=1e-3 bash ko/train_main.sh
```

B를 택하면 학습 로그와 평가 결과가 `ko/results/*v2*`에 다시 써진다. 기존 결과와 숫자가 같은지 확인하고, 다르면 덮어쓰지 말고 이름을 바꿔 보관한다.

## 2단계. 은닉 상태 추출

```bash
export XLA_PYTHON_CLIENT_PREALLOCATE=false
.venv/bin/python ko/decision_head.py extract --adapter ko/out/ko_v2_lora.safetensors \
  --data ko/data/train.jsonl ko/data/test_human.jsonl ko/data/test_extra.jsonl --name v2
.venv/bin/python ko/decision_head.py extract --adapter none \
  --data ko/data/train.jsonl ko/data/test_human.jsonl ko/data/test_extra.jsonl --name base
```

- 문장마다 두 가지 입력을 처리한다: 문장만(`query`, 길이 192), 도구 정의 + 문장(`prompt`, 길이 512). 층은 10·15·20.
- 결과는 `ko/out/features/<name>_<split>.npz` (git 제외). 출력 줄의 `ms/row`가 GPU 처리 속도다.
- 메모리가 부족하면 `--batch 16`이나 `--batch 8`로 줄인다.

## 3단계. 판단 헤드 학습과 평가

```bash
.venv/bin/python ko/decision_head.py fit --name v2
.venv/bin/python ko/decision_head.py fit --name base
```

- 학습 데이터 3,000행 중 300행은 확신도 보정용으로 떼어 두고 나머지로 학습한다.
- 입력 종류(query/prompt) × 층(10/15/20/all) × 질문(사건 유형/위험)마다 정확도와 보정 오차(ECE)를 출력하고 `ko/results/dhead_<name>.json`에 저장한다.

## 결과 해석

- **v2 + prompt**가 사건 유형에서 생성 방식(0.906 / 0.812)에 가까우면: Needle이 "한 번에 답하는 판단 모델"로 쓰일 수 있다는 근거다. 이 경우 처리 비용은 생성 없이 입력만 읽는 비용(prefill)뿐이다.
- **v2 + query**도 높으면: 도구 정의 없이 문장만 넣어도 된다는 뜻이라 훨씬 빠르다(입력 토큰 약 30개).
- **base**는 한국어를 모르므로 낮을 것으로 예상한다. v2와의 차이가 "한국어 파인튜닝이 표현 자체를 바꿨다"는 증거가 된다.
- 확신도 0.9 이상 응답의 정확도(`acc_conf_ge_0.9`)와 그 비율도 본다. 안전 게이트로 쓸 수 있는지의 지표다.

## 시간이 남으면

1. **엔진에서 쓰기**: 좋은 설정(입력 종류·층)이 정해지면, 그 판단 헤드를 Pi에서 돌릴 방법을 찾는다. 후보: (a) Needle의 임베딩 헤드 구조를 직접 학습해 `.cact`에 넣기, (b) 엔진 출력 대신 별도 경량 런타임에서 은닉 상태 추출. 어느 쪽이든 하루 이상 걸릴 수 있다.
2. **Laya와 같은 조건 비교**: 같은 평가셋으로 Laya 다국어판의 사건 유형 정확도를 잰다 (`~/Develop/edge-ko-command`의 Laya 환경 사용).

## 커밋할 것

`ko/results/dhead_*.json`, 추출 단계 출력(콘솔 로그를 `ko/results/dhead_<name>_extract.log`로 저장), 1단계 B를 했다면 학습 로그. 특징 파일(`ko/out/`)은 커밋하지 않는다.

## 5080의 Claude Code에 줄 요청 예

```
NeedleTest 저장소의 CLAUDE.md와 DECISION_HEAD.md를 먼저 읽어줘.
DECISION_HEAD.md의 준비 단계부터 3단계까지 진행해줘. v2 어댑터는 (A: 3070에서 복사해 둠 / B: 다시 학습) 으로 할게.
단계마다 결과를 요약해서 보여주고, 다음 단계로 넘어가기 전에 나한테 확인받아줘.
끝나면 결과 파일을 커밋하고 푸시해줘.
```
