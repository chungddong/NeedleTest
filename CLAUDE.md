# NeedleTest

Cactus Compute의 온디바이스 툴콜링 모델 **Needle 3**(1.21억 파라미터, 35MB)를 라즈베리파이 5에서 평가하고, **한국어 재난 정전·설비 피해 신고 정형화**용으로 파인튜닝하는 프로젝트입니다. 사용자와는 한국어로 대화합니다.

## 목표

- **대회**: 한전KDN「2026 빛가람 AI·ICT 경진대회」. 접수 마감 **2026-10-11**, 본선 전시 2026-11-11. 심사 기준은 독창성, 기술성, 완성도, 전달력, 사업화 가능성.
- **작품(가칭 BlackoutLink)**: 태풍·폭우·산불 때 주민 신고(사투리, 외국어 포함), 현장 작업자 보고, 스마트미터 정전 이벤트를 단말에서 하나의 스키마로 정형화하고 병합합니다. 통신이 불안정해도 단말에서 처리하고 수십 바이트만 전송합니다.
- **이 저장소의 역할**: Needle 3를 한국어로 파인튜닝해 신고 문장을 `report_incident` 호출로 바꾸는 모델을 만들고, 원본 대비 개선을 측정합니다.

## 현재 상태 (2026-10-02)

완료
- Pi 5에서 원본 모델 벤치마크: 영어 90%, 한국어 17%, 지연 361ms, 메모리 74MB(C 런타임). 결과는 [RESULTS.md](RESULTS.md).
- 한국어 파인튜닝 가능성 확인: 토크나이저에 한글 조각이 없고(바이트 단위, 글자당 2.46토큰), Pi CPU에서 LoRA 학습 시 loss가 떨어지는 것까지 확인. [FINETUNE.md](FINETUNE.md).
- 원본 모델 기준점: 사람이 쓴 평가셋 32개 중 정답 7개, **한국어 신고 24건은 0건** (`ko/results/base-20L-w2_test_human.json`).
- **GPU 스모크 통과** (RTX 3070 WSL2, Ubuntu 24.04, 저장소 `~/NeedleTest`): 학습 → `.cact` 내보내기 → 엔진 추론까지 동작. 학습한 24문장에서 엔진 정답 **24/24**, 오류 0 (`ko/results/smoke-20L_smoke.json`). 50에폭 300스텝 loss 0.0006.
- **처음 두 번 실패한 원인**: 엔진은 도구 호출 전에 항상 `<think>` 추론 블록을 쓰는데, 학습 데이터에 `reasoning`이 없었음. 10에폭은 4/24(유형도 미학습), 50에폭은 JAX 24/24인데 `<think>`를 붙이면 JAX 12/24, 엔진 11/24 (`ko/results/noreason*`). `reasoning`을 넣자 해결.
- 사람이 쓴 평가셋(참고용, 24문장만 학습): 정답 9/32, 오류 0. 한국어 신고 24건 중 호출 23, 유형 9, 장소 10, 정답 3 (원본은 정답 0). 무관 문의 거절 5/6. 추론 문장은 스모크의 24개 구절을 그대로 재사용하는 수준이라, 다양한 데이터가 필요함.
- 라벨 규칙 위반 수정: 스모크 1건, 평가셋 1건(“저수지 가는 길…”)에서 위험을 직접 말하지 않았는데 붙어 있던 `hazard`를 제거. 평가셋 결과 4개는 `ko/rescore.py`로 저장된 출력을 다시 채점함(점수 변화 없음, 파일의 `rescored`에 기록).

- **학습 데이터 3,000행 생성** (`ko/data/train.jsonl`): Claude Code 워크플로(`ko/gen_workflow.js`)로 3,500행을 만들고 묶음마다 독립 검수 둘을 거친 뒤 조립. 원본과 검수 결과는 `ko/data/gen/`에 있고 그대로 재현됨. 생성·검수 에이전트는 평가셋을 열지 않음.
- **1차 본 학습** (`ko/train_main.sh`, 3에폭, 배치 8, 길이 512):
  - 학습률 탐색(학습 데이터에서 떼어 둔 검증 300행, 엔진 채점). 정답 1e-4 27, 3e-4 58, **1e-3 90**. 유형 87/116/133, 장소 69/119/174 (각 /270).
  - 1e-3으로 3,000행 전체 학습. loss 0.259 → 0.205 → 0.132 (`ko/results/train.log`).
  - 평가셋 32개 정답은 12L 5, 16L 4, **20L 8**. 20L 한국어 신고 24건은 호출 23, 장소 16, 유형 6, 정답 4. 같은 PC의 원본 4비트 20L은 정답 2/32, 유형 7/26, 장소 3/26 (`ko/results/ko-*L_test_human.json`, `base-20L-w4_test_human.json`).
  - 영어 회귀(`bench.py`, 92문항)는 파인튜닝 20L 0.674, 원본 4비트 0.772. multi 0.80 대 1.00, refusal 0.27 대 0.53 (`results/ko-20L.json`, `results/pc-base-20L-w4.json`).
  - **약점은 신고 유형.** 검증 오답을 보면 모델이 먼저 쓰는 영어 근거 구절이 입력과 무관한 흔한 문구로 쏠리고(변압기 소음에 "power out, 40 households stated"), 호출이 그 문구를 따라감.
- **후속 실험** (같은 검증 300행, 1e-3, `ko/exp_reasoning.sh`, `ko/results/exp_*`): 정답은 근거+라벨 3에폭 90, 라벨만 3에폭 108, 근거+라벨 8에폭 167, **라벨만 8에폭 213**(유형 250/270, 판단 293/300). 근거 구절을 빼는 것과 에폭을 늘리는 것 모두 효과가 있음. `reasoning`을 라벨만 쓰는 형식으로 바꿔 `train.jsonl`을 재조립(행·순서 동일).
- **현재 최선 모델 v2** (라벨만 `reasoning`, 1e-3, 8에폭, 3,000행; `TAG=v2 EPOCHS=8 LR=1e-3 bash ko/train_main.sh`; 어댑터 `ko/out/ko_v2_lora.safetensors`, `ko/out/needle3-ko-v2-20L.cact` 63MB):
  - 평가셋 32개(학습에 안 쓴 사람이 쓴 문장): 20L **정답 19/32**, 판단 31/32. 한국어 신고 24건은 정답 12, **유형 21**, 장소 14. 무관 문의 거절 6/6. 12L 정답 2, 16L 8 (`ko/results/ko-v2-*L_test_human.json`).
  - 유형 정확도가 검증 93%와 평가셋 88%로 비슷함(1차는 49% 대 31%).
  - **남은 약점은 장소**: 대부분 범위를 넓게 자름("시청 사거리 신호등이랑", "대촌동 변압기 소음 민원 현장 확인"). 유형 오답 2건("불 다 꺼졌어요"를 `fire`로), 신고 누락 1건("정전요 행복빌라").
  - 영어 회귀 0.739 (1차 0.674, 원본 4비트 0.772). multi 1.00, refusal 0.67(원본 0.53)이지만 basic 0.90, args 0.85, bench 한국어 세트 0/12 (`results/ko-v2-20L.json`).
  - 확신도 구간 평가(`ko/calib.py`, JAX에서 출력 전체 확률): 평균 확신도 0.85 대 정확도 0.59로 과신(ECE 0.31). 0.95 이상 18행은 정확도 89%, 0.50~0.95 11행은 27%. 0.95 이상만 자동 접수하면 56%를 정확도 89%로 처리 (`ko/results/calib_v2_test_human.json`, 32행이라 구간별 표본이 작음).

다음 할 일
1. `ko/out/needle3-ko-v2-20L.cact`를 Pi로 복사해 지연·메모리 측정.
2. 장소 범위 자르기 개선(학습 데이터 장소 규칙 점검, 장소가 길게 잘린 실패 유형 보강).
3. 평가셋을 100~200문장으로 확대(32문장은 구간이 넓음).
4. 확신도 게이트를 단말에서 쓰려면 엔진이 토큰 확률을 내줘야 함(현재 JAX에서만 계산).

데이터 생성은 OpenRouter 대신 이 세션의 Claude 에이전트로 합니다(사용자 결정). 외부 유료 서비스나 API 키가 필요하면 먼저 이유를 설명합니다.

## 핵심 결정과 이유

- **처음부터 학습하지 않고 LoRA 파인튜닝**: 사전학습은 대회 일정 안에 불가능. 영어 능력도 유지됨.
- **스키마의 키와 enum 값은 영어**(`incident_type: "pole_down"`), 한국어는 신고자 표현을 그대로 옮기는 `location`에만 씀: 한국어는 글자당 바이트 3개로 생성되므로 출력할 한국어를 최소화.
- **라벨 규칙**: `hazard`는 감전·화재 위험을 직접 말했을 때만(“불꽃이 튀어요”만으로는 넣지 않음, “불날 것 같아요”면 `fire`). `households`는 가구·세대 수를 말했을 때만. 무관한 문의와 “정전 아님”은 `answers: []`.
- **확신도 헤드가 빠짐**: `needle build --lora`는 확신도 헤드를 제거해 confidence가 `None`이 됩니다. 거절 예시 학습, SDK 검증, 규칙 검사로 보완합니다.
- **모든 학습 행에 `reasoning` 한 줄**: 엔진이 항상 `<think>`부터 생성하므로 이 블록을 학습시켜야 합니다. 형식은 `ko/schema.py`의 `reasoning()`: 영어 근거 구절 + 라벨에서 자동으로 만든 판단(`power out, 30 households stated -> outage; households 30; hazard none`, 거절은 `... -> no report`). 라벨과 어긋날 수 없고, 한국어를 쓰지 않아 토큰이 적습니다.
- **파인튜닝 모델은 `auto_date=False`**: 학습 프롬프트에 system 턴이 없으므로 SDK의 자동 날짜 문구를 끕니다(`ko/try_model.py`, 앱에서도 동일).
- **학습과 엔진 중 어디가 문제인지 분리**: `ko/jax_score.py`로 어댑터를 엔진 없이 JAX에서 채점합니다(기본값은 엔진처럼 `<think>` 강제).
- **4비트 내보내기**: 공개 도구는 4비트만 지원(20레이어 63MB). 깊이 12/16/20을 만들어 정확도·크기를 비교합니다.

## 규칙

- `ko/data/test_human.jsonl`은 **평가 전용**입니다. 학습 데이터에 섞거나, 이 문장을 바탕으로 학습 데이터를 만들지 않습니다. 라벨을 고치면 `ko/rescore.py`로 기존 결과를 다시 채점합니다(모델 재실행 불필요).
- API 키(OpenRouter, GitHub 토큰 등)는 터미널 `export`로만 설정하고 파일에 쓰거나 커밋하지 않습니다.
- 텔레메트리는 끕니다: `NEEDLE_TELEMETRY=0` (스크립트에 설정되어 있음).
- `.cact`, `.safetensors`, `ko/out/`은 git에 넣지 않습니다(`.gitignore`). 결과 공유는 `ko/results/`의 JSON과 로그로 합니다.
- 측정 수치를 문서에 쓸 때는 결과 파일에서 확인한 값만 씁니다. 추정치는 추정이라고 표시합니다.

## 작업 흐름과 장비

| 장비 | 역할 |
|---|---|
| RTX 3070 Windows PC (WSL2 Ubuntu-24.04) | 학습, 내보내기, 1차 평가. 저장소는 WSL 홈(`~/NeedleTest`, `.venv` 설치됨). Claude Code는 Windows 저장소(`F:\Develop\NeedleTest`)에서 `wsl -d Ubuntu-24.04`로 작업하고, 커밋·푸시는 GitHub 인증이 있는 Windows 저장소에서 함 |
| Raspberry Pi 5 (`chungserver`) | 벤치마크(지연·메모리), 결과 검토. `models/`에 `.cact`를 받아 측정 |
| GitHub `chungddong/NeedleTest` | 코드, 데이터, 평가 결과 공유 |

결과 공유: `ko/results/`, `results/ko-*.json`, `ko/data/train.jsonl`을 커밋·푸시 → Pi에서 `git pull`. 모델 파일은 `scp ko/out/needle3-ko-*.cact chungman@<Pi 주소>:~/Develop/NeedleTest/models/`.

## 자주 쓰는 명령 (저장소 루트, WSL)

```bash
bash ko/gpu_smoke.sh                                              # 첫 실행: 설치 + 스모크
.venv/bin/python ko/try_model.py <model.cact|-> ko/data/test_human.jsonl --out ko/results/<이름>.json
.venv/bin/python ko/jax_score.py ko/out/<adapter>.safetensors ko/data/smoke.jsonl   # 엔진 없이 어댑터 채점
.venv/bin/python ko/calib.py ko/out/<adapter>.safetensors <data.jsonl> --name <이름>   # 확신도 구간별 정확도
.venv/bin/needle finetune ko/data/train.jsonl --epochs 3 --batch-size 8 --lr 1e-4 --max-len 512 --out ko/out/ko_lora.safetensors
.venv/bin/needle build checkpoints/needle3.safetensors --lora ko/out/ko_lora.safetensors --layers 20 --out ko/out/needle3-ko-20L.cact
.venv/bin/python bench.py --weights ko/out/needle3-ko-20L.cact --label ko-20L   # 영어 회귀 확인
```

8GB GPU라 `XLA_PYTHON_CLIENT_PREALLOCATE=false`를 설정하고, 메모리 부족이면 `--batch-size`를 4로 낮춥니다.

## 문서 지도

- [FINETUNE.md](FINETUNE.md): 파인튜닝 확인 사항, 제약, WSL2 설치, 단계별 명령, 데이터 원칙, 라이선스(Apache 2.0)와 명명
- [RESULTS.md](RESULTS.md): Pi 5 원본 모델 벤치마크 결과와 진행 기록
- [RUNPOD.md](RUNPOD.md): 클라우드 GPU 대안 (RunPod 서버에서 GitHub 차단 이력 있음)
- `ko/schema.py`: 신고 스키마. `ko/gen_data.py`: 학습 데이터 생성 프롬프트(화자 스타일 목록 포함)
