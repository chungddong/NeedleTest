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
- **GPU 스모크 통과** (RTX 3070 WSL2, Ubuntu 24.04, 저장소 `~/NeedleTest`): 학습 → `.cact` 내보내기 → 엔진 추론까지 동작. 학습한 24문장에서 엔진 정답 **24/24**, 오류 0 (`ko/results/smoke-20L_smoke.json`). 50에폭 300스텝 loss 0.0014.
- **처음 두 번 실패한 원인**: 엔진은 도구 호출 전에 항상 `<think>` 추론 블록을 쓰는데, 학습 데이터에 `reasoning`이 없었음. 10에폭은 4/24(유형도 미학습), 50에폭은 JAX 24/24인데 `<think>`를 붙이면 JAX 12/24, 엔진 11/24 (`ko/results/noreason*`). `reasoning`을 넣자 해결.
- 사람이 쓴 평가셋(참고용, 24문장만 학습): 정답 7/32, 오류 0. 한국어 신고 24건 중 호출 23, 유형 10, 장소 9, 정답 3 (원본은 정답 0). 추론 문장은 스모크의 24개 구절을 그대로 재사용하는 수준이라, 다양한 데이터가 필요함.

다음 할 일
1. 사용자가 요청하면 학습 데이터 생성(`ko/gen_data.py`, 이제 `reasoning` 포함), 본 학습, 깊이별 내보내기, 평가.
2. 결과를 커밋·푸시하고, `.cact`는 Pi로 복사해 지연·메모리 측정.

본 학습과 데이터 생성은 **사용자가 요청할 때** 진행합니다. 데이터 생성은 OpenRouter 요금이 듭니다.

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

- `ko/data/test_human.jsonl`은 **평가 전용**입니다. 학습 데이터에 섞거나, 이 문장을 바탕으로 학습 데이터를 만들지 않습니다.
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
