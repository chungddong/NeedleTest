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
- RTX 3070 WSL2(Ubuntu 24.04, 저장소 `~/NeedleTest`)에서 `ko/gpu_smoke.sh` 완주: JAX가 `CudaDevice` 사용, 60스텝 loss 0.958 → 0.069, `.cact` 내보내기와 엔진 추론까지 동작 (`ko/results/smoke_*`).
- **스모크 기준 미달**: 학습한 24문장에서 정답 4/24. 한국어 신고는 호출 형식과 `location`은 일부 맞히지만(`location` 6/17) `incident_type`이 거의 틀리고(1/17), 오류 8건(토큰 예산 초과 5, UTF-8 깨짐 3). 원인(학습 부족인지 파이프라인 문제인지)은 아직 확인하지 않음.

다음 할 일
1. 스모크 기준 미달 원인 확인 (예: 에폭을 늘려 24문장을 외우는지 재확인).
2. 사용자와 확인한 뒤 학습 데이터 생성(`ko/gen_data.py`), 본 학습, 깊이별 내보내기, 평가.
3. 결과를 커밋·푸시하고, `.cact`는 Pi로 복사해 지연·메모리 측정.

본 학습과 데이터 생성은 **사용자가 요청할 때** 진행합니다. 데이터 생성은 OpenRouter 요금이 듭니다.

## 핵심 결정과 이유

- **처음부터 학습하지 않고 LoRA 파인튜닝**: 사전학습은 대회 일정 안에 불가능. 영어 능력도 유지됨.
- **스키마의 키와 enum 값은 영어**(`incident_type: "pole_down"`), 한국어는 신고자 표현을 그대로 옮기는 `location`에만 씀: 한국어는 글자당 바이트 3개로 생성되므로 출력할 한국어를 최소화.
- **라벨 규칙**: `hazard`는 감전·화재 위험을 직접 말했을 때만(“불꽃이 튀어요”만으로는 넣지 않음, “불날 것 같아요”면 `fire`). `households`는 가구·세대 수를 말했을 때만. 무관한 문의와 “정전 아님”은 `answers: []`.
- **확신도 헤드가 빠짐**: `needle build --lora`는 확신도 헤드를 제거해 confidence가 `None`이 됩니다. 거절 예시 학습, SDK 검증, 규칙 검사로 보완합니다.
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
| RTX 3070 Windows PC (WSL2 Ubuntu) | 학습, 내보내기, 1차 평가. 저장소는 WSL 홈 폴더에 둠 |
| Raspberry Pi 5 (`chungserver`) | 벤치마크(지연·메모리), 결과 검토. `models/`에 `.cact`를 받아 측정 |
| GitHub `chungddong/NeedleTest` | 코드, 데이터, 평가 결과 공유 |

결과 공유: `ko/results/`, `results/ko-*.json`, `ko/data/train.jsonl`을 커밋·푸시 → Pi에서 `git pull`. 모델 파일은 `scp ko/out/needle3-ko-*.cact chungman@<Pi 주소>:~/Develop/NeedleTest/models/`.

## 자주 쓰는 명령 (저장소 루트, WSL)

```bash
bash ko/gpu_smoke.sh                                              # 첫 실행: 설치 + 스모크
.venv/bin/python ko/try_model.py <model.cact|-> ko/data/test_human.jsonl --out ko/results/<이름>.json
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
