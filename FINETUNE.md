# Needle 3 한국어 파인튜닝 가이드

재난 정전·설비 피해 신고 정형화(BlackoutLink)를 위해 Needle 3를 한국어로 파인튜닝하는 계획과, 그 전에 Pi에서 확인한 사실을 정리했습니다. 학습은 **RTX 3070 Windows PC의 WSL2**에서 진행합니다(RunPod는 대안). 작성일은 2026-10-02입니다.

## 요약

- 한국어 파인튜닝은 **가능합니다.** 한국어 데이터로 LoRA 학습을 돌렸을 때 loss가 정상적으로 떨어졌습니다.
- 다만 이 모델은 영어 전용으로 설계돼서 **한국어를 바이트 단위로만 읽고 씁니다.** 학습으로 정확도는 올릴 수 있지만 토큰 효율은 그대로입니다.
- 공개 도구로 파인튜닝하면 **확신도(confidence) 헤드가 빠지고 4비트로 내보내집니다.** 이 두 가지는 설계에 반영해야 합니다.
- 처음부터 다시 학습하는 것보다 파인튜닝이 훨씬 현실적입니다(아래 “처음부터 학습 vs 파인튜닝”).

## Pi에서 확인한 것

| 항목 | 결과 |
|---|---|
| 토크나이저 | SentencePiece, 어휘 8,192개. **한글 조각 0개**, 바이트 대체(byte fallback) 256개 |
| 한국어 토큰 수 | 글자당 **2.46토큰** (한 글자 = UTF-8 3바이트). 영어는 0.28토큰으로 약 9배 차이 |
| 학습 시퀀스 길이 | 신고 문장 + 툴 스키마 + 정답이 최대 317토큰. `--max-len 512`로 충분 |
| 원본 모델 성능 | 한국어 신고 18건 **0건 처리** (확신도 0.02~0.07로 전부 거절, 1건은 UTF-8 오류). 영어 3건 중 1건 정답 |
| 학습 동작 | Pi CPU에서 LoRA 학습 시작, 15스텝 동안 loss 0.95 → 0.23~0.29. 기기 재부팅으로 중단 (스텝당 약 1분) |
| 학습 방식 | LoRA rank 16, 어텐션 투영(q, k, v, gate, out)에만 적용. 임베딩과 MLP는 고정 |
| 모델 크기 | 파라미터 1.21억 개, 컨텍스트 8,192토큰 |
| 라이선스 | **Apache 2.0** (가중치·코드). 수정·재배포 가능, 고지 유지 필요 |

**GPU 장비에서 확인한 것** (RTX 3070, WSL2, 2026-10-02): 학습 → `.cact` 내보내기 → 엔진 추론이 끝까지 동작합니다. 24문장을 외우게 한 스모크 모델이 엔진에서 24/24를 맞혔습니다(`ko/results/smoke-20L_smoke.json`). 다만 학습 데이터에 `reasoning`이 있어야 합니다(아래 제약 5). **아직 확인하지 못한 것**: 실제 데이터로 학습했을 때의 정확도 향상 폭, Pi에서 파인튜닝 모델의 지연.

## 설계에 반영할 제약

1. **한국어는 바이트로 처리됩니다.** 토크나이저에 한글이 없어서 모델은 한 글자를 바이트 3개로 만들어 냅니다. 원본 모델이 깨진 UTF-8을 낸 것도 이 때문입니다. 그래서 스키마의 **키와 enum 값은 영어**(`incident_type: "pole_down"`)로 두고, 한국어는 신고자의 표현을 그대로 옮기는 `location`에만 씁니다. 출력할 한국어가 적을수록 빠르고 정확합니다.
2. **확신도 헤드가 빠집니다.** `needle build --lora`는 로컬에서 학습되지 않은 확신도 헤드를 제거하고, 파인튜닝 모델은 confidence로 `None`을 보고합니다. “모르면 실행하지 않고 넘긴다”는 안전 장치를 그대로 쓸 수 없으므로 거절 예시를 충분히 학습시키고, SDK 검증(근거 없음·부정문 표시)과 규칙 검사로 보완해야 합니다. Cactus 호스팅 파인튜닝(`needle platform finetune`, cactuscompute.com 계정과 API 키 필요)은 헤드를 유지한다고 SDK에 적혀 있습니다. 요금제별 작업 횟수 제한이 있어 사용 전에 확인이 필요합니다.
3. **4비트로 내보내집니다.** 공개 도구는 4비트만 지원해서 20레이어 모델이 공식 2비트(35MB)보다 큰 63MB가 됩니다. Pi 측정에서 16레이어 4비트는 51MB, 메모리 137MB(Python SDK)였습니다. 현장 단말 기준으로도 충분히 작습니다.
4. **LoRA는 어텐션만 학습합니다.** 정확도가 기대보다 낮게 멈추면 MLP나 바이트 임베딩까지 학습하도록 학습 코드를 수정할 수 있습니다. 코드는 JAX(`needle/model/finetune.py`)로 공개되어 있습니다.
5. **엔진은 도구 호출 전에 항상 `<think>` 추론을 씁니다.** 응답의 `reasoning` 필드가 그 내용입니다. `needle finetune`은 학습 행에 `reasoning`이 있을 때만 이 블록을 학습시키고, 공식 데이터 생성 템플릿도 행마다 `reasoning` 한 줄을 요구합니다. 이게 없으면 파인튜닝 모델이 추론 단계에서 원본 모델처럼 행동하다 답을 망칩니다. 무관 문의도 신고로 처리하고, 장소를 자르고, `hazard`와 `households`를 빠뜨립니다. 스모크에서 측정한 값(학습한 24문장):

   | | 정답 |
   |---|---|
   | `reasoning` 없음, 10에폭, 엔진 | 4/24 |
   | `reasoning` 없음, 50에폭, JAX (학습 프롬프트 그대로) | 24/24 |
   | `reasoning` 없음, 50에폭, JAX (`<think>` 강제) | 12/24 |
   | `reasoning` 없음, 50에폭, 엔진 | 11/24 |
   | **`reasoning` 있음, 50에폭, 엔진** | **24/24** |

   결과 파일은 `ko/results/noreason*`, `ko/results/smoke*`입니다. 추론 문장 형식은 `ko/schema.py`의 `reasoning()`입니다. 영어 근거 구절에 라벨에서 자동으로 만든 판단을 붙입니다(예: `power out, 30 households stated -> outage; households 30; hazard none`, 거절은 `asks about the bill -> no report`). 이렇게 하면 추론과 라벨이 어긋나지 않고, 한국어를 출력하지 않아 토큰이 적게 듭니다. 파인튜닝 모델은 `needle.Needle(..., auto_date=False)`로 씁니다. 학습 프롬프트에 system 턴이 없기 때문입니다. 날짜 문구를 켜고 끈 차이는 작았습니다(엔진 11/24 → 12/24).

## 학습 장비 준비

JAX의 GPU 버전은 Linux 전용이라, Windows에서는 **WSL2(Windows 안의 Ubuntu)**에서 학습합니다. 모델이 1.21억 파라미터라 RTX 3070(8GB)으로 충분합니다. 클라우드를 쓰려면 [RUNPOD.md](RUNPOD.md)를 보세요(단, 이전에 쓴 RunPod 서버는 GitHub 접속이 막혀 있었습니다).

**Windows 쪽 (한 번만)**

1. NVIDIA 그래픽 드라이버를 최신으로 업데이트합니다. WSL2의 CUDA는 Windows 드라이버를 그대로 씁니다. **WSL 안에는 NVIDIA 드라이버를 설치하지 않습니다.**
2. 관리자 권한 PowerShell에서 Ubuntu를 설치하고 재부팅합니다.

```powershell
wsl --install -d Ubuntu-24.04
```

**WSL Ubuntu 안에서 (한 번만)**

```bash
nvidia-smi
```

여기서 RTX 3070이 보여야 합니다. 그다음 기본 도구를 설치하고, GitHub에 푸시할 수 있게 로그인합니다.

```bash
sudo apt-get update && sudo apt-get install -y python3-venv git gh
```

```bash
gh auth login
```

```bash
gh auth setup-git
```

저장소는 **WSL 홈 폴더**에 받습니다. `/mnt/c/...`(Windows 드라이브)에 받으면 파일 접근이 매우 느립니다.

```bash
cd ~ && git clone https://github.com/chungddong/NeedleTest.git && cd NeedleTest
```

Claude Code도 이 WSL 터미널의 저장소 폴더에서 실행해야 GPU 학습 환경을 그대로 씁니다.

## 진행 순서

저장소 루트에서 실행합니다.

**1. 스모크 테스트** (설치부터 학습, 내보내기, 엔진 추론까지 한 번에 확인)

```bash
bash ko/gpu_smoke.sh
```

출력 처음의 `jax devices:`에 `CudaDevice`가 보여야 GPU를 쓰는 것입니다. 24개 문장을 50에폭 동안 일부러 외우게 한 뒤 두 번 채점합니다. 먼저 `ko/jax_score.py`로 엔진 없이 JAX에서 채점하고, 그다음 `.cact`로 엔진에서 채점합니다. 둘 다 대부분 맞히면 파이프라인이 정상입니다. JAX만 맞히고 엔진이 못 맞히면 내보내기나 엔진 쪽 문제입니다. 10에폭으로는 loss가 0.07까지 내려가도 신고 유형을 거의 학습하지 못했습니다. loss는 출력 토큰 전체의 평균이라, 쉬운 JSON 구조와 장소를 옮겨 적는 바이트가 대부분을 차지하기 때문입니다. 사람이 쓴 테스트셋 점수는 이 단계에서는 낮게 나오는 게 정상입니다. 결과는 `ko/results/smoke*`에 저장되니 커밋해서 공유합니다.

**2. 학습 데이터 생성** (OpenRouter API 키 필요, 사용량만큼 과금. GPU는 쓰지 않음)

```bash
export OPENROUTER_API_KEY=...
.venv/bin/python ko/gen_data.py --num 3000 --out ko/data/train.jsonl
```

생성된 데이터에서 100개쯤은 직접 읽고 라벨 규칙(아래)에 맞는지 확인하세요. 틀린 라벨이 많으면 프롬프트(`ko/gen_data.py`의 `PROMPT`)를 고친 뒤 다시 생성합니다.

**3. 학습**

```bash
export XLA_PYTHON_CLIENT_PREALLOCATE=false
.venv/bin/needle finetune ko/data/train.jsonl --epochs 3 --batch-size 8 --lr 1e-4 \
  --max-len 512 --out ko/out/ko_lora.safetensors 2>&1 | tee ko/results/train.log
```

학습이 끝나면 10% 검증 데이터의 정확도(4비트 기준)를 출력합니다. RTX 3070(8GB)이라 배치를 8로 잡았습니다. 메모리 부족(OOM)이 나면 4로 낮추고, 여유가 있으면 16으로 올립니다.

**4. 내보내기** (깊이별로 여러 개 만들어 비교)

```bash
for L in 12 16 20; do
  .venv/bin/needle build checkpoints/needle3.safetensors --lora ko/out/ko_lora.safetensors \
    --layers $L --out ko/out/needle3-ko-${L}L.cact
done
```

**5. 평가**

```bash
# 사람이 쓴 한국어 테스트셋 (학습에 절대 쓰지 않음). 깊이별로 반복
.venv/bin/python ko/try_model.py ko/out/needle3-ko-20L.cact ko/data/test_human.jsonl \
  --out ko/results/ko-20L_test_human.json
# 영어 능력이 무너지지 않았는지 기존 벤치마크로 확인 (results/ko-20L.json 생성)
.venv/bin/python bench.py --weights ko/out/needle3-ko-20L.cact --label ko-20L
```

**6. 결과 공유**: `ko/results/`(평가 JSON, 학습 로그), `results/ko-*.json`, `ko/data/train.jsonl`을 커밋하고 푸시합니다. `.cact` 모델 파일은 git에 넣지 않고 같은 네트워크의 Pi로 복사합니다. 지연·메모리는 Pi에서 `bench.py`, `cbench.py`로 다시 잽니다.

```bash
scp ko/out/needle3-ko-*.cact chungman@<Pi 주소>:~/Develop/NeedleTest/models/
```

## 데이터 원칙

- **라벨 규칙을 먼저 고정합니다.** 예: `hazard`는 신고자가 감전·화재 위험을 직접 말했을 때만 넣고, “불꽃이 튀어요”만 있으면 넣지 않습니다. `households`는 가구·세대 수를 말했을 때만 넣습니다. 규칙이 흔들리면 모델도 흔들립니다.
- **평가셋은 사람이 따로 씁니다.** 생성 데이터와 같은 모델이 만든 문장으로 평가하면 점수가 부풀려집니다. `ko/data/test_human.jsonl`(32개)을 뉴스·커뮤니티의 실제 신고 표현으로 100~200개까지 늘리는 것을 권합니다.
- **영어와 거절 예시를 섞습니다.** 외국인 신고(영어)와 무관한 문의(요금, 명의 변경)를 넣어야 영어 능력이 유지되고 엉뚱한 신고 접수가 줄어듭니다.
- **화자 다양성**: 표준어, 경상·전라·충청 사투리, 오타·STT 오류, 현장 작업자 약어, 서툰 한국어, 영어 (`gen_data.py`의 `STYLES`).
- **모든 행에 `reasoning`** (제약 5): `gen_data.py`는 생성 모델에게 영어 근거 구절(`evidence`, 2~8단어)만 받고, `reasoning`은 라벨에서 조립합니다. 근거 구절이 없거나 한글이 섞인 행은 버립니다. 24문장 스모크 모델은 평가셋에서 처음 보는 문장에도 스모크의 근거 구절을 그대로 재사용했습니다. 실제 데이터에서는 근거 구절도 다양해야 합니다.

## 처음부터 학습 vs 파인튜닝

| | 처음부터 학습 (자체 K-Needle) | 파인튜닝 |
|---|---|---|
| 필요 자원 | Needle 1 기준 사전학습 2,000억 토큰, TPU v6e 16개로 27시간 (공개 자료). 대규모 한국어 말뭉치, 증류용 대형 모델, 2비트 양자화 학습, 확신도 헤드 학습 별도 | GPU 1대, 데이터 수천 건, 수 시간 이내 예상 (미측정) |
| 한국어 토크나이저 | 새로 만들 수 있음 (토큰 효율 크게 개선) | 기존 바이트 방식 그대로 |
| 영어·다국어 | 다시 학습해야 함 | 원본 능력 대부분 유지 (영어 데이터를 섞을 때) |
| 대회 일정 안에 가능? | 불가능에 가까움 | 가능 |

중간 선택지로 **어휘 확장**(한글 조각을 토크나이저에 추가하고 임베딩을 이어서 학습)이 있습니다. 한국어 토큰 효율 문제를 풀 수 있지만, 공개 내보내기 도구가 원본 토크나이저를 그대로 복사하고 엔진의 문법 제약이 토크나이저에 묶여 있어서 도구와 엔진 쪽 수정이 필요합니다. 본선 이후 과제로 남겨두는 것을 권합니다.

## 라이선스와 명명

- Needle 3는 Apache 2.0입니다. 파인튜닝한 모델을 배포할 때는 라이선스 사본을 함께 두고, 원본이 Cactus Compute의 Needle 3이며 무엇을 바꿨는지 밝혀야 합니다.
- 이름은 자유롭게 붙일 수 있지만, 설명은 **“Needle 3 기반 한국어 재난 신고 특화 파인튜닝 모델”**처럼 원본을 밝혀야 합니다. “자체 개발 파운데이션 모델”이나 “처음부터 학습한 모델”이라고 하면 사실과 다릅니다.
- 우리 기여로 내세울 수 있는 것: 한국어 재난 신고 데이터셋과 라벨 규칙, 학습 레시피, 사람이 쓴 평가셋과 측정 결과(원본 0% → 파인튜닝 후 X%), 온디바이스 배포와 시스템 통합.

## 파일

| 경로 | 내용 |
|---|---|
| `ko/schema.py` | 신고 스키마 `report_incident` |
| `ko/make_smoke_data.py` | 파이프라인 확인용 24개 (학습용, 행마다 근거 구절 → `reasoning`) |
| `ko/jax_score.py` | 어댑터를 `.cact` 없이 JAX에서 채점 (기본값은 엔진처럼 `<think>` 강제, `--no-think`) |
| `ko/make_test_data.py` | 사람이 쓴 평가셋 32개 (학습 금지) |
| `ko/gen_data.py` | OpenRouter로 한국어 학습 데이터 생성 |
| `ko/try_model.py` | `.cact` 모델로 데이터셋을 돌려 채점, `--out`으로 결과 JSON 저장 |
| `ko/gpu_smoke.sh` | GPU 장비 첫 실행: 설치 + 스모크 학습·내보내기·추론, 결과를 `ko/results/`에 저장 |
| `ko/results/` | 평가 결과와 학습 로그 (커밋 대상). `base-20L-w2_test_human.json`은 원본 모델 기준점 |
