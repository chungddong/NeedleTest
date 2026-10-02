# RunPod 세팅 가이드

[FINETUNE.md](FINETUNE.md)의 한국어 파인튜닝을 RunPod GPU 서버에서 돌리기 위한 준비 과정입니다. 현재 주 학습 장비는 RTX 3070 PC(WSL2)이고 RunPod는 대안입니다. 2026-10-02에 쓴 RunPod 서버는 GitHub 접속이 차단되어 있었으니, Pod를 띄우면 먼저 `curl -sI https://github.com`으로 확인하세요. RunPod 화면 이름은 2026-10 기준 [공식 문서](https://docs.runpod.io/get-started)를 따랐습니다. 화면이 바뀌었으면 문서를 우선하세요.

## 필요한 사양

| 항목 | 권장 | 이유 |
|---|---|---|
| GPU | VRAM 16GB 이상 (RTX 4090, RTX A5000, L4 등) | 모델이 1.21억 파라미터라 큰 GPU는 필요 없음. 패키지가 `jax[cuda12]`를 쓰므로 CUDA 12 지원 GPU |
| 템플릿 | RunPod 공식 PyTorch 템플릿 | Python, CUDA 드라이버, Jupyter, 웹 터미널이 갖춰져 있음. PyTorch 자체는 쓰지 않음 |
| Volume disk | 20GB 이상 | 가상환경(CUDA 포함 JAX) 수 GB, 체크포인트 242MB, 결과 모델 파일 |
| 비용 | 목록에 표시된 시간당 요금 확인 | GPU 종류와 클라우드 종류(Secure/Community)에 따라 다름 |

## 1. 계정과 결제

1. https://www.console.runpod.io/signup 에서 가입하고 이메일 인증, 2단계 인증을 설정합니다.
2. 결제 수단을 등록하고 크레딧을 충전합니다. 스모크 테스트와 본 학습은 몇 시간 안에 끝날 것으로 예상하니 소액부터 시작하세요.

## 2. SSH 키 등록 (선택)

웹 터미널(Jupyter Lab)만 써도 학습은 할 수 있습니다. Pi에서 직접 접속하거나 `scp`로 파일을 받으려면 키를 등록하세요.

Pi에서 키를 만들고 공개키를 출력합니다:

```bash
ssh-keygen -t ed25519
```

```bash
cat ~/.ssh/id_ed25519.pub
```

출력된 한 줄(`ssh-ed25519 AAAA...`)을 RunPod 콘솔의 **Credentials** 페이지에 추가합니다. 비밀키(`id_ed25519`, `.pub` 없는 파일)는 절대 올리지 마세요.

## 3. Pod 만들기

1. 콘솔 오른쪽 위 **+ New** → **Pod** (또는 왼쪽 **Pods** 메뉴)
2. **Workload**에서 PyTorch 템플릿을 고릅니다. 검색창이나 **Explore all**에서 찾을 수 있습니다.
3. **Compute**의 **Available** 탭에서 VRAM 16GB 이상 GPU를 1개 고릅니다.
4. 저장소는 **Volume disk**를 20GB 이상으로 잡습니다. `/workspace`에 마운트되고 Pod를 멈췄다 다시 켜도 남습니다. 여러 Pod에서 같이 쓰거나 Pod를 지운 뒤에도 보관하려면 **Network volume**을 씁니다.
5. 배포합니다.

**주의**: `/workspace` 밖(컨테이너 디스크)에 둔 파일은 Pod를 멈추면 지워집니다. 저장소, 가상환경, 데이터, 결과물은 모두 `/workspace` 안에 둡니다.

## 4. 접속

- **웹**: Pods 페이지에서 Pod 선택 → HTTP Services의 **Jupyter Lab** → Terminal
- **SSH (기본)**: Pod 화면의 SSH 접속 명령을 복사해서 실행합니다. `ssh.runpod.io`를 거치는 방식이라 `scp`는 지원하지 않습니다.
- **SSH (공인 IP)**: 공인 IP와 TCP 22번 포트가 열린 Pod라면 `ssh root@<IP> -p <포트>`로 접속하고 `scp`도 쓸 수 있습니다.

## 5. 저장소 받기와 스모크 테스트

```bash
cd /workspace
```

```bash
git clone https://github.com/chungddong/NeedleTest.git
```

```bash
cd NeedleTest && bash ko/gpu_smoke.sh
```

스크립트가 하는 일: 가상환경 생성 → `cactus-needle[train,gpu]` 설치 → JAX가 GPU를 인식하는지 출력 → 24문장 과적합 학습 → 모델 파일 내보내기 → 한국어 추론 확인.

- 처음 출력 중 `jax devices:`에 `CudaDevice`가 보여야 GPU를 쓰는 것입니다. `CpuDevice`만 보이면 GPU를 못 잡은 것이니 아래 문제 해결을 보세요.
- `python3 -m venv`에서 오류가 나면 `apt-get update && apt-get install -y python3-venv`를 실행한 뒤 다시 돌립니다.

## 6. 본 학습

**API 키는 터미널에서만 설정합니다.** 파일이나 저장소에 저장하지 마세요.

```bash
export OPENROUTER_API_KEY=여기에_본인_키
```

학습 데이터 생성, 학습, 내보내기, 평가는 [FINETUNE.md](FINETUNE.md)의 2~5단계를 따릅니다.

오래 걸리는 작업은 브라우저 창을 닫아도 계속 돌도록 백그라운드로 실행하고 로그를 남깁니다:

```bash
nohup .venv/bin/needle finetune ko/data/train.jsonl --epochs 3 --batch-size 16 --lr 1e-4 --max-len 512 --out ko/out/ko_lora.safetensors > ko/out/train.log 2>&1 &
```

```bash
tail -f ko/out/train.log
```

## 7. 결과를 Pi로 가져오기

학습한 모델 파일(`ko/out/*.cact`)은 `.gitignore`에 들어 있어서 git으로는 옮겨지지 않습니다.

**방법 A: runpodctl** (SSH 설정 없이 가능)

Pod에서 실행하면 일회용 코드가 나옵니다:

```bash
runpodctl send ko/out/needle3-ko-20L.cact
```

Pi에서 [runpodctl](https://docs.runpod.io/runpodctl/overview)을 설치한 뒤 그 코드로 받습니다:

```bash
runpodctl receive <코드>
```

**방법 B: scp** (공인 IP SSH Pod인 경우)

```bash
scp -P <포트> root@<IP>:/workspace/NeedleTest/ko/out/needle3-ko-20L.cact models/
```

학습 데이터(`ko/data/train.jsonl`)와 학습 로그는 git으로 커밋해 두면 기록이 남습니다. 데이터셋도 대회에서 내세울 우리 기여입니다.

## 8. 끝낼 때

| 동작 | 결과 | 비용 |
|---|---|---|
| **Stop** | GPU 반납, Volume disk(`/workspace`) 유지, 컨테이너 디스크 삭제 | 저장 공간 요금만 (문서 기준 GB당 월 $0.20) |
| **Terminate** | Pod와 Volume disk 모두 삭제 (Network volume은 유지) | 요금 없음 |

작업이 끝나면 바로 **Stop**하고, 결과를 Pi로 옮긴 걸 확인한 뒤 **Terminate**하세요. 켜둔 채로 두면 GPU 시간 요금이 계속 나갑니다.

## 문제 해결

| 증상 | 확인할 것 |
|---|---|
| `jax devices`에 CPU만 나옴 | `nvidia-smi`로 GPU가 보이는지 확인. 보이면 `.venv/bin/pip install -U "jax[cuda12]"` 후 재시도 |
| 학습 중 메모리 부족 (OOM) | `--batch-size`를 8이나 4로 낮춤 |
| 체크포인트 다운로드 실패 | Hugging Face 일시 오류일 수 있음. 다시 실행하면 이어서 받음 |
| Pod를 껐다 켰더니 설치한 게 없음 | 저장소를 `/workspace` 밖에 받았는지 확인 |
