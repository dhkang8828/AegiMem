# CXL Memory RL 프로젝트 배포 및 실행 가이드

## 회사 환경에서 실행하기

이 가이드는 로컬 개발 환경에서 개발한 코드를 회사의 CXL 테스트 환경에서 실행하는 방법을 설명합니다.

---

## 1. 사전 준비

### 1.1 필수 요구사항

- **Python**: 3.8 이상
- **CUDA**: 11.0 이상 (GPU 사용 시)
- **메모리**: 최소 8GB RAM (GPU 사용 시 더 많이 필요)
- **MBIST 바이너리**: Montage MBIST 실행 파일

### 1.2 Git에서 코드 다운로드

```bash
# 프로젝트 클론
git clone git@github.com:dhkang8828/AegiMem.git
cd AegiMem

# 또는 pull로 최신 업데이트
git pull origin main
```

---

## 2. 환경 설정

### 2.1 가상환경 생성 및 활성화

```bash
# 가상환경 생성
python3 -m venv venv

# 가상환경 활성화 (Linux/Mac)
source venv/bin/activate

# 가상환경 활성화 (Windows)
venv\Scripts\activate
```

### 2.2 의존성 설치

```bash
# pip 업그레이드
pip install --upgrade pip

# 의존성 설치
pip install -r requirements.txt
```

### 2.3 설치 확인

```bash
# CUDA 사용 가능 여부 확인
python3 -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"

# 전체 환경 테스트
python3 test_setup.py
```

---

## 3. 설정 파일 구성

### 3.1 기본 설정 파일 복사

```bash
# 설정 파일을 복사하여 수정
cp config/default_config.yaml config/company_config.yaml
```

### 3.2 회사 환경에 맞게 설정 수정

`config/company_config.yaml` 파일을 편집:

```yaml
environment:
  # MBIST 바이너리 경로를 실제 경로로 수정
  mbist_binary_path: "/actual/path/to/mbist_smbus.exe"

  # 메모리 크기 (실제 CXL 모듈 크기로 조정)
  memory_size: 0x100000000  # 4GB

  # 안전 모드 (실제 하드웨어: false, 시뮬레이션: true)
  safety_mode: true

training:
  # 훈련 에피소드 수 조정
  total_episodes: 1000

output:
  # 모델 저장 경로
  model_save_dir: "./models"

  # 로그 저장 경로
  log_dir: "./logs"
```

---

## 4. 실행 방법

### 4.1 시뮬레이션 모드로 테스트 (안전)

실제 하드웨어 없이 알고리즘 테스트:

```bash
cd src
python3 train_agent.py --config ../config/company_config.yaml --safety-mode --episodes 100
```

### 4.2 실제 하드웨어 테스트 (주의!)

**⚠️ 경고**: 실제 CXL 하드웨어에서 실행하기 전에 반드시 시뮬레이션 모드로 충분히 테스트하세요!

```bash
cd src
python3 train_agent.py --config ../config/company_config.yaml --no-safety-mode --mbist-path /path/to/mbist
```

### 4.3 Quick Test

전체 훈련 전에 빠른 통합 테스트:

```bash
python3 quick_test.py
```

---

## 5. 실행 옵션

### 5.1 주요 커맨드라인 옵션

```bash
# 설정 파일 지정
--config <path>               # YAML 설정 파일 경로

# 환경 설정
--safety-mode                 # 안전 모드 활성화 (시뮬레이션)
--no-safety-mode             # 안전 모드 비활성화 (실제 하드웨어)
--mbist-path <path>          # MBIST 바이너리 경로

# 훈련 설정
--episodes <num>             # 훈련 에피소드 수
--experiment-name <name>     # 실험 이름
--learning-rate <float>      # 학습률
--batch-size <int>           # 배치 크기
```

### 5.2 실행 예제

```bash
# 예제 1: 빠른 테스트 (100 에피소드, 시뮬레이션)
python3 train_agent.py --episodes 100 --safety-mode --experiment-name "quick_test"

# 예제 2: 전체 훈련 (사용자 정의 설정)
python3 train_agent.py --config ../config/company_config.yaml --episodes 2000

# 예제 3: 실제 하드웨어 (주의!)
python3 train_agent.py \
    --no-safety-mode \
    --mbist-path /company/path/to/mbist \
    --episodes 500 \
    --experiment-name "hardware_test_v1"
```

---

## 6. 결과 확인

### 6.1 로그 파일 위치

훈련 중 생성되는 파일들:

```
logs/
├── <experiment_name>_config.yaml          # 사용된 설정
├── <experiment_name>_progress.png         # 훈련 진행 그래프
├── <experiment_name>_checkpoint_*.json    # 체크포인트 데이터
└── <experiment_name>_final_results.json   # 최종 결과

models/
├── <experiment_name>_best.pt              # 최고 성능 모델
└── <experiment_name>_ep_*.pt              # 에피소드별 체크포인트
```

### 6.2 실시간 모니터링

터미널에서 실시간으로 훈련 진행 상황 확인:

```
Episode   10 | Reward:   245.32 | Avg Reward:   198.45 | Faults:   5 | Steps:  87 | Time: 12.34s | Epsilon: 0.980
Episode   20 | Reward:   312.56 | Avg Reward:   267.89 | Faults:   8 | Steps:  95 | Time: 11.23s | Epsilon: 0.960
...
```

### 6.3 결과 분석

최종 결과 파일 (`*_final_results.json`) 확인:

```bash
cat logs/<experiment_name>_final_results.json | python3 -m json.tool
```

---

## 7. 문제 해결

### 7.1 CUDA 메모리 부족

```bash
# 배치 크기 줄이기
python3 train_agent.py --batch-size 32

# 또는 CPU 사용
# config.yaml에서 device: "cpu"로 설정
```

### 7.2 MBIST 바이너리 실행 권한 오류

```bash
chmod +x /path/to/mbist_smbus.exe
```

### 7.3 모듈 import 오류

```bash
# 가상환경이 활성화되었는지 확인
which python3

# 의존성 재설치
pip install -r requirements.txt --force-reinstall
```

### 7.4 훈련이 너무 느림

- GPU가 실제로 사용되는지 확인: `nvidia-smi`
- 배치 크기 증가 고려
- episode 수를 줄여서 테스트

---

## 8. 안전 수칙 (실제 하드웨어 사용 시)

⚠️ **중요**: 실제 CXL 하드웨어에서 실행할 때는 다음 사항을 반드시 준수하세요:

1. **시뮬레이션 먼저**: 항상 `--safety-mode`로 충분히 테스트
2. **작은 규모부터**: 처음에는 적은 episode 수로 시작
3. **모니터링**: 실행 중 하드웨어 상태 지속 모니터링
4. **백업**: 중요한 데이터는 사전 백업
5. **권한**: 실제 하드웨어 접근 권한 확인
6. **로그 확인**: 이상 징후 발견 시 즉시 중단

---

## 9. 프로젝트 업데이트

새로운 코드가 GitHub에 push되면:

```bash
# 로컬 변경사항 백업 (있는 경우)
git stash

# 최신 코드 받기
git pull origin main

# 변경사항 적용 (있는 경우)
git stash pop

# 의존성 업데이트
pip install -r requirements.txt --upgrade
```

---

## 10. 결과 공유

회사 환경에서 실행 후 결과를 공유할 때:

```bash
# 결과 파일 압축
tar -czf results_<date>.tar.gz logs/ models/

# 또는 JSON 결과만
cp logs/*_final_results.json ./results/
```

결과 파일을 개발 환경으로 전송하고, 프롬프트로 내용을 알려주시면 분석 및 개선을 진행하겠습니다.

---

## 문의사항

문제가 발생하거나 질문이 있으면:

1. 로그 파일 내용을 확인
2. 에러 메시지를 정확히 기록
3. 실행 명령어와 설정 파일 내용을 함께 공유

GitHub Issues: https://github.com/dhkang8828/AegiMem/issues
