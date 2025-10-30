# CXL Type3 메모리 불량 검출을 위한 강화학습 프로젝트

## 프로젝트 개요

이 프로젝트는 삼성전자 CXL Type3 메모리 모듈에서 DRAM Memory Cell 불량을 검출하기 위한 강화학습 기반 테스트 프로그램입니다.

### 핵심 기술
- **강화학습**: PPO (Proximal Policy Optimization) 알고리즘
- **하드웨어**: Montage 컨트롤러의 MBIST 엔진 활용
- **최적화 목표**: 검출 정확도 향상 및 새로운 불량 패턴 발견

## 프로젝트 구조

```
cxl_memory_rl_project/
├── src/                    # 소스 코드
│   ├── rl_environment.py   # 강화학습 환경
│   ├── ppo_agent.py       # PPO 에이전트
│   └── train_agent.py     # 훈련 스크립트
├── models/                # 훈련된 모델 저장
├── logs/                  # 훈련 로그 및 결과
├── data/                  # 데이터 파일
├── config/                # 설정 파일
├── tests/                 # 테스트 코드
├── requirements.txt       # Python 의존성
├── test_setup.py         # 설정 테스트
├── test_cuda.py          # CUDA 테스트
└── README.md             # 프로젝트 문서
```

## 설치 및 설정

### 1. 가상환경 생성 및 의존성 설치

```bash
# 가상환경 생성
python3 -m venv venv
source venv/bin/activate

# 의존성 설치
pip install -r requirements.txt
```

### 2. 설정 검증

```bash
# 전체 설정 테스트
python3 test_setup.py

# CUDA 성능 테스트
python3 test_cuda.py
```

## 사용법

### 1. 기본 훈련 실행

```bash
source venv/bin/activate
python3 src/train_agent.py --episodes 1000 --safety-mode
```

### 2. 실제 하드웨어 테스트 (주의: 안전 모드 해제)

```bash
python3 src/train_agent.py --episodes 100 --mbist-path /path/to/mbist_binary
```

### 3. 훈련 파라미터 설정

```bash
python3 src/train_agent.py \
    --episodes 2000 \
    --experiment-name "cxl_optimization_v1" \
    --safety-mode
```

## 강화학습 환경 설계

### State (상태)
- **Memory Map**: 64x64 그리드로 표현된 메모리 영역
- **Fault Map**: 발견된 불량 위치 정보
- **Coverage Map**: 테스트 커버리지 현황
- **Confidence Map**: 테스트 신뢰도 정보

### Action (행동)
- **Algorithm ID**: 14가지 MBIST 알고리즘 선택 (March C+, MATS+, etc.)
- **Pattern ID**: 6가지 데이터 패턴 선택 (55h, AAh, Checkerboard, etc.)
- **Start Region**: 테스트 시작 메모리 영역
- **End Region**: 테스트 종료 메모리 영역

### Reward (보상)
- **새로운 불량 발견**: +100점
- **커버리지 증가**: +1점 (영역 크기에 비례)
- **효율성 보너스**: +10점 (시간당 불량 발견 수)
- **중복 테스트 패널티**: -5점
- **시간 패널티**: -0.1점 (테스트 시간에 비례)
- **안전 위반 패널티**: -50점

## 지원하는 MBIST 알고리즘

| ID | 알고리즘 | 설명 |
|----|----------|------|
| 0x00 | PPR | Post Package Repair |
| 0x01 | WRC | Write, Read and Compare |
| 0x02 | Checkerboard | 체커보드 패턴 테스트 |
| 0x03 | March C+ | 메모리 셀 테스트 알고리즘 |
| 0x04 | MATS | Memory Array Test Sequence |
| 0x05 | MATS+ | 향상된 MATS |
| 0x06 | Walking 1s | 워킹 원 패턴 |
| 0x07 | Walking 0s | 워킹 제로 패턴 |
| 0x08 | March X | March X 알고리즘 |
| 0x09 | March Y | March Y 알고리즘 |
| 0x0a | Marching 1/0 | 마칭 패턴 |
| 0x0c | TPH | Temperature Pattern Hold |
| 0x0e | Write, Read 1BL16 | 1BL16 테스트 |
| 0x0f | MRW, MRR | 메모리 읽기/쓰기 |

## 안전 기능

### 하드웨어 보호
- **최대 연속 테스트 제한**: 1000회
- **테스트 간 쿨다운**: 1초
- **최대 테스트 시간**: 300초
- **연속 실패 제한**: 10회

### 시뮬레이션 모드
- 실제 하드웨어 없이도 알고리즘 개발 가능
- 다양한 불량 패턴 시뮬레이션
- 안전한 테스트 환경 제공

## 모니터링 및 로깅

### 훈련 메트릭
- 에피소드별 보상
- 발견된 불량 수
- 테스트 커버리지
- 에이전트 성능 지표

### 로그 파일
- `logs/`: 훈련 진행 상황 및 결과
- `models/`: 저장된 모델 파일
- 시각화된 진행 상황 그래프

## 성능 최적화

### GPU 가속
- NVIDIA RTX 4060 Ti 지원
- CUDA 12.8 활용
- 자동 GPU/CPU 전환

### 메모리 효율성
- 배치 처리 최적화
- 경험 버퍼 관리
- 가중치 공유

## 문제 해결

### 일반적인 문제

1. **MBIST 바이너리 권한 오류**
   ```bash
   chmod +x /path/to/mbist_binary
   ```

2. **CUDA 메모리 부족**
   - 배치 크기 줄이기
   - GPU 메모리 정리: `torch.cuda.empty_cache()`

3. **느린 훈련 속도**
   - GPU 사용 확인: `nvidia-smi`
   - 배치 크기 조정

## 향후 개발 계획

1. **알고리즘 개선**
   - 다른 RL 알고리즘 비교 (SAC, A3C)
   - 하이퍼파라미터 최적화

2. **하드웨어 통합**
   - 실제 CXL Type3 모듈 테스트
   - 실시간 불량 검출

3. **분석 도구**
   - 불량 패턴 분석
   - 예측 모델 개발

## 라이선스

이 프로젝트는 Samsung Electronics의 내부 연구용으로 개발되었습니다.

## 연락처

프로젝트 관련 문의사항이 있으시면 담당자에게 연락하시기 바랍니다.
