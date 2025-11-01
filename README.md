# CXL Type3 메모리 불량 검출을 위한 강화학습 프로젝트

## 프로젝트 개요 (2025-11-01 업데이트)

삼성전자 CXL Type3 메모리 모듈에서 DRAM Memory Cell 불량을 검출하기 위한 **혁명적인 강화학습 기반** 테스트 프로그램입니다.

### 핵심 혁신: Low-level DRAM Command 직접 제어

기존 접근법과의 차이:

```
❌ 기존 방식 (제한적):
   action = select_algorithm([March_C+, MATS+, Checkerboard, ...])
   → 고정된 알고리즘 조합만 가능

✅ 새로운 방식 (혁명적):
   action = {
       'primitive': WRITE_READ_ROW | ROW_HAMMER | BANK_THRASH | ...,
       'rank': 0-3,
       'bank_group': 0-7,
       'bank': 0-3,
       'row': 0-262143,
       'pattern': PRBS | Checkerboard | Walking_1s | ...
   }
   → RL이 DRAM command 시퀀스를 직접 생성!
   → 새로운 테스트 패턴을 "발명"할 수 있음!
```

### 문제 정의

**발견된 사실:**
```
stressapptest (범용, 10시간) → PASS
  ↓
실제로는 불량 존재!
  ↓
MRAT (회사 자체 패턴) → FAIL 검출 성공
```

**핵심 통찰:**
- stressapptest는 불완전함 (일부 불량 놓침)
- MRAT은 효과적이지만, **다른 불량은 다른 패턴 필요**
- **각 불량 타입마다 최적의 DRAM command 시퀀스가 다름**

## 프로젝트 목표

### Phase 1: MRAT 수준 도달 (현재 단계)
```
입력: 불량 device (MRAT FAIL 또는 stressapptest FAIL)
목표: 이 device를 FAIL로 만드는 command 시퀀스 발견
검증: 발견한 시퀀스를 반복 실행 → 매번 FAIL 검출

성공 기준:
  - MRAT FAIL device → RL도 FAIL 검출
  - 검출률 100% 목표 (False Negative = 0)
  - 재현성: 5회 테스트 중 5회 검출
```

### Phase 2: MRAT 초과 (목표)
```
목표: RL이 MRAT보다 더 많은 불량 발견
측정:
  - MRAT PASS + stressapptest PASS → RL이 불량 발견
  - 검증: 추가 검사로 실제 불량임을 확인
```

### Phase 3: 범용 패턴 (최종)
```
목표: 다양한 불량 타입을 하나의 policy로 검출
측정:
  - 여러 불량 타입 (MRAT, 기타) 모두 검출
  - 양산 라인 적용 가능 수준
```

## 프로젝트 구조

```
cxl_memory_rl_project/
├── src/
│   ├── rl_environment.py      # ✨ 최신 Low-level DRAM Command RL Environment
│   ├── mbist_interface.py     # MBIST C library Python wrapper
│   ├── ppo_agent.py           # PPO 에이전트
│   ├── dqn_agent.py           # DQN 에이전트
│   ├── train_agent.py         # PPO 훈련 스크립트
│   ├── train_dqn.py           # DQN 훈련 스크립트
│   ├── config_loader.py       # 설정 파일 로더
│   └── c_library/             # MBIST C library integration
├── docs/
│   ├── REVISED_PROJECT_DESIGN.md        # ✨ 최신 프로젝트 설계 (필독!)
│   ├── MBIST_INTEGRATION_PLAN.md        # MBIST 통합 계획
│   ├── MBIST_COMMAND_ENCODING_ANALYSIS.md
│   └── JAX_SETUP.md
├── config/
│   └── default_config.yaml
├── models/                    # 훈련된 모델
├── logs/                      # 훈련 로그
├── tests/                     # 테스트 코드
└── README.md
```

## 강화학습 환경 설계 (최신)

### Action Space: Low-level DRAM Primitives

**12가지 DRAM Primitives:**

| ID | Primitive | 설명 | 용도 |
|----|-----------|------|------|
| 0 | WRITE_READ_CELL | 단일 셀 W/R | 기본 셀 테스트 |
| 1 | WRITE_READ_ROW | 전체 row W/R | Row 단위 테스트 |
| 2 | WRITE_READ_BANK | 전체 bank W/R | Bank 단위 테스트 |
| 3 | ROW_HAMMER | 특정 row 반복 액세스 | Rowhammer 불량 검출 |
| 4 | BANK_THRASH | Bank 간 빠른 전환 | Bank 간섭 검출 |
| 5 | REFRESH_STRESS | Refresh 지연/스킵 | Refresh 관련 불량 |
| 6 | CHECKERBOARD_WR | Checkerboard 패턴 | 인접 셀 간섭 |
| 7 | WALKING_ONES | Walking 1s | 비트 패턴 민감도 |
| 8 | PRBS_PATTERN | PRBS 랜덤 패턴 | 랜덤 데이터 민감도 |
| 9 | ASCENDING_MARCH | 주소 증가 순서 | March 알고리즘 |
| 10 | DESCENDING_MARCH | 주소 감소 순서 | Reverse March |
| 11 | WRITE_DELAY_READ | 쓰기 → 대기 → 읽기 | Retention 불량 |

**Action 구성:**
```python
action = [
    primitive,   # 0-11 (12 choices)
    rank,        # 0-3 (4 ranks)
    bank_group,  # 0-7 (8 bank groups)
    bank,        # 0-3 (4 banks)
    row_start,   # 0-255 (256 row groups)
    row_end,     # 0-255
    pattern,     # 0-7 (8 data patterns)
    repeat       # 0-9 (1-10 repeats)
]

Total action space: 12 × 4 × 8 × 4 × 256 × 256 × 8 × 10 = ~2억
```

### State Space: 실제 DRAM 구조 반영

```python
state = {
    'memory_map': (4, 8, 4, 256),      # [rank, bg, ba, row_group]
    'fault_map': (4, 8, 4, 256),       # 발견된 불량 위치
    'coverage': (4, 8, 4, 256),        # 테스트 커버리지
    'recent_commands': (10, 8),        # 최근 10개 command
    'metadata': (4,)                   # [tests, faults, coverage_ratio, confidence]
}

Total state dimensions: 4×8×4×256 = 32,768 regions
```

### Reward 함수: Phase 1 불량 발견에 집중

```python
# 1. 불량 발견 (최우선!)
if test_failed:
    reward += 10000  # 🎯 매우 높은 보상!
    reward += len(error_addresses) * 100

# 2. 패스 (정보 제공)
else:
    reward += 1  # 작은 보상 (커버리지)

# 3. 탐색 보너스
if new_region:
    reward += 10

# 4. 효율성 (적은 테스트로 불량 발견)
if test_failed:
    reward += 1000 / total_tests

# 5. 다양성 보너스
if diverse_action:
    reward += 5
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

### 2. 환경 테스트

```bash
# 전체 설정 테스트
python3 test_setup.py

# CUDA 성능 테스트 (GPU 있는 경우)
python3 test_cuda.py
```

## 사용법

### 1. 시뮬레이션 모드로 RL 훈련 (추천: 시작점)

실제 하드웨어 없이도 알고리즘 개발 가능합니다.

```bash
source venv/bin/activate

# DQN으로 훈련 (추천)
python3 src/train_dqn.py --safety-mode --episodes 1000

# PPO로 훈련
python3 src/train_agent.py --safety-mode --episodes 1000
```

### 2. 환경 동작 확인

```python
from src.rl_environment import make_env, DRAMPrimitive

# 환경 생성
env = make_env(safety_mode=True, max_tests=1000)

# Reset
obs, info = env.reset()

# Random action
action = env.action_space.sample()
obs, reward, terminated, truncated, info = env.step(action)

# Render
env.render()
```

### 3. 실제 하드웨어 테스트 (주의!)

```bash
# MBIST 라이브러리 경로 지정
export MBIST_LIB_PATH=/home/dhkang/data3/mbist_sample_code-gen2_es

# 실제 하드웨어에서 훈련
python3 src/train_dqn.py \
    --no-safety-mode \
    --mbist-path $MBIST_LIB_PATH \
    --device-id CXL-001 \
    --episodes 500
```

## 시뮬레이션 모드

### 특징
- **실제 하드웨어 없이 개발 가능**
- **다양한 불량 패턴 시뮬레이션**
- **안전한 테스트 환경**

### 시뮬레이션된 불량 패턴

```python
# Pattern 1: Stuck-at fault
{
    'type': 'stuck_at',
    'rank': 0, 'bg': 0, 'ba': 0,
    'row': 100 * 1024,
    'detection_primitives': [WRITE_READ_ROW, WRITE_READ_BANK]
}

# Pattern 2: Row hammer victim
{
    'type': 'rowhammer',
    'rank': 0, 'bg': 1, 'ba': 2,
    'row': 150 * 1024,
    'detection_primitives': [ROW_HAMMER]
}

# Pattern 3: Retention failure
{
    'type': 'retention',
    'rank': 1, 'bg': 3, 'ba': 1,
    'row': 200 * 1024,
    'detection_primitives': [WRITE_DELAY_READ, REFRESH_STRESS]
}
```

RL Agent는 이러한 불량을 찾는 최적의 command 시퀀스를 학습합니다!

## 데이터 패턴

| ID | 패턴 | 설명 | 용도 |
|----|------|------|------|
| 0 | FIXED_00 | All 0x00 | Stuck-at-1 검출 |
| 1 | FIXED_FF | All 0xFF | Stuck-at-0 검출 |
| 2 | FIXED_55 | 0x55555555... | 체커보드 변형 |
| 3 | FIXED_AA | 0xAAAAAAAA... | 체커보드 변형 |
| 4 | CHECKERBOARD | 0x55/0xAA 교대 | 인접 셀 간섭 |
| 5 | PRBS | Pseudo-Random | 랜덤 데이터 민감도 |
| 6 | WALKING_1S | Walking 1s | 비트 패턴 민감도 |
| 7 | WALKING_0S | Walking 0s | 비트 패턴 민감도 |

## 알고리즘 비교: PPO vs DQN

| 특징 | PPO | DQN |
|------|-----|-----|
| **타입** | Policy-based | Value-based |
| **학습 안정성** | 높음 (Clipping) | 중간 (Target Network) |
| **샘플 효율성** | 낮음 (On-policy) | 높음 (Off-policy) |
| **탐색 방법** | Stochastic policy | Epsilon-greedy |
| **메모리 요구량** | 낮음 | 높음 (Replay Buffer) |
| **수렴 속도** | 빠름 | 느림 |
| **Phase 1 추천** | 안정성 중시 시 | 샘플 효율 중시 시 |

### 권장사항
1. **처음 시작**: DQN으로 시작 (구현 단순, 샘플 효율 좋음)
2. **안정성 중시**: PPO 사용
3. **최적 성능**: 두 알고리즘 모두 실험 후 비교

## 모니터링 및 로깅

### 훈련 중 메트릭
- **Episode Reward**: 에피소드별 누적 보상
- **Faults Found**: 발견한 불량 수
- **Coverage**: 테스트한 메모리 영역 비율
- **Test Efficiency**: 테스트 당 불량 발견 비율
- **Command Diversity**: Action 다양성 지표

### 로그 예시
```
Episode 100/1000
  Reward: 10125.5
  Faults Found: 1 🎯
  Coverage: 45.2% (4,563/32,768 regions)
  Tests: 248
  Success Sequence: [WRITE_READ_ROW(r=0,bg=0,ba=0,rows=100-100), ...]
```

## 성능 최적화

### GPU 가속
- NVIDIA RTX 4060 Ti 지원
- CUDA 12.8 활용
- 자동 GPU/CPU 전환

### 메모리 효율성
- Dict observation space로 구조화된 state
- Efficient numpy operations
- Batch processing (DQN replay buffer)

## 다음 단계 (개발 로드맵)

### 단기 (1-2개월)
- [ ] MBIST C library Python wrapper 완성
- [ ] 실제 하드웨어 테스트 환경 구축
- [ ] Phase 1: 첫 번째 불량 device에서 불량 재현 성공

### 중기 (3-6개월)
- [ ] 여러 불량 device에서 테스트
- [ ] Transfer learning (한 device 학습 → 다른 device 적용)
- [ ] Phase 2: MRAT 초과하는 패턴 발견

### 장기 (6개월-1년)
- [ ] Phase 3: 범용 불량 검출 policy 개발
- [ ] 양산 라인 적용
- [ ] stressapptest 완전 대체

## 기술 스택

- **Python**: 3.12+
- **RL Framework**: Gymnasium (OpenAI Gym successor)
- **Deep Learning**: PyTorch 2.x
- **CUDA**: 12.8 (GPU 가속)
- **Hardware Interface**: MBIST C library (ctypes/pybind11)
- **컨트롤러**: Montage MBIST Engine

## 참고 문서

- **docs/REVISED_PROJECT_DESIGN.md**: 전체 프로젝트 설계 (필독!)
- **docs/MBIST_INTEGRATION_PLAN.md**: MBIST 통합 계획
- **docs/MBIST_COMMAND_ENCODING_ANALYSIS.md**: Command encoding 분석
- **CLAUDE.md**: 프로젝트 배경 및 컨텍스트

## 핵심 통찰

> **"MRAT, March C+ 등은 모두 특정 DRAM command 시퀀스일 뿐이다.
> RL이 command 레벨에서 제어하면 새로운 패턴을 발명할 수 있다."**

이 프로젝트는 단순히 기존 알고리즘을 선택하는 것이 아니라,
**RL이 DRAM의 물리적 특성을 이해하고 최적의 테스트 시퀀스를 생성**하도록 합니다.

## 기대 효과

### Phase 1 성공 시
```
✓ RL Agent가 불량 device에서 불량 검출 command 시퀀스 발견
✓ "왜 MRAT이 효과적인지" 이해
✓ 다른 불량에도 적용 가능한 원리 발견
```

### Phase 2 성공 시
```
✓ MRAT보다 더 효과적인 새로운 패턴 발견
✓ 더 많은 불량 검출
✓ 불량품 출하 방지
✓ 양산 수율 향상
```

### 최종 목표
```
✓ 하나의 RL policy로 다양한 불량 타입 검출
✓ stressapptest 완전 대체
✓ 불량 검출률 100% 달성
✓ 테스트 시간 최적화
```

## 라이선스

이 프로젝트는 Samsung Electronics의 내부 연구용으로 개발되었습니다.

## 연락처

프로젝트 관련 문의사항이 있으시면 담당자에게 연락하시기 바랍니다.

---

**마지막 업데이트**: 2025-11-01
**버전**: 2.0 (Low-level DRAM Command Control)
**상태**: Phase 1 개발 중
