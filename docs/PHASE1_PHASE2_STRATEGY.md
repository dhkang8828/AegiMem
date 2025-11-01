# Phase 1 & Phase 2 전략

## 프로젝트 목표 재정의

### 최종 목표
**stressapptest를 대체하는 RL 기반 메모리 불량 검출 프로그램**

### 단계별 접근
- **Phase 1**: 기존 알고리즘(stressapptest, MRAT) 패턴 발견
- **Phase 2**: 새로운 불량 패턴 발견 및 확장

---

## Phase 1: Pattern Discovery (Week 1-6)

### 목표
**stressapptest와 MRAT이 사용하는 패턴 조합을 RL이 스스로 발견**

### 핵심 아이디어
```
RL Agent에게 물어보기:
"이 불량 device를 FAIL로 만드는 패턴 시퀀스는?"

→ Agent가 찾은 답: [⇑(W0), ⇑(R0,W1), ⇓(R1)]
→ 이게 바로 MRAT의 핵심 패턴!
```

### Action Space

```python
# 1536 Actions = 6 operations × 256 patterns
action = operation_type * 256 + pattern_byte

# 6가지 Operation Types:
OperationType = {
    WRITE_ASC: 0        # ^(W pattern) - Ascending Write only
    READ_ASC: 1         # ^(R pattern) - Ascending Read & Compare
    WRITE_DESC: 2       # v(W pattern) - Descending Write only
    READ_DESC: 3        # v(R pattern) - Descending Read & Compare
    WRITE_READ_DESC: 4  # v(W+R pattern) - Descending Write then Read
    WRITE_READ_ASC: 5   # ^(W+R pattern) - Ascending Write then Read
}

# 256 Data Patterns: 0x00 ~ 0xFF
# RL이 효과적인 패턴을 학습으로 발견
# 예: 0x00, 0xFF, 0x55, 0xAA, 0x33, 0xCC, ...

# 고정 파라미터
start_address = 0x0
end_address = ENTIRE_MEMORY  # 전체 영역 스캔
step_size = 1  # 순차 접근
```

### 주요 Data Patterns (예시)

```python
# 전통적 패턴
0x00 = 0b00000000  # All zeros
0xFF = 0b11111111  # All ones
0x55 = 0b01010101  # Checkerboard
0xAA = 0b10101010  # Inverse checkerboard

# RL이 발견할 수 있는 새로운 패턴
0x33 = 0b00110011
0xCC = 0b11001100
0xF0 = 0b11110000
0x0F = 0b00001111
... (총 256가지)
```

### Episode 구조

```python
# 한 에피소드 = 패턴 시퀀스 (최대 10개)
episode = [
    action_0,  # 예: ⇑(W0)
    action_1,  # 예: ⇑(R0)
    action_2,  # 예: ⇓(R1)
    ...
]

# 각 action = 전체 메모리 한 번 스캔
# 시퀀스 = 여러 스캔의 조합
```

### State Space

```python
observation = {
    'sequence_history': [action_0, action_1, ..., action_n],  # 최근 10개
    'last_test_result': 0 or 1,  # PASS/FAIL
    'sequence_length': n,
    'error_count': count  # 발견된 에러 개수
}
```

### Reward Function

```python
def calculate_reward(result, sequence_length):
    """
    Phase 1: 불량 발견에만 집중
    """
    if result['test_failed']:
        # 불량 발견! (짧은 시퀀스일수록 높은 보상)
        return 1000.0 / sequence_length
    else:
        # 불량 못 찾음 (작은 페널티)
        return -0.1
```

### 성공 기준

#### 정량적 목표
```
1. 검출률: MRAT이 찾은 불량을 100% 재발견
2. 시퀀스 길이: 10 step 이하
3. 재현성: 동일 불량 5/5회 검출
```

#### 정성적 목표
```
1. 학습된 패턴이 MRAT과 유사한지 분석
2. 왜 그 패턴이 효과적인지 이해
3. stressapptest가 놓치는 이유 파악
```

### 훈련 전략

#### 데이터셋
```python
# 불량 device 필요
faulty_devices = [
    {
        'id': 'CXL-001',
        'known_fault': 'MRAT FAIL',
        'stressapptest': 'PASS',  # 놓친 케이스!
        'fault_type': 'unknown'
    },
    {
        'id': 'CXL-002',
        'known_fault': 'MRAT FAIL',
        'stressapptest': 'PASS',
        'fault_type': 'unknown'
    },
    # ... 최소 5-10개 필요
]
```

#### 훈련 과정
```python
for device in faulty_devices:
    env = Phase1Environment(device)
    agent = DQNAgent(state_dim=..., action_dim=1536)

    for episode in range(max_episodes):
        state = env.reset()
        sequence = []

        while not done:
            action = agent.select_action(state)
            next_state, reward, done, info = env.step(action)

            sequence.append(action)
            agent.train()

            if info['test_failed']:
                # 성공! 패턴 저장
                save_successful_pattern(device.id, sequence)
                decoded = decode_action(action)
                print(f"✓ Found pattern for {device.id}: {decoded}")
                break
```

#### 학습 가속
```python
# 1536개 action space
# - Epsilon-greedy로 exploration
# - Experience replay로 샘플 효율 극대화
# - 짧은 에피소드 (최대 10 step)
# - Action space가 커졌지만 구조화되어 있음:
#   * Operation type만 6가지
#   * Pattern은 연속적 (0x00 ~ 0xFF)

예상 학습 시간:
- 에피소드당: 10 actions × (메모리 스캔 시간)
- 총 에피소드: 1000-5000회 (action space 증가로 더 필요)
- Device당 학습: 1-3일
```

### 제약사항 및 해결

#### 제약 1: 전체 메모리 스캔 시간
```
문제: 68B 주소 모두 스캔 = 시간 오래 걸림

해결책:
1. 샘플링: 전체의 1-10%만 테스트 (대표성 있게)
2. MBIST 하드웨어 속도 활용 (아마 빠를 것)
3. 병렬 실행 (가능하다면)
```

#### 제약 2: 불량 device 필요
```
문제: MRAT FAIL device 확보 필요

해결책:
1. 양산 라인에서 불량품 수집
2. 시뮬레이터에서 불량 주입 (초기 개발)
3. 최소 5-10개 확보 목표
```

#### 제약 3: 하드웨어 손상 위험
```
문제: 반복 테스트로 하드웨어 손상 가능

해결책:
1. 이미 불량인 device 사용 (추가 손상 무관)
2. Safety limit 설정 (최대 테스트 횟수)
3. 온도 모니터링
```

---

## Phase 2: Advanced Discovery (Week 7+)

### 진입 조건
```
Phase 1에서 다음 조건 만족 시:
✅ MRAT 불량 재발견율 95% 이상
✅ 패턴 시퀀스 이해 완료
✅ 실제 하드웨어 검증 완료
```

### 목표
**MRAT도 못 찾는 새로운 불량 패턴 발견**

### Action Space 확장

```python
# Phase 1의 32 actions에 추가

# 1. Address Control (추가)
action = {
    # 기존
    'direction': 'ASCENDING' | 'DESCENDING',
    'operation': 'READ' | 'WRITE',
    'pattern': 0-7,

    # 새로 추가
    'address_mode': 'FULL' | 'SPARSE' | 'TARGETED',
    'rank': 0-3,         # 특정 rank만
    'bank_group': 0-7,   # 특정 BG만
    'step_size': [1, 2, 4, 8, 16, 32, 64, 128]
}

# 2. Advanced Operations (추가)
advanced_ops = {
    'ROW_HAMMER': {
        'target_row': 0-262143,
        'hammer_count': 1000-100000
    },
    'RETENTION_TEST': {
        'delay_ms': 10-10000
    },
    'BANK_THRASHING': {
        'bank_switches': 100-1000
    }
}

# Total actions: 수천~수만 개
```

### 접근 방법

#### 1. Transfer Learning
```python
# Phase 1에서 학습한 policy를 base로 사용
phase2_agent = load_model('phase1_best_policy.pth')

# Fine-tuning with expanded action space
phase2_agent.expand_action_space(new_actions)
```

#### 2. Hierarchical RL
```python
# High-level policy: 어떤 전략?
high_level_actions = [
    'MARCH_LIKE',      # Phase 1 패턴 사용
    'ROW_HAMMER',      # Rowhammer 공격
    'RETENTION_TEST',  # 보유 시간 테스트
    'TARGETED_SCAN'    # 특정 영역 집중
]

# Low-level policy: 구체적 실행
low_level_policy(high_level_action)
```

#### 3. Curriculum Learning
```python
# 단계적 난이도 증가
curriculum = [
    'Stage 1: Phase 1 patterns (32 actions)',
    'Stage 2: + Address control (256 actions)',
    'Stage 3: + Row hammer (1024 actions)',
    'Stage 4: + All advanced ops (10000+ actions)'
]
```

### 새로운 불량 발견 시나리오

```python
# Scenario 1: stressapptest PASS + MRAT PASS
# → RL이 불량 발견!
device = {
    'stressapptest': 'PASS',
    'MRAT': 'PASS',
    'RL_Phase2': 'FAIL'  # 새로운 발견!
}

# Scenario 2: 특정 조건에서만 발생하는 불량
rl_pattern = [
    'ROW_HAMMER(row=1234, count=50000)',  # Rowhammer 공격
    'READ(victim_row=1233)',               # 인접 row 확인
    'FAIL detected!'                       # 발견!
]
```

### 검증 방법

```python
# Phase 2에서 발견한 불량이 진짜인지 검증

1. 재현성 테스트:
   - 동일 device에서 5회 반복
   - 다른 device에서도 테스트

2. 물리적 검증:
   - 상세 에러 분석
   - 불량 위치 확인
   - 실패 메커니즘 분석

3. 전문가 검토:
   - 하드웨어 엔지니어 확인
   - 불량 타입 분류
```

---

## 구현 우선순위

### Milestone 1: Phase 1 Environment (Week 1-2)
```python
✅ Phase1Environment 구현
✅ 1536-action space (6 operations × 256 patterns)
✅ Pattern executor (6가지 operation types)
✅ Reward function
✅ State management
```

### Milestone 2: Agent Training (Week 3-4)
```python
✅ DQN agent 구현
✅ Experience replay
✅ 불량 device에서 훈련
✅ Pattern 저장/분석
```

### Milestone 3: Validation (Week 5-6)
```python
✅ 실제 하드웨어 테스트
✅ MRAT 비교
✅ 재현성 확인
✅ Phase 1 완료 보고서
```

### Milestone 4: Phase 2 Planning (Week 7+)
```python
⏳ Phase 1 결과 분석
⏳ Phase 2 action space 설계
⏳ Advanced operations 구현
⏳ 새로운 불량 발견
```

---

## 기대 효과

### Phase 1 성공 시
```
✅ MRAT 수준 검출 능력 검증
✅ RL 접근법 타당성 증명
✅ 패턴 조합의 중요성 이해
✅ 불량 검출 원리 학습

→ 프로젝트 계속 진행 근거 확보
```

### Phase 2 성공 시
```
✅ MRAT를 넘어서는 검출 능력
✅ 새로운 불량 패턴 발견
✅ 양산 적용 가능성
✅ 테스트 시간 단축

→ 실제 산업 가치 창출
```

---

## 리스크 관리

### Phase 1 리스크

#### R1-1: 불량 device 확보 실패
```
확률: 중
영향: 높음
대응: 시뮬레이터로 대체, 양산팀 협조 요청
```

#### R1-2: RL이 패턴을 못 찾음
```
확률: 낮음 (32 actions만 있어서 쉬움)
영향: 높음
대응: Reward function 조정, Expert demonstration 활용
```

#### R1-3: 메모리 스캔 시간 과다
```
확률: 중
영향: 중
대응: 샘플링, 하드웨어 가속 활용
```

### Phase 2 리스크

#### R2-1: Action space 너무 커짐
```
확률: 높음
영향: 중
대응: Hierarchical RL, Curriculum learning
```

#### R2-2: False positive (가짜 불량 발견)
```
확률: 중
영향: 높음
대응: 엄격한 검증 절차, 전문가 리뷰
```

---

## 성공 지표

### Phase 1 KPI

| 지표 | 목표 | 측정 방법 |
|------|------|----------|
| MRAT 불량 재발견율 | 100% | Known fault detection |
| 시퀀스 길이 | ≤10 steps | Episode length |
| 학습 시간 | ≤1000 episodes | Training log |
| 재현성 | 5/5회 | Repeated tests |

### Phase 2 KPI

| 지표 | 목표 | 측정 방법 |
|------|------|----------|
| 새 불량 발견 | ≥1개 | Validated new faults |
| False positive rate | <10% | Expert review |
| 양산 적용 | Pilot line | Production integration |

---

## 다음 액션

### 즉시 (이번 주)
1. Phase 1 Environment 구현 시작
2. 32-action space 코드 작성
3. Pattern executor 구현
4. 불량 device 확보 계획 수립

### 단기 (2주 내)
1. DQN agent 구현
2. 시뮬레이터에서 초기 테스트
3. 실제 하드웨어 연동 준비

### 중기 (1개월 내)
1. Phase 1 훈련 완료
2. MRAT 비교 분석
3. Phase 2 설계 시작

---

## 참고 문서

- `REVISED_PROJECT_DESIGN.md` - 전체 프로젝트 설계
- `MBIST_INTEGRATION_PLAN.md` - MBIST 통합 계획
- `C_LIBRARY_BUILD.md` - C 라이브러리 빌드
- `PROJECT_PLAN.md` - 기존 프로젝트 계획

---

## 변경 이력

- 2025-11-01: 초안 작성
- 2025-11-01: Phase 1/2 전략 확정
  - Phase 1: 32 actions (simplified)
  - Phase 2: Extended action space
  - 명확한 목표 및 성공 기준 정의
- 2025-11-01: Action space 재설계
  - 1536 actions = 6 operations × 256 patterns
  - Operation types: WRITE_ASC, READ_ASC, WRITE_DESC, READ_DESC, WRITE_READ_DESC, WRITE_READ_ASC
  - Pattern space: 0x00 ~ 0xFF (RL이 최적 패턴 학습)
  - MBIST user-specify 패턴 기능 활용
