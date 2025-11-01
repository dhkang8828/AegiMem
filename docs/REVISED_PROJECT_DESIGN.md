# CXL Type3 메모리 불량 검출 RL 프로젝트 재설계

## 프로젝트 개요 (2025-11-01 업데이트)

### 문제 정의

**발견된 사실:**
```
stressapptest (범용, 10시간) → PASS
실제로는 불량 존재!
MRAT (회사 자체 패턴) → FAIL 검출 성공
```

**핵심 통찰:**
- stressapptest는 불완전함 (일부 불량 놓침)
- MRAT은 효과적이지만, 다른 불량은 다른 패턴 필요
- **각 불량 타입마다 최적의 DRAM command 시퀀스가 다름**

### 혁명적 접근법

기존 (제한적):
```python
action = select_algorithm([March_C+, MATS+, ...])  # 고정된 알고리즘 조합
```

새로운 (혁명적):
```python
action = {
    'command': ACT | WR | RD | PRE,
    'rank': 0-3,
    'bank_group': 0-7,
    'bank': 0-3,
    'row': 0-262143,
    'column': 0-2047,
    'data_pattern': PRBS | Checkerboard | ...
}
# RL이 DRAM command 시퀀스를 직접 생성!
# → 새로운 테스트 패턴을 "발명"할 수 있음!
```

## 프로젝트 목표

### 주목표
**stressapptest를 대체하는 MBIST 기반 불량 검출 프로그램 개발**

### 성공 기준

#### Phase 1: MRAT 수준 도달 (필수)
```
목표: RL Agent가 MRAT이 찾은 불량을 재발견
측정:
  - MRAT FAIL device → RL도 FAIL 검출
  - 검출률 100% 목표 (False Negative = 0)
  - 재현성: 5회 테스트 중 5회 검출
```

#### Phase 2: MRAT 초과 (목표)
```
목표: RL이 MRAT보다 더 많은 불량 발견
측정:
  - MRAT PASS + stressapptest PASS → RL이 불량 발견
  - 검증: 추가 검사로 실제 불량임을 확인
```

#### Phase 3: 범용 패턴 (최종)
```
목표: 다양한 불량 타입을 하나의 policy로 검출
측정:
  - 여러 불량 타입 (MRAT, 기타) 모두 검출
  - 양산 라인 적용 가능 수준
```

## MBIST Engine 기능 (Montage)

### API 위치
```
/home/dhkang/data3/mbist_sample_code-gen2_es/
```

### 핵심 기능

#### 1. Low-level DRAM Command
```c
typedef enum tag_CMD_TYPE {
    ACT,        // ACTIVATE
    WR, WRA,    // WRITE, WRITE with Auto-precharge
    RD, RDA,    // READ, READ with Auto-precharge
    PREab,      // PRECHARGE all banks
    PREsb,      // PRECHARGE single bank
    REFab,      // REFRESH all banks
    REFsb,      // REFRESH single bank
    MRW, MRR,   // Mode Register Write/Read
    // ... 더 많은 명령어
} CMD_TYPE;
```

#### 2. 주소 공간
```c
typedef struct tagMBIST_ADDRS_CXL {
    uint64_t bg : 3;      // Bank Group (0-7)
    uint64_t ba : 2;      // Bank Address (0-3)
    uint64_t rank : 2;    // Rank (0-3)
    uint64_t column : 11; // Column (0-2047)
    uint64_t row : 18;    // Row (0-262143)
    uint64_t cid : 4;     // Chip ID (0-15)
    uint64_t ch : 2;      // Channel
} MBIST_ADDRS_CXL_T;

총 주소 공간: 8 × 4 × 4 × 2048 × 262144 × 16 = ~280 테라 주소
```

#### 3. 데이터 패턴
```c
- PRBS (Pseudo-Random Binary Sequence)
- 고정 640-bit 패턴
- Checkerboard (0x55, 0xAA 교대)
- Walking 1s/0s
- Per-DQ 독립 패턴 (40개 DQ 각각 다른 패턴)
```

#### 4. 에러 분석
```c
// 에러 주소 로깅 (최대 16개)
mt_get_err_addrs(MBIST_ERR_ADDRS_T *addrs, channel)

// DQ별 에러 카운트 (40개 DQ)
mt_get_dq_error_cnt(MBIST_DQ_ERR_CNT_T *dq, overflow, channel)

// 에러 발생 데이터 (640 bits)
mt_get_failure_data(MBIST_BL16_DATA_T *bl16, number, channel)

// 테스트 결과
mt_get_test_result(channel)  // 0: pass, 1: fail
```

## Phase 1 상세 설계

### 목표
**이미 불량이 발생한 device에서 어떤 DRAM command 조합이 불량을 발현시키는지 RL이 찾기**

### 전제 조건
```
입력: 불량 device (MRAT FAIL 또는 stressapptest FAIL)
목표: 이 device를 FAIL로 만드는 command 시퀀스 발견
검증: 발견한 시퀀스를 반복 실행 → 매번 FAIL 검출
```

### Action Space 설계

#### 옵션 A: Primitive Commands (권장)
```python
class DRAMPrimitive(Enum):
    # Basic operations
    WRITE_READ_CELL = 0      # 단일 셀 W/R
    WRITE_READ_ROW = 1       # 전체 row W/R
    WRITE_READ_BANK = 2      # 전체 bank W/R

    # Stress operations
    ROW_HAMMER = 3           # 특정 row 반복 액세스 (rowhammer)
    BANK_THRASH = 4          # Bank 간 빠른 전환
    REFRESH_STRESS = 5       # Refresh 지연/스킵

    # Pattern operations
    CHECKERBOARD_WR = 6      # Checkerboard 패턴 쓰기
    WALKING_ONES = 7         # Walking 1s
    PRBS_PATTERN = 8         # PRBS 랜덤 패턴

    # March-like operations
    ASCENDING_MARCH = 9      # 주소 증가 순서
    DESCENDING_MARCH = 10    # 주소 감소 순서

    # Retention test
    WRITE_DELAY_READ = 11    # 쓰기 → 대기 → 읽기

action = {
    'primitive': DRAMPrimitive,     # 12 choices
    'rank': 0-3,                    # 4 choices
    'bank_group': 0-7,              # 8 choices
    'bank': 0-3,                    # 4 choices
    'row_start': 0-255,             # 256 groups (262144 rows / 1024)
    'row_end': 0-255,               # 256 groups
    'pattern': 0-7,                 # 8 pattern types
    'repeat': 1-10                  # 반복 횟수
}

# Action space size: 12 × 4 × 8 × 4 × 256 × 256 × 8 × 10 = ~2억
# 하지만 continuous하지 않으므로 학습 가능
```

### State Space 설계

```python
class DRAMState:
    """RL Agent가 관찰하는 상태"""

    def __init__(self):
        # Memory structure (coarse-grained)
        # [rank, bank_group, bank, row_group]
        self.memory_map = np.zeros((4, 8, 4, 256), dtype=np.float32)

        # Fault detection (발견된 불량 위치)
        self.fault_detected = np.zeros((4, 8, 4, 256), dtype=bool)

        # Test coverage (테스트한 영역)
        self.test_coverage = np.zeros((4, 8, 4, 256), dtype=int)

        # Recent command sequence (시퀀스 중요!)
        self.recent_commands = []  # Last 10 commands

        # Confidence (신뢰도)
        self.confidence = np.zeros((4, 8, 4, 256), dtype=float)

        # Metadata
        self.total_tests = 0
        self.faults_found = 0
        self.current_hypothesis = None  # 현재 가설 (어디에 불량?)
```

### Reward 함수

```python
class RewardCalculator:
    """Phase 1: 불량 발견에 집중"""

    def calculate(self, action, result):
        reward = 0.0

        # 1. 불량 발견 (최우선!)
        if result.test_failed:  # mt_get_test_result() == 1
            reward += 10000  # 매우 높은 보상

            # 에러 주소 정보가 있으면 추가 보상
            if result.error_addresses:
                reward += len(result.error_addresses) * 100

            print(f"🎯 FAULT DETECTED! Command: {action}")
            print(f"   Error addresses: {result.error_addresses}")

        # 2. 패스 (정보 제공)
        else:
            reward += 1  # 작은 보상 (커버리지)

        # 3. 탐색 보너스 (초기)
        if self.is_new_region(action):
            reward += 10

        # 4. 효율성 (적은 테스트로 불량 발견)
        if result.test_failed:
            reward += 1000 / self.total_tests  # 빨리 찾을수록 좋음

        # 5. 다양성 보너스
        if self.is_diverse_action(action):
            reward += 5

        return reward
```

### RL Environment 구현

```python
class DRAMCommandRLEnvironment(gym.Env):
    """Low-level DRAM Command RL Environment for Phase 1"""

    def __init__(self, mbist_lib_path, faulty_device_id):
        super().__init__()

        # MBIST Engine 인터페이스
        self.mbist = MBISTInterface(mbist_lib_path)
        self.device_id = faulty_device_id

        # Action space
        self.action_space = gym.spaces.MultiDiscrete([
            12,   # primitives
            4,    # ranks
            8,    # bank groups
            4,    # banks
            256,  # row_start (grouped)
            256,  # row_end (grouped)
            8,    # patterns
            10    # repeat count
        ])

        # Observation space
        self.observation_space = gym.spaces.Dict({
            'memory_map': gym.spaces.Box(
                low=0, high=1,
                shape=(4, 8, 4, 256),
                dtype=np.float32
            ),
            'fault_map': gym.spaces.Box(
                low=0, high=1,
                shape=(4, 8, 4, 256),
                dtype=np.float32
            ),
            'coverage': gym.spaces.Box(
                low=0, high=1,
                shape=(4, 8, 4, 256),
                dtype=np.float32
            ),
            'recent_commands': gym.spaces.Box(
                low=0, high=1,
                shape=(10, 8),  # Last 10 commands
                dtype=np.float32
            )
        })

        self.reward_calculator = RewardCalculator()

    def step(self, action):
        """Execute DRAM command primitive"""

        # Decode action
        primitive, rank, bg, ba, row_start, row_end, pattern, repeat = action

        # Execute via MBIST Engine
        result = self._execute_primitive(
            primitive=DRAMPrimitive(primitive),
            rank=rank,
            bank_group=bg,
            bank=ba,
            row_start=row_start * 1024,  # Ungroup
            row_end=row_end * 1024,
            pattern=pattern,
            repeat=repeat + 1
        )

        # Calculate reward
        reward = self.reward_calculator.calculate(action, result)

        # Update state
        self._update_state(action, result)

        # Termination
        done = (
            self.total_tests >= self.max_tests or
            self.faults_found >= 1  # Phase 1: 하나만 찾으면 성공
        )

        info = {
            'test_result': result.test_passed,
            'error_addresses': result.error_addresses,
            'command_sequence': self.command_history
        }

        return self._get_observation(), reward, done, info

    def _execute_primitive(self, primitive, rank, bank_group, bank,
                          row_start, row_end, pattern, repeat):
        """Execute primitive via MBIST C library"""

        if primitive == DRAMPrimitive.WRITE_READ_ROW:
            # 1. ACTIVATE
            self.mbist.send_command('ACT', rank, bank_group, bank, row_start)

            # 2. WRITE with pattern
            for col in range(0, 2048, 64):  # 64 columns at a time
                self.mbist.write_data(
                    rank, bank_group, bank, row_start, col,
                    pattern=self._get_pattern(pattern)
                )

            # 3. PRECHARGE
            self.mbist.send_command('PRE', rank, bank_group, bank)

            # 4. ACTIVATE again
            self.mbist.send_command('ACT', rank, bank_group, bank, row_start)

            # 5. READ and compare
            errors = []
            for col in range(0, 2048, 64):
                data_read = self.mbist.read_data(rank, bank_group, bank, row_start, col)
                if not self.mbist.compare_data(data_read, self._get_pattern(pattern)):
                    errors.append((rank, bank_group, bank, row_start, col))

            # 6. PRECHARGE
            self.mbist.send_command('PRE', rank, bank_group, bank)

            # 7. Check test result
            test_result = self.mbist.get_test_result()

            return TestResult(
                test_passed=(test_result == 0),
                error_addresses=errors if errors else None
            )

        elif primitive == DRAMPrimitive.ROW_HAMMER:
            # Rowhammer attack
            target_row = row_start

            for _ in range(repeat * 10000):  # Hammer many times
                self.mbist.send_command('ACT', rank, bank_group, bank, target_row)
                self.mbist.send_command('PRE', rank, bank_group, bank)

            # Check victim rows (target ± 1)
            errors = []
            for victim_row in [target_row - 1, target_row + 1]:
                if victim_row >= 0 and victim_row < 262144:
                    # Read victim row
                    self.mbist.send_command('ACT', rank, bank_group, bank, victim_row)
                    for col in range(0, 2048, 64):
                        data_read = self.mbist.read_data(rank, bank_group, bank, victim_row, col)
                        # Check if data corrupted
                        if self.mbist.check_corruption(data_read):
                            errors.append((rank, bank_group, bank, victim_row, col))
                    self.mbist.send_command('PRE', rank, bank_group, bank)

            test_result = self.mbist.get_test_result()
            return TestResult(
                test_passed=(test_result == 0 and len(errors) == 0),
                error_addresses=errors if errors else None
            )

        # ... 다른 primitives 구현
```

### 훈련 전략

```python
# Phase 1: Fault Detection Training

# 1. 준비
faulty_devices = [
    {'id': 'CXL-001', 'known_fault': 'MRAT_FAIL', 'location': 'unknown'},
    {'id': 'CXL-002', 'known_fault': 'stressapptest_FAIL', 'location': 'unknown'},
    # ... more
]

# 2. 각 불량 device마다 훈련
for device in faulty_devices:
    env = DRAMCommandRLEnvironment(
        mbist_lib_path='/home/dhkang/data3/mbist_sample_code-gen2_es',
        faulty_device_id=device['id']
    )

    agent = DQNAgent(
        state_dim=env.observation_space,
        action_dim=env.action_space
    )

    # 목표: 불량을 검출하는 command 시퀀스 찾기
    for episode in range(max_episodes):
        state = env.reset()
        done = False

        while not done:
            action = agent.select_action(state)
            next_state, reward, done, info = env.step(action)

            agent.store_experience(state, action, reward, next_state, done)
            agent.train()

            if info['test_result'] == False:  # FAIL detected!
                print(f"✓ Device {device['id']} fault detected!")
                print(f"  Command sequence: {info['command_sequence']}")

                # Save successful sequence
                save_fault_detection_pattern(device['id'], info['command_sequence'])
                break

            state = next_state
```

## 다음 단계

1. **MBIST Python 인터페이스 구현** (우선순위: 높음)
   - C library를 Python에서 호출
   - ctypes 또는 pybind11 사용

2. **Primitive 함수 구현** (우선순위: 높음)
   - WRITE_READ_ROW
   - ROW_HAMMER
   - 기타 12개 primitives

3. **Phase 1 Environment 구현** (우선순위: 높음)
   - DRAMCommandRLEnvironment
   - Reward calculator
   - State management

4. **불량 device 확보** (우선순위: 최고!)
   - MRAT FAIL device
   - 실제 테스트 가능한 환경

5. **Baseline 측정**
   - MRAT이 찾는 시간
   - stressapptest 결과

## 기대 효과

### Phase 1 성공 시
```
RL Agent가 불량 device에서 불량을 검출하는 command 시퀀스 발견
→ "왜 MRAT이 효과적인지" 이해
→ 다른 불량에도 적용 가능한 원리 발견
```

### Phase 2 성공 시
```
MRAT보다 더 효과적인 새로운 패턴 발견
→ 더 많은 불량 검출
→ 불량품 출하 방지
→ 양산 수율 향상
```

### 최종 목표
```
하나의 RL policy로 다양한 불량 타입 검출
→ stressapptest 완전 대체
→ 10시간 → ?시간 (시간은 부차적, 검출률이 중요)
→ 불량 검출률 100% 달성
```

## 리스크 및 대응

### 리스크 1: Action space 너무 큼
**대응**:
- Hierarchical RL 사용
- Curriculum learning (쉬운 것부터)
- Transfer learning (한 device에서 학습 → 다른 device 적용)

### 리스크 2: 하드웨어 손상
**대응**:
- 시뮬레이터 먼저 구현
- 불량 device에서만 테스트 (이미 불량이므로 손상 무관)
- Safety limit 설정

### 리스크 3: 학습 시간 오래 걸림
**대응**:
- GPU 가속
- Batch training
- Experience replay 효율화

## 결론

**핵심 통찰:**
- MRAT, March C+ 등은 모두 특정 DRAM command 시퀀스
- RL이 command 레벨에서 제어하면 새로운 패턴 발명 가능
- Phase 1만 성공해도 큰 가치 (불량 검출 원리 이해)

**현실적 목표:**
- Phase 1: MRAT 수준 도달 (6개월)
- Phase 2: MRAT 초과 (1년)
- Phase 3: 범용 패턴 (1.5년)

**프로젝트 타당성: ✅ 매우 높음**
