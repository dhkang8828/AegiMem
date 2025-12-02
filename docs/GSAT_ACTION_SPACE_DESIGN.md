# GSAT-like Action Space Design for Reinforcement Learning

## StressAppTest 핵심 분석

### 참고 자료
- [StressAppTest GitHub](https://github.com/stressapptest/stressapptest)
- [Pattern Header](https://github.com/stressapptest/stressapptest/blob/master/src/pattern.h)
- [Pattern Implementation](https://github.com/stressapptest/stressapptest/blob/master/src/pattern.cc)

## 1. StressAppTest의 15가지 표준 패턴

### Walking Patterns (Bit Position Testing)
- **walkingOnes**: `0x00000001 → 0x00000002 → ... → 0x80000000`
- **walkingInvOnes**: Walking ones with complement
- **walkingZeros**: Walking zeros pattern

### Simple Patterns (Basic Stress)
- **JustZero**: `0x00000000` (All zeros)
- **JustOne**: `0xffffffff` (All ones)
- **JustFive**: `0x55555555` (01010101 pattern)
- **JustA**: `0xaaaaaaaa` (10101010 pattern)
- **OneZero**: Alternates `0x00000000` and `0xffffffff`

### Checkerboard Patterns (Adjacent Cell Stress)
- **FiveA**: Alternates `0x55555555` and `0xaaaaaaaa`
- **FiveA8**: `[0x5aa5a55a, 0xa55a5aa5, 0xa55a5aa5, 0x5aa5a55a]`

### 8b10b Encoding Patterns (Signal Integrity)
- **Long8b10b**: `0x16161616` (Low transition density)
- **Short8b10b**: `0xb5b5b5b5` (High transition density)
- **Checker8b10b**: Alternates `0xb5b5b5b5` and `0x4a4a4a4a`

### Edge Case Patterns (Anomaly Detection)
- **Five7**: `0x55555557`, `0x55575555` (Single-bit anomalies)
- **Zero2fd**: `0x00020002`, `0xfffdfffd` (Word boundary variations)

## 2. 동작 타입 (Operations)

현재 `gsat_like_memory_agent.c`에 구현된 4가지:

```c
OP_FILL   = 0,  // Thermal stress (Simple write)
OP_INVERT = 1,  // Switching noise (Write → Invert → Write) [RECOMMENDED]
OP_COPY   = 2,  // Bandwidth saturation (Memcpy)
OP_CHECK  = 3   // Read disturb (Read only)
```

## 3. 강화학습 Action Space 설계

### Option A: 간소화된 액션 스페이스 (권장)
**Total: 64 actions**

```python
# 4 Operations × 16 Key Patterns = 64 actions
OPERATIONS = ['FILL', 'INVERT', 'COPY', 'CHECK']

# 16 핵심 패턴 (StressAppTest 기반)
KEY_PATTERNS = [
    0x00,       # All zeros
    0xFF,       # All ones
    0x55,       # 01010101
    0xAA,       # 10101010
    0xF0,       # 11110000
    0x0F,       # 00001111
    0xCC,       # 11001100
    0x33,       # 00110011
    0x01,       # Walking ones start
    0x80,       # Walking ones end
    0x16,       # 8b10b low transition
    0xB5,       # 8b10b high transition
    0x4A,       # Checker pattern
    0x57,       # Edge case 1
    0x02,       # Edge case 2
    0xFD,       # Edge case 3
]

# Action encoding: action = operation_id * 16 + pattern_id
# 예시:
# - action=0: OP_FILL with 0x00
# - action=17: OP_INVERT with 0x55 (RECOMMENDED for CE detection)
# - action=63: OP_CHECK with 0xFD
```

### Option B: 중간 규모 액션 스페이스
**Total: 256 actions**

```python
# 4 Operations × 64 Patterns = 256 actions
OPERATIONS = ['FILL', 'INVERT', 'COPY', 'CHECK']

# 64 패턴: 16 핵심 + 48 랜덤 샘플링
# - 16 핵심 패턴 (고정)
# - 16 Walking patterns (0x01, 0x02, 0x04, ..., 0x80)
# - 32 추가 탐색 패턴 (에이전트가 학습)
```

### Option C: 기존과 유사한 규모
**Total: 1024 actions**

```python
# 4 Operations × 256 Patterns = 1024 actions
# 모든 가능한 8-bit 패턴 조합
```

## 4. 권장 구현 전략

### Phase 1: 간소화된 64 액션 (빠른 학습)
- **장점**: 빠른 수렴, 핵심 패턴에 집중
- **단점**: 표현력 제한
- **적합**: 초기 CE 발생 확인용

### Phase 2: 256 액션 (균형)
- **장점**: 충분한 탐색 공간, 학습 가능
- **단점**: 약간 느린 수렴
- **적합**: 본격 학습용

### Phase 3: 1024 액션 (완전 탐색)
- **장점**: 최대 표현력
- **단점**: 매우 느린 수렴
- **적합**: 최종 최적화용

## 5. 구현 우선순위

### 우선 테스트할 Action 조합 (CE 발생 가능성 높음)

```python
HIGH_PRIORITY_ACTIONS = [
    # OP_INVERT (1) with critical patterns
    (1, 0x55),  # INVERT + 01010101
    (1, 0xAA),  # INVERT + 10101010
    (1, 0xFF),  # INVERT + all ones
    (1, 0x00),  # INVERT + all zeros
    (1, 0xF0),  # INVERT + 11110000
    (1, 0x0F),  # INVERT + 00001111

    # OP_FILL (0) with thermal stress
    (0, 0xFF),  # FILL + all ones (max heat)
    (0, 0xAA),  # FILL + alternating

    # OP_COPY (2) with bandwidth stress
    (2, 0x55),  # COPY + pattern
    (2, 0xCC),  # COPY + different pattern
]
```

## 6. 다음 단계

1. **`gsat_like_memory_agent.c` 수정**
   - 16 핵심 패턴 상수 정의
   - Pattern lookup table 추가

2. **Python Environment 작성**
   - `phase1_gsat_like_environment_distributed.py` 완성
   - 64 action space로 시작
   - 기존 environment와 인터페이스 호환

3. **Training Script 수정**
   - `num_actions=64`로 변경
   - 학습 후 256/1024로 확장 가능

4. **우선순위 기반 초기화**
   - Epsilon-greedy 대신 HIGH_PRIORITY_ACTIONS로 warm-start
   - 빠른 CE 발견 가능성 증가

## 7. 기대 효과

- **지속적 부하 (5초)** + **Bit flipping (INVERT)** + **핵심 패턴**
- 기존 대비 **100배 이상의 스트레스**
- CE 발생 가능성 대폭 증가
- 강화학습 신호 강화 → 빠른 학습 수렴
