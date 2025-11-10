# 프로젝트 아키텍처 수정: MBIST → devdax

**작성일**: 2025-11-01
**사유**: MBIST의 ECC OFF 요구사항으로 인한 실용성 문제

---

## 🔥 변경 이유

### MBIST 방식의 문제점

```
❌ 치명적 제약사항: ECC OFF 필수
   - Montage MBIST는 ECC를 꺼야만 정상 동작
   - ECC OFF = 실제 운영 환경과 완전히 다름
   - CE(Correctable Error) 감지 불가능
   - 양산 라인 적용 불가 (ECC는 항상 켜져 있음)
   - MRAT도 동일한 문제

❌ 복잡성
   - Low-level command 이해 필요 (ACT, WR, RD, PRE)
   - Vendor-specific API
   - 컨트롤러 의존적

❌ 범용성 부족
   - Montage 컨트롤러에만 적용
   - 다른 CXL 디바이스에 사용 불가
```

### devdax 방식의 장점

```
✅ 실제 운영 환경과 동일
   - ECC ON 상태 유지
   - CE 발생 → ECC가 정정 → CE count 증가
   - 실제 불량을 제대로 감지 가능
   - 양산 라인에 바로 적용 가능

✅ 표준 Linux 인터페이스
   - /dev/dax* 디바이스 사용
   - mmap(), read(), write() 표준 API
   - 어떤 CXL 디바이스에도 적용 가능

✅ 단순성
   - Application level에서 접근
   - No vendor-specific API
   - CE count는 CXL Mailbox 명령어로 확인
```

---

## 🏗️ 새로운 아키텍처

### 전체 구조

```
┌──────────────────────────────────────────────────────┐
│                    RL Agent                          │
│  ┌──────────────────────────────────────────────┐   │
│  │ Policy Network                               │   │
│  │ - 입력: CE count history, test pattern       │   │
│  │ - 출력: 다음 테스트 패턴 선택                  │   │
│  └──────────────────────────────────────────────┘   │
└─────────────────────┬────────────────────────────────┘
                      │
                      ▼
┌──────────────────────────────────────────────────────┐
│              Pattern Executor                        │
│  ┌──────────────────────────────────────────────┐   │
│  │ def execute_pattern():                       │   │
│  │   1. Write pattern to devdax                 │   │
│  │   2. Read back (optional stress)             │   │
│  │   3. Check CE count delta                    │   │
│  │   4. Return CE increase as reward signal     │   │
│  └──────────────────────────────────────────────┘   │
└────────────┬──────────────────┬──────────────────────┘
             │                  │
             ▼                  ▼
┌─────────────────────┐  ┌─────────────────────────────┐
│  devdax Interface   │  │  CXL Mailbox Interface      │
│  /dev/dax0.0        │  │  (cxl command)              │
│                     │  │                             │
│  - mmap()           │  │  - cxl list                 │
│  - memset()         │  │  - cxl get-health-info      │
│  - memcpy()         │  │  - Parse CE count           │
└──────────┬──────────┘  └─────────────┬───────────────┘
           │                           │
           ▼                           │
┌──────────────────────────────────────┼───────────────┐
│       CXL Type3 Memory (CMM-D)       │               │
│       ┌──────────────────────────────┘               │
│       │ ECC ON                                       │
│       │ - Weak cell → CE 발생                        │
│       │ - ECC 정정 → Data 올바름                     │
│       │ - CE count 누적                              │
└──────────────────────────────────────────────────────┘
```

### devdax 인터페이스

```c
// /dev/dax0.0 사용 예시
#include <sys/mman.h>
#include <fcntl.h>

int fd = open("/dev/dax0.0", O_RDWR);
size_t size = get_device_size(fd);  // e.g., 128GB

// Memory map
void *addr = mmap(NULL, size, PROT_READ | PROT_WRITE,
                  MAP_SHARED, fd, 0);

// Write pattern
uint8_t pattern = 0xAA;
memset(addr, pattern, size);

// Read back (stress)
uint8_t *data = malloc(size);
memcpy(data, addr, size);

// Verify (optional, ECC will correct errors)
for (size_t i = 0; i < size; i++) {
    if (data[i] != pattern) {
        // Data는 정상이지만 CE가 발생했을 수 있음
    }
}

munmap(addr, size);
close(fd);
```

### CE Count 확인

```bash
# CXL command line tool 사용
cxl list
# {
#   "memdev":"mem0",
#   "pmem_size":"128.00 GiB (137.44 GB)",
#   ...
# }

# Health info 조회
cxl get-health-info mem0
# {
#   "health_status":"ok",
#   "media_status":"ok",
#   "life_used_percentage":0,
#   "correctable_errors":1234,    # ← CE count!
#   "uncorrectable_errors":0,
#   ...
# }
```

또는 Python으로:

```python
import subprocess
import json

def get_ce_count(memdev='mem0'):
    """Get CE count from CXL device"""
    result = subprocess.run(
        ['cxl', 'get-health-info', memdev, '--json'],
        capture_output=True,
        text=True
    )

    health_info = json.loads(result.stdout)
    ce_count = health_info.get('correctable_errors', 0)

    return ce_count
```

---

## 🎯 불량 검출 메커니즘

### 핵심 아이디어

```
불량 cell은 ECC로 정정되지만 CE를 발생시킨다!

정상 cell: Data 안정 → CE 없음
불량 cell: Data 불안정 → CE 발생 → ECC 정정 → CE count ↑

→ CE count 증가 = 불량 신호!
```

### 테스트 프로세스

```python
def test_pattern(device, pattern, operation):
    """
    Test a specific pattern and return CE delta

    Args:
        device: /dev/dax0.0
        pattern: 0x00 ~ 0xFF
        operation: WRITE_READ_ASC, etc.

    Returns:
        ce_delta: CE count increase
    """

    # 1. Get baseline CE count
    ce_before = get_ce_count('mem0')

    # 2. Execute test operation
    if operation == 'WRITE_READ_ASC':
        write_ascending(device, pattern)
        read_ascending(device)
    elif operation == 'WRITE_READ_DESC':
        write_descending(device, pattern)
        read_descending(device)
    elif operation == 'WRITE_PAUSE_READ':
        write_ascending(device, pattern)
        time.sleep(1.0)  # Retention test
        read_ascending(device)
    # ... more operations

    # 3. Get CE count after test
    ce_after = get_ce_count('mem0')

    # 4. Calculate delta
    ce_delta = ce_after - ce_before

    return ce_delta
```

### Reward 함수

```python
def calculate_reward(ce_delta):
    """
    CE 증가량에 비례한 reward

    불량을 더 많이 자극하는 패턴 = 더 높은 reward
    """
    if ce_delta > 0:
        # CE 발생! 불량을 찾았다
        return 100.0 * ce_delta
    else:
        # CE 없음
        return -0.1  # 작은 페널티
```

---

## 📊 새로운 Action Space

### 정의

```python
# Total: 1,536 actions (기존과 동일)
action_index = operation_type * 256 + pattern_byte

operation_type: 0-5 (6가지)
  0: WRITE_READ_ASC       # [^(W pat), ^(R pat)]
  1: WRITE_READ_DESC      # [v(W pat), v(R pat)]
  2: WRITE_PAUSE_READ     # [W pat → delay → R pat]
  3: REPEATED_READ        # [R pat] × N
  4: WRITE_READ_WRITE     # [W pat → R pat → W pat']
  5: ALTERNATING          # [W/R alternating]

pattern_byte: 0x00 ~ 0xFF (256가지)
```

### Operations 상세

#### 1. WRITE_READ_ASC (March-like)
```python
# Ascending order write then read
for addr in range(0, device_size, block_size):
    write(addr, pattern)

for addr in range(0, device_size, block_size):
    read(addr)
    # CE 발생 가능
```

#### 2. WRITE_PAUSE_READ (Retention test)
```python
# Write, wait, then read (retention test)
for addr in range(0, device_size, block_size):
    write(addr, pattern)

time.sleep(pause_duration)  # 1초 ~ 10초

for addr in range(0, device_size, block_size):
    read(addr)
    # Weak cell → retention failure → CE
```

#### 3. REPEATED_READ (Read stress)
```python
# Repeated read to stress cells
for iteration in range(repeat_count):
    for addr in range(0, device_size, block_size):
        read(addr)
        # Read disturb → CE
```

#### 4. ALTERNATING (Write/Read interleaved)
```python
# Alternating write/read
for addr in range(0, device_size, block_size):
    write(addr, pattern)
    read(addr)
    write(addr, ~pattern)  # Inverse
    read(addr)
    # Pattern switching stress → CE
```

---

## 🔧 구현 계획

### Phase 1: devdax Interface 구현

**Week 1-2**:

```python
# src/devdax_interface.py

class DevDaxInterface:
    """devdax를 통한 CXL 메모리 접근"""

    def __init__(self, device_path='/dev/dax0.0'):
        self.device = device_path
        self.fd = None
        self.mmap_addr = None
        self.size = self._get_device_size()

    def open(self):
        """Open device and mmap"""
        self.fd = os.open(self.device, os.O_RDWR)
        self.mmap_addr = mmap.mmap(
            self.fd,
            self.size,
            mmap.MAP_SHARED,
            mmap.PROT_READ | mmap.PROT_WRITE
        )

    def write_pattern(self, pattern, start=0, length=None):
        """Write pattern to memory"""
        if length is None:
            length = self.size

        # Use memoryview for efficiency
        view = memoryview(self.mmap_addr)[start:start+length]
        pattern_byte = pattern.to_bytes(1, 'little')
        view[:] = pattern_byte * length

    def read_memory(self, start=0, length=None):
        """Read memory region"""
        if length is None:
            length = self.size

        data = self.mmap_addr[start:start+length]
        return bytes(data)

    def close(self):
        """Close device"""
        if self.mmap_addr:
            self.mmap_addr.close()
        if self.fd:
            os.close(self.fd)
```

```python
# src/cxl_mailbox.py

class CXLMailbox:
    """CXL Mailbox 명령어 인터페이스"""

    def __init__(self, memdev='mem0'):
        self.memdev = memdev

    def get_ce_count(self):
        """Get correctable error count"""
        import subprocess
        import json

        result = subprocess.run(
            ['cxl', 'get-health-info', self.memdev, '--json'],
            capture_output=True,
            text=True,
            check=True
        )

        health = json.loads(result.stdout)
        ce_count = health.get('correctable_errors', 0)

        return ce_count

    def get_health_info(self):
        """Get full health information"""
        # Similar implementation
        pass
```

### Phase 2: Pattern Executor

**Week 2-3**:

```python
# src/pattern_executor.py

class PatternExecutor:
    """패턴 테스트 실행"""

    def __init__(self, dax_interface, mailbox):
        self.dax = dax_interface
        self.mailbox = mailbox

    def execute_pattern(self, operation_type, pattern_byte):
        """Execute test pattern and return CE delta"""

        # Get baseline CE
        ce_before = self.mailbox.get_ce_count()

        # Execute operation
        if operation_type == 0:  # WRITE_READ_ASC
            self._write_read_ascending(pattern_byte)
        elif operation_type == 1:  # WRITE_READ_DESC
            self._write_read_descending(pattern_byte)
        # ... other operations

        # Get CE after
        ce_after = self.mailbox.get_ce_count()

        # Return delta
        return ce_after - ce_before

    def _write_read_ascending(self, pattern):
        """Ascending write then read"""
        self.dax.write_pattern(pattern, start=0)
        data = self.dax.read_memory(start=0)
        # CE may occur during read

    def _write_read_descending(self, pattern):
        """Descending write then read"""
        # Implement descending order
        pass
```

### Phase 3: RL Environment 수정

**Week 3-4**:

```python
# src/phase1_environment_devdax.py

class Phase1EnvironmentDevDax(gym.Env):
    """devdax 기반 Phase 1 Environment"""

    def __init__(self, device_path='/dev/dax0.0', memdev='mem0'):
        self.dax = DevDaxInterface(device_path)
        self.mailbox = CXLMailbox(memdev)
        self.executor = PatternExecutor(self.dax, self.mailbox)

        # Action space: 1536 (6 ops × 256 patterns)
        self.action_space = gym.spaces.Discrete(1536)

    def step(self, action):
        """Execute action and return CE delta as reward"""

        # Decode action
        operation_type = action // 256
        pattern_byte = action % 256

        # Execute test
        ce_delta = self.executor.execute_pattern(
            operation_type,
            pattern_byte
        )

        # Reward = CE delta
        reward = 100.0 * ce_delta if ce_delta > 0 else -0.1

        # Done if significant CE found
        done = (ce_delta > 100)  # Threshold

        info = {
            'ce_delta': ce_delta,
            'operation': operation_type,
            'pattern': pattern_byte
        }

        return self._get_observation(), reward, done, False, info
```

---

## 🎯 성공 기준 (수정)

### Phase 1 목표

```
기존: MRAT이 찾은 불량을 RL이 재발견
새로운: RL이 CE를 최대한 많이 발생시키는 패턴 발견

측정:
- CE delta per pattern
- 최적 패턴 조합 발견
- Weak cell 위치 특정
```

### KPI

| 지표 | 목표 | 측정 |
|------|------|------|
| CE 발견 | >0 CE | get_ce_count() |
| 패턴 효율성 | CE/test 최대화 | Reward tracking |
| 학습 시간 | <1000 episodes | Training log |
| 재현성 | 5/5회 | Repeated tests |

---

## 📝 다음 액션

### 즉시 (Week 1)

1. **devdax 디바이스 확인**
   ```bash
   ls -l /dev/dax*
   # /dev/dax0.0, /dev/dax1.0, ...

   # Device size 확인
   cat /sys/devices/dax*/size
   ```

2. **CXL command 설치 및 테스트**
   ```bash
   # cxl-cli 설치
   sudo apt install cxl-cli  # or build from source

   # CXL 디바이스 확인
   cxl list

   # Health info 확인
   cxl get-health-info mem0
   ```

3. **간단한 테스트 프로그램**
   ```python
   # test_devdax_basic.py
   import mmap
   import os

   device = '/dev/dax0.0'
   size = 1024 * 1024  # 1MB test

   fd = os.open(device, os.O_RDWR)
   mm = mmap.mmap(fd, size, mmap.MAP_SHARED)

   # Write pattern
   mm[:] = b'\xAA' * size

   # Read back
   data = mm[:]
   assert data == b'\xAA' * size

   mm.close()
   os.close(fd)
   print("✓ devdax basic test passed")
   ```

---

## 🔄 마이그레이션 계획

### 기존 코드 재사용

```
✅ Action space 정의: 거의 동일 (1536 actions)
✅ RL Agent (DQN): 그대로 사용 가능
✅ Reward 함수: CE delta로 변경
✅ Episode 구조: 동일

❌ MBIST interface: 완전 교체 → devdax
❌ Pattern executor: 재구현
❌ 에러 감지: mt_get_test_result() → CE count
```

### 문서 업데이트

- [ ] ARCHITECTURE_REVISION_DEVDAX.md (이 문서)
- [ ] PHASE1_IMPLEMENTATION_SCHEDULE.md 수정
- [ ] PHASE1_PHASE2_STRATEGY.md 수정
- [ ] README.md 업데이트

---

## 🎉 기대 효과

### 기술적

✅ **실제 환경 테스트**: ECC ON 상태
✅ **범용성**: 모든 CXL Type3 디바이스 적용 가능
✅ **단순성**: 표준 Linux API
✅ **신뢰성**: CE count = 명확한 불량 신호

### 비즈니스

✅ **양산 적용 가능**: ECC를 끌 필요 없음
✅ **실용성**: 실제 불량을 제대로 감지
✅ **확장성**: 다른 제품에도 적용 가능

---

**작성자**: AI Assistant
**검토 필요**: 프로젝트 리드
**다음 단계**: devdax 환경 확인 및 기본 테스트
