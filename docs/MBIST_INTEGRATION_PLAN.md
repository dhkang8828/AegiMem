# MBIST Engine Integration Plan

## 목표
RL Agent가 DRAM의 특정 rank/bank/row/column에 대해 ACT, WR, RD, PRE 등을 세세하게 제어

## 설계: 2-Tier Primitives

### Tier 1: Low-level Atomic Commands

```c
// include/mt_rl_primitives.h

// Sequence mode control
void mt_rl_begin_sequence(uint8_t channel);
void mt_rl_end_sequence(uint8_t channel);

// Atomic DRAM commands
int mt_rl_send_activate(uint8_t rank, uint8_t bg, uint8_t ba, uint32_t row);
int mt_rl_send_write(uint8_t rank, uint8_t bg, uint8_t ba, uint32_t row,
                     uint16_t col, uint8_t pattern_type);
int mt_rl_send_read(uint8_t rank, uint8_t bg, uint8_t ba, uint32_t row,
                    uint16_t col);
int mt_rl_send_precharge(uint8_t rank, uint8_t bg, uint8_t ba);
int mt_rl_send_precharge_all(uint8_t rank);
int mt_rl_send_refresh(uint8_t rank, uint8_t bg, uint8_t ba);
int mt_rl_send_refresh_all(uint8_t rank);

// Result checking
int mt_rl_get_test_result(uint8_t channel);
int mt_rl_get_error_count(uint8_t channel);
```

### Tier 2: Mid-level Composed Operations

```c
// include/mt_rl_composed.h

// Common patterns
int mt_rl_write_read_cell(uint8_t rank, uint8_t bg, uint8_t ba,
                          uint32_t row, uint16_t col, uint8_t pattern);
int mt_rl_write_read_row(uint8_t rank, uint8_t bg, uint8_t ba,
                         uint32_t row, uint8_t pattern);
int mt_rl_write_read_bank(uint8_t rank, uint8_t bg, uint8_t ba,
                          uint32_t row_start, uint32_t row_end, uint8_t pattern);

// Stress operations
int mt_rl_row_hammer(uint8_t rank, uint8_t bg, uint8_t ba,
                     uint32_t target_row, uint32_t count);
int mt_rl_bank_thrash(uint8_t rank, uint8_t bg, uint32_t count);

// Retention test
int mt_rl_retention_test(uint8_t rank, uint8_t bg, uint8_t ba,
                         uint32_t row, uint32_t delay_us, uint8_t pattern);
```

## 구현 세부사항

### 1. Command Pattern Encoding

기존 알고리즘 코드 분석:
```c
// src/mt_algo_pattern.c에서 참고
// March C+, MATS+ 등이 어떻게 command를 encoding하는지 확인
```

필요한 helper 함수:
```c
// src/mt_rl_command_builder.c

static uint64_t build_activate_pattern(uint8_t rank, uint8_t bg,
                                       uint8_t ba, uint32_t row) {
    // CMD_ACT structure 채우기
    CMD_ACT act_cmd = {0};
    act_cmd.cap0_2_5R0_3 = row & 0xF;
    act_cmd.cap0_6_7BA0_1 = ba;
    act_cmd.cap0_8_10BG0_2 = bg;
    // ... rank, row 상위 비트 등

    // TYPE0_CA_PATTERN으로 변환
    // ...

    return pattern;
}

static uint64_t build_write_pattern(uint8_t rank, uint8_t bg, uint8_t ba,
                                    uint32_t row, uint16_t col,
                                    uint8_t pattern_type) {
    // CMD_WR structure 채우기
    // ...
}

static uint64_t build_read_pattern(...) {
    // CMD_RD structure 채우기
    // ...
}

static uint64_t build_precharge_pattern(...) {
    // CMD_PREsb or CMD_PREab
    // ...
}
```

### 2. Sequence Mode 구현

```c
// src/mt_rl_primitives.c

static int g_sequence_mode = 0;
static uint8_t g_current_channel = 0;

void mt_rl_begin_sequence(uint8_t channel) {
    g_sequence_mode = 1;
    g_current_channel = channel;
    mt_clear_sram();
}

void mt_rl_end_sequence(uint8_t channel) {
    g_sequence_mode = 0;

    // Execute queued commands
    mt_load_one_test();

    // Wait for completion
    while (!mt_get_test_end_status(channel)) {
        usleep(100);
    }
}

int mt_rl_send_activate(uint8_t rank, uint8_t bg, uint8_t ba, uint32_t row) {
    uint64_t cmd = build_activate_pattern(rank, bg, ba, row);

    if (g_sequence_mode) {
        // Queue mode: just add to SRAM
        mt_add_sram(cmd);
    } else {
        // Immediate mode: clear, add, execute
        mt_clear_sram();
        mt_add_sram(cmd);
        mt_load_one_test();
        while (!mt_get_test_end_status(g_current_channel)) {
            usleep(100);
        }
    }

    return 0;
}
```

### 3. Data Pattern 처리

```c
// Pattern types
typedef enum {
    RL_PATTERN_FIXED_00,    // All 0x00
    RL_PATTERN_FIXED_FF,    // All 0xFF
    RL_PATTERN_FIXED_55,    // 0x55555555...
    RL_PATTERN_FIXED_AA,    // 0xAAAAAAAA...
    RL_PATTERN_CHECKERBOARD,
    RL_PATTERN_PRBS,
    RL_PATTERN_WALKING_1S,
    RL_PATTERN_WALKING_0S
} RL_PATTERN_TYPE;

int mt_rl_set_data_pattern(RL_PATTERN_TYPE pattern_type) {
    switch (pattern_type) {
        case RL_PATTERN_FIXED_55:
            mt_fix_pat_wr(...);  // 기존 함수 활용
            break;
        case RL_PATTERN_PRBS:
            mt_prbs_pat(0x12345678);  // seed
            break;
        case RL_PATTERN_CHECKERBOARD:
            mt_checkerboard_pat(0);
            break;
        // ...
    }
    return 0;
}
```

### 4. Error Detection

```c
// src/mt_rl_error.c

typedef struct {
    uint32_t test_result;      // 0: pass, 1: fail
    uint32_t error_count;      // Total errors
    MBIST_ERR_ADDRS_T err_addrs;  // Error addresses
    MBIST_DQ_ERR_CNT_T dq_err;    // Per-DQ error count
} RL_TEST_RESULT;

int mt_rl_get_detailed_result(uint8_t channel, RL_TEST_RESULT *result) {
    result->test_result = mt_get_test_result(channel);

    if (result->test_result != 0) {  // FAIL
        mt_get_err_addrs(&result->err_addrs, channel);

        MBIST_DQ_T overflow;
        mt_get_dq_error_cnt(&result->dq_err, &overflow, channel);

        result->error_count = 0;
        for (int i = 0; i < 40; i++) {
            result->error_count += result->dq_err.dq[i];
        }
    } else {
        result->error_count = 0;
    }

    return 0;
}
```

## Python Wrapper

### ctypes Binding

```python
# src/mbist_interface.py

import ctypes
import time
from enum import IntEnum

class PatternType(IntEnum):
    FIXED_00 = 0
    FIXED_FF = 1
    FIXED_55 = 2
    FIXED_AA = 3
    CHECKERBOARD = 4
    PRBS = 5
    WALKING_1S = 6
    WALKING_0S = 7

class TestResult(ctypes.Structure):
    _fields_ = [
        ("test_result", ctypes.c_uint32),
        ("error_count", ctypes.c_uint32),
        # ... more fields
    ]

class MBISTInterface:
    """Python interface to MBIST C library"""

    def __init__(self, lib_path="./bin/mbist_smbus"):
        # Load library
        self.lib = ctypes.CDLL(lib_path)

        # Setup function signatures
        self._setup_functions()

        # Initialize MBIST
        self._initialize()

    def _setup_functions(self):
        """Setup ctypes function signatures"""

        # Sequence control
        self.lib.mt_rl_begin_sequence.argtypes = [ctypes.c_uint8]
        self.lib.mt_rl_begin_sequence.restype = None

        self.lib.mt_rl_end_sequence.argtypes = [ctypes.c_uint8]
        self.lib.mt_rl_end_sequence.restype = None

        # Atomic commands
        self.lib.mt_rl_send_activate.argtypes = [
            ctypes.c_uint8,  # rank
            ctypes.c_uint8,  # bg
            ctypes.c_uint8,  # ba
            ctypes.c_uint32  # row
        ]
        self.lib.mt_rl_send_activate.restype = ctypes.c_int

        self.lib.mt_rl_send_write.argtypes = [
            ctypes.c_uint8,  # rank
            ctypes.c_uint8,  # bg
            ctypes.c_uint8,  # ba
            ctypes.c_uint32, # row
            ctypes.c_uint16, # col
            ctypes.c_uint8   # pattern
        ]
        self.lib.mt_rl_send_write.restype = ctypes.c_int

        # ... more functions

    def _initialize(self):
        """Initialize MBIST hardware"""
        # Call initialization functions
        # umxc_lib_init(0)
        # mt_mbist_init_reg()
        # mt_init_sram()
        pass

    # High-level API
    def begin_sequence(self, channel=0):
        """Start command sequence mode"""
        self.lib.mt_rl_begin_sequence(channel)

    def end_sequence(self, channel=0):
        """End sequence and execute"""
        self.lib.mt_rl_end_sequence(channel)

    def send_activate(self, rank, bg, ba, row):
        """Send ACTIVATE command"""
        return self.lib.mt_rl_send_activate(rank, bg, ba, row)

    def send_write(self, rank, bg, ba, row, col, pattern=PatternType.FIXED_AA):
        """Send WRITE command"""
        return self.lib.mt_rl_send_write(rank, bg, ba, row, col, pattern)

    def send_read(self, rank, bg, ba, row, col):
        """Send READ command"""
        return self.lib.mt_rl_send_read(rank, bg, ba, row, col)

    def send_precharge(self, rank, bg, ba):
        """Send PRECHARGE command"""
        return self.lib.mt_rl_send_precharge(rank, bg, ba)

    def get_test_result(self, channel=0):
        """Get test result (0: pass, 1: fail)"""
        return self.lib.mt_rl_get_test_result(channel)

    def get_detailed_result(self, channel=0):
        """Get detailed error information"""
        result = TestResult()
        self.lib.mt_rl_get_detailed_result(channel, ctypes.byref(result))
        return result
```

### High-level Primitives

```python
# src/dram_primitives.py

from mbist_interface import MBISTInterface, PatternType

class DRAMPrimitives:
    """High-level DRAM operations for RL"""

    def __init__(self, mbist: MBISTInterface):
        self.mbist = mbist

    # Tier 2: Composed operations
    def write_read_cell(self, rank, bg, ba, row, col, pattern=PatternType.FIXED_AA):
        """Write and read a single cell"""
        self.mbist.begin_sequence()
        self.mbist.send_activate(rank, bg, ba, row)
        self.mbist.send_write(rank, bg, ba, row, col, pattern)
        self.mbist.send_precharge(rank, bg, ba)
        self.mbist.send_activate(rank, bg, ba, row)
        self.mbist.send_read(rank, bg, ba, row, col)
        self.mbist.send_precharge(rank, bg, ba)
        self.mbist.end_sequence()

        return self.mbist.get_test_result()

    def write_read_row(self, rank, bg, ba, row, pattern=PatternType.FIXED_AA):
        """Write and read entire row"""
        self.mbist.begin_sequence()
        self.mbist.send_activate(rank, bg, ba, row)

        # Write all columns (burst length 16, so 2048/16 = 128 bursts)
        for col in range(0, 2048, 16):
            self.mbist.send_write(rank, bg, ba, row, col, pattern)

        self.mbist.send_precharge(rank, bg, ba)
        self.mbist.send_activate(rank, bg, ba, row)

        # Read and verify
        for col in range(0, 2048, 16):
            self.mbist.send_read(rank, bg, ba, row, col)

        self.mbist.send_precharge(rank, bg, ba)
        self.mbist.end_sequence()

        return self.mbist.get_test_result()

    def row_hammer(self, rank, bg, ba, target_row, count=10000):
        """Row hammer attack"""
        # Hammer는 너무 많은 command → 배치 처리
        batch_size = 500  # SRAM limit 512
        num_batches = count // batch_size

        for _ in range(num_batches):
            self.mbist.begin_sequence()
            for _ in range(batch_size):
                self.mbist.send_activate(rank, bg, ba, target_row)
                self.mbist.send_precharge(rank, bg, ba)
            self.mbist.end_sequence()

        # Check victim rows
        return self._check_victim_rows(rank, bg, ba, target_row)

    def _check_victim_rows(self, rank, bg, ba, target_row):
        """Check rows adjacent to hammered row"""
        results = []

        for victim_row in [target_row - 1, target_row + 1]:
            if 0 <= victim_row < 262144:
                result = self.write_read_row(rank, bg, ba, victim_row)
                results.append({
                    'victim_row': victim_row,
                    'test_result': result
                })

        return results

    def custom_sequence(self, commands):
        """Execute custom command sequence"""
        self.mbist.begin_sequence()

        for cmd in commands:
            if cmd['type'] == 'ACT':
                self.mbist.send_activate(cmd['rank'], cmd['bg'], cmd['ba'], cmd['row'])
            elif cmd['type'] == 'WR':
                self.mbist.send_write(cmd['rank'], cmd['bg'], cmd['ba'],
                                     cmd['row'], cmd['col'], cmd['pattern'])
            elif cmd['type'] == 'RD':
                self.mbist.send_read(cmd['rank'], cmd['bg'], cmd['ba'],
                                    cmd['row'], cmd['col'])
            elif cmd['type'] == 'PRE':
                self.mbist.send_precharge(cmd['rank'], cmd['bg'], cmd['ba'])
            elif cmd['type'] == 'DELAY':
                self.mbist.end_sequence()  # Execute accumulated
                time.sleep(cmd['delay_sec'])
                self.mbist.begin_sequence()  # Start new batch

        self.mbist.end_sequence()
        return self.mbist.get_test_result()
```

## 구현 단계

### Phase 1: Command Builder (Week 1-2)
- [ ] 기존 알고리즘 코드 분석 (March C+, MATS+ 등)
- [ ] build_activate_pattern() 구현
- [ ] build_write_pattern() 구현
- [ ] build_read_pattern() 구현
- [ ] build_precharge_pattern() 구현
- [ ] 단위 테스트

### Phase 2: Low-level Primitives (Week 2-3)
- [ ] mt_rl_primitives.c 구현
- [ ] Sequence mode 구현
- [ ] 즉시 실행 모드 구현
- [ ] Error handling

### Phase 3: Python Wrapper (Week 3-4)
- [ ] ctypes binding 구현
- [ ] MBISTInterface 클래스
- [ ] 단위 테스트 (mock device)

### Phase 4: High-level Primitives (Week 4-5)
- [ ] DRAMPrimitives 클래스
- [ ] write_read_row, row_hammer 등 구현
- [ ] 실제 하드웨어 테스트

### Phase 5: RL Integration (Week 5-6)
- [ ] RL Environment에 통합
- [ ] CE 감지 기능 통합 (Mailbox get-health-info)
- [ ] End-to-end 테스트

## 테스트 계획

### Unit Tests
```python
# tests/test_mbist_primitives.py

def test_activate():
    mbist = MBISTInterface(mock=True)
    result = mbist.send_activate(rank=0, bg=0, ba=0, row=100)
    assert result == 0

def test_write_read():
    mbist = MBISTInterface(mock=True)
    mbist.begin_sequence()
    mbist.send_activate(0, 0, 0, 100)
    mbist.send_write(0, 0, 0, 100, 0, PatternType.FIXED_AA)
    mbist.send_read(0, 0, 0, 100, 0)
    mbist.send_precharge(0, 0, 0)
    mbist.end_sequence()

    assert mbist.get_test_result() == 0  # PASS
```

### Integration Tests
```python
# tests/test_dram_primitives.py

def test_row_hammer_real_device():
    """Test on actual CXL device"""
    mbist = MBISTInterface()
    prims = DRAMPrimitives(mbist)

    # Hammer a known good row
    result = prims.row_hammer(rank=0, bg=0, ba=0, target_row=1000, count=100000)

    # Should not cause errors on good device
    assert all(r['test_result'] == 0 for r in result)
```

## 다음 액션

1. **기존 알고리즘 코드 분석**
   ```bash
   cd /home/dhkang/data3/mbist_sample_code-gen2_es
   # March C+가 어떻게 command를 생성하는지 확인
   grep -r "March" src/
   ```

2. **Command Builder 프로토타입**
   ```c
   // 간단한 프로토타입으로 시작
   uint64_t build_activate_pattern(uint8_t rank, uint8_t bg, uint8_t ba, uint32_t row) {
       // TODO: 구현
   }
   ```

3. **Python Mock 테스트**
   ```python
   # 실제 하드웨어 없이 로직 테스트
   class MockMBIST:
       def send_activate(self, ...):
           print(f"ACT: rank={rank}, row={row}")
           return 0
   ```
