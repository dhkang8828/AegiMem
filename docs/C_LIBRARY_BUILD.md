# C Library Build and Integration Guide

## Overview

Memory Agent C library는 고성능 메모리 테스트를 위한 저수준 라이브러리입니다. devdax를 통한 직접 메모리 접근과 umxc를 통한 CE 감지를 제공합니다.

## Location

```
src/MemoryAgent/c_library/
├── memory_agent.c           # C 구현
├── libmemory_agent.so      # 컴파일된 공유 라이브러리
└── Makefile                # 빌드 스크립트
```

## Build Process

### Prerequisites

- GCC compiler
- Linux system (GNR-SP)
- Root 권한 (devdax 접근용)
- umxc tool 설치

### Build Commands

```bash
# 1. C library 디렉토리로 이동
cd src/MemoryAgent/c_library

# 2. Clean previous build
make clean

# 3. Build shared library
make

# 4. Verify compilation
ls -la libmemory_agent.so
```

Expected output:
```
-rwxrwxr-x 1 user user 21064 Nov 20 13:22 libmemory_agent.so
```

## Implemented Functions

### Initialization

```c
int ma_init(const char* devdax_path,
            size_t memory_size_mb,
            double sampling_rate);
```

**Parameters:**
- `devdax_path`: devdax 경로 (e.g., "/dev/dax0.0")
- `memory_size_mb`: 메모리 크기 (MB)
- `sampling_rate`: 샘플링 비율 (0.0-1.0)

**Returns:** 0 on success, -1 on failure

### Execute Action

```c
int ma_execute_action(int action, ActionResult* result);
```

**Parameters:**
- `action`: 0-1535 (operation_type * 256 + pattern)
- `result`: 결과를 저장할 구조체

**Action Encoding:**
```c
operation_type = action / 256;  // 0-5
pattern = action % 256;         // 0x00-0xFF
```

**Operation Types:**
- 0: WR_ASC_ASC - Write ascending, Read ascending
- 1: WR_DESC_DESC - Write descending, Read descending
- 2: WR_ASC_DESC - Write ascending, Read descending
- 3: WR_DESC_ASC - Write descending, Read ascending
- 4: WR_DESC_SINGLE - Write+Read descending (single pass)
- 5: WR_ASC_SINGLE - Write+Read ascending (single pass)

### CE Detection

```c
int ma_get_ce_info(CEInfo* ce_info);
```

**Returns:** Current CE count information from umxc

### Reset Baseline

```c
int ma_reset_baseline();
```

**Returns:** 0 on success

## Data Structures

### CEInfo

```c
typedef struct {
    int volatile_count;      // Corrected volatile errors
    int persistent_count;    // Corrected persistent errors
    int total_count;         // Total CE count
    int temperature;         // Device temperature (°C)
    int health_status;       // Health status code
} CEInfo;
```

### ActionResult

```c
typedef struct {
    CEInfo ce_info;          // CE information
    int success;             // Operation success flag
} ActionResult;
```

## Implementation Details

### devdax Access

```c
static int devdax_fd = -1;

// Open devdax device
int fd = open(devdax_path, O_RDWR | O_SYNC);

// Write to memory
lseek(fd, dpa, SEEK_SET);
write(fd, data, size);

// Read from memory
lseek(fd, dpa, SEEK_SET);
read(fd, buffer, size);
```

### umxc CE Detection

```c
static int execute_umxc(CEInfo* ce_info) {
    FILE* fp = popen("umxc mbox -H", "r");

    // Parse output:
    // [0Ah] Corrected Volatile Error Count: X
    // [0Eh] Corrected Persistent Error Count: X

    // Calculate delta from baseline
    ce_info->volatile_count = current - baseline_volatile;
    ce_info->persistent_count = current - baseline_persistent;
    ce_info->total_count = ce_info->volatile_count +
                           ce_info->persistent_count;

    pclose(fp);
    return 0;
}
```

### Memory Operations

#### Write Ascending
```c
static int write_ascending(unsigned char pattern) {
    for (uint64_t dpa = 0; dpa < total_size; dpa += BLOCK_SIZE) {
        lseek(devdax_fd, dpa, SEEK_SET);
        write(devdax_fd, data, BLOCK_SIZE);
    }
    return 0;
}
```

#### Read Ascending with Verification
```c
static int read_ascending(unsigned char expected_pattern) {
    for (uint64_t dpa = 0; dpa < total_size; dpa += BLOCK_SIZE) {
        lseek(devdax_fd, dpa, SEEK_SET);
        read(devdax_fd, buffer, BLOCK_SIZE);

        // Verification happens at hardware level
        // CE detection via umxc
    }
    return 0;
}
```

## Python Integration

### Using ctypes

```python
from ctypes import CDLL, Structure, c_int, c_uint64, c_char_p, c_double

class CEInfo(Structure):
    _fields_ = [
        ('volatile_count', c_int),
        ('persistent_count', c_int),
        ('total_count', c_int),
        ('temperature', c_int),
        ('health_status', c_int)
    ]

class MemoryAgentC:
    def __init__(self, library_path='c_library/libmemory_agent.so'):
        self.lib = CDLL(library_path)

        # Setup function signatures
        self.lib.ma_init.argtypes = [c_char_p, c_uint64, c_double]
        self.lib.ma_init.restype = c_int

        self.lib.ma_execute_action.argtypes = [c_int, POINTER(ActionResult)]
        self.lib.ma_execute_action.restype = c_int

    def init(self, devdax_path, memory_size_mb, sampling_rate):
        return self.lib.ma_init(
            devdax_path.encode('utf-8'),
            memory_size_mb,
            sampling_rate
        )

    def execute_action(self, action):
        result = ActionResult()
        ret = self.lib.ma_execute_action(action, byref(result))
        return result.ce_info, (ret == 0)
```

### Usage in Memory Agent Server

```python
# src/MemoryAgent/memory_agent_server.py
from memory_agent_c_wrapper import MemoryAgentC

# Initialize
memory_agent = MemoryAgentC()
memory_agent.init("/dev/dax0.0", 128000, 0.01)

# Execute action
@app.route('/execute_action', methods=['POST'])
def execute_action():
    action = request.json['action']
    ce_info, success = memory_agent.execute_action(action)

    return jsonify({
        'success': success,
        'ce_detected': ce_info.has_errors(),
        'ce_total': ce_info.total_count
    })
```

## Testing

### Compilation Test

```bash
cd src/MemoryAgent/c_library
make clean && make

# Check for errors
echo $?  # Should be 0
```

### Standalone Test (requires root)

```bash
# Create simple test program
cat > test_simple.c << 'EOF'
#include <stdio.h>
#include "memory_agent.h"

int main() {
    printf("Initializing...\n");
    int ret = ma_init("/dev/dax0.0", 128000, 0.01);
    printf("Init result: %d\n", ret);
    return 0;
}
EOF

gcc -o test_simple test_simple.c -L. -lmemory_agent
sudo ./test_simple
```

### Integration Test with Python

```python
# test_integration.py
from memory_agent_c_wrapper import MemoryAgentC

agent = MemoryAgentC()
print("Initializing...")
agent.init("/dev/dax0.0", 128000, 0.01)

print("Executing action 0...")
ce_info, success = agent.execute_action(0)
print(f"Success: {success}")
print(f"CE Total: {ce_info.total_count}")
```

```bash
sudo python3 test_integration.py
```

## Performance Considerations

### Block Size

현재 설정: 64 bytes (cache line aligned)

```c
#define BLOCK_SIZE 64
```

### Sampling Rate

전체 메모리를 테스트하는 대신 샘플링:

```c
// 1% sampling
ma_init("/dev/dax0.0", 128000, 0.01);

// 실제 테스트 크기 = 128GB * 0.01 = 1.28GB
```

### Memory Access Pattern

- Sequential access for better cache utilization
- 64-byte aligned for devdax requirements
- O_SYNC flag for immediate hardware write

## Troubleshooting

### Build Errors

**Error**: `stdarg.h: No such file or directory`

**Solution**:
```bash
sudo apt-get install build-essential
```

**Error**: Permission denied on devdax

**Solution**: Run with sudo or add user to appropriate group:
```bash
sudo chmod 666 /dev/dax0.0
```

### Runtime Issues

**Issue**: CE count not updating

**Check**:
1. umxc tool installed and in PATH
2. Baseline properly initialized
3. Sufficient memory access to trigger CE

**Issue**: Segmentation fault

**Check**:
1. devdax properly initialized before use
2. Memory size not exceeding device capacity
3. Proper structure alignment in ctypes

## Makefile

```makefile
CC = gcc
CFLAGS = -Wall -Wextra -fPIC -O2
LDFLAGS = -shared

TARGET = libmemory_agent.so
SOURCES = memory_agent.c
OBJECTS = $(SOURCES:.c=.o)

all: $(TARGET)

$(TARGET): $(OBJECTS)
	$(CC) $(LDFLAGS) -o $@ $^

%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -f $(OBJECTS) $(TARGET)

.PHONY: all clean
```

## References

- **Implementation**: `src/MemoryAgent/c_library/memory_agent.c`
- **Python Wrapper**: `src/MemoryAgent/memory_agent_c_wrapper.py`
- **REST API**: `src/MemoryAgent/memory_agent_server.py`
- **Architecture**: `docs/DISTRIBUTED_ARCHITECTURE.md`
- **DPA Mapping**: `docs/DPA_ADDRESS_MAPPING_EXPLAINED.md`

## Changelog

### 2024-11-20
- ✅ Migrated to devdax-based architecture
- ✅ Removed MBIST dependency
- ✅ Added umxc CE detection
- ✅ Implemented 6 operation types
- ✅ Created Python ctypes wrapper
- ✅ Integrated with Flask REST API server

### 2024-11-17
- ✅ Initial C library implementation
- ✅ Successfully compiled libmemory_agent.so
- ✅ Fixed compilation warnings

---

**Last Updated**: 2024-11-20
**Version**: 2.0 (devdax-based)
