# DevDax Interface Design

## Overview

DevDax 기반 CXL 메모리 접근 인터페이스 설계 문서입니다. MBIST 엔진 대신 Linux devdax (`/dev/dax*`)를 사용하여 ECC ON 상태에서 메모리 테스트를 수행합니다.

## Architecture

```
Phase1Environment
    ↓
DevDaxInterface (Python)
    ↓
/dev/dax* (Linux kernel)
    ↓
CXL Memory (CMM-D) with ECC ON
    ↓
CE Count monitoring → Fault detection
```

## Key Components

### 1. DevDaxInterface Class

```python
class DevDaxInterface:
    """
    Interface for CXL memory access via devdax (/dev/dax*)

    Features:
    - Direct memory access with ECC ON
    - CE (Correctable Error) count monitoring
    - DPA to DRAM address translation
    - DRAM address to DPA reverse translation
    """

    def __init__(self, device_path: str, dpa_translator):
        """
        Args:
            device_path: Path to devdax device (e.g., "/dev/dax0.0")
            dpa_translator: DPATranslator instance for address conversion
        """

    def write_dpa(self, dpa: int, data: bytes) -> int:
        """Write data to DPA address"""

    def read_dpa(self, dpa: int, length: int) -> bytes:
        """Read data from DPA address"""

    def write_dram_cell(self, rank, bg, ba, row, col, data: bytes) -> int:
        """Write to specific DRAM cell (via reverse translation)"""

    def read_dram_cell(self, rank, bg, ba, row, col, length: int) -> bytes:
        """Read from specific DRAM cell (via reverse translation)"""

    def get_ce_count(self, device: str = "mem0") -> int:
        """Get current CE count from CXL mailbox"""

    def execute_pattern_test(self, operation_type, pattern_byte,
                            start_dram, end_dram) -> int:
        """
        Execute pattern test and return CE delta

        Returns:
            CE count difference (faults detected)
        """
```

### 2. DPATranslator Class

```python
class DPATranslator:
    """
    Bidirectional DPA ↔ DRAM address translation

    Based on collected mapping data from umxc ei -t
    """

    def __init__(self, mapping_file: str):
        """Load mapping data and infer translation rules"""

    def dpa_to_dram(self, dpa: int) -> Dict:
        """
        Forward translation: DPA → DRAM address

        Returns:
            {'subch': int, 'dimm': int, 'rank': int,
             'bg': int, 'ba': int, 'row': int, 'col': int}
        """

    def dram_to_dpa(self, rank, bg, ba, row, col,
                    dimm=0, subch=0) -> int:
        """
        Reverse translation: DRAM address → DPA

        This is the KEY function for March algorithm implementation
        """
```

### 3. CECountMonitor Class

```python
class CECountMonitor:
    """
    Monitor CE (Correctable Error) count from CXL mailbox

    Uses CXL mailbox command to read ECC error counters
    """

    def __init__(self, device: str = "mem0"):
        """
        Args:
            device: CXL device name (e.g., "mem0")
        """

    def get_ce_count(self) -> int:
        """
        Get current CE count

        Implementation options:
        1. Read from sysfs: /sys/bus/cxl/devices/mem0/...
        2. Use CXL mailbox command via ioctl
        3. Parse dmesg/kernel logs
        """

    def reset_ce_count(self) -> int:
        """Reset CE counter (if supported)"""

    def get_ce_delta(self, baseline: int) -> int:
        """Get CE count difference from baseline"""
```

## Implementation Strategy

### Phase 1: Basic Infrastructure (Current)
- [x] DPA mapping data collection tool
- [x] Mapping visualization and analysis
- [ ] Infer mapping rules from data
- [ ] Implement forward translation (DPA → DRAM)
- [ ] Implement reverse translation (DRAM → DPA)

### Phase 2: DevDax Interface
- [ ] Implement DevDaxInterface class
- [ ] Test read/write via /dev/dax*
- [ ] Verify mmap vs direct I/O performance
- [ ] Handle alignment requirements (cache line)

### Phase 3: CE Count Monitoring
- [ ] Research CXL mailbox CE count command
- [ ] Implement CECountMonitor class
- [ ] Test CE count collection
- [ ] Verify CE count reset capability

### Phase 4: Integration
- [ ] Update Phase1Environment to use DevDaxInterface
- [ ] Replace MBIST calls with devdax operations
- [ ] Update reward calculation (MBIST failure → CE delta)
- [ ] Test with mock mode and real hardware

## Technical Details

### devdax Access Methods

#### Option 1: mmap (Recommended)
```python
import mmap
import os

fd = os.open("/dev/dax0.0", os.O_RDWR | os.O_SYNC)
mm = mmap.mmap(fd, length=size, offset=dpa_offset)

# Write
mm[0:64] = data_bytes

# Read
data = mm[0:64]
```

**Pros**: Fast, direct memory mapping
**Cons**: Alignment restrictions

#### Option 2: Direct read/write
```python
fd = os.open("/dev/dax0.0", os.O_RDWR | os.O_SYNC)
os.lseek(fd, dpa_offset, os.SEEK_SET)
os.write(fd, data_bytes)
os.lseek(fd, dpa_offset, os.SEEK_SET)
data = os.read(fd, length)
```

**Pros**: Simple, flexible
**Cons**: System call overhead

### CE Count Sources

#### Option 1: sysfs (Preferred if available)
```bash
cat /sys/bus/cxl/devices/mem0/ras/correctable_error_count
```

#### Option 2: CXL Mailbox Command
```python
# Get ECC Error Log (Opcode 0x0101)
# Implementation needed
```

#### Option 3: EDAC (Error Detection And Correction)
```bash
cat /sys/devices/system/edac/mc/mc0/ce_count
```

### Address Alignment

- **Cache line alignment**: 64 bytes (0x40)
- DPA addresses should be aligned to cache line boundaries
- Column addresses in DRAM are typically cache line granularity

### March Algorithm with devdax

**Challenge**: Sequential DPA access may not map to sequential DRAM cells

**Example**:
```
DPA 0x0    → rank=0, bg=0, ba=0, row=0, col=0
DPA 0x40   → rank=0, bg=0, ba=0, row=0, col=0, dimm=1  # Different DIMM!
DPA 0x80   → rank=0, bg=0, ba=0, row=0, col=1, dimm=0  # Back to dimm 0
```

**Solution**: Use reverse translation
```python
# March ascending on specific bank
for row in range(0, max_row):
    for col in range(0, max_col):
        dpa = translator.dram_to_dpa(rank=0, bg=0, ba=0, row=row, col=col)
        devdax.write_dpa(dpa, pattern)
        devdax.read_dpa(dpa, 64)
```

## Comparison: MBIST vs devdax

| Feature | MBIST | devdax |
|---------|-------|--------|
| **ECC** | OFF (required) | ON (normal operation) |
| **Fault Detection** | Direct data verification | CE count delta |
| **DRAM Control** | Direct (rank/bg/ba/row/col) | Indirect (via DPA translation) |
| **Performance** | Fast | Slower (translation overhead) |
| **Real Environment** | ❌ Not production-like | ✅ Production-like |
| **Complexity** | Low | Medium (translation needed) |

## Open Questions

### Q1: DPA mapping consistency
- Is the mapping static or dynamic?
- Does it change with different configurations?

### Q2: CE count granularity
- Can we get CE count per rank/bank/row?
- Or only global CE count for entire device?

### Q3: devdax performance
- What is the latency of read/write operations?
- Can we batch operations for better performance?

### Q4: Translation accuracy
- Can we achieve single-cell precision with reverse translation?
- Are there DRAM cells that are not accessible via any DPA?

## Next Steps

1. **Analyze collected DPA mapping data** (waiting for GNR-CRB collection)
2. **Infer and implement translation functions**
3. **Research CE count access methods** on GNR-CRB board
4. **Prototype DevDaxInterface** with basic read/write
5. **Test March algorithm feasibility** with reverse translation

## References

- CXL 2.0/3.0 Specification
- Linux kernel devdax documentation
- CXL mailbox commands specification
- EDAC subsystem documentation
