# C Library Build and Integration Guide

## Overview

The C library (`mt_rl_primitives`) provides low-level DRAM command primitives for the RL Agent. This library is implemented within the MBIST codebase and compiled as part of the MBIST build process.

## Location

The RL primitives are located in the MBIST codebase:

```
/home/dhkang/data3/mbist_sample_code-gen2_es/
├── include/
│   └── mt_rl_primitives.h          # Header file with function declarations
└── src/
    └── mt_rl_primitives.c          # Implementation
```

## Build Process

### Prerequisites

- GCC compiler
- MBIST development environment

### Build Commands

1. **Clean previous build**:
   ```bash
   cd /home/dhkang/data3/mbist_sample_code-gen2_es
   make clean
   ```

2. **Build the library**:
   ```bash
   make smbus
   ```

3. **Verify compilation**:
   ```bash
   ls -la target/smbus/mt_rl_primitives.o
   ```

   Expected output: Object file successfully compiled.

### Build Notes

- The compilation will succeed even if linking fails (due to missing hardware libraries)
- We only need the object file (`mt_rl_primitives.o`) for Python integration
- The warning about unused `pattern_type` parameter is expected and will be fixed in future

## Implemented Functions

### Sequence Mode Control

```c
void mt_rl_begin_sequence(uint8_t channel);
void mt_rl_end_sequence(uint8_t channel);
```

### Low-level DRAM Commands

```c
int mt_rl_send_activate(uint8_t rank, uint8_t bg, uint8_t ba, uint32_t row);
int mt_rl_send_write(uint8_t rank, uint8_t bg, uint8_t ba, uint32_t row,
                     uint16_t col, uint8_t pattern_type);
int mt_rl_send_read(uint8_t rank, uint8_t bg, uint8_t ba, uint32_t row,
                    uint16_t col);
int mt_rl_send_precharge(uint8_t rank, uint8_t bg, uint8_t ba, uint8_t all_banks);
int mt_rl_send_refresh(uint8_t rank, uint8_t bg, uint8_t ba, uint8_t all_banks);
```

### Result Checking

```c
int mt_rl_get_test_result(uint8_t channel);
int mt_rl_get_error_count(uint8_t channel);
int mt_rl_get_error_addresses(uint8_t channel, MBIST_ERR_ADDRS_T *addrs);
```

## Command Encoding Details

### ACTIVATE Command
- Encodes: rank, bank group, bank, row address
- Row address split: bits [0:3], [4:16], [17]
- Chip select based on rank

### WRITE Command
- Encodes: rank, bank group, bank, row, column
- Column encoding: bits [3:10]
- Pattern set separately via data pattern functions

### READ Command
- Similar to WRITE but with different command opcode
- Column encoding: bits [2:10] (9 bits)

### PRECHARGE Command
- Single bank: Encodes rank and bank address only
- All banks: No address encoding needed
- **Note**: Bank group is NOT encoded in PREsb (hardware limitation)

### REFRESH Command
- Similar to PRECHARGE
- **Note**: Bank group is NOT encoded in REFsb (hardware limitation)

## Known Issues and Fixes

### Issue 1: Column Address Encoding

**Problem**: Original code used `cap1_1_8C3_10` field which doesn't exist.

**Fix**: Changed to `cap1_0_8C2_10` with correct bit shift:
```c
// Before
cmd_out->rd.cap1_1_8C3_10 = (col >> 3) & 0xFF;

// After
cmd_out->rd.cap1_0_8C2_10 = (col >> 2) & 0x1FF;
```

### Issue 2: Bank Group in PRECHARGE/REFRESH Single Bank

**Problem**: Original code tried to set bank group field which doesn't exist.

**Fix**: Removed bank group encoding for PREsb and REFsb:
```c
// PREsb structure doesn't have BG field
// Only BA (bank address) is supported
cmd_out->presb.cap_6_7BA0_1 = ba & 0x3;
```

### Issue 3: Missing Header Include

**Problem**: `MBIST_ERR_ADDRS_T` type not found.

**Fix**: Added include in header file:
```c
#include "mt_err.h"
```

## Testing

### Unit Test Program

A standalone test program is available:

```bash
cd /home/dhkang/cxl_memory_rl_project/src/c_library
gcc -o test_rl_primitives test_rl_primitives.c
./test_rl_primitives
```

Expected output:
```
=== RL Primitives Test Program ===
Test 1: ACTIVATE rank=0, bg=0, ba=0, row=100
✓ Test 1 PASSED
...
=== All Tests Complete ===
```

## Python Integration

The Python interface (`src/mbist_interface.py`) uses ctypes to call these C functions.

### Current Status

- Python interface implemented with Mock mode
- Ready for integration with compiled library
- Will require building shared library (.so) from object files

### Next Steps

1. Create shared library from object files
2. Load library in Python using ctypes
3. Test with actual hardware
4. Integrate with RL environment

## Address Space

### DRAM Organization

```
Rank: 0-3 (4 ranks)
Bank Group: 0-7 (8 bank groups)
Bank: 0-3 (4 banks per group)
Row: 0-262143 (262K rows, 18 bits)
Column: 0-2047 (2K columns, 11 bits)
```

### Total Addressable Space

```
4 ranks × 8 BG × 4 BA × 262144 rows × 2048 columns
= ~68 billion addresses
```

## Performance Considerations

### Sequence Mode

- **Queue mode**: Commands batched in SRAM (up to 512 commands)
- **Immediate mode**: Single command executed immediately
- Use sequence mode for efficiency when running test patterns

### SRAM Limitations

- Maximum 512 commands per sequence
- For longer sequences (e.g., row hammer), use batching:
  ```c
  batch_size = 500;
  for (int i = 0; i < total_count; i += batch_size) {
      mt_rl_begin_sequence(channel);
      // Add up to 500 commands
      mt_rl_end_sequence(channel);
  }
  ```

## Troubleshooting

### Build Errors

**Error**: `cannot find -lumxc_devlib`

**Solution**: This is expected when building full executable. We only need object files.

**Error**: `unknown type name 'MBIST_ERR_ADDRS_T'`

**Solution**: Ensure `mt_err.h` is included in header file.

### Runtime Issues

**Issue**: Commands not executing

**Check**:
1. Sequence mode properly closed with `mt_rl_end_sequence()`
2. Hardware initialization completed
3. Channel parameter correct (0 or 1)

## References

- MBIST API Documentation: `/home/dhkang/data3/mbist_sample_code-gen2_es/docs/`
- Command Encoding Analysis: `docs/MBIST_COMMAND_ENCODING_ANALYSIS.md`
- Integration Plan: `docs/MBIST_INTEGRATION_PLAN.md`

## Changelog

### 2025-11-01
- ✅ Initial implementation
- ✅ Fixed column address encoding bug
- ✅ Fixed bank group encoding in PREsb/REFsb
- ✅ Added header includes for missing types
- ✅ Successfully compiled object files
- ✅ Created and ran unit tests
