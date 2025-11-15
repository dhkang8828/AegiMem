# DPA to DRAM Address Mapping Rules

## CMM-D Device Specification

- **Capacity**: 128GB
- **Total DPA Range**: 0x0 ~ 0x2000000000
- **Configuration**:
  - Subchannel: 2 (0-1)
  - DIMM: 2 (0-1)
  - Rank: 1 (0)
  - Bank Group (BG): 8 (0-7)
  - Bank Address (BA): 4 (0-3)
  - Row: Variable
  - Column: 2048 (0-0x7ff)

## Interleaving Hierarchy

The DPA address is decomposed into DRAM components in the following order (from smallest to largest):

| Layer | Component | Size | Quantity | Range |
|-------|-----------|------|----------|-------|
| 1 | DIMM | 64B (0x40) | 2 | 0-1 |
| 2 | Bank Group (BG) | 128B (0x80) | 8 | 0-7 |
| 3 | Column | 1KB (0x400) | 2048 | 0-0x7ff |
| 4 | Bank Address (BA) | 128KB (0x20000) | 4 | 0-3 |
| 5 | Subchannel | 512KB (0x80000) | 2 | 0-1 |
| 6 | Row | 1MB (0x100000) | Variable | 0-... |

## Mapping Formula

### Forward Translation: DPA → DRAM Address

```
dimm    = (DPA / 0x40) % 2
bg      = (DPA / 0x80) % 8
col     = (DPA / 0x400) % 0x800
ba      = (DPA / 0x20000) % 4
subch   = (DPA / 0x80000) % 2
row     = DPA / 0x100000
rank    = 0 (fixed)
```

### Reverse Translation: DRAM Address → DPA

```
DPA = row × 0x100000 +
      subch × 0x80000 +
      ba × 0x20000 +
      col × 0x400 +
      bg × 0x80 +
      dimm × 0x40
```

## Examples

### Example 1: DPA 0x0

```
DPA: 0x0
→ dimm=0, bg=0, ba=0, row=0, col=0, subch=0, rank=0
```

### Example 2: DPA 0x40 (64 bytes)

```
DPA: 0x40
→ dimm=1, bg=0, ba=0, row=0, col=0, subch=0, rank=0

Change: DIMM interleaving
```

### Example 3: DPA 0x80 (128 bytes)

```
DPA: 0x80
→ dimm=0, bg=1, ba=0, row=0, col=0, subch=0, rank=0

Change: BG increases (after 2 DIMMs)
```

### Example 4: DPA 0x400 (1KB)

```
DPA: 0x400
→ dimm=0, bg=0, ba=0, row=0, col=1, subch=0, rank=0

Change: Column increases (after 2 DIMMs × 8 BGs)
```

### Example 5: DPA 0x20000 (128KB)

```
DPA: 0x20000
→ dimm=0, bg=0, ba=1, row=0, col=0, subch=0, rank=0

Change: BA increases (after 2048 Columns)
```

### Example 6: DPA 0x80000 (512KB)

```
DPA: 0x80000
→ dimm=0, bg=0, ba=0, row=0, col=0, subch=1, rank=0

Change: Subchannel switches (after 4 BAs)
```

### Example 7: DPA 0x100000 (1MB)

```
DPA: 0x100000
→ dimm=0, bg=0, ba=0, row=1, col=0, subch=0, rank=0

Change: Row increases (after 2 Subchannels)
```

## Interleaving Pattern

### Within 1KB (16 cache lines)

```
DPA         DIMM  BG   BA   Col  SubCh  Row
0x000       0     0    0    0    0      0
0x040       1     0    0    0    0      0    (DIMM switch)
0x080       0     1    0    0    0      0    (BG +1)
0x0C0       1     1    0    0    0      0    (DIMM switch)
...
0x380       0     7    0    0    0      0
0x3C0       1     7    0    0    0      0
0x400       0     0    0    1    0      0    (Column +1)
```

### Progression Summary

```
1 cache line (64B) → DIMM interleaving
2 cache lines (128B) → BG +1
16 cache lines (1KB) → Column +1
2048 Columns (2MB) → BA +1
4 BAs (8MB total across subchannels) → Subchannel switch
2 Subchannels (1MB per row) → Row +1
```

## Key Observations

1. **Cache Line Alignment**: All meaningful DPA values are 64B aligned (0x40)
2. **DIMM Interleaving**: Every cache line alternates between DIMM 0 and 1
3. **BG Interleaving**: Bank Groups interleave every 2 cache lines
4. **Column Sequential**: Columns increment sequentially after all BG/DIMM combinations
5. **BA Sequential**: Bank Addresses increment after all columns
6. **Subchannel Sequential**: Subchannels alternate after all BAs
7. **Row Sequential**: Rows increment after all lower-level components

## Physical Memory Layout

```
128GB Device = 2 Subchannels × 64GB per subchannel
             = 2 Subchannels × 4 BAs × 16GB per BA
             = 2 Subchannels × 4 BAs × 2048 Cols × 8KB per col
             = 2 Subchannels × 4 BAs × 2048 Cols × (8 BGs × 2 DIMMs × 64B)
```

## Validation Data

| DPA | SubCh | DIMM | Rank | BG | BA | Row | Col |
|-----|-------|------|------|----|----|----|-----|
| 0x0 | 0 | 0 | 0 | 0 | 0 | 0x0 | 0x0 |
| 0x40 | 0 | 1 | 0 | 0 | 0 | 0x0 | 0x0 |
| 0x80 | 0 | 0 | 0 | 1 | 0 | 0x0 | 0x0 |
| 0x400 | 0 | 0 | 0 | 0 | 0 | 0x0 | 0x1 |
| 0x20000 | 0 | 0 | 0 | 0 | 1 | 0x0 | 0x0 |
| 0x80000 | 1 | 0 | 0 | 0 | 0 | 0x0 | 0x0 |
| 0x100000 | 0 | 0 | 0 | 0 | 0 | 0x1 | 0x0 |

## Usage in RL Environment

The reverse translation (DRAM → DPA) is critical for implementing March algorithms:

```python
# Example: Test Column 0 to Column 100 in Row 0, BA 0, BG 0
for col in range(0, 101):
    dpa = dram_to_dpa(dimm=0, bg=0, ba=0, row=0, col=col)
    devdax.write(dpa, pattern_data)
    devdax.read(dpa, 64)
```

This allows precise DRAM cell-level access patterns while using devdax interface.
