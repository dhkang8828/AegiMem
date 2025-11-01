# MBIST Command Encoding Analysis

## 분석일: 2025-11-01

## 핵심 발견

### 1. Command Encoding 흐름

```
사용자 요청 (ACT, WR, RD, PRE...)
    ↓
CMD_TRUTH_TABLE 구조체 생성 및 초기화
    ↓
spec_fmtcmd_*() 함수로 command 필드 채우기
    ↓
mt_add_type0_ca_pattern() 호출
    ↓
Type0_CA_pattern_t 생성 (64-bit pattern)
    ↓
mt_add_sram() - SRAM에 추가
```

### 2. 핵심 구조체

#### CMD_TRUTH_TABLE
```c
// include/mt_mbist_patdef.h
typedef union tagCMD_TRUTH_TABLE {
    CA_PATTERN_CMD ca_pattern;    // 공통
    CMD_ACT act;                  // ACTIVATE
    CMD_WR wr;                    // WRITE
    CMD_WRA wra;                  // WRITE with Auto-precharge
    CMD_RD rd;                    // READ
    CMD_RDA rda;                  // READ with Auto-precharge
    CMD_PREab preab;              // PRECHARGE all banks
    CMD_PREsb presb;              // PRECHARGE single bank
    CMD_REFab refab;              // REFRESH all banks
    CMD_REFsb refsb;              // REFRESH single bank
    // ... more
    CMD_COMMON common;
} CMD_TRUTH_TABLE;
```

#### CMD_ACT (ACTIVATE 예시)
```c
typedef struct tagCMD_ACT {
    uint32_t cap0_0_1 : 2;
    uint32_t cap0_2_5R0_3 : 4;       // Row[0:3]
    uint32_t cap0_6_7BA0_1 : 2;      // Bank Address[0:1]
    uint32_t cap0_8_10BG0_2 : 3;     // Bank Group[0:2]
    uint32_t cap0_11_13CID0_2 : 3;   // Chip ID[0:2]
    uint32_t cap1_0_12R4_16 : 13;    // Row[4:16]
    uint32_t cap1_13R17CID3 : 1;     // Row[17] or CID[3]
} CMD_ACT;
```

### 3. Command Type Array

```c
// src/pat_api.c:161
CMD_TYPE_SPECIFY cmd_type_arr[] = {
    {0x00, "ACT",    spec_fmtcmd_act,       NULL},
    {0x0d, "WR",     spec_fmtcmd_wr,        NULL},
    {0x0d, "WRA",    spec_fmtcmd_wra,       NULL},
    {0x1d, "RD",     spec_fmtcmd_rd,        NULL},
    {0x1d, "RDA",    spec_fmtcmd_rda,       NULL},
    {0x0b, "PREab",  spec_fmtcmd_preab,     NULL},
    {0x0b, "PREsb",  spec_fmtcmd_presb,     NULL},
    {0x13, "REFab",  spec_fmtcmd_refab,     NULL},
    {0x13, "REFsb",  spec_fmtcmd_refsb,     NULL},
    // ...
};
```

### 4. Spec Format Functions

#### ACTIVATE
```c
// src/pat_api.c:18
static void spec_fmtcmd_act(uint8_t cmd, CMD_TRUTH_TABLE *cmd_truth_table) {
    cmd_truth_table->act.cap0_0_1 = cmd;
}
```

#### WRITE
```c
// src/pat_api.c:49
static void spec_fmtcmd_wr(uint8_t cmd, CMD_TRUTH_TABLE *cmd_truth_table) {
    cmd_truth_table->wr.cap0_0_4     = cmd;
    cmd_truth_table->wr.cap0_5BL     = 1;        // Burst Length
    cmd_truth_table->wr.cap1_10H     = 1;
    cmd_truth_table->wr.cap1_11WRP   = 1;
}
```

#### READ
```c
// src/pat_api.c:65
static void spec_fmtcmd_rd(uint8_t cmd, CMD_TRUTH_TABLE *cmd_truth_table) {
    cmd_truth_table->rd.cap0_0_4     = cmd;
    cmd_truth_table->rd.cap0_5BL     = 1;
    cmd_truth_table->rd.cap1_10H     = 1;
}
```

#### PRECHARGE
```c
// src/pat_api.c:118
static void spec_fmtcmd_preab(uint8_t cmd, CMD_TRUTH_TABLE *cmd_truth_table) {
    cmd_truth_table->preab.cap0_0_4     = cmd;
    cmd_truth_table->preab.cap_10L      = 0;
}

static void spec_fmtcmd_presb(uint8_t cmd, CMD_TRUTH_TABLE *cmd_truth_table) {
    cmd_truth_table->presb.cap0_0_4     = cmd;
    cmd_truth_table->presb.cap_10H      = 1;
}
```

### 5. 기존 알고리즘 예시 (March C+)

```c
// src/mt_algo_pattern.c

// WRITE operation
void prgm_write(uint8_t dq_inv, ...) {
    CMD_TRUTH_TABLE cmd_truth_table;
    memset(&cmd_truth_table, 0, sizeof(cmd_truth_table));

    // Add WRITE command
    mt_add_type0_ca_pattern(replace_en, WR, dq_inv, wait_cycle, &cmd_truth_table);
}

// READ operation
void prgm_read(uint8_t dq_inv, ...) {
    CMD_TRUTH_TABLE cmd_truth_table;
    memset(&cmd_truth_table, 0, sizeof(cmd_truth_table));

    // Add READ command
    mt_add_type0_ca_pattern(replace_en, RD, dq_inv, wait_cycle, &cmd_truth_table);
}
```

## Command Builder 설계

### 필요한 Helper 함수

```c
// 1. ACTIVATE command 생성
int build_activate_cmd(uint8_t rank, uint8_t bg, uint8_t ba, uint32_t row,
                      CMD_TRUTH_TABLE *cmd_out);

// 2. WRITE command 생성
int build_write_cmd(uint8_t rank, uint8_t bg, uint8_t ba, uint32_t row,
                   uint16_t col, CMD_TRUTH_TABLE *cmd_out);

// 3. READ command 생성
int build_read_cmd(uint8_t rank, uint8_t bg, uint8_t ba, uint32_t row,
                  uint16_t col, CMD_TRUTH_TABLE *cmd_out);

// 4. PRECHARGE command 생성
int build_precharge_cmd(uint8_t rank, uint8_t bg, uint8_t ba,
                       uint8_t all_banks, CMD_TRUTH_TABLE *cmd_out);

// 5. REFRESH command 생성
int build_refresh_cmd(uint8_t rank, uint8_t bg, uint8_t ba,
                     uint8_t all_banks, CMD_TRUTH_TABLE *cmd_out);
```

### Address Encoding

DRAM 주소를 CMD_ACT 구조체로 변환하는 방법:

```
Input:
  rank = 0-3
  bg = 0-7 (3 bits)
  ba = 0-3 (2 bits)
  row = 0-262143 (18 bits)

Output (CMD_ACT):
  cap0_2_5R0_3 = row & 0xF               // row[0:3]
  cap0_6_7BA0_1 = ba                     // bank[0:1]
  cap0_8_10BG0_2 = bg                    // bank group[0:2]
  cap0_11_13CID0_2 = rank & 0x7          // rank[0:2] (or CID)
  cap1_0_12R4_16 = (row >> 4) & 0x1FFF  // row[4:16]
  cap1_13R17CID3 = (row >> 17) & 0x1    // row[17]
```

### WRITE의 경우 Column 추가

```c
typedef struct tagCMD_WR {
    uint32_t cap0_0_4: 5;
    uint32_t cap0_5BL : 1;
    uint32_t cap0_6_7BA0_1 : 2;      // Bank Address
    uint32_t cap0_8_10BG0_2 : 3;     // Bank Group
    uint32_t cap0_11_13CID0_2 : 3;   // Chip ID/Rank
    uint32_t cap1_0V : 1;
    uint32_t cap1_1_8C3_10 : 8;      // Column[3:10]
    uint32_t cap1_9V : 1;
    uint32_t cap1_10H : 1;
    uint32_t cap1_11WRP : 1;
    uint32_t cap1_12V : 1;
    uint32_t cap1_13CID3 : 1;        // CID[3] or Rank[3]
} CMD_WR;

Encoding:
  cap0_6_7BA0_1 = ba
  cap0_8_10BG0_2 = bg
  cap0_11_13CID0_2 = rank & 0x7
  cap1_1_8C3_10 = (col >> 3) & 0xFF  // column[3:10]
  cap1_13CID3 = (rank >> 3) & 0x1
```

## 주의사항

### 1. Replace Enable
```c
uint8_t replace_en = mt_set_replace_en();
```
- 이 플래그는 address field를 자동으로 채울지 여부를 결정
- RL에서는 수동으로 주소를 설정하므로 적절히 처리 필요

### 2. Chip Select (CS)
```c
type0_ca_pattern.cs_p0 = 0xe;  // Default
```
- Rank 선택은 CS를 통해 이루어짐
- Rank 0 = CS0, Rank 1 = CS1, etc.

### 3. Timing (Wait Cycles)
```c
mt_add_type0_ca_pattern(replace_en, WR, dq_inv, wait_cycle, &cmd_truth_table);
```
- wait_cycle: command 간 대기 시간
- DRAM timing 파라미터 (tCCD, tRCD, tRP 등)를 고려해야 함

## 다음 단계

1. ✅ 기존 코드 분석 완료
2. 🔄 Command Builder C 코드 구현
3. ⏳ Python wrapper
4. ⏳ 단위 테스트
