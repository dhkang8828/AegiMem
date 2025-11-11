# DPA to DRAM Address Mapping 분석

**목적**: devdax 방식의 타당성 검증
**핵심 질문**: /dev/dax 주소가 실제 DRAM 주소로 어떻게 매핑되는가?

---

## 📋 용어 정리

### DPA (Device Physical Address)
```
- /dev/dax0.0를 통해 접근하는 주소
- User space에서 보이는 선형 주소 공간
- 예: 0x0000000000000000 ~ 0x0000001FFFFFFFFF (128GB)
```

### HPA (Host Physical Address)
```
- 시스템 메모리 맵에서의 물리 주소
- CPU가 인식하는 주소
- CXL.mem을 통해 CXL device로 전달
```

### DRAM Address
```
- 실제 DRAM 칩의 물리적 주소
- 구성 요소:
  * Rank: 0-3
  * Bank Group (BG): 0-7
  * Bank Address (BA): 0-3
  * Row: 0-262143 (18-bit)
  * Column: 0-2047 (11-bit)
  * Chip ID (CID): 0-15
```

---

## 🗺️ Address Mapping Chain

```
┌─────────────────────────────────────────────────────┐
│  User Application                                   │
│  addr = 0x1000 (DPA in /dev/dax0.0)                │
└──────────────────┬──────────────────────────────────┘
                   │ mmap() / read() / write()
                   ▼
┌─────────────────────────────────────────────────────┐
│  Kernel (DAX driver)                                │
│  DPA → HPA translation                              │
└──────────────────┬──────────────────────────────────┘
                   │ Memory transaction
                   ▼
┌─────────────────────────────────────────────────────┐
│  CXL.mem Protocol                                   │
│  HPA carried in CXL.mem packet                      │
└──────────────────┬──────────────────────────────────┘
                   │ CXL transaction
                   ▼
┌─────────────────────────────────────────────────────┐
│  CXL Device (CMM-D)                                 │
│  HPA → DPA (via HDM decoder)                        │
│  DPA → DRAM Address (controller logic)             │
└──────────────────┬──────────────────────────────────┘
                   │ DRAM command
                   ▼
┌─────────────────────────────────────────────────────┐
│  DRAM (Physical Memory)                             │
│  rank/bg/ba/row/column                              │
└─────────────────────────────────────────────────────┘
```

---

## 📐 Mapping 정보 (사용자 제공 예정)

### CXL Device 구성

```
Device: CMM-D (CXL Type3 Memory Module)
Capacity: (TBD)
DRAM Configuration:
  - Ranks: (TBD)
  - Bank Groups: (TBD)
  - Banks per group: (TBD)
  - Rows per bank: (TBD)
  - Columns per row: (TBD)
  - Data width: (TBD)
```

### DPA Bit Layout

```
DPA 구조 (사용자 제공 예정):

Bit [?:?]: Rank selection
Bit [?:?]: Bank Group
Bit [?:?]: Bank Address
Bit [?:?]: Row
Bit [?:?]: Column
Bit [?:?]: Byte offset

예시 (가정):
DPA[63:0]
  [63:36]: Reserved
  [35:34]: Rank (2-bit, 4 ranks)
  [33:31]: Bank Group (3-bit, 8 BG)
  [30:29]: Bank Address (2-bit, 4 BA)
  [28:11]: Row (18-bit, 262K rows)
  [10:6]:  Column (5-bit, 32 columns × 64B = 2K columns)
  [5:0]:   Byte offset (6-bit, 64B cache line)
```

### Address Translation Function

```python
def dpa_to_dram_address(dpa):
    """
    DPA를 DRAM address로 변환

    (사용자가 제공할 실제 매핑 로직)
    """

    # Bit extraction (예시, 실제 값은 사용자 제공)
    rank = (dpa >> 34) & 0x3
    bg = (dpa >> 31) & 0x7
    ba = (dpa >> 29) & 0x3
    row = (dpa >> 11) & 0x3FFFF
    col = (dpa >> 6) & 0x1F
    byte_offset = dpa & 0x3F

    return {
        'rank': rank,
        'bank_group': bg,
        'bank_address': ba,
        'row': row,
        'column': col,
        'byte_offset': byte_offset
    }
```

---

## 🔬 타당성 검증 항목

### 1. 순차 접근 패턴

**질문**: DPA를 순차적으로 증가시키면 DRAM 주소가 어떻게 변하는가?

```python
# Test case
for dpa in range(0, 1024 * 1024, 64):  # 1MB, 64B 간격
    dram_addr = dpa_to_dram_address(dpa)
    print(f"DPA 0x{dpa:x} -> Rank {dram_addr['rank']}, "
          f"BG {dram_addr['bank_group']}, BA {dram_addr['bank_address']}, "
          f"Row {dram_addr['row']}, Col {dram_addr['column']}")

# 기대되는 패턴:
# - Column이 먼저 증가? (Row buffer 내에서 이동)
# - Row가 먼저 증가? (Row 간 이동)
# - Bank interleaving?
```

**중요성**:
- March 알고리즘은 순차 접근을 가정
- Column 내 → Row 내 → Bank 내 순서가 중요
- 순차성이 보장되지 않으면 March-like 패턴 불가능

### 2. Row Buffer 활용

**질문**: 같은 Row의 연속된 Column을 접근할 수 있는가?

```python
# 같은 row의 다른 column 접근
dpa1 = get_dpa_for(rank=0, bg=0, ba=0, row=100, col=0)
dpa2 = get_dpa_for(rank=0, bg=0, ba=0, row=100, col=1)

delta = dpa2 - dpa1
print(f"Column delta in DPA: {delta} bytes")

# 기대: delta가 작고 예측 가능
# → Row buffer hit 최적화 가능
```

**중요성**:
- Row buffer locality 활용
- 성능 최적화
- DRAM timing 특성 활용

### 3. Bank/Row 경계

**질문**: Bank나 Row가 바뀔 때 DPA가 어떻게 변하는가?

```python
# Row boundary
last_col_in_row = get_dpa_for(rank=0, bg=0, ba=0, row=100, col=2047)
first_col_next_row = get_dpa_for(rank=0, bg=0, ba=0, row=101, col=0)
row_boundary = first_col_next_row - last_col_in_row

# Bank boundary
last_row_in_bank = get_dpa_for(rank=0, bg=0, ba=0, row=262143, col=2047)
first_row_next_bank = get_dpa_for(rank=0, bg=0, ba=1, row=0, col=0)
bank_boundary = first_row_next_bank - last_row_in_bank
```

**중요성**:
- Ascending/Descending 패턴 구현
- Bank thrashing 구현
- Address wrap-around 처리

### 4. Interleaving/Scrambling

**질문**: Address interleaving이나 scrambling이 적용되는가?

```
가능한 시나리오:

Scenario A: Direct mapping (이상적)
  DPA[10:6] → Column[4:0]
  순차 DPA → 순차 Column

Scenario B: Bank interleaving
  DPA[8:6] → Bank
  DPA[10:9] → Column[4:3]
  순차 DPA → Bank가 번갈아 바뀜

Scenario C: Address scrambling
  Column = scramble(DPA[10:6])
  순차 DPA → 무작위 Column
```

**중요성**:
- Interleaving: 성능 최적화용, 예측 가능
- Scrambling: 보안용, 예측 불가 → devdax 방식 어려움

---

## 🎯 devdax 타당성 판단 기준

### ✅ devdax 사용 가능한 경우

```
1. DPA → DRAM 매핑이 명확하고 예측 가능
2. 순차 DPA 접근이 의미있는 DRAM 패턴 생성
3. Ascending/Descending 구현 가능
4. Row/Bank 경계 제어 가능
5. March-like 알고리즘 구현 가능
```

**결론**: devdax로 전환 ✅

### ❌ devdax 사용 불가능한 경우

```
1. Address scrambling으로 인해 매핑 예측 불가
2. 순차 접근이 무작위 DRAM 접근으로 변환
3. Row/Bank 제어 불가능
4. March 알고리즘 구현 불가
```

**결론**: MBIST 방식 유지 또는 다른 대안 필요 ❌

### ⚠️ 부분적 사용 가능한 경우

```
1. 매핑은 예측 가능하지만 복잡
2. 특정 패턴만 구현 가능
3. 추가 계산/변환 필요
```

**결론**: 구현 복잡도 vs 이점 비교 필요 ⚠️

---

## 📊 검증 실험 계획

### Experiment 1: Sequential Access Pattern

```python
"""DPA 순차 접근 시 DRAM 주소 변화 관찰"""

import mmap
import os

device = '/dev/dax0.0'
block_size = 4096  # 4KB
num_blocks = 100

fd = os.open(device, os.O_RDWR)
mm = mmap.mmap(fd, block_size * num_blocks, mmap.MAP_SHARED)

# Write sequential pattern
for i in range(num_blocks):
    offset = i * block_size
    pattern = i.to_bytes(8, 'little')
    mm[offset:offset+8] = pattern

# 동시에 DRAM address 모니터링 (방법 TBD)
# - Mailbox command?
# - Debug register?
# - External analyzer?

mm.close()
os.close(fd)
```

### Experiment 2: Row Boundary Detection

```python
"""Row 경계 감지"""

# Strategy:
# 1. Sequential write로 전체 메모리 초기화
# 2. 특정 패턴으로 read
# 3. CE count 변화 관찰
# 4. CE 발생 위치 = Row 경계 추정?

# (구체적 방법은 매핑 정보 확인 후)
```

### Experiment 3: Bank Interleaving Check

```python
"""Bank interleaving 확인"""

# Write pattern to consecutive DPA
# Check if banks are interleaved

# 방법:
# - Performance counter 사용?
# - Memory bandwidth 측정?
# - CE pattern 분석?
```

---

## 📝 사용자 제공 필요 정보

다음 정보를 제공해주시면 분석을 완성하겠습니다:

### 1. CMM-D 사양
```
- [ ] 전체 용량
- [ ] DRAM 구성 (rank/bg/ba/row/col)
- [ ] Data width
- [ ] ECC 구성
```

### 2. DPA Bit Layout
```
- [ ] DPA에서 각 DRAM address 필드의 bit 위치
- [ ] Byte offset, column, row, bank, rank 매핑
- [ ] Reserved bits
```

### 3. Address Translation
```
- [ ] DPA → DRAM 변환 함수 또는 규칙
- [ ] Interleaving 방식 (있다면)
- [ ] Scrambling 여부
```

### 4. 제약사항
```
- [ ] 특정 주소 범위 제약
- [ ] Alignment 요구사항
- [ ] Access granularity
```

### 5. 검증 방법
```
- [ ] DRAM address 확인 방법
- [ ] Debug register 접근 방법
- [ ] Monitoring tool 존재 여부
```

---

## 🚦 다음 단계

### Step 1: 정보 수집 (현재 단계)
- [x] 문서 템플릿 준비
- [ ] 사용자로부터 매핑 정보 수신
- [ ] 정보 정리 및 분석

### Step 2: 이론적 검증
- [ ] 매핑 규칙 이해
- [ ] March 알고리즘 구현 가능성 판단
- [ ] 제약사항 파악

### Step 3: 실험적 검증
- [ ] Sequential access 테스트
- [ ] Row/Bank 경계 확인
- [ ] CE 발생 패턴 관찰

### Step 4: 의사결정
- [ ] devdax 타당성 최종 판단
- [ ] MBIST vs devdax 비교표 작성
- [ ] 프로젝트 방향 확정

---

**작성자**: AI Assistant
**상태**: 사용자 입력 대기
**중요도**: 🔴 CRITICAL - 프로젝트 방향 결정

---

## 💬 질문 & 답변 공간

### Q1: DPA bit layout은?
**A**: (사용자 제공 예정)

### Q2: Interleaving 방식은?
**A**: (사용자 제공 예정)

### Q3: DRAM address 확인 방법은?
**A**: (사용자 제공 예정)

---

*이 문서는 사용자가 매핑 정보를 제공하면 업데이트됩니다.*
