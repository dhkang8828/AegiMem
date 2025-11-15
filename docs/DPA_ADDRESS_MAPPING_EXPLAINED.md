# CXL 메모리 주소 매핑 완전 가이드

## 📌 목차

1. [왜 주소 매핑이 중요한가?](#왜-주소-매핑이-중요한가)
2. [기본 개념: 3가지 주소 공간](#기본-개념-3가지-주소-공간)
3. [DPA-to-DRAM Address 변환](#dpa-to-dram-address-변환)
4. [devdax를 통한 메모리 접근](#devdax를-통한-메모리-접근)
5. [실제 예시](#실제-예시)
6. [왜 RL 프로젝트에 필요한가?](#왜-rl-프로젝트에-필요한가)

---

## 왜 주소 매핑이 중요한가?

### 문제 상황

**CXL 메모리 불량 검출을 위해 특정 DRAM 셀을 테스트해야 합니다.**

하지만:
- 우리가 사용하는 인터페이스: **devdax** (리눅스 OS)
- devdax는 **DPA (Device Physical Address)** 사용
- 실제 하드웨어: **DRAM address** (Rank, BG, BA, Row, Col)

**즉, OS 주소 ↔ DRAM 주소 변환이 필수!**

### 왜 필요한가?

✅ **March 알고리즘 구현**
- March 알고리즘: Row 0 → Row 1 → ... 순차 접근
- 하지만 devdax는 DPA만 이해
- **DPA ↔ DRAM 변환 없이는 구현 불가능**

✅ **정확한 테스트 범위 설정**
- 특정 Bank Group만 테스트하고 싶다면?
- 특정 Row 범위만 테스트하고 싶다면?
- **DRAM address를 알아야 가능**

✅ **불량 위치 추적**
- 불량 발견 시 정확한 DRAM 셀 위치 파악
- 제조 공정 분석에 필수

---

## 기본 개념: 3가지 주소 공간

### 1️⃣ Virtual Address (VA) - 응용 프로그램 관점

```
응용 프로그램이 보는 주소
예: ptr = malloc(1024); → VA: 0x7FFE12345678
```

- 프로세스마다 독립적
- MMU가 Physical Address로 변환

### 2️⃣ Device Physical Address (DPA) - OS/devdax 관점

```
CXL 장치의 물리 주소
예: /dev/dax0.0의 주소: DPA: 0x100000
```

- CXL 컨트롤러가 이해하는 주소
- devdax를 통해 직접 접근 가능
- **우리가 코드에서 사용하는 주소**

### 3️⃣ DRAM Address - 하드웨어 관점

```
실제 DRAM 칩 내부 주소
예: Rank=0, BG=2, BA=1, Row=0x1234, Col=0x56
```

- DRAM 물리적 구조 그대로
- 메모리 컨트롤러가 사용
- **불량 분석에 필요한 주소**

---

## DPA-to-DRAM Address 변환

### DRAM 주소 구조 (128GB CMM-D 기준)

```
┌─────────────────────────────────────────────────┐
│  DRAM Address                                   │
├─────────────────────────────────────────────────┤
│  Subchannel (2)  │  DIMM (2)  │  Rank (1)      │
│  BG (8)  │  BA (4)  │  Row (가변)  │  Col (2048) │
└─────────────────────────────────────────────────┘
```

**각 필드 설명:**
- **Subchannel**: 서브채널 번호 (0-1)
- **DIMM**: DIMM 모듈 번호 (0-1)
- **Rank**: Rank 번호 (고정: 0)
- **BG (Bank Group)**: 뱅크 그룹 (0-7)
- **BA (Bank Address)**: 뱅크 주소 (0-3)
- **Row**: 행 주소 (0-0x1FFFF)
- **Col**: 열 주소 (0x0, 0x10, 0x20, ... 0x7F0) - **인코딩됨!**

### 매핑 계층 구조

**DPA는 계층적으로 매핑됩니다:**

```
     DPA 0x12345678
         │
    ┌────▼────────────────────────────┐
    │ Layer 6: Row (1MB per row)      │
    │   → Row = DPA / 0x100000        │
    └────┬────────────────────────────┘
         │
    ┌────▼────────────────────────────┐
    │ Layer 5: SubCh (512KB)          │
    │   → SubCh = remainder / 0x80000 │
    └────┬────────────────────────────┘
         │
    ┌────▼────────────────────────────┐
    │ Layer 4: BA (128KB)             │
    │   → BA = remainder / 0x20000    │
    └────┬────────────────────────────┘
         │
    ┌────▼────────────────────────────┐
    │ Layer 3: Col (1KB, 인코딩!)     │
    │   → Col = (remainder/0x400)*0x10│
    └────┬────────────────────────────┘
         │
    ┌────▼────────────────────────────┐
    │ Layer 2: BG (128B)              │
    │   → BG = remainder / 0x80       │
    └────┬────────────────────────────┘
         │
    ┌────▼────────────────────────────┐
    │ Layer 1: DIMM (64B)             │
    │   → DIMM = remainder / 0x40     │
    └─────────────────────────────────┘
```

### 변환 공식

#### Forward: DPA → DRAM

```python
row   = dpa // 0x100000
temp  = dpa % 0x100000

subch = temp // 0x80000
temp  = temp % 0x80000

ba    = temp // 0x20000
temp  = temp % 0x20000

col   = (temp // 0x400) * 0x10  # ⚠️ 인코딩 주의!
temp  = temp % 0x400

bg    = temp // 0x80
temp  = temp % 0x80

dimm  = temp // 0x40
rank  = 0  # 고정
```

#### Reverse: DRAM → DPA

```python
dpa = row * 0x100000         # Layer 6: Row
    + subch * 0x80000        # Layer 5: Subchannel
    + ba * 0x20000           # Layer 4: BA
    + (col // 0x10) * 0x400  # Layer 3: Col (디코딩!)
    + bg * 0x80              # Layer 2: BG
    + dimm * 0x40            # Layer 1: DIMM
```

### ⚠️ Column Address 인코딩

**중요: Column 주소는 특별히 인코딩됩니다!**

```
DPA 증가    →  Column 값
0x000           0x0
0x400 (1KB)     0x10
0x800 (2KB)     0x20
0xC00 (3KB)     0x30
...
0x1FC00         0x7F0
```

**규칙:**
- DPA가 1KB (0x400) 증가할 때마다
- Column은 0x10씩 증가

**왜 이렇게?**
- DRAM 내부 주소 인코딩 방식
- Burst length와 관련
- 우리는 알고리즘으로 처리하면 됨

---

## devdax를 통한 메모리 접근

### devdax란?

**devdax (Direct Access):**
- 리눅스 커널 드라이버
- CXL 메모리를 파일처럼 접근
- 경로: `/dev/dax0.0`, `/dev/dax0.1`, ...

```bash
# devdax 확인
ls -lh /dev/dax*
# crw------- 1 root root 241, 0 Nov 15 09:00 /dev/dax0.0
```

### 메모리 접근 흐름

```
┌──────────────────┐
│  Python Code     │
│  write(dpa, data)│
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│  devdax Driver   │  ← DPA 주소 사용
│  /dev/dax0.0     │
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│  CXL Controller  │  ← DPA → DRAM 변환
│  (MTG Chip)      │
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│  DRAM Chip       │  ← DRAM Address로 접근
│  (삼성 CMM-D)    │     (Rank,BG,BA,Row,Col)
└──────────────────┘
```

### 실제 코드 예시

```python
import os

# devdax 열기
fd = os.open('/dev/dax0.0', os.O_RDWR | os.O_SYNC)

# DPA 0x100000에 64바이트 쓰기
dpa = 0x100000
data = b'\xAA' * 64

os.lseek(fd, dpa, os.SEEK_SET)
os.write(fd, data)

# 같은 위치에서 읽기
os.lseek(fd, dpa, os.SEEK_SET)
result = os.read(fd, 64)

os.close(fd)
```

**이 코드가 실제로 하는 일:**
1. DPA 0x100000 접근
2. CXL 컨트롤러가 DPA → DRAM 변환
3. DRAM 주소: Row=1, SubCh=0, BA=0, BG=0, Col=0x0, DIMM=0
4. 실제 DRAM 셀에 쓰기/읽기

### 주소 정렬 (Alignment)

**중요: devdax는 64바이트 정렬 필수!**

```python
# ✓ OK: 64바이트 정렬
dpa = 0x0, 0x40, 0x80, 0xC0, 0x100, ...

# ✗ ERROR: 정렬 안 됨
dpa = 0x1, 0x32, 0x77, ...
```

**이유:**
- Cache line 크기 = 64 바이트
- DIMM 인터리빙 단위 = 64 바이트
- 하드웨어 제약

---

## 실제 예시

### 예시 1: 간단한 DPA

**DPA: 0x0**

```
Forward Translation:
row   = 0x0 / 0x100000 = 0
subch = 0
ba    = 0
col   = 0
bg    = 0
dimm  = 0
rank  = 0

→ DRAM Address: (0, 0, 0, 0, 0x0, 0, 0)
```

### 예시 2: Row 증가

**DPA: 0x100000 (1MB)**

```
Forward Translation:
row   = 0x100000 / 0x100000 = 1  ← Row 증가!
subch = 0
ba    = 0
col   = 0
bg    = 0
dimm  = 0

→ DRAM Address: (0, 0, 0, 1, 0x0, 0, 0)
```

### 예시 3: 복잡한 주소

**DPA: 0x7416F4C0**

```
1. Row 추출:
   row = 0x7416F4C0 / 0x100000 = 0x741
   나머지 = 0x6F4C0

2. Subchannel 추출:
   subch = 0x6F4C0 / 0x80000 = 0
   나머지 = 0x6F4C0

3. BA 추출:
   ba = 0x6F4C0 / 0x20000 = 3
   나머지 = 0xF4C0

4. Column 추출 (인코딩!):
   col = (0xF4C0 / 0x400) * 0x10 = 0x3D * 0x10 = 0x3D0
   나머지 = 0xC0

5. BG 추출:
   bg = 0xC0 / 0x80 = 1
   나머지 = 0x40

6. DIMM 추출:
   dimm = 0x40 / 0x40 = 1

→ DRAM Address:
  SubCh=0, DIMM=1, Rank=0, BG=1, BA=3, Row=0x741, Col=0x3D0
```

**검증 (Reverse):**
```
dpa = 0x741 * 0x100000     = 0x74100000
    + 0 * 0x80000          = 0x0
    + 3 * 0x20000          = 0x60000
    + (0x3D0/0x10) * 0x400 = 0xF400
    + 1 * 0x80             = 0x80
    + 1 * 0x40             = 0x40
    ────────────────────────────────
    = 0x7416F4C0 ✓
```

### 예시 4: March 알고리즘 - Row 순차 접근

**목표: Row 0 → Row 1 → Row 2 순서로 접근**

```python
from dpa_translator import DPATranslator

translator = DPATranslator(mock_mode=True)

# Row 0, 1, 2에 접근하고 싶음
# 다른 필드는 모두 0
for row in range(3):
    dpa = translator.dram_to_dpa(
        rank=0, bg=0, ba=0,
        row=row, col=0,
        dimm=0, subch=0
    )
    print(f"Row {row} → DPA 0x{dpa:X}")

    # devdax로 실제 접근
    access_memory(dpa, pattern)

# 출력:
# Row 0 → DPA 0x0
# Row 1 → DPA 0x100000
# Row 2 → DPA 0x200000
```

**핵심:**
- DPA translator 없이는 불가능!
- Row 순차 접근 = DPA 1MB씩 증가

---

## 왜 RL 프로젝트에 필요한가?

### 1. March 알고리즘 구현 필수

**March C-:**
```
⇑(w0); ⇑(r0,w1); ⇑(r1,w0); ⇓(r0,w1); ⇓(r1,w0); ⇓(r0)
```

- `⇑`: Row 증가 방향 순차 접근
- `⇓`: Row 감소 방향 순차 접근

**DPA translator 없으면?**
- Row 순서를 알 수 없음
- March 알고리즘 구현 불가능
- 단순 랜덤 접근만 가능

**DPA translator 있으면?**
```python
# Row 증가 순서
for row in range(max_row):
    dpa = dram_to_dpa(..., row=row, ...)
    test_memory(dpa)

# Row 감소 순서
for row in range(max_row, -1, -1):
    dpa = dram_to_dpa(..., row=row, ...)
    test_memory(dpa)
```

### 2. RL Agent의 State 정의

**State에 DRAM 정보 포함:**
```python
state = {
    'current_row': 0x1234,
    'current_bg': 2,
    'current_ba': 1,
    'tested_cells': 50000,
    'ce_count': 5
}
```

- DPA만으로는 "어느 Row인지" 알 수 없음
- DRAM address를 알아야 의미있는 state 생성

### 3. Action Space 정의

**의미있는 Action:**
```python
actions = [
    "test_next_row",      # Row + 1
    "test_next_bg",       # BG + 1
    "test_same_col",      # 같은 Col, 다른 Row
    "test_random_cell"    # 랜덤
]
```

- DPA만으로는 "다음 Row"의 의미를 정의 불가
- DRAM address 이해 필수

### 4. Reward Function

**정밀한 Reward 계산:**
```python
if new_ce_detected:
    if same_row_as_previous_ce:
        reward = +10  # Row 집중 탐색 성공
    elif same_bg_as_previous_ce:
        reward = +5   # BG 집중 탐색
    else:
        reward = +1   # 새로운 영역 발견
```

- 불량 셀의 상관관계 분석
- "같은 Row", "같은 BG" 판단 필요
- DPA만으로는 불가능

### 5. 제조 공정 피드백

**불량 위치 정확한 보고:**
```
불량 발견!
DPA: 0x7416F4C0
  ↓ 변환
DRAM: Row=0x741, BG=1, BA=3, DIMM=1
  ↓ 분석
"Row 0x741 주변에 불량 집중"
  ↓ 제조팀 피드백
"특정 Row 제조 공정 문제 가능성"
```

---

## 정리

### DPA-to-DRAM Address 변환이 중요한 이유

1. ✅ **March 알고리즘 구현** - 순차 접근 필수
2. ✅ **RL State/Action 정의** - 의미있는 상태/행동
3. ✅ **정확한 불량 위치 파악** - 제조 공정 피드백
4. ✅ **테스트 범위 제어** - 특정 영역 집중 테스트

### devdax의 역할

1. ✅ **직접 메모리 접근** - OS 우회, 빠른 속도
2. ✅ **DPA 주소 사용** - 커널이 관리
3. ✅ **64바이트 단위** - Cache line 정렬

### 우리 프로젝트의 핵심

```
devdax (DPA) ←→ DPA Translator ←→ DRAM Address
     ↓                                  ↓
  OS Level                         Hardware Level
  우리 코드                        실제 메모리 셀
```

**DPA Translator = 양쪽 세계의 다리!**

---

## 참고 자료

- **구현**: `src/dpa_translator.py`
- **매핑 규칙**: `docs/DPA_MAPPING_RULES.md`
- **검증 도구**: `tools/validate_dpa_translation.py`
- **테스트**: `tests/test_dpa_translator.py`

---

**작성일**: 2024-11-15
**검증 상태**: umxc 기반 140,000 샘플 검증 중
