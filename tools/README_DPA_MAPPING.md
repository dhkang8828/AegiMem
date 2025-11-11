# DPA to DRAM Address Mapping Tools

GNR-CRB 보드에서 Montage `umxc` 툴을 사용하여 DPA to DRAM address 매핑 데이터를 수집하고 분석하는 도구입니다.

## 파일 구성

- **dpa_mapping_collector.py**: GNR-CRB 보드에서 실행, umxc로 매핑 데이터 수집
- **dpa_mapping_visualizer.py**: 수집된 데이터 시각화 및 분석 (로컬/보드 모두 가능)
- **README_DPA_MAPPING.md**: 이 문서

## 사용 방법

### Step 1: GNR-CRB 보드로 스크립트 전송

```bash
# 로컬 개발 머신에서
cd /home/dhkang/cxl_memory_rl_project/tools
scp dpa_mapping_collector.py user@gnr-crb:/tmp/
```

### Step 2: GNR-CRB 보드에서 데이터 수집

```bash
# GNR-CRB 보드에 SSH 접속
ssh user@gnr-crb

# 기본 실행 (1MB 범위, 64B 간격)
python3 /tmp/dpa_mapping_collector.py --output /tmp/dpa_mapping.csv

# 커스텀 범위 지정
python3 /tmp/dpa_mapping_collector.py \
    --start 0x0 \
    --end 0x10000000 \
    --step 0x1000 \
    --output /tmp/dpa_mapping_16mb.csv

# JSON으로도 저장
python3 /tmp/dpa_mapping_collector.py \
    --output /tmp/dpa_mapping.csv \
    --json /tmp/dpa_mapping.json

# umxc 경로 지정 (PATH에 없는 경우)
python3 /tmp/dpa_mapping_collector.py \
    --umxc /path/to/umxc \
    --output /tmp/dpa_mapping.csv
```

#### 주요 옵션

- `--start`: 시작 DPA 주소 (기본값: 0x0)
- `--end`: 종료 DPA 주소 (기본값: 0x100000 = 1MB)
- `--step`: 샘플링 간격 (기본값: 0x40 = 64B)
- `--output`: 출력 CSV 파일명
- `--json`: JSON 파일로도 저장
- `--umxc`: umxc 실행 파일 경로
- `--quiet`: 진행 상황 출력 안 함

#### 권장 샘플링 범위

```bash
# 1. 빠른 테스트 (1MB, 64B 간격) - ~16K 샘플
python3 dpa_mapping_collector.py --end 0x100000 --step 0x40

# 2. 중간 범위 (256MB, 4KB 간격) - ~64K 샘플
python3 dpa_mapping_collector.py --end 0x10000000 --step 0x1000

# 3. 전체 범위 (전체 메모리, 64KB 간격) - 매우 많은 샘플
# 주의: CXL 메모리 크기에 따라 조정 필요
python3 dpa_mapping_collector.py --end 0x400000000 --step 0x10000
```

### Step 3: 로컬로 데이터 다운로드

```bash
# 로컬 개발 머신에서
cd /home/dhkang/cxl_memory_rl_project/tools
scp user@gnr-crb:/tmp/dpa_mapping.csv ./
```

### Step 4: 데이터 분석 및 시각화

```bash
# 로컬 개발 머신에서

# 1. 테이블 출력 + 패턴 분석
python3 dpa_mapping_visualizer.py dpa_mapping.csv

# 2. 테이블만 출력 (최대 100행)
python3 dpa_mapping_visualizer.py dpa_mapping.csv --table --max-rows 100

# 3. 패턴 분석만
python3 dpa_mapping_visualizer.py dpa_mapping.csv --analyze

# 4. 그래프 생성 (matplotlib 필요)
python3 dpa_mapping_visualizer.py dpa_mapping.csv --plot

# 5. 전체 분석 (테이블 + 패턴 + 그래프)
python3 dpa_mapping_visualizer.py dpa_mapping.csv --table --analyze --plot
```

## 출력 예시

### CSV 파일 형식

```csv
dpa,subch,dimm,rank,bg,ba,row,col
0,0,0,0,0,0,0,0
64,0,1,0,0,0,0,0
4096,0,0,0,0,0,0,64
...
```

### 분석 결과 예시

```
Mapping Pattern Analysis
================================================================================

1. DPA Increment Analysis
----------------------------------------
  DIMM 0→1                                           → DPA +0x40 (64 bytes)
  Col 0x0→0x40                                       → DPA +0x1000 (4096 bytes)
  Row 0x0→0x1                                        → DPA +0x100000 (1048576 bytes)

2. Interleaving Pattern
----------------------------------------
  Subchannels: [0] (count: 1)
  DIMMs: [0, 1] (count: 2)
  Ranks: [0] (count: 1)
  Bank Groups: [0, 1, 2, 3] (count: 4)
  Banks: [0, 1, 2, 3] (count: 4)

3. Address Ranges
----------------------------------------
  Max DPA: 0xffffff0 (0.25 GB)
  Max Row: 0x7fff (32767)
  Max Col: 0x3ff (1023)

4. Inferred Mapping Rules
----------------------------------------
  DIMM interleaving granularity: 64 bytes (0x40)
  Column increment: 4096 bytes (0x1000)
  Row increment: 1048576 bytes (0x100000)
```

## 다음 단계

수집된 매핑 데이터를 분석한 후:

1. **역변환 함수 구현**: DRAM address → DPA 변환 함수 추론
2. **devdax 타당성 검증**: Sequential DPA가 의미있는 DRAM 패턴을 만드는지 확인
3. **March 알고리즘 가능성**: Ascending/Descending DRAM access가 DPA로 가능한지 검증
4. **Phase1Environment 통합**: 역변환 함수를 활용한 정밀 DRAM cell 접근

## 트러블슈팅

### umxc 명령어 오류
```bash
# umxc가 PATH에 없는 경우
which umxc
# 없으면 --umxc 옵션으로 절대 경로 지정
```

### Python 버전 확인
```bash
python3 --version  # Python 3.6 이상 필요
```

### matplotlib 설치 (그래프 기능 사용 시)
```bash
pip3 install matplotlib
# 또는
pip3 install matplotlib --user
```

### SSH 키 설정 (비밀번호 없이 접속)
```bash
# 로컬에서
ssh-keygen -t rsa
ssh-copy-id user@gnr-crb
```

## 참고

- Montage umxc 툴 매뉴얼
- CXL Type3 메모리 사양
- DRAM addressing 구조 (rank, bank group, bank, row, column)
