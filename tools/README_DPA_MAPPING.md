# DPA to DRAM Address Mapping Tools

GNR-CRB 보드에서 Montage `umxc` 툴을 사용하여 DPA to DRAM address 매핑 데이터를 수집하고 분석하는 도구입니다.

## 📁 파일 구성

- **dpa_mapping_collector.py**: GNR-CRB 보드에서 실행, umxc ei -t 명령으로 매핑 데이터 수집
- **dpa_mapping_visualizer.py**: 수집된 데이터 시각화 및 분석 (로컬/보드 모두 가능)
- **README_DPA_MAPPING.md**: 이 사용 가이드
- **umxc_output**: umxc 명령어 출력 샘플 (참고용)

## 🚀 빠른 시작 가이드

### Step 1: GNR-CRB 보드로 스크립트 전송

```bash
# 로컬 개발 머신에서
cd ~/cxl_memory_rl_project/tools
scp dpa_mapping_collector.py user@gnr-crb:/tmp/
```

### Step 2: GNR-CRB 보드에서 데이터 수집

```bash
# 보드에 SSH 접속
ssh user@gnr-crb

# 기본 실행 (1MB 범위, 64B 간격, 약 16K 샘플)
python3 /tmp/dpa_mapping_collector.py --output /tmp/dpa_mapping.csv
```

**진행 상황 확인**: 10개 샘플마다 progress가 업데이트됩니다
```
Progress: 0/16384 (0.0%) - DPA: 0x0
Progress: 10/16384 (0.1%) - DPA: 0x280
Progress: 20/16384 (0.1%) - DPA: 0x500
...
```

### Step 3: 로컬로 데이터 가져오기

```bash
# 로컬 머신에서
cd ~/cxl_memory_rl_project/tools
scp user@gnr-crb:/tmp/dpa_mapping.csv ./
```

### Step 4: 데이터 시각화 및 분석

```bash
# 기본 분석 (테이블 + 패턴 분석)
python3 dpa_mapping_visualizer.py dpa_mapping.csv

# 전체 분석 (테이블 + 패턴 + 그래프)
python3 dpa_mapping_visualizer.py dpa_mapping.csv --table --analyze --plot
```

## 📊 상세 사용법

### 데이터 수집 옵션

#### 기본 사용
```bash
python3 /tmp/dpa_mapping_collector.py --output /tmp/dpa_mapping.csv
```

#### 커스텀 범위 지정
```bash
# 16MB 범위, 4KB 간격 (약 4K 샘플, 빠름)
python3 /tmp/dpa_mapping_collector.py \
    --start 0x0 \
    --end 0x1000000 \
    --step 0x1000 \
    --output /tmp/dpa_mapping_16mb.csv

# 256MB 범위, 64KB 간격 (약 4K 샘플, 더 넓은 범위)
python3 /tmp/dpa_mapping_collector.py \
    --start 0x0 \
    --end 0x10000000 \
    --step 0x10000 \
    --output /tmp/dpa_mapping_256mb.csv
```

#### JSON 형식으로도 저장
```bash
python3 /tmp/dpa_mapping_collector.py \
    --output /tmp/dpa_mapping.csv \
    --json /tmp/dpa_mapping.json
```

#### umxc 경로 지정 (PATH에 없는 경우)
```bash
python3 /tmp/dpa_mapping_collector.py \
    --umxc /usr/local/bin/umxc \
    --output /tmp/dpa_mapping.csv
```

### 주요 옵션

- `--start 0x0`: 시작 DPA 주소 (기본값: 0x0)
- `--end`: 종료 DPA 주소 (기본값: 0x100000 = 1MB)
- `--step`: 샘플링 간격 (기본값: 0x40 = 64B)
- `--output`: 출력 CSV 파일명
- `--json`: JSON 파일로도 저장
- `--umxc`: umxc 실행 파일 경로 (PATH에 없는 경우)
- `--quiet`: 진행 상황 출력 안 함

### 시각화 옵션

```bash
# 기본: 테이블 + 패턴 분석
python3 dpa_mapping_visualizer.py dpa_mapping.csv

# 테이블만 출력 (최대 행 수 지정)
python3 dpa_mapping_visualizer.py dpa_mapping.csv --table --max-rows 100

# 패턴 분석만
python3 dpa_mapping_visualizer.py dpa_mapping.csv --analyze

# 그래프 생성 (matplotlib 필요)
python3 dpa_mapping_visualizer.py dpa_mapping.csv --plot

# 전체 (테이블 + 패턴 + 그래프)
python3 dpa_mapping_visualizer.py dpa_mapping.csv --table --analyze --plot
```

## 📈 출력 예시

### 1. CSV 파일 형식

수집된 데이터는 다음과 같은 CSV 형식으로 저장됩니다:

```csv
dpa,subch,dimm,rank,bg,ba,row,col
0,0,0,0,0,0,0,0
64,0,1,0,0,0,0,0
128,0,0,0,0,0,0,1
4096,0,0,0,0,0,0,64
...
```

### 2. 테이블 출력

```
DPA to DRAM Address Mapping Table
====================================================================================================
         DPA | SubCh | DIMM | Rank | BG | BA |      Row |    Col
----------------------------------------------------------------------------------------------------
0x0000000000 |     0 |    0 |    0 |  0 |  0 | 0x000000 | 0x0000
0x0000000040 |     0 |    1 |    0 |  0 |  0 | 0x000000 | 0x0000
0x0000000080 |     0 |    0 |    0 |  0 |  0 | 0x000000 | 0x0001
0x0000001000 |     0 |    0 |    0 |  0 |  0 | 0x000000 | 0x0040
...
```

### 3. 패턴 분석 결과

```
Mapping Pattern Analysis
================================================================================

1. DPA Increment Analysis
----------------------------------------
  DIMM 0→1              → DPA +0x40 (64 bytes)
  Col 0x0→0x1           → DPA +0x80 (128 bytes)
  Col 0x0→0x40          → DPA +0x1000 (4096 bytes)
  Row 0x0→0x1           → DPA +0x100000 (1048576 bytes)
  BA 0→1                → DPA +0x4000000 (64 MB)
  BG 0→1                → DPA +0x10000000 (256 MB)

2. Interleaving Pattern
----------------------------------------
  Subchannels: [0] (count: 1)
  DIMMs: [0, 1] (count: 2)
  Ranks: [0, 1] (count: 2)
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

### 4. 그래프 출력 (--plot 옵션 시)

matplotlib를 설치하면 다음 6개 그래프가 생성됩니다:
- DIMM interleaving pattern
- Rank distribution
- Bank Group interleaving
- Bank Address distribution
- Row address progression
- Column address progression

그래프는 `dpa_mapping_plot.png` 파일로 저장됩니다.

## 🎯 데이터 활용

수집된 매핑 데이터는 다음 용도로 사용됩니다:

1. **역변환 함수 구현**: DRAM address → DPA 변환 함수 개발
   - `src/dpa_translator.py`에서 사용
   - March 알고리즘 구현에 필수

2. **devdax 타당성 검증**: Sequential DPA 접근이 의미있는 DRAM 패턴을 만드는지 확인

3. **Phase1Environment 통합**: 역변환 함수를 활용한 정밀 DRAM cell 접근

4. **메모리 테스트 전략 최적화**: Interleaving 패턴을 고려한 효율적인 테스트 설계

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

### 수집 진행 상황 확인 안 될 때

최신 버전(69d7ec2)은 10개 샘플마다 progress를 출력합니다. 만약 진행 상황이 보이지 않는다면:

1. **스크립트 버전 확인**: 최신 버전을 보드에 다시 전송했는지 확인
2. **umxc 실행 시간**: umxc 명령어가 느릴 수 있음 (각 샘플당 수 초 소요 가능)
3. **출력 버퍼링**: `--quiet` 옵션 없이 실행했는지 확인

## 💡 팁

### 효율적인 샘플링 전략

**목적에 따른 권장 설정:**

```bash
# 빠른 패턴 파악 (5-10분)
--end 0x1000000 --step 0x10000    # 16MB, 64KB 간격, ~256 샘플

# 상세한 매핑 분석 (30분-1시간)
--end 0x10000000 --step 0x1000    # 256MB, 4KB 간격, ~64K 샘플

# 초정밀 분석 (시간이 매우 오래 걸림)
--end 0x100000 --step 0x40        # 1MB, 64B 간격, ~16K 샘플
```

### 수집 중 중단 시

Ctrl+C로 중단해도 괜찮습니다. 이미 수집된 데이터는 보존되지 않으므로, 다시 시작해야 합니다.

### 데이터 백업

```bash
# 여러 범위의 데이터를 수집하여 비교
scp user@gnr-crb:/tmp/dpa_mapping_*.csv ~/cxl_memory_rl_project/data/
```

## 📚 참고

- Montage umxc 툴 매뉴얼
- CXL 2.0/3.0 Specification - Address Translation
- DRAM addressing 구조 (rank, bank group, bank, row, column)
- DDR4/DDR5 Memory Architecture
