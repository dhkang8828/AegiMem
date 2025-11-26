# CXL Type3 메모리 불량 검출을 위한 강화학습 프로젝트

## 프로젝트 개요

삼성전자 CXL Type3 메모리 모듈(128GB CMM-D)에서 DRAM Memory Cell 불량을 검출하기 위한 **강화학습 기반** 테스트 프로그램입니다.

### 핵심 목표

**불량 검출 정확도 향상**
- 기존 데이터 없이 실제 하드웨어 테스트하며 학습 (Cold Start)
- 강화학습으로 최적의 테스트 패턴 발견
- 새로운 불량 패턴 자동 탐색

## 시스템 아키텍처

### 분산 시스템 구조

```
┌─────────────────────────────────┐
│   Windows PC (RL Agent)         │
│   - 강화학습 알고리즘 (DQN/PPO) │
│   - 정책 학습 및 의사결정       │
│   - TensorBoard 모니터링        │
└────────────┬────────────────────┘
             │ REST API
             │ (HTTP/JSON)
             ↓
┌─────────────────────────────────┐
│   GNR-SP Linux (Memory Agent)   │
│   - 128GB CXL Type3 CMM-D       │
│   - devdax 직접 제어            │
│   - CE (Correctable Error) 검출 │
│   - umxc 기반 불량 감지         │
└─────────────────────────────────┘
```

### 네트워크 설정
- **Windows PC**: `12.26.204.100` (RL Agent)
- **GNR-SP Linux**: `192.168.3.20` (Memory Agent Server)
- **통신**: REST API (Port 5000)

## 프로젝트 구조

```
cxl_memory_rl_project/
├── src/
│   ├── MemoryAgent/              # GNR-SP 리눅스에서 실행
│   │   ├── c_library/
│   │   │   ├── memory_agent.c    # devdax 제어 C 라이브러리
│   │   │   ├── Makefile
│   │   │   └── DEVDAX_MMAP_MIGRATION.md
│   │   ├── memory_agent_c_wrapper.py  # C 라이브러리 Python wrapper
│   │   ├── memory_agent_server.py     # REST API 서버
│   │   └── README.md
│   │
│   └── RLAgent/                  # Windows PC에서 실행
│       ├── phase1_environment_distributed.py  # RL 환경
│       ├── dqn_agent.py          # DQN 알고리즘
│       ├── ppo_agent.py          # PPO 알고리즘
│       └── README.md
│
├── docs/
│   ├── DPA_ADDRESS_MAPPING_EXPLAINED.md  # DPA 주소 매핑 설명
│   ├── PHASE1_IMPROVEMENTS.md            # Phase 1 개선 방안
│   └── ARCHITECTURE.md                   # 시스템 아키텍처
│
├── CLAUDE.md                     # 프로젝트 배경
└── README.md                     # 이 파일
```

## 강화학습 설계

### Action Space (1,536 actions)

**6가지 메모리 Operation × 256가지 패턴**

| Operation | 설명 |
|-----------|------|
| `write_ascending` | 주소 증가 순서로 쓰기 → 읽기 |
| `write_descending` | 주소 감소 순서로 쓰기 → 읽기 |
| `read_ascending` | 주소 증가 순서로 읽기 |
| `read_descending` | 주소 감소 순서로 읽기 |
| `write_read_ascending` | 증가 순서 쓰기 직후 읽기 |
| `write_read_descending` | 감소 순서 쓰기 직후 읽기 |

**패턴 (0x00 ~ 0xFF)**: 256가지 데이터 패턴 테스트

**Action 인코딩**:
```python
action = operation_index * 256 + pattern
# 예: action=0 → write_ascending, pattern=0x00
#     action=512 → read_ascending, pattern=0x00
```

### State Space

```python
state = {
    'ce_volatile': int,      # 일시적 CE 카운트
    'ce_persistent': int,    # 영구적 CE 카운트
    'ce_total': int,         # 총 CE 카운트
    'temperature': int,      # 메모리 온도
    'tests_done': int,       # 수행한 테스트 수
    'unique_actions': int,   # 사용한 고유 action 수
}
```

### Reward Function (Phase 1)

```python
# 1. CE 발견 (핵심!)
reward = ce_delta * 100  # 새로운 CE 발견 시 큰 보상

# 2. 탐색 보너스
if action not in used_actions:
    reward += 10  # 새로운 action 시도 시

# 3. 효율성 페널티
reward -= 1  # 각 테스트마다 작은 페널티 (빠른 발견 유도)
```

## 설치 및 실행

### 1. GNR-SP Linux (Memory Agent)

#### Prerequisites
- Linux (Ubuntu/CentOS)
- Python 3.8+
- GCC compiler
- CXL Type3 device (`/dev/dax1.0`)

#### 설치
```bash
cd /home/dhkang/cxl_memory_rl_project

# C 라이브러리 컴파일
cd src/MemoryAgent/c_library
make clean
make

# Python 의존성 설치
cd ..
pip install flask requests
```

#### 서버 실행
```bash
sudo python3 memory_agent_server.py \
    --devdax /dev/dax1.0 \
    --memory-size 128000 \
    --port 5000
```

**서버 로그 예시**:
```
============================================================
Memory Agent REST API Server
============================================================
Server: 0.0.0.0:5000
devdax: /dev/dax1.0
Memory size: 128000 MB
Test coverage: 100% (entire memory)
============================================================
Memory Agent initialized:
  Device: /dev/dax1.0
  Memory size: 128000 MB (134217728000 bytes)
```

### 2. Windows PC (RL Agent)

#### Prerequisites
- Windows 10/11
- Python 3.8+
- PyTorch (CPU/GPU)
- Network access to GNR-SP

#### 설치
```bash
# Python 패키지 설치
pip install torch numpy requests gymnasium
```

#### 실행 예시
```python
import requests

# 프록시 비활성화
session = requests.Session()
session.trust_env = False

MEMORY_AGENT_URL = "http://192.168.3.20:5000"

# Health check
response = session.get(f"{MEMORY_AGENT_URL}/health")
print(response.json())

# Action 실행
action = 0  # write_ascending, pattern=0x00
response = session.post(
    f"{MEMORY_AGENT_URL}/execute_action",
    json={'action': action},
    timeout=30
)
print(response.json())
```

## 기술적 세부사항

### devdax 2MB Alignment

CXL Type3 devdax는 **2MB alignment 요구사항**이 있습니다:
- `write()/read()` 시스템 콜 사용 불가
- **mmap()** 사용 필수
- offset은 2MB 배수여야 함

**구현 방식**:
```c
#define ALIGN_SIZE (2 * 1024 * 1024)  // 2MB

uint64_t aligned_start = (offset / ALIGN_SIZE) * ALIGN_SIZE;
void* mapped = mmap(NULL, size, PROT_READ|PROT_WRITE,
                    MAP_SHARED, fd, aligned_start);

// 매핑된 메모리 내에서는 64B 단위로 자유롭게 접근!
uint8_t* ptr = (uint8_t*)mapped + (offset - aligned_start);
```

**중요**: RL Agent는 여전히 **64B 단위**로 메모리를 제어할 수 있습니다!

### DPA (Device Physical Address) 접근

Memory Agent는 **Virtual Address를 거치지 않고 DPA를 직접 접근**합니다:

```
일반 Application:  VA → MMU → HPA → DRAM
Memory Agent:      DPA 직접 → DRAM (devdax)
```

### CE (Correctable Error) 검출

- **umxc** 도구로 CE 정보 수집
- Volatile CE와 Persistent CE 구분
- 온도 정보 포함

## API 명세

### Health Check
```
GET /health
Response: {
    "status": "healthy",
    "initialized": true
}
```

### Execute Action
```
POST /execute_action
Body: {"action": 0-1535}
Response: {
    "success": true,
    "ce_detected": false,
    "ce_volatile": 0,
    "ce_persistent": 0,
    "ce_total": 0,
    "temperature": 35,
    "operation": "write_ascending",
    "pattern": "0x00"
}
```

### Reset Baseline
```
POST /reset_baseline
Response: {"success": true}
```

### Get CE Info
```
GET /get_ce_info
Response: {
    "volatile_count": 0,
    "persistent_count": 0,
    "total_count": 0,
    "temperature": 35,
    "health_status": "HEALTHY"
}
```

## 개발 로드맵

### Phase 1: 기본 불량 검출 (현재)
- [x] 분산 아키텍처 구축
- [x] devdax mmap 방식 구현
- [x] REST API 서버 완성
- [x] 기본 RL 환경 구현
- [ ] DQN/PPO 에이전트 훈련
- [ ] 첫 번째 CE 검출 성공

### Phase 2: 성능 개선
- [ ] Shaped reward function
- [ ] Batch execution (10-30x speedup)
- [ ] Curriculum learning
- [ ] Prior knowledge injection

### Phase 3: 범용화
- [ ] Transfer learning
- [ ] Multi-device testing
- [ ] Production deployment

## 참고 문서

### 필독 문서
- **docs/DPA_ADDRESS_MAPPING_EXPLAINED.md**: DPA 주소 체계 이해
- **docs/PHASE1_IMPROVEMENTS.md**: Phase 1 개선 방안
- **CLAUDE.md**: 프로젝트 배경 및 동기

### 기술 문서
- **src/MemoryAgent/c_library/DEVDAX_MMAP_MIGRATION.md**: devdax mmap 마이그레이션 가이드
- **src/MemoryAgent/README.md**: Memory Agent 상세 설명
- **src/RLAgent/README.md**: RL Agent 상세 설명

## 문제 해결

### Windows에서 Connection Timeout
```python
# 프록시 비활성화 필요!
session = requests.Session()
session.trust_env = False
```

### GNR-SP에서 devdax EINVAL
- 2MB alignment 확인: `cat /sys/bus/dax/devices/dax1.0/align`
- mmap 사용 확인 (write()/read() 불가)

### CE 검출 안 됨
- umxc 실행 권한 확인
- baseline reset 확인
- 메모리 실제 불량 여부 확인

## 기술 스택

### GNR-SP Linux
- **OS**: Linux (Ubuntu/CentOS)
- **Language**: C, Python
- **Libraries**: Flask, ctypes
- **Hardware**: 128GB CXL Type3 CMM-D

### Windows PC
- **OS**: Windows 10/11
- **Language**: Python
- **Libraries**: PyTorch, Gymnasium, NumPy
- **Hardware**: RTX 4060 Ti (선택)

## 라이선스

이 프로젝트는 Samsung Electronics의 내부 연구용으로 개발되었습니다.

---

**프로젝트명**: AegiMem (Aegis Memory - 방패막이 메모리)
**마지막 업데이트**: 2025-11-26
**버전**: 3.0 (Distributed Architecture with devdax)
**상태**: Phase 1 개발 중
