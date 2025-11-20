# AegiMem Distributed Architecture

## Overview

AegiMem 프로젝트는 **2개의 독립적인 Agent**로 구성됩니다:

1. **RL Agent**: Windows PC에서 강화학습 수행
2. **Memory Agent**: GNR-SP (Linux)에서 실제 메모리 테스트 수행

## Architecture Diagram

```
┌──────────────────────────────────────────────────────┐
│  Windows PC (RL Agent)                               │
│  ─────────────────────────────────────               │
│                                                      │
│  src/RLAgent/                                        │
│  ├── phase1_environment_distributed.py              │
│  │   └─ Gym environment (REST API client)           │
│  │                                                   │
│  ├── dqn_agent.py / ppo_agent.py                    │
│  │   └─ 강화학습 알고리즘                             │
│  │                                                   │
│  └── config_loader.py                                │
│      └─ 설정 관리                                     │
│                                                      │
└───────────────────┬──────────────────────────────────┘
                    │
                    │ REST API (HTTP/JSON)
                    │ http://192.168.3.20:5000
                    │
                    ▼
┌──────────────────────────────────────────────────────┐
│  GNR-SP Linux (Memory Agent)                         │
│  ─────────────────────────────────────               │
│                                                      │
│  src/MemoryAgent/                                    │
│  ├── memory_agent_server.py                         │
│  │   └─ Flask REST API Server                       │
│  │                                                   │
│  ├── memory_agent_c_wrapper.py                      │
│  │   └─ Python ctypes wrapper                       │
│  │                                                   │
│  ├── c_library/                                      │
│  │   ├── libmemory_agent.so                         │
│  │   └─ C library for performance                   │
│  │       ├─ devdax I/O (/dev/dax0.0)                │
│  │       └─ umxc CE detection                       │
│  │                                                   │
│  ├── dpa_translator.py                               │
│  │   └─ DPA ↔ DRAM address conversion              │
│  │                                                   │
│  └── ce_monitor.py                                   │
│      └─ CE 모니터링                                   │
│                                                      │
└───────────────────┬──────────────────────────────────┘
                    │
                    ▼
            [CXL Memory Device]
            128GB CMM-D (DDR5)
```

## Component Details

### 1. RL Agent (Windows PC)

**위치**: `src/RLAgent/`

**역할**:
- 강화학습 알고리즘 수행 (DQN, PPO)
- Action 선택 및 REST API로 전송
- Reward 계산 및 학습
- 모델 저장/로드

**주요 파일**:

#### phase1_environment_distributed.py
```python
class Phase1EnvironmentDistributed(gym.Env):
    """
    Phase#1 RL Environment

    - Action Space: 1536 discrete (6 operations × 256 patterns)
    - REST API로 Memory Agent와 통신
    - CE 발생 여부로 reward 계산
    """

    def __init__(self, memory_agent_url="http://192.168.3.20:5000"):
        self.memory_agent_url = memory_agent_url
        self.action_space = gym.spaces.Discrete(1536)

    def step(self, action):
        # REST API로 action 전송
        response = requests.post(
            f"{self.memory_agent_url}/execute_action",
            json={'action': action}
        )

        # CE 발생 여부로 reward 계산
        result = response.json()
        reward = +10 if result['ce_detected'] else -1

        return observation, reward, done, info
```

#### dqn_agent.py / ppo_agent.py
```python
class DQNAgent:
    """Deep Q-Network Agent for Phase#1"""

    def __init__(self, state_dim, action_dim=1536):
        self.q_network = QNetwork(state_dim, action_dim)
        self.target_network = QNetwork(state_dim, action_dim)
        self.replay_buffer = ReplayBuffer(max_size=100000)

    def select_action(self, state, epsilon=0.1):
        """Epsilon-greedy action selection"""
        if random.random() < epsilon:
            return random.randint(0, 1535)
        return self.q_network(state).argmax().item()

    def train_step(self, batch_size=64):
        """Experience replay로 학습"""
        batch = self.replay_buffer.sample(batch_size)
        # Q-learning update
        ...
```

### 2. Memory Agent (GNR-SP Linux)

**위치**: `src/MemoryAgent/`

**역할**:
- REST API 서버 제공
- devdax를 통한 메모리 접근
- umxc로 CE 감지
- Action 실행 및 결과 반환

**주요 파일**:

#### memory_agent_server.py
```python
from flask import Flask, request, jsonify
from memory_agent_c_wrapper import MemoryAgentC

app = Flask(__name__)
memory_agent = MemoryAgentC()

@app.route('/execute_action', methods=['POST'])
def execute_action():
    """
    Action 실행

    Request: {"action": 0-1535}
    Response: {
        "success": true,
        "ce_detected": bool,
        "ce_total": int,
        "temperature": int
    }
    """
    action = request.json['action']
    ce_info, success = memory_agent.execute_action(action)

    return jsonify({
        'success': success,
        'ce_detected': ce_info.has_errors(),
        'ce_total': ce_info.total_count,
        'temperature': ce_info.temperature
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
```

#### memory_agent_c_wrapper.py
```python
from ctypes import CDLL, Structure, c_uint64, c_int

class CEInfo(Structure):
    _fields_ = [
        ('volatile_count', c_int),
        ('persistent_count', c_int),
        ('total_count', c_int),
        ('temperature', c_int),
        ('health_status', c_int)
    ]

class MemoryAgentC:
    """C library wrapper using ctypes"""

    def __init__(self, library_path='c_library/libmemory_agent.so'):
        self.lib = CDLL(library_path)
        self._setup_function_signatures()

    def execute_action(self, action: int):
        """Execute memory test action via C library"""
        result = ActionResult()
        ret = self.lib.ma_execute_action(action, byref(result))
        return result.ce_info, (ret == 0)
```

#### c_library/memory_agent.c
```c
// C library for high-performance memory operations

int ma_execute_action(int action, ActionResult* result) {
    // Decode action
    int operation_type = action / 256;
    int pattern = action % 256;

    // Execute operation via devdax
    switch (operation_type) {
        case WR_ASC_ASC:
            write_ascending(pattern);
            read_ascending(pattern);
            break;
        // ... other operations
    }

    // Detect CE via umxc
    execute_umxc(&result->ce_info);

    return 0;
}
```

## Communication Protocol

### REST API Endpoints

**Base URL**: `http://192.168.3.20:5000`

#### 1. Health Check
```
GET /health

Response:
{
    "status": "healthy",
    "initialized": true
}
```

#### 2. Execute Action
```
POST /execute_action

Request:
{
    "action": 0-1535  // operation_type * 256 + pattern
}

Response:
{
    "success": true,
    "ce_detected": false,
    "ce_volatile": 0,
    "ce_persistent": 0,
    "ce_total": 0,
    "temperature": 42,
    "operation": "WR_ASC_ASC",
    "pattern": "0xAA"
}
```

#### 3. Reset CE Baseline
```
POST /reset_baseline

Response:
{
    "success": true,
    "message": "Baseline reset successfully"
}
```

#### 4. Get CE Info
```
GET /get_ce_info

Response:
{
    "volatile_count": 0,
    "persistent_count": 0,
    "total_count": 0,
    "temperature": 42,
    "health_status": 0
}
```

## Action Space (Phase#1)

**Total: 1536 actions** (6 operations × 256 patterns)

### 6 Operation Types

| Operation | Code | Description |
|-----------|------|-------------|
| WR_ASC_ASC | 0 | Write ascending → Read ascending |
| WR_DESC_DESC | 1 | Write descending → Read descending |
| WR_ASC_DESC | 2 | Write ascending → Read descending |
| WR_DESC_ASC | 3 | Write descending → Read ascending |
| WR_DESC_SINGLE | 4 | Write+Read descending (single pass) |
| WR_ASC_SINGLE | 5 | Write+Read ascending (single pass) |

### 256 Data Patterns

- 0x00 ~ 0xFF (모든 8비트 패턴)

### Action Encoding

```
action = operation_type * 256 + pattern

예시:
- action 0 = WR_ASC_ASC with pattern 0x00
- action 255 = WR_ASC_ASC with pattern 0xFF
- action 256 = WR_DESC_DESC with pattern 0x00
- action 1535 = WR_ASC_SINGLE with pattern 0xFF
```

## Deployment

### Memory Agent (GNR-SP)

```bash
# 1. C library 컴파일
cd src/MemoryAgent/c_library
make clean && make

# 2. 서버 실행 (root 권한 필요)
sudo python3 src/MemoryAgent/memory_agent_server.py \
    --devdax /dev/dax0.0 \
    --memory-size 128000 \
    --port 5000
```

### RL Agent (Windows PC)

```bash
# 1. 환경 설정
cd C:\path\to\cxl_memory_rl_project

# 2. Training 실행
python src\RLAgent\train_phase1.py \
    --memory-agent-url http://192.168.3.20:5000 \
    --algorithm dqn \
    --episodes 1000
```

## Network Configuration

- **Windows PC IP**: 자동 (DHCP)
- **GNR-SP IP**: 192.168.3.20 (고정)
- **Port**: 5000 (Flask)
- **Protocol**: HTTP (REST API)

## Data Flow

```
1. RL Agent: Action 선택 (0-1535)
   ↓
2. REST API: POST /execute_action {"action": 123}
   ↓
3. Memory Agent: Action 실행
   - C library 호출
   - devdax 메모리 접근
   - umxc CE 감지
   ↓
4. REST API: Response {ce_detected: true, ...}
   ↓
5. RL Agent: Reward 계산 및 학습
   - CE detected: +10 reward
   - No CE: -1 reward
```

## File Organization

```
src/
├── MemoryAgent/          # GNR-SP에서 실행
│   ├── c_library/
│   │   ├── memory_agent.c
│   │   ├── libmemory_agent.so
│   │   └── Makefile
│   ├── memory_agent_server.py
│   ├── memory_agent_c_wrapper.py
│   ├── dpa_translator.py
│   ├── ce_monitor.py
│   ├── devdax_interface.py
│   ├── __init__.py
│   └── README.md
│
└── RLAgent/              # Windows PC에서 실행
    ├── phase1_environment_distributed.py
    ├── dqn_agent.py
    ├── ppo_agent.py
    ├── config_loader.py
    ├── dpa_translator.py
    ├── __init__.py
    └── README.md
```

---

**Last Updated**: 2024-11-20
**Architecture Version**: 2.0 (Distributed with REST API)
