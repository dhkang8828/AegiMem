# AegiMem Distributed Architecture Design

## Overview

AegiMem 프로젝트를 **2개의 독립적인 Agent**로 분리하여 구현합니다:

1. **Memory Test Agent**: CXL 서버(GNR-CRB)에서 실제 메모리 테스트 수행
2. **RL Policy Agent**: 로컬 개발 머신에서 강화학습 수행

## Architecture Diagram

```
┌─────────────────────────────────────────────────────┐
│  RL Policy Agent (로컬 개발 머신)                     │
│  ┌─────────────┐  ┌─────────────┐                  │
│  │   DQN       │  │    PPO      │                  │
│  │  Agent      │  │   Agent     │                  │
│  └──────┬──────┘  └──────┬──────┘                  │
│         │                │                          │
│         └────────┬───────┘                          │
│                  │                                  │
│         ┌────────▼────────┐                         │
│         │  Policy Manager │                         │
│         │  - Action 선택   │                         │
│         │  - 학습 수행     │                         │
│         └────────┬────────┘                         │
│                  │                                  │
│         ┌────────▼────────┐                         │
│         │ Experience      │                         │
│         │ Buffer (JSONL)  │                         │
│         └────────┬────────┘                         │
│                  │                                  │
│         ┌────────▼────────┐                         │
│         │  REST Client    │                         │
│         └────────┬────────┘                         │
└──────────────────┼──────────────────────────────────┘
                   │ HTTP POST
                   │ {"operation": 0, "pattern": 0xAA}
                   │
                   ▼
┌─────────────────────────────────────────────────────┐
│  Memory Test Agent (GNR-CRB 서버)                    │
│         ┌────────────────┐                          │
│         │  REST Server   │                          │
│         │  (bottle.py)   │                          │
│         └────────┬───────┘                          │
│                  │                                  │
│         ┌────────▼────────┐                         │
│         │  Test Executor  │                         │
│         │  - devdax I/O   │                         │
│         │  - CE monitor   │                         │
│         └────────┬────────┘                         │
│                  │                                  │
│         ┌────────▼────────┐                         │
│         │ DevDaxInterface │                         │
│         │ DPATranslator   │                         │
│         │ CECountMonitor  │                         │
│         └────────┬────────┘                         │
└──────────────────┼──────────────────────────────────┘
                   │
                   ▼
            [CXL Memory Device]
```

## Component Details

### 1. RL Policy Agent (로컬 머신)

**위치**: `/home/dhkang/cxl_memory_rl_project/src/rl_agent/`

**역할**:
- 강화학습 policy 학습 및 관리
- Action 선택 및 전송
- Experience 수집 및 저장
- 학습 데이터 분석

**주요 컴포넌트**:

```python
# src/rl_agent/policy_manager.py
class PolicyManager:
    """
    RL policy 관리자
    - DQN, PPO 알고리즘 통합
    - Action 선택
    - 학습 스케줄링
    """

    def __init__(self, algorithm='dqn'):
        if algorithm == 'dqn':
            self.agent = DQNAgent(...)
        elif algorithm == 'ppo':
            self.agent = PPOAgent(...)

    def select_action(self, state):
        """Current policy로 action 선택"""

    def train_step(self):
        """Experience buffer에서 샘플링하여 학습"""

# src/rl_agent/dqn_agent.py
class DQNAgent:
    """
    Deep Q-Network Agent
    - Off-policy 학습
    - Experience replay
    - Target network
    """

# src/rl_agent/ppo_agent.py
class PPOAgent:
    """
    Proximal Policy Optimization Agent
    - On-policy 학습
    - Actor-Critic
    - Clipped surrogate objective
    """

# src/rl_agent/experience_buffer.py
class ExperienceBuffer:
    """
    JSONL 기반 experience 저장
    - 영구 보존
    - 분석 용이
    """

    def add(self, state, action, reward, next_state, done):
        with open(self.file_path, 'a') as f:
            f.write(json.dumps({
                'timestamp': time.time(),
                'state': state,
                'action': action,
                'reward': reward,
                'next_state': next_state,
                'done': done
            }) + '\n')

# src/rl_agent/memory_agent_client.py
class MemoryAgentClient:
    """
    Memory Test Agent와 통신
    """

    def __init__(self, base_url='http://gnr-crb:8000'):
        self.base_url = base_url

    def execute_action(self, action):
        """Action 전송 및 결과 수신"""
        response = requests.post(
            f'{self.base_url}/api/execute',
            json=action
        )
        return response.json()
```

**필요 패키지**:
- PyTorch 또는 TensorFlow (RL 알고리즘)
- NumPy (수치 계산)
- pandas (데이터 분석)
- matplotlib (시각화)
- requests (HTTP 클라이언트)

### 2. Memory Test Agent (GNR-CRB 서버)

**위치**: `/tmp/memory_test_agent/` (GNR-CRB 서버)

**역할**:
- REST API 서버 운영
- 메모리 테스트 실행
- CE count 수집
- 결과 리포트

**주요 컴포넌트**:

```python
# memory_test_agent.py (단일 파일)
import bottle
from bottle import route, run, request
import json
import os
import subprocess
import time

# DevDax, DPA translator, CE monitor import
# (같은 디렉토리에 복사된 파일들)

@route('/api/execute', method='POST')
def execute_action():
    """
    RL Agent로부터 action 받아서 실행

    Request:
    {
        "operation_type": 0-5,
        "pattern": 0x00-0xFF,
        "start_dram": {"rank": 0, "bg": 0, "ba": 0, "row": 0, "col": 0},
        "end_dram": {"rank": 0, "bg": 0, "ba": 0, "row": 100, "col": 100}
    }

    Response:
    {
        "ce_delta": 5,
        "execution_time": 1.23,
        "status": "success",
        "timestamp": "2025-01-13T10:30:00"
    }
    """
    try:
        action = request.json

        # CE count 초기값
        ce_before = ce_monitor.get_ce_count()
        start_time = time.time()

        # 테스트 실행
        devdax.execute_pattern_test(
            operation_type=action['operation_type'],
            pattern_byte=action['pattern'],
            start_dram=action['start_dram'],
            end_dram=action['end_dram']
        )

        # CE count 최종값
        ce_after = ce_monitor.get_ce_count()
        execution_time = time.time() - start_time

        return {
            'ce_delta': ce_after - ce_before,
            'execution_time': execution_time,
            'status': 'success',
            'timestamp': time.strftime('%Y-%m-%dT%H:%M:%S')
        }

    except Exception as e:
        return {
            'status': 'error',
            'error': str(e),
            'timestamp': time.strftime('%Y-%m-%dT%H:%M:%S')
        }

@route('/api/status', method='GET')
def get_status():
    """Agent 상태 확인"""
    return {
        'status': 'running',
        'device': '/dev/dax0.0',
        'ce_count': ce_monitor.get_ce_count()
    }

if __name__ == '__main__':
    # DevDax, CE monitor 초기화
    global devdax, ce_monitor

    devdax = DevDaxInterface(
        device_path='/dev/dax0.0',
        dpa_translator=translator,
        ce_monitor=ce_monitor
    )

    ce_monitor = CECountMonitor(device='mem0')

    print("Memory Test Agent starting on port 8000...")
    run(host='0.0.0.0', port=8000)
```

**필요 파일** (GNR-CRB로 복사):
- `bottle.py` (단일 파일, 70KB)
- `memory_test_agent.py`
- `devdax_interface.py`
- `dpa_translator.py`
- `ce_count_monitor.py`
- `dpa_mapping.csv` (수집된 매핑 데이터)

**의존성**: Python 3.10.12 표준 라이브러리만

## Communication Protocol

### REST API Specification

#### 1. Execute Action

**Endpoint**: `POST /api/execute`

**Request**:
```json
{
    "operation_type": 0,
    "pattern": 170,
    "start_dram": {
        "rank": 0,
        "bg": 0,
        "ba": 0,
        "row": 0,
        "col": 0
    },
    "end_dram": {
        "rank": 0,
        "bg": 0,
        "ba": 0,
        "row": 100,
        "col": 100
    }
}
```

**Response**:
```json
{
    "ce_delta": 5,
    "execution_time": 1.234,
    "status": "success",
    "timestamp": "2025-01-13T10:30:00",
    "metadata": {
        "ce_before": 100,
        "ce_after": 105
    }
}
```

#### 2. Get Status

**Endpoint**: `GET /api/status`

**Response**:
```json
{
    "status": "running",
    "device": "/dev/dax0.0",
    "ce_count": 105,
    "uptime": 3600
}
```

## Data Flow

### 1. Training Loop

```python
# RL Agent (로컬)
for episode in range(num_episodes):
    state = env.reset()
    done = False

    while not done:
        # 1. Action 선택
        action = policy.select_action(state)

        # 2. Memory Agent에 전송
        result = memory_client.execute_action(action)

        # 3. Reward 계산
        reward = result['ce_delta']

        # 4. Experience 저장
        experience_buffer.add(state, action, reward, next_state, done)

        # 5. 학습
        if buffer.size() > batch_size:
            policy.train_step()

        state = next_state
```

### 2. Experience Storage

**파일 위치**: `data/experiences/experiment_001.jsonl`

**Format**:
```json
{"timestamp": 1705132200.123, "state": {...}, "action": {...}, "reward": 5, "next_state": {...}, "done": false}
{"timestamp": 1705132205.456, "state": {...}, "action": {...}, "reward": 3, "next_state": {...}, "done": false}
{"timestamp": 1705132210.789, "state": {...}, "action": {...}, "reward": 0, "next_state": {...}, "done": true}
```

## Project Structure

```
cxl_memory_rl_project/
├── src/
│   ├── rl_agent/                    # RL Policy Agent (로컬)
│   │   ├── __init__.py
│   │   ├── policy_manager.py        # Policy 관리
│   │   ├── dqn_agent.py             # DQN 구현
│   │   ├── ppo_agent.py             # PPO 구현
│   │   ├── experience_buffer.py     # JSONL 버퍼
│   │   ├── memory_agent_client.py   # REST 클라이언트
│   │   └── training_loop.py         # 학습 루프
│   │
│   ├── memory_agent/                # Memory Test Agent (GNR-CRB)
│   │   ├── memory_test_agent.py     # Main server
│   │   ├── devdax_interface.py      # DevDax 인터페이스
│   │   ├── dpa_translator.py        # DPA 변환
│   │   └── ce_count_monitor.py      # CE count 수집
│   │
│   └── common/                      # 공통 모듈
│       ├── state_representation.py  # State 정의
│       └── action_space.py          # Action space 정의
│
├── data/
│   ├── experiences/                 # JSONL experience files
│   │   ├── dqn_exp001.jsonl
│   │   └── ppo_exp001.jsonl
│   ├── models/                      # 학습된 모델
│   └── dpa_mapping/                 # DPA 매핑 데이터
│
├── configs/
│   ├── dqn_config.yaml
│   └── ppo_config.yaml
│
├── scripts/
│   ├── deploy_memory_agent.sh      # GNR-CRB 배포 스크립트
│   ├── start_training_dqn.sh
│   └── start_training_ppo.sh
│
└── docs/
    ├── DISTRIBUTED_ARCHITECTURE.md  # 이 문서
    ├── DQN_IMPLEMENTATION.md
    └── PPO_IMPLEMENTATION.md
```

## Deployment

### Memory Test Agent 배포

```bash
# 로컬에서
cd /home/dhkang/cxl_memory_rl_project

# 1. bottle.py 다운로드
wget https://raw.githubusercontent.com/bottlepy/bottle/master/bottle.py -P src/memory_agent/

# 2. GNR-CRB로 전송
scp -r src/memory_agent user@gnr-crb:/tmp/
scp data/dpa_mapping/dpa_mapping.csv user@gnr-crb:/tmp/memory_agent/

# 3. GNR-CRB에서 실행
ssh user@gnr-crb
cd /tmp/memory_agent
python3 memory_test_agent.py
# Memory Test Agent starting on port 8000...
```

### RL Policy Agent 실행

```bash
# 로컬에서
cd /home/dhkang/cxl_memory_rl_project

# DQN 학습
python3 src/rl_agent/training_loop.py --algorithm dqn --episodes 1000

# PPO 학습
python3 src/rl_agent/training_loop.py --algorithm ppo --episodes 1000
```

## RL Algorithm Comparison

| Feature | DQN | PPO |
|---------|-----|-----|
| **Type** | Off-policy | On-policy |
| **Experience Replay** | ✅ Yes | ❌ No |
| **Sample Efficiency** | 🟢 High | 🟡 Medium |
| **Stability** | 🟡 Medium | 🟢 High |
| **Continuous Action** | ❌ No | ✅ Yes |
| **Implementation** | 🟢 Simple | 🟡 Complex |
| **우리 프로젝트** | Discrete (1536 actions) | Discrete (1536 actions) |

**Both suitable for our discrete action space!**

## State Representation

```python
state = {
    # 현재까지 테스트한 패턴 정보
    'tested_patterns': [0xAA, 0x55, ...],  # 최근 N개

    # 각 operation별 효율성
    'operation_efficiency': [0.8, 0.6, ...],  # 6개

    # 현재 메모리 영역 상태
    'current_region': {
        'rank': 0, 'bg': 0, 'ba': 0,
        'row_range': (0, 1000)
    },

    # 누적 통계
    'total_ce_found': 123,
    'total_tests': 456,
    'avg_ce_per_test': 0.27
}
```

## Action Space

```python
action = {
    'operation_type': 0-5,  # 6 operations
    'pattern': 0x00-0xFF,   # 256 patterns

    # Total: 6 × 256 = 1,536 discrete actions
}

# Mapping
action_id = operation_type * 256 + pattern
# Example: action_id = 0 * 256 + 170 = 170
#          → [^(W 0xAA), ^(R 0xAA)]
```

## Next Steps

1. ✅ Architecture design complete
2. ⏭️ Implement CE count collection mechanism
3. ⏭️ Implement Memory Test Agent
4. ⏭️ Implement DQN Agent
5. ⏭️ Implement PPO Agent
6. ⏭️ Analyze collected DPA mapping data
7. ⏭️ Integration testing
8. ⏭️ Real hardware deployment

## Open Questions

1. **State representation 세부사항**:
   - 어떤 정보를 state에 포함?
   - State 차원은?

2. **Reward shaping**:
   - CE delta만?
   - 시간 패널티?
   - Exploration bonus?

3. **Episode 정의**:
   - Episode 종료 조건?
   - 몇 step per episode?

4. **Hyperparameters**:
   - Learning rate
   - Batch size
   - Network architecture

이러한 세부사항은 구현하면서 결정하겠습니다!
