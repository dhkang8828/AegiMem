# DQN Training Quick Start

## 개요

Simple DQN으로 CXL 메모리 불량 검출 학습

- **Action Space**: 1536 discrete actions (6 operations × 256 patterns)
- **State**: 6-dimensional vector (flattened from dict)
- **Goal**: CE (Correctable Error) 검출

## 사전 준비

### 1. GNR-SP에서 Memory Agent 실행

```bash
cd /home/dhkang/cxl_memory_rl_project/src/MemoryAgent
sudo python3 memory_agent_server.py \
    --devdax /dev/dax1.0 \
    --memory-size 128000 \
    --port 5000
```

### 2. Windows PC에서 Python 패키지 설치

```bash
pip install torch numpy requests gymnasium
```

## 훈련 실행

### 기본 사용법

```bash
cd /path/to/cxl_memory_rl_project/src/RLAgent
python train_dqn_distributed.py
```

### 파라미터 커스터마이징

```bash
python train_dqn_distributed.py \
    --memory-agent-url http://192.168.3.20:5000 \
    --episodes 100 \
    --max-steps 50 \
    --save-dir models \
    --save-interval 10 \
    --log-file training.log
```

### 파라미터 설명

| 파라미터 | 기본값 | 설명 |
|---------|--------|------|
| `--memory-agent-url` | `http://192.168.3.20:5000` | Memory Agent REST API URL |
| `--episodes` | `100` | 훈련 episode 수 |
| `--max-steps` | `50` | Episode당 최대 step 수 |
| `--save-dir` | `models` | 모델 저장 디렉토리 |
| `--save-interval` | `10` | N episode마다 체크포인트 저장 |
| `--log-file` | `None` | 로그 파일 경로 (선택) |

## 훈련 시간 예상

**1 action = 45초** (128GB 전체 테스트, cache flush 포함)

| Episodes | Max Steps | 예상 시간 |
|----------|-----------|----------|
| 10 | 50 | 6.25시간 |
| 50 | 50 | 31.25시간 |
| 100 | 50 | 62.5시간 |
| 100 | 20 | 25시간 |

**추천**: 먼저 10 episodes × 20 steps로 시작 (약 2.5시간)

```bash
python train_dqn_distributed.py --episodes 10 --max-steps 20
```

## 모델 파일

훈련 완료 후 `models/` 디렉토리에 저장됨:

- `best_model.pt` - 최고 성능 모델
- `final_model.pt` - 마지막 모델
- `checkpoint_episode_N.pt` - N episode 체크포인트
- `training_stats_episode_N.json` - 훈련 통계
- `final_training_stats.json` - 최종 통계

## 훈련 모니터링

### 콘솔 출력 예시

```
============================================================
Episode 1/100
Epsilon: 1.0000
Step 1/50: Action 742
  Reward: -1.00, CE total: 0
Step 2/50: Action 1234
  Reward: -1.00, CE total: 0
...
============================================================
Episode 1 Summary:
  Total reward: -50.00
  Steps: 50
  Avg loss: 0.0234
  CE detected: 0
  Duration: 2250.5s
  Epsilon: 0.9950
```

### 로그 파일 사용

```bash
python train_dqn_distributed.py --log-file training.log
tail -f training.log  # 실시간 모니터링
```

## 훈련 통계 분석

```python
import json
import matplotlib.pyplot as plt

# Load stats
with open('models/final_training_stats.json', 'r') as f:
    stats = json.load(f)

# Plot rewards
plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.plot(stats['episode_rewards'])
plt.xlabel('Episode')
plt.ylabel('Total Reward')
plt.title('Episode Rewards')

plt.subplot(1, 3, 2)
plt.plot(stats['episode_losses'])
plt.xlabel('Episode')
plt.ylabel('Loss')
plt.title('Training Loss')

plt.subplot(1, 3, 3)
plt.plot(stats['episode_ce_counts'])
plt.xlabel('Episode')
plt.ylabel('CE Count')
plt.title('CE Detected per Episode')

plt.tight_layout()
plt.savefig('training_progress.png')
plt.show()
```

## 체크포인트에서 재개

```python
from dqn_simple import SimpleDQNAgent

# Load checkpoint
agent = SimpleDQNAgent()
agent.load('models/checkpoint_episode_50.pt')

# Continue training...
```

## 문제 해결

### Connection Timeout

```python
# Windows에서 프록시 비활성화 필요
import requests
session = requests.Session()
session.trust_env = False
```

### Memory Agent 응답 없음

GNR-SP에서 확인:
```bash
# 서버 실행 중인지 확인
ps aux | grep memory_agent_server.py

# 포트 확인
sudo netstat -tulpn | grep 5000

# 로그 확인
sudo journalctl -f | grep memory_agent
```

### 훈련이 너무 느림

- `--max-steps`를 줄이기 (50 → 20)
- Episode 수 줄이기
- 나중에 batch execution으로 10-30배 가속 가능

## 다음 단계

1. **10 episodes × 20 steps로 빠른 테스트**
2. **훈련 통계 분석**
3. **Hyperparameter tuning**
4. **더 긴 훈련 (100+ episodes)**
5. **Phase 1 개선사항 적용** (docs/PHASE1_IMPROVEMENTS.md)

## 참고 문서

- **src/RLAgent/dqn_simple.py** - DQN 구현
- **src/RLAgent/phase1_environment_distributed.py** - 환경 구현
- **docs/PHASE1_IMPROVEMENTS.md** - 성능 개선 방안
