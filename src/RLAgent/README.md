# RL Agent

Windows PC에서 실행되는 강화학습 에이전트입니다.

## 구성 요소

- **phase1_environment_distributed.py** - Phase#1 RL 환경
  - REST API를 통해 Memory Agent와 통신
  - Action space: 1536 actions (6 operations × 256 patterns)

- **dqn_agent.py** - DQN (Deep Q-Network) 에이전트

- **ppo_agent.py** - PPO (Proximal Policy Optimization) 에이전트

- **config_loader.py** - 설정 파일 로더

- **dpa_translator.py** - DPA ↔ DRAM 주소 변환 (참고용)

## 실행 방법

```bash
# Windows PC에서 실행
cd C:\path\to\cxl_memory_rl_project
python src\RLAgent\train_phase1.py --config configs\phase1_config.yaml
```

## 요구사항

- Windows 10/11
- Python 3.8+
- PyTorch
- gymnasium
- numpy
- requests

## Memory Agent 연결

기본 설정:
- Memory Agent URL: `http://192.168.3.20:5000`
- Timeout: 30초
