# GSAT-like Memory Agent Quick Start Guide

## Overview

This guide helps you switch from the existing memory agent to the GSAT-style agent with 64-action space based on Google StressAppTest patterns.

**Key Benefits:**
- ✅ Sustained stress (5 seconds per action) instead of single-pass
- ✅ Verified patterns from StressAppTest (15+ years of production use)
- ✅ Smaller action space (64 vs 1536) for faster learning
- ✅ Higher CE detection probability

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                   Training Machine                           │
│  ┌────────────────────────────────────────────────────────┐ │
│  │  Python RL Agent (train_dqn_gsat.py)                   │ │
│  │  - 64 actions (4 ops × 16 patterns)                    │ │
│  │  - DQN with 6-dim state space                          │ │
│  └────────────────────────────────────────────────────────┘ │
│                          ↓ HTTP                              │
└──────────────────────────┼───────────────────────────────────┘
                           ↓
┌──────────────────────────┼───────────────────────────────────┐
│                  GNR-SP Linux Server                         │
│  ┌────────────────────────────────────────────────────────┐ │
│  │  Memory Agent REST API (Flask)                         │ │
│  └────────────────────────────────────────────────────────┘ │
│                          ↓                                   │
│  ┌────────────────────────────────────────────────────────┐ │
│  │  libgsat_memory_agent.so (C Library)                   │ │
│  │  - 16 threads × 5 seconds stress                       │ │
│  │  - FILL/INVERT/COPY/CHECK operations                   │ │
│  └────────────────────────────────────────────────────────┘ │
│                          ↓                                   │
│  ┌────────────────────────────────────────────────────────┐ │
│  │  /dev/dax0.0 (CXL Type3 Device)                        │ │
│  └────────────────────────────────────────────────────────┘ │
│                          ↓                                   │
│  ┌────────────────────────────────────────────────────────┐ │
│  │  umxc (CXL Mailbox - CE monitoring)                    │ │
│  └────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
```

## Step 1: Build C Library (GNR-SP Server)

```bash
# Navigate to c_library directory
cd ~/cxl_memory_rl_project/src/MemoryAgent/c_library

# Build GSAT-like library
make -f Makefile_gsat quick

# Verify
ls -lh libgsat_memory_agent.so

# Expected output:
# -rwxr-xr-x 1 user user 50K Dec 2 10:00 libgsat_memory_agent.so
```

## Step 2: Update Memory Agent Server (GNR-SP)

The existing `memory_agent_server.py` needs to be updated to use the new library.

**Option A: Create new server (recommended)**

```bash
cd ~/cxl_memory_rl_project/src/MemoryAgent

# Copy and modify
cp memory_agent_server.py gsat_memory_agent_server.py
```

Edit `gsat_memory_agent_server.py`:

```python
# Change library import
from gsat_memory_agent_c_wrapper import GSATMemoryAgentC

# Update initialization
agent = GSATMemoryAgentC()
agent.init("/dev/dax0.0", memory_size_mb=1024)
```

**Option B: Modify existing server**

Just update the import in `memory_agent_server.py`:

```python
# OLD:
from memory_agent_c_wrapper import MemoryAgentC

# NEW:
from gsat_memory_agent_c_wrapper import GSATMemoryAgentC as MemoryAgentC
```

## Step 3: Restart Memory Agent Server

```bash
# Stop old server
pkill -f memory_agent_server

# Start new server (using gsat library)
cd ~/cxl_memory_rl_project/src/MemoryAgent
python3 gsat_memory_agent_server.py

# Or if you modified existing:
python3 memory_agent_server.py

# Verify in logs:
# "MA Initialized: /dev/dax0.0, 1024 MB"
```

## Step 4: Test Connection (Training Machine)

```bash
cd ~/cxl_memory_rl_project/src/RLAgent

# Test environment
python3 phase1_gsat_like_environment_distributed.py

# Expected output:
# ===========================================================
# Testing GSAT-style Environment
# ===========================================================
# [ENV] Connecting to Memory Agent at http://192.168.3.20:5000...
# [ENV] Connected: {'status': 'healthy'}
#
# Action space: Discrete(64)
# Observation space: Box(...)
#
# Example action decoding:
#   Action  0: FILL    with pattern 0x00
#   Action 17: INVERT  with pattern 0x55  <- BEST for CE
#   Action 19: INVERT  with pattern 0xAA  <- BEST for CE
#   Action 32: COPY    with pattern 0x00
#   Action 63: CHECK   with pattern 0xFD
```

## Step 5: Start Training

```bash
cd ~/cxl_memory_rl_project/src/RLAgent

# Start training with logging
python3 train_dqn_gsat.py \
    --memory-agent-url http://192.168.3.20:5000 \
    --episodes 200 \
    --max-steps 50 \
    --save-dir models_gsat \
    --save-interval 10 \
    --log-file training_gsat_$(date +%Y%m%d_%H%M%S).log

# Training will create:
# - models_gsat/best_model.pt (best reward)
# - models_gsat/best_ce_model.pt (best CE detection)
# - models_gsat/checkpoint_episode_*.pt
# - models_gsat/training_stats_episode_*.json
# - training_gsat_YYYYMMDD_HHMMSS.log
```

## Action Space Reference

### 64 Actions = 4 Operations × 16 Patterns

**Operations:**
- 0: FILL (Thermal stress)
- 1: INVERT (Switching noise) ← **RECOMMENDED for CE**
- 2: COPY (Bandwidth saturation)
- 3: CHECK (Read disturb)

**16 Key Patterns (StressAppTest):**
```
Index  Hex    Binary       Description
-----  ----   --------     -----------
  0    0x00   00000000     All zeros
  1    0xFF   11111111     All ones
  2    0x55   01010101     Checkerboard A ← BEST
  3    0xAA   10101010     Checkerboard B ← BEST
  4    0xF0   11110000     Half-half
  5    0x0F   00001111     Half-half inv
  6    0xCC   11001100     2-bit pattern
  7    0x33   00110011     2-bit pattern inv
  8    0x01   00000001     Walking ones start
  9    0x80   10000000     Walking ones end
 10    0x16   00010110     8b10b low transition
 11    0xB5   10110101     8b10b high transition
 12    0x4A   01001010     Checker pattern
 13    0x57   01010111     Edge case 1
 14    0x02   00000010     Edge case 2
 15    0xFD   11111101     Edge case 3
```

**Action Encoding:**
```
action = operation_id × 16 + pattern_id

Examples:
- Action  0 = FILL   with 0x00
- Action 17 = INVERT with 0x55 ← Top priority for CE
- Action 19 = INVERT with 0xAA ← Top priority for CE
- Action 32 = COPY   with 0x00
- Action 63 = CHECK  with 0xFD
```

## High-Priority Actions for CE Detection

If you want to manually test these actions first:

```bash
# Test action 17 (INVERT with 0x55)
curl -X POST http://192.168.3.20:5000/execute_action \
     -H "Content-Type: application/json" \
     -d '{"action": 17}'

# Test action 19 (INVERT with 0xAA)
curl -X POST http://192.168.3.20:5000/execute_action \
     -H "Content-Type: application/json" \
     -d '{"action": 19}'

# Check CE count
curl http://192.168.3.20:5000/get_ce_info
```

## Monitoring Training

```bash
# Watch log in real-time
tail -f training_gsat_*.log

# Look for:
# [ENV] *** CE DETECTED! +100.0 reward ***
# [ENV] New best CE detector! CE count: X

# Check models directory
ls -lh models_gsat/

# Analyze stats
cat models_gsat/training_stats_episode_10.json
```

## Comparison: Old vs GSAT

| Feature | Old Agent | GSAT Agent |
|---------|-----------|------------|
| Action Space | 1536 (6 ops × 256) | 64 (4 ops × 16) |
| Stress Duration | Single pass (~0.1s) | 5 seconds sustained |
| Patterns | All 256 bytes | 16 verified patterns |
| Operations | 6 directional | 4 stress types |
| Learning Speed | Slow (large space) | Fast (small space) |
| CE Probability | Low | **High** |

## Troubleshooting

### 1. Library not found

```bash
# Check library exists
ls ~/cxl_memory_rl_project/src/MemoryAgent/c_library/libgsat_memory_agent.so

# If not, rebuild:
cd ~/cxl_memory_rl_project/src/MemoryAgent/c_library
make -f Makefile_gsat clean
make -f Makefile_gsat quick
```

### 2. Server connection failed

```bash
# Check server is running
ps aux | grep memory_agent_server

# Check port is open
netstat -tulpn | grep 5000

# Restart server
cd ~/cxl_memory_rl_project/src/MemoryAgent
python3 gsat_memory_agent_server.py
```

### 3. Action validation error

```
Error: Invalid action: 64 (must be 0-63)
```

This means the training script is still using old action space. Verify:

```python
# In train_dqn_gsat.py, line ~75:
num_actions=64,  # Must be 64, not 1536
```

### 4. No CE detected after many episodes

This is expected - CE detection is challenging! Try:

1. Increase stress duration in C code:
   ```c
   #define STRESS_DURATION_SEC 10  // Increase from 5 to 10
   ```

2. Manually test high-priority actions:
   ```bash
   # Test each priority action 10 times
   for i in {1..10}; do
       curl -X POST http://192.168.3.20:5000/execute_action -d '{"action": 17}'
       sleep 1
   done
   ```

## Next Steps

1. **Run 200 episodes** with GSAT agent
2. **Compare CE detection rate** vs old agent (150 episodes with 0 CE)
3. **If CE detected**: Analyze which action found it
4. **If no CE**: Consider increasing `STRESS_DURATION_SEC` to 10 or 30 seconds

## Expected Timeline

- **Build & Setup**: 10 minutes
- **Training (200 episodes × 50 steps × 5s)**: ~14 hours
- **Analysis**: 30 minutes

## References

- StressAppTest: https://github.com/stressapptest/stressapptest
- Pattern Design: `docs/GSAT_ACTION_SPACE_DESIGN.md`
- C Implementation: `src/MemoryAgent/c_library/gsat_like_memory_agent.c`
