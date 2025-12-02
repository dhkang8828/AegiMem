"""
CXL Memory RL Environment (Distributed) - GSAT Style
Phase 1: StressAppTest-like Action Space with 64 actions

Based on Google StressAppTest: https://github.com/stressapptest/stressapptest
Action Space: 4 Operations × 16 Key Patterns = 64 actions
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import requests
import time
import json


class Phase1GSATEnvironmentDistributed(gym.Env):
    """
    GSAT-style CXL Memory Testing Environment

    Action Space: 64 discrete actions
        - 4 Operations: FILL, INVERT, COPY, CHECK
        - 16 Key Patterns: StressAppTest verified patterns

    Observation Space: [6 features]
        - CE Volatile Count (delta from baseline)
        - CE Persistent Count (delta)
        - Total CE Count
        - Temperature
        - Current Step
        - Unique Actions Tried

    Reward Function:
        - Primary: +100 * CE_count (CE detection)
        - Exploration: +5 for new action
        - Operation bonus: +1 for INVERT (most effective)
        - Step penalty: -0.1 per step
    """

    metadata = {'render_modes': ['human']}

    # StressAppTest key patterns
    KEY_PATTERNS = [
        0x00, 0xFF, 0x55, 0xAA, 0xF0, 0x0F, 0xCC, 0x33,
        0x01, 0x80, 0x16, 0xB5, 0x4A, 0x57, 0x02, 0xFD
    ]

    OPERATION_NAMES = ['FILL', 'INVERT', 'COPY', 'CHECK']

    def __init__(self,
                 memory_agent_url="http://192.168.3.20:5000",
                 max_sequence_length=50,
                 verbose=True):
        """
        Initialize GSAT-style environment

        Args:
            memory_agent_url: URL of Memory Agent REST API
            max_sequence_length: Maximum steps per episode
            verbose: Print debug information
        """
        super().__init__()

        self.memory_agent_url = memory_agent_url
        self.max_sequence_length = max_sequence_length
        self.verbose = verbose

        # Action Space: 64 discrete actions (4 ops × 16 patterns)
        self.action_space = spaces.Discrete(64)

        # Observation Space: 6 features
        # [0] CE Volatile Count (delta)
        # [1] CE Persistent Count (delta)
        # [2] Total CE Count
        # [3] Device Temperature (°C)
        # [4] Current Step
        # [5] Unique Actions Tried
        self.observation_space = spaces.Box(
            low=np.array([0, 0, 0, 0, 0, 0], dtype=np.float32),
            high=np.array([1000, 1000, 2000, 100, max_sequence_length, 64], dtype=np.float32),
            dtype=np.float32
        )

        # Episode state
        self.current_step = 0
        self.used_actions = set()
        self.total_ce_count = 0
        self.episode_ce_history = []

        # HTTP session
        self.session = requests.Session()
        self.session.trust_env = False  # Disable proxy

        # Initial connection check
        self._check_connection()

    def _check_connection(self):
        """Test connection to Memory Agent"""
        try:
            if self.verbose:
                print(f"[ENV] Connecting to Memory Agent at {self.memory_agent_url}...")

            resp = self.session.get(f"{self.memory_agent_url}/health", timeout=5)
            resp.raise_for_status()

            health = resp.json()
            if self.verbose:
                print(f"[ENV] Connected: {health}")

        except Exception as e:
            print(f"[ENV ERROR] Failed to connect to Memory Agent: {e}")
            raise RuntimeError(f"Memory Agent connection failed: {e}")

    def reset(self, seed=None, options=None):
        """
        Reset environment for new episode

        Returns:
            observation, info
        """
        super().reset(seed=seed)

        # Reset episode state
        self.current_step = 0
        self.used_actions.clear()
        self.total_ce_count = 0
        self.episode_ce_history = []

        if self.verbose:
            print(f"\n[ENV] Resetting environment (episode start)...")

        # Reset hardware baseline
        try:
            resp = self.session.post(
                f"{self.memory_agent_url}/reset_baseline",
                timeout=10
            )
            resp.raise_for_status()

            if self.verbose:
                print(f"[ENV] Baseline reset successful")

            # Get initial CE info
            ce_info = self._get_ce_info()
            observation = self._build_observation(ce_info)

            info = {
                'ce_volatile': 0,
                'ce_persistent': 0,
                'ce_total': 0,
                'temperature': ce_info.get('temperature', 0),
                'step': 0
            }

            return observation, info

        except Exception as e:
            print(f"[ENV ERROR] Reset failed: {e}")
            # Return zero state on failure
            return np.zeros(6, dtype=np.float32), {'error': str(e)}

    def step(self, action):
        """
        Execute action on hardware

        Args:
            action: Integer in [0, 63]

        Returns:
            observation, reward, terminated, truncated, info
        """
        self.current_step += 1

        if not (0 <= action < 64):
            raise ValueError(f"Invalid action: {action} (must be 0-63)")

        # Decode action
        operation = action // 16
        pattern_idx = action % 16
        pattern_byte = self.KEY_PATTERNS[pattern_idx]

        if self.verbose:
            op_name = self.OPERATION_NAMES[operation]
            print(f"\n[ENV] Step {self.current_step}: Action={action} "
                  f"({op_name} with 0x{pattern_byte:02X})")

        # Execute action on hardware
        # NOTE: Each action runs for 5 seconds (STRESS_DURATION_SEC in C code)
        start_time = time.time()

        try:
            resp = self.session.post(
                f"{self.memory_agent_url}/execute_action",
                json={'action': int(action)},
                timeout=60  # 5s stress + margin
            )
            resp.raise_for_status()
            result = resp.json()

            execution_time = time.time() - start_time

            ce_volatile = result.get('ce_volatile', 0)
            ce_persistent = result.get('ce_persistent', 0)
            ce_total = ce_volatile + ce_persistent
            temperature = result.get('temperature', 0)

            if self.verbose:
                print(f"[ENV] Result: CE={ce_total} (V={ce_volatile}, P={ce_persistent}), "
                      f"Temp={temperature}°C, Time={execution_time:.1f}s")

            # Update episode state
            self.used_actions.add(action)
            self.total_ce_count += ce_total
            self.episode_ce_history.append((action, ce_total))

        except Exception as e:
            print(f"[ENV ERROR] Action execution failed: {e}")
            ce_volatile = 0
            ce_persistent = 0
            ce_total = 0
            temperature = 0
            result = {'error': str(e)}

        # Build observation
        observation = np.array([
            float(ce_volatile),
            float(ce_persistent),
            float(ce_total),
            float(temperature),
            float(self.current_step),
            float(len(self.used_actions))
        ], dtype=np.float32)

        # Calculate reward
        reward = self._calculate_reward(action, ce_total, ce_volatile, ce_persistent)

        # Check termination
        terminated = False
        truncated = self.current_step >= self.max_sequence_length

        # Build info
        info = {
            'action': action,
            'operation': self.OPERATION_NAMES[operation],
            'pattern': f"0x{pattern_byte:02X}",
            'ce_volatile': ce_volatile,
            'ce_persistent': ce_persistent,
            'ce_total': ce_total,
            'temperature': temperature,
            'step': self.current_step,
            'unique_actions': len(self.used_actions),
            'total_ce_episode': self.total_ce_count
        }

        if self.verbose:
            print(f"[ENV] Reward: {reward:.2f}, CE Total (Episode): {self.total_ce_count}")

        return observation, reward, terminated, truncated, info

    def _calculate_reward(self, action, ce_total, ce_volatile, ce_persistent):
        """
        Calculate reward based on action result

        Reward components:
        1. CE detection (primary goal): +100 per CE
        2. Exploration bonus: +5 for new action
        3. Operation preference: +1 for INVERT (most effective)
        4. Step penalty: -0.1 per step (encourage efficiency)
        """
        reward = 0.0

        # 1. Primary: CE detection (huge reward)
        if ce_total > 0:
            reward += 100.0 * ce_total

            if self.verbose:
                print(f"[ENV] *** CE DETECTED! +{100.0 * ce_total} reward ***")

        # 2. Exploration bonus (encourage trying new actions)
        if action not in self.used_actions:
            reward += 5.0

            if self.verbose:
                print(f"[ENV] New action explored: +5.0")

        # 3. Operation preference (INVERT is best for CE)
        operation = action // 16
        if operation == 1:  # OP_INVERT
            reward += 1.0

        # 4. Step penalty (encourage finding CE quickly)
        reward -= 0.1

        return reward

    def _get_ce_info(self):
        """Get current CE information from hardware"""
        try:
            resp = self.session.get(
                f"{self.memory_agent_url}/get_ce_info",
                timeout=5
            )
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            if self.verbose:
                print(f"[ENV WARNING] Failed to get CE info: {e}")
            return {}

    def _build_observation(self, ce_info):
        """Build observation from CE info"""
        return np.array([
            float(ce_info.get('volatile_count', 0)),
            float(ce_info.get('persistent_count', 0)),
            float(ce_info.get('total_count', 0)),
            float(ce_info.get('temperature', 0)),
            float(self.current_step),
            float(len(self.used_actions))
        ], dtype=np.float32)

    def render(self, mode='human'):
        """Render environment state"""
        if mode == 'human':
            print(f"\n{'='*60}")
            print(f"Episode Progress:")
            print(f"  Step: {self.current_step}/{self.max_sequence_length}")
            print(f"  Unique actions tried: {len(self.used_actions)}/64")
            print(f"  Total CE detected: {self.total_ce_count}")
            print(f"{'='*60}\n")

    def close(self):
        """Close environment"""
        self.session.close()

        if self.verbose:
            print(f"[ENV] Environment closed")


if __name__ == "__main__":
    # Test environment
    print("="*60)
    print("Testing GSAT-style Environment")
    print("="*60)

    # Create environment
    env = Phase1GSATEnvironmentDistributed(
        memory_agent_url="http://192.168.3.20:5000",
        max_sequence_length=10,
        verbose=True
    )

    print(f"\nAction space: {env.action_space}")
    print(f"Observation space: {env.observation_space}")

    # Test decode
    print("\nExample action decoding:")
    test_actions = [0, 17, 19, 32, 63]
    for action in test_actions:
        op = action // 16
        pat_idx = action % 16
        pat_byte = env.KEY_PATTERNS[pat_idx]
        op_name = env.OPERATION_NAMES[op]
        print(f"  Action {action:2d}: {op_name:7s} with pattern 0x{pat_byte:02X}")

    print("\n" + "="*60)
    print("Note: To run full test, ensure Memory Agent server is running")
    print("="*60)
