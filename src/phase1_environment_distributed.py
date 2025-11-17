"""
Phase 1 Environment: Distributed Architecture (REST API Client)

Runs on Windows PC and communicates with Memory Agent on GNR-SP via REST API.

Goal: Find pattern sequences that detect known faults (MRAT, stressapptest)
Action Space: 1536 actions (6 operations × 256 patterns)
"""

import gymnasium as gym
import numpy as np
import requests
from enum import IntEnum
from typing import Dict, List, Tuple, Optional
import time


class OperationType(IntEnum):
    """Memory test operation types - All guarantee Write before Read"""
    WR_ASC_ASC = 0      # [^(W pat), ^(R pat)] - Ascending W → Ascending R
    WR_DESC_DESC = 1    # [v(W pat), v(R pat)] - Descending W → Descending R
    WR_ASC_DESC = 2     # [^(W pat), v(R pat)] - Ascending W → Descending R (cross!)
    WR_DESC_ASC = 3     # [v(W pat), ^(R pat)] - Descending W → Ascending R (cross!)
    WR_DESC_SINGLE = 4  # [v(W pat, R pat)] - Descending single-pass W+R
    WR_ASC_SINGLE = 5   # [^(W pat, R pat)] - Ascending single-pass W+R


class Phase1EnvironmentDistributed(gym.Env):
    """
    Phase 1 RL Environment - Distributed Architecture

    Architecture:
        Windows PC (RL Agent) ←→ REST API ←→ GNR-SP (Memory Agent)

    Action Space: 1536 discrete actions
        - 6 operation types (all guarantee Write before Read):
          * 0: [^(W pat), ^(R pat)] - Ascending W → Ascending R
          * 1: [v(W pat), v(R pat)] - Descending W → Descending R
          * 2: [^(W pat), v(R pat)] - Ascending W → Descending R (cross-direction)
          * 3: [v(W pat), ^(R pat)] - Descending W → Ascending R (cross-direction)
          * 4: [v(W+R pat)] - Descending single-pass W+R
          * 5: [^(W+R pat)] - Ascending single-pass W+R
        - 256 data patterns (0x00 ~ 0xFF)
        - Encoding: action = operation_type * 256 + pattern_byte

    Observation Space:
        - sequence_history: Last N actions taken
        - last_test_result: Previous test result (PASS/FAIL)
        - sequence_length: Current sequence length
        - cumulative_errors: Total errors found so far
    """

    metadata = {'render.modes': ['human']}

    def __init__(
        self,
        memory_agent_url: str = "http://192.168.3.20:5000",
        device_id: str = "unknown",
        max_sequence_length: int = 10,
        timeout: float = 30.0,
        verbose: bool = True
    ):
        """
        Initialize Phase 1 Environment (Distributed)

        Args:
            memory_agent_url: URL of Memory Agent REST API
            device_id: Device identifier for logging
            max_sequence_length: Maximum actions per episode
            timeout: REST API request timeout in seconds
            verbose: Print detailed logs
        """
        super().__init__()

        self.memory_agent_url = memory_agent_url.rstrip('/')
        self.device_id = device_id
        self.max_seq_len = max_sequence_length
        self.timeout = timeout
        self.verbose = verbose

        # Action space: 1536 discrete actions (6 operations × 256 patterns)
        # Encoding: action = operation_type * 256 + pattern_byte
        self.action_space = gym.spaces.Discrete(1536)

        # Observation space
        self.observation_space = gym.spaces.Dict({
            'sequence_history': gym.spaces.Box(
                low=0, high=1535,
                shape=(max_sequence_length,),
                dtype=np.int32
            ),
            'last_test_result': gym.spaces.Discrete(2),  # 0=PASS, 1=FAIL
            'sequence_length': gym.spaces.Discrete(max_sequence_length + 1),
            'cumulative_errors': gym.spaces.Box(
                low=0, high=np.inf,
                shape=(1,),
                dtype=np.float32
            )
        })

        # Episode state
        self.current_sequence = []
        self.last_result = 0
        self.total_errors = 0
        self.step_count = 0

        # Statistics
        self.episode_count = 0
        self.successful_sequences = []

        # Test connection to Memory Agent
        self._test_connection()

        self.log(f"Phase1EnvironmentDistributed initialized")
        self.log(f"  Memory Agent: {self.memory_agent_url}")
        self.log(f"  Device: {device_id}")
        self.log(f"  Max sequence length: {max_sequence_length}")
        self.log(f"  Action space: {self.action_space.n} actions")

    def _test_connection(self):
        """Test connection to Memory Agent"""
        try:
            response = requests.get(
                f"{self.memory_agent_url}/health",
                timeout=5.0
            )
            response.raise_for_status()
            health = response.json()
            self.log(f"Connected to Memory Agent: {health}")
        except Exception as e:
            raise ConnectionError(
                f"Cannot connect to Memory Agent at {self.memory_agent_url}: {e}"
            )

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        """Reset environment for new episode"""
        super().reset(seed=seed)

        self.current_sequence = []
        self.last_result = 0
        self.total_errors = 0
        self.step_count = 0
        self.episode_count += 1

        # Reset CE baseline on Memory Agent
        try:
            requests.post(
                f"{self.memory_agent_url}/reset_baseline",
                timeout=self.timeout
            )
        except Exception as e:
            self.log(f"Warning: Failed to reset baseline: {e}")

        self.log(f"\n=== Episode {self.episode_count} Start ===")

        return self._get_observation(), {}

    def step(self, action: int):
        """
        Execute one action via REST API to Memory Agent

        Args:
            action: Action index (0-1535)

        Returns:
            observation, reward, terminated, truncated, info
        """
        # Decode action for logging
        cmd = self._decode_action(action)

        self.log(f"\nStep {self.step_count + 1}: {self._action_to_string(action)}")

        # Execute action via REST API
        start_time = time.time()
        result = self._execute_action_remote(action)
        execution_time = time.time() - start_time

        # Update state
        self.current_sequence.append(action)
        self.last_result = 1 if result['ce_detected'] else 0
        self.total_errors += result['ce_total']
        self.step_count += 1

        # Calculate reward
        reward = self._calculate_reward(result, self.step_count)

        # Check termination
        terminated = result['ce_detected']  # Found fault!
        truncated = self.step_count >= self.max_seq_len  # Max length reached

        # Info
        info = {
            'ce_detected': result['ce_detected'],
            'ce_total': result['ce_total'],
            'ce_volatile': result['ce_volatile'],
            'ce_persistent': result['ce_persistent'],
            'sequence': self.current_sequence.copy(),
            'execution_time': execution_time,
            'action_decoded': cmd
        }

        # Log result
        if result['ce_detected']:
            self.log(f"  ✓ CE DETECTED! Total: {result['ce_total']} "
                    f"(V:{result['ce_volatile']}, P:{result['ce_persistent']})")
            self.log(f"  Winning sequence: {[self._action_to_string(a) for a in self.current_sequence]}")
            self.successful_sequences.append(self.current_sequence.copy())
        else:
            self.log(f"  - Test PASS (no CE)")

        self.log(f"  Reward: {reward:.2f}, Time: {execution_time:.3f}s")

        if terminated or truncated:
            self.log(f"\n=== Episode {self.episode_count} End ===")
            self.log(f"  Result: {'SUCCESS' if terminated else 'TRUNCATED'}")
            self.log(f"  Sequence length: {len(self.current_sequence)}")
            self.log(f"  Total successful sequences: {len(self.successful_sequences)}")

        return self._get_observation(), reward, terminated, truncated, info

    def _execute_action_remote(self, action: int) -> Dict:
        """
        Execute action via REST API to Memory Agent

        Args:
            action: Action index (0-1535)

        Returns:
            Dict with CE detection results
        """
        try:
            response = requests.post(
                f"{self.memory_agent_url}/execute_action",
                json={'action': action},
                timeout=self.timeout
            )
            response.raise_for_status()
            result = response.json()

            if not result.get('success', False):
                raise RuntimeError(f"Memory Agent error: {result.get('error_message')}")

            return result

        except requests.Timeout:
            raise TimeoutError(f"Memory Agent timeout (>{self.timeout}s)")
        except requests.RequestException as e:
            raise ConnectionError(f"Memory Agent communication error: {e}")

    def _decode_action(self, action: int) -> Dict:
        """
        Decode action index to command components

        Action encoding: operation_type * 256 + pattern_byte

        Args:
            action: Action index (0-1535)

        Returns:
            Dict with operation_type and pattern_byte
        """
        operation_type = action // 256
        pattern_byte = action % 256

        return {
            'operation_type': OperationType(operation_type),
            'pattern_byte': pattern_byte
        }

    def _action_to_string(self, action: int) -> str:
        """Convert action to human-readable string"""
        cmd = self._decode_action(action)
        op = cmd['operation_type'].name
        pat = f"0x{cmd['pattern_byte']:02X}"
        return f"Action[{action}]: {op}, pattern={pat}"

    def _get_observation(self) -> Dict:
        """Get current observation"""
        # Pad sequence history
        history = np.zeros(self.max_seq_len, dtype=np.int32)
        if self.current_sequence:
            seq_len = min(len(self.current_sequence), self.max_seq_len)
            history[-seq_len:] = self.current_sequence[-seq_len:]

        return {
            'sequence_history': history,
            'last_test_result': self.last_result,
            'sequence_length': len(self.current_sequence),
            'cumulative_errors': np.array([self.total_errors], dtype=np.float32)
        }

    def _calculate_reward(self, result: Dict, step_num: int) -> float:
        """
        Calculate reward for this step

        Reward structure for Phase 1:
            - CE detected: +1000 (main goal!)
            - Additional CE count bonus: +10 per CE
            - Efficiency bonus: +100 / step_num (shorter sequence = higher reward)
            - No CE: +1 (exploration bonus)

        Args:
            result: Test result dict
            step_num: Current step number

        Returns:
            Reward value
        """
        reward = 0.0

        if result['ce_detected']:
            # Main success reward
            reward += 1000.0

            # Bonus for error count
            reward += result['ce_total'] * 10.0

            # Efficiency bonus (shorter sequence = better)
            reward += 100.0 / step_num

            self.log(f"  Reward breakdown: base=1000, count={result['ce_total']*10}, "
                    f"efficiency={100.0/step_num:.2f}")
        else:
            # Small exploration bonus
            reward += 1.0

        return reward

    def render(self, mode='human'):
        """Render environment state"""
        if mode == 'human':
            print(f"\n{'='*60}")
            print(f"Episode {self.episode_count} - Step {self.step_count}")
            print(f"Current sequence: {self.current_sequence}")
            print(f"Total errors: {self.total_errors}")
            print(f"Successful sequences found: {len(self.successful_sequences)}")
            print(f"{'='*60}\n")

    def log(self, message: str):
        """Print log message if verbose"""
        if self.verbose:
            print(message)


# Helper function
def make_distributed_env(
    memory_agent_url: str = "http://192.168.3.20:5000",
    device_id: str = "CMM-D-001",
    max_sequence_length: int = 10,
    verbose: bool = True
) -> Phase1EnvironmentDistributed:
    """
    Create Phase 1 Distributed Environment

    Args:
        memory_agent_url: Memory Agent REST API URL
        device_id: Device identifier
        max_sequence_length: Maximum actions per episode
        verbose: Enable verbose logging

    Returns:
        Configured environment
    """
    return Phase1EnvironmentDistributed(
        memory_agent_url=memory_agent_url,
        device_id=device_id,
        max_sequence_length=max_sequence_length,
        verbose=verbose
    )


if __name__ == "__main__":
    # Test distributed environment
    print("Testing Distributed Phase1 Environment...")

    # Create environment
    env = make_distributed_env(
        memory_agent_url="http://192.168.3.20:5000",
        device_id="CMM-D-TEST",
        max_sequence_length=5,
        verbose=True
    )

    # Run one episode
    print("\nRunning test episode...")
    obs, info = env.reset()

    for step in range(3):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)

        if terminated or truncated:
            break

    print("\nTest completed!")
