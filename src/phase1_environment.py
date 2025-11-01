"""
Phase 1 Environment: Pattern Sequence Discovery

Goal: Find pattern sequences that detect known faults (MRAT, stressapptest)
Action Space: 1536 actions (6 operations × 256 patterns)
"""

import gymnasium as gym
import numpy as np
from enum import IntEnum
from typing import Dict, List, Tuple, Optional
import time


class OperationType(IntEnum):
    """Memory test operation types"""
    WRITE_ASC = 0        # ^(W pattern) - Ascending Write only
    READ_ASC = 1         # ^(R pattern) - Ascending Read & Compare
    WRITE_DESC = 2       # v(W pattern) - Descending Write only
    READ_DESC = 3        # v(R pattern) - Descending Read & Compare
    WRITE_READ_DESC = 4  # v(W pattern, R pattern) - Descending Write then Read
    WRITE_READ_ASC = 5   # ^(W pattern, R pattern) - Ascending Write then Read


class Phase1Environment(gym.Env):
    """
    Phase 1 RL Environment for Pattern Sequence Discovery

    Action Space: 1536 discrete actions
        - 6 operation types:
          * 0: ^(W pattern) - Ascending Write
          * 1: ^(R pattern) - Ascending Read
          * 2: v(W pattern) - Descending Write
          * 3: v(R pattern) - Descending Read
          * 4: v(W+R pattern) - Descending Write then Read
          * 5: ^(W+R pattern) - Ascending Write then Read
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
        mbist_interface,
        device_id: str = "unknown",
        max_sequence_length: int = 10,
        memory_sampling_rate: float = 1.0,  # 1.0 = full scan, 0.01 = 1%
        verbose: bool = True
    ):
        """
        Initialize Phase 1 Environment

        Args:
            mbist_interface: MBIST hardware interface
            device_id: Device identifier for logging
            max_sequence_length: Maximum actions per episode
            memory_sampling_rate: Fraction of memory to scan (1.0 = full)
            verbose: Print detailed logs
        """
        super().__init__()

        self.mbist = mbist_interface
        self.device_id = device_id
        self.max_seq_len = max_sequence_length
        self.sampling_rate = memory_sampling_rate
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

        self.log(f"Phase1Environment initialized for device: {device_id}")
        self.log(f"  Max sequence length: {max_sequence_length}")
        self.log(f"  Memory sampling: {memory_sampling_rate * 100:.1f}%")
        self.log(f"  Action space: {self.action_space.n} actions")

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        """Reset environment for new episode"""
        super().reset(seed=seed)

        self.current_sequence = []
        self.last_result = 0
        self.total_errors = 0
        self.step_count = 0
        self.episode_count += 1

        self.log(f"\n=== Episode {self.episode_count} Start ===")

        return self._get_observation(), {}

    def step(self, action: int):
        """
        Execute one action (pattern test on entire memory)

        Args:
            action: Action index (0-1535)

        Returns:
            observation, reward, terminated, truncated, info
        """
        # Decode action
        cmd = self._decode_action(action)

        self.log(f"\nStep {self.step_count + 1}: {self._action_to_string(action)}")

        # Execute pattern test on entire memory
        start_time = time.time()
        result = self._execute_pattern_test(
            operation_type=cmd['operation_type'],
            pattern_byte=cmd['pattern_byte']
        )
        execution_time = time.time() - start_time

        # Update state
        self.current_sequence.append(action)
        self.last_result = 1 if result['test_failed'] else 0
        self.total_errors += result['error_count']
        self.step_count += 1

        # Calculate reward
        reward = self._calculate_reward(result, self.step_count)

        # Check termination
        terminated = result['test_failed']  # Found fault!
        truncated = self.step_count >= self.max_seq_len  # Max length reached

        # Info
        info = {
            'test_failed': result['test_failed'],
            'error_count': result['error_count'],
            'sequence': self.current_sequence.copy(),
            'execution_time': execution_time,
            'action_decoded': cmd
        }

        # Log result
        if result['test_failed']:
            self.log(f"  ✓ FAULT DETECTED! Errors: {result['error_count']}")
            self.log(f"  Winning sequence: {[self._action_to_string(a) for a in self.current_sequence]}")
            self.successful_sequences.append(self.current_sequence.copy())
        else:
            self.log(f"  - Test PASS (no faults)")

        self.log(f"  Reward: {reward:.2f}, Time: {execution_time:.3f}s")

        if terminated or truncated:
            self.log(f"\n=== Episode {self.episode_count} End ===")
            self.log(f"  Result: {'SUCCESS' if terminated else 'TRUNCATED'}")
            self.log(f"  Sequence length: {len(self.current_sequence)}")
            self.log(f"  Total successful sequences: {len(self.successful_sequences)}")

        return self._get_observation(), reward, terminated, truncated, info

    def _decode_action(self, action: int) -> Dict:
        """
        Decode action index to command components

        Action encoding: operation_type * 256 + pattern_byte

        Args:
            action: Action index (0-1535)

        Returns:
            Dict with operation_type, pattern_byte
        """
        pattern_byte = action % 256
        operation_type = action // 256

        return {
            'operation_type': OperationType(operation_type),
            'pattern_byte': pattern_byte
        }

    def _action_to_string(self, action: int) -> str:
        """Convert action to readable string"""
        cmd = self._decode_action(action)

        op_type = cmd['operation_type']
        pattern = cmd['pattern_byte']

        # Map operation type to notation
        op_map = {
            OperationType.WRITE_ASC: f"^(W 0x{pattern:02X})",
            OperationType.READ_ASC: f"^(R 0x{pattern:02X})",
            OperationType.WRITE_DESC: f"v(W 0x{pattern:02X})",
            OperationType.READ_DESC: f"v(R 0x{pattern:02X})",
            OperationType.WRITE_READ_DESC: f"v(W 0x{pattern:02X}, R 0x{pattern:02X})",
            OperationType.WRITE_READ_ASC: f"^(W 0x{pattern:02X}, R 0x{pattern:02X})"
        }

        return op_map[op_type]

    def _execute_pattern_test(
        self,
        operation_type: OperationType,
        pattern_byte: int
    ) -> Dict:
        """
        Execute pattern test on entire memory

        Performs one of 6 operation types:
        - WRITE_ASC: ^(W pattern) - Ascending Write
        - READ_ASC: ^(R pattern) - Ascending Read & Compare
        - WRITE_DESC: v(W pattern) - Descending Write
        - READ_DESC: v(R pattern) - Descending Read & Compare
        - WRITE_READ_DESC: v(W+R pattern) - Descending Write then Read
        - WRITE_READ_ASC: ^(W+R pattern) - Ascending Write then Read

        Args:
            operation_type: One of 6 operation types
            pattern_byte: Data pattern (0x00 ~ 0xFF)

        Returns:
            Dict with test_failed, error_count
        """
        # Determine direction
        is_ascending = operation_type in [
            OperationType.WRITE_ASC,
            OperationType.READ_ASC,
            OperationType.WRITE_READ_ASC
        ]

        # Generate address sequence
        addresses = self._generate_address_sequence(is_ascending)

        # Begin MBIST sequence mode
        self.mbist.begin_sequence()

        # Execute based on operation type
        if operation_type == OperationType.WRITE_ASC or operation_type == OperationType.WRITE_DESC:
            # Write only: ^(W pattern) or v(W pattern)
            for rank, bg, ba, row in addresses:
                self.mbist.send_activate(rank, bg, ba, row)
                for col in range(0, 2048, 16):
                    self.mbist.send_write(rank, bg, ba, row, col, pattern_byte)
                self.mbist.send_precharge(rank, bg, ba, all_banks=False)

        elif operation_type == OperationType.READ_ASC or operation_type == OperationType.READ_DESC:
            # Read only: ^(R pattern) or v(R pattern)
            # Assumes memory was previously written with this pattern
            for rank, bg, ba, row in addresses:
                self.mbist.send_activate(rank, bg, ba, row)
                for col in range(0, 2048, 16):
                    self.mbist.send_read(rank, bg, ba, row, col)
                self.mbist.send_precharge(rank, bg, ba, all_banks=False)

        elif operation_type == OperationType.WRITE_READ_ASC or operation_type == OperationType.WRITE_READ_DESC:
            # Write then Read: ^(W+R pattern) or v(W+R pattern)
            # First pass: Write
            for rank, bg, ba, row in addresses:
                self.mbist.send_activate(rank, bg, ba, row)
                for col in range(0, 2048, 16):
                    self.mbist.send_write(rank, bg, ba, row, col, pattern_byte)
                self.mbist.send_precharge(rank, bg, ba, all_banks=False)

            # Second pass: Read (same direction)
            for rank, bg, ba, row in addresses:
                self.mbist.send_activate(rank, bg, ba, row)
                for col in range(0, 2048, 16):
                    self.mbist.send_read(rank, bg, ba, row, col)
                self.mbist.send_precharge(rank, bg, ba, all_banks=False)

        # Execute and wait for completion
        self.mbist.end_sequence()

        # Get test result
        test_result = self.mbist.get_test_result()
        error_count = self.mbist.get_error_count()

        return {
            'test_failed': (test_result != 0),
            'error_count': error_count
        }

    def _generate_address_sequence(self, is_ascending: bool) -> List[Tuple]:
        """
        Generate address sequence for memory scan

        Args:
            is_ascending: True for ascending (⇑), False for descending (⇓)

        Returns:
            List of (rank, bg, ba, row) tuples
        """
        addresses = []

        # Full address space
        for rank in range(4):
            for bg in range(8):
                for ba in range(4):
                    for row in range(262144):
                        addresses.append((rank, bg, ba, row))

        # Apply sampling if needed
        if self.sampling_rate < 1.0:
            sample_size = int(len(addresses) * self.sampling_rate)
            # Systematic sampling to maintain coverage
            step = len(addresses) // sample_size
            addresses = addresses[::step]

        # Apply direction
        if not is_ascending:
            addresses.reverse()

        return addresses

    def _calculate_reward(self, result: Dict, sequence_length: int) -> float:
        """
        Calculate reward for current step

        Phase 1 reward strategy:
        - High reward for finding fault (earlier = better)
        - Small penalty for each step (encourage efficiency)

        Args:
            result: Test result dict
            sequence_length: Current sequence length

        Returns:
            Reward value
        """
        if result['test_failed']:
            # Found fault! Reward inversely proportional to sequence length
            # Shorter sequences = better
            return 1000.0 / sequence_length
        else:
            # No fault found, small penalty
            return -0.1

    def _get_observation(self) -> Dict:
        """Get current observation"""
        # Pad sequence history to fixed length
        seq_history = np.zeros(self.max_seq_len, dtype=np.int32)
        if len(self.current_sequence) > 0:
            seq_len = min(len(self.current_sequence), self.max_seq_len)
            seq_history[:seq_len] = self.current_sequence[:seq_len]

        return {
            'sequence_history': seq_history,
            'last_test_result': self.last_result,
            'sequence_length': len(self.current_sequence),
            'cumulative_errors': np.array([self.total_errors], dtype=np.float32)
        }

    def log(self, message: str):
        """Print log message if verbose"""
        if self.verbose:
            print(message)

    def get_statistics(self) -> Dict:
        """Get training statistics"""
        return {
            'total_episodes': self.episode_count,
            'successful_sequences': len(self.successful_sequences),
            'success_rate': len(self.successful_sequences) / max(1, self.episode_count),
            'sequences': self.successful_sequences
        }

    def render(self, mode='human'):
        """Render environment state"""
        if mode == 'human':
            print(f"\n--- Environment State ---")
            print(f"Device: {self.device_id}")
            print(f"Episode: {self.episode_count}")
            print(f"Step: {self.step_count}/{self.max_seq_len}")
            print(f"Current sequence:")
            for i, action in enumerate(self.current_sequence):
                print(f"  {i+1}. {self._action_to_string(action)}")
            print(f"Last result: {'FAIL' if self.last_result else 'PASS'}")
            print(f"Total errors: {self.total_errors}")
            print(f"------------------------\n")


# Helper function to create environment
def make_phase1_env(
    mbist_interface,
    device_id: str = "unknown",
    **kwargs
) -> Phase1Environment:
    """
    Create Phase 1 environment with default settings

    Args:
        mbist_interface: MBIST hardware interface
        device_id: Device identifier
        **kwargs: Additional environment arguments

    Returns:
        Configured Phase1Environment
    """
    return Phase1Environment(
        mbist_interface=mbist_interface,
        device_id=device_id,
        **kwargs
    )


if __name__ == "__main__":
    """Test environment with mock MBIST"""
    from mbist_interface import MBISTInterface

    # Create mock MBIST interface
    mbist = MBISTInterface()  # Runs in mock mode

    # Create environment
    env = Phase1Environment(
        mbist_interface=mbist,
        device_id="TEST-DEVICE",
        max_sequence_length=3,
        memory_sampling_rate=0.0001,  # 0.01% sampling for fast testing
        verbose=True
    )

    print("=== Phase 1 Environment Test ===\n")

    # Test random episode
    obs, info = env.reset()
    done = False

    while not done:
        # Random action
        action = env.action_space.sample()

        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

    # Print statistics
    stats = env.get_statistics()
    print("\n=== Statistics ===")
    print(f"Episodes: {stats['total_episodes']}")
    print(f"Successful: {stats['successful_sequences']}")
    print(f"Success rate: {stats['success_rate']:.1%}")
