"""
CXL Memory Fault Detection Reinforcement Learning Environment
Phase 1: Low-level DRAM Command Control

최신 설계 (2025-11-01):
- Action: Low-level DRAM primitives (ACT, WR, RD, PRE 조합)
- Goal: 불량 device에서 불량을 재현하는 command 시퀀스 발견
- Reward: 불량 발견에 집중 (10000점)
"""

import gymnasium as gym
import numpy as np
import time
from typing import Dict, List, Tuple, Any, Optional
from enum import IntEnum
import logging

logger = logging.getLogger(__name__)


class DRAMPrimitive(IntEnum):
    """Low-level DRAM operation primitives for RL"""
    # Basic operations
    WRITE_READ_CELL = 0      # 단일 셀 W/R
    WRITE_READ_ROW = 1       # 전체 row W/R
    WRITE_READ_BANK = 2      # 전체 bank W/R

    # Stress operations
    ROW_HAMMER = 3           # 특정 row 반복 액세스 (rowhammer)
    BANK_THRASH = 4          # Bank 간 빠른 전환
    REFRESH_STRESS = 5       # Refresh 지연/스킵

    # Pattern operations
    CHECKERBOARD_WR = 6      # Checkerboard 패턴 쓰기
    WALKING_ONES = 7         # Walking 1s
    PRBS_PATTERN = 8         # PRBS 랜덤 패턴

    # March-like operations
    ASCENDING_MARCH = 9      # 주소 증가 순서
    DESCENDING_MARCH = 10    # 주소 감소 순서

    # Retention test
    WRITE_DELAY_READ = 11    # 쓰기 → 대기 → 읽기


class DataPattern(IntEnum):
    """Data patterns for DRAM testing"""
    FIXED_00 = 0        # All 0x00
    FIXED_FF = 1        # All 0xFF
    FIXED_55 = 2        # 0x55555555...
    FIXED_AA = 3        # 0xAAAAAAAA...
    CHECKERBOARD = 4    # Alternating 0x55/0xAA
    PRBS = 5            # Pseudo-Random Binary Sequence
    WALKING_1S = 6      # Walking 1s
    WALKING_0S = 7      # Walking 0s


class TestResult:
    """MBIST test result container"""
    def __init__(self, test_passed: bool = True, error_addresses: Optional[List] = None,
                 duration: float = 0.0, error_count: int = 0):
        self.test_passed = test_passed
        self.error_addresses = error_addresses or []
        self.duration = duration
        self.error_count = error_count


class RewardCalculator:
    """Phase 1: 불량 발견에 집중하는 보상 함수"""

    def __init__(self):
        self.total_tests = 0
        self.tested_regions = set()

    def calculate(self, action: np.ndarray, result: TestResult) -> float:
        """Calculate reward based on test result"""
        reward = 0.0

        # 1. 불량 발견 (최우선!)
        if not result.test_passed:  # FAIL detected
            reward += 10000  # 매우 높은 보상

            # 에러 주소 정보가 있으면 추가 보상
            if result.error_addresses:
                reward += len(result.error_addresses) * 100

            logger.info(f"🎯 FAULT DETECTED! Reward: {reward}, Errors: {len(result.error_addresses)}")

        # 2. 패스 (정보 제공)
        else:
            reward += 1  # 작은 보상 (커버리지)

        # 3. 탐색 보너스 (초기)
        primitive, rank, bg, ba, row_start, row_end, pattern, repeat = action
        region_key = (rank, bg, ba, row_start // 16)  # 16 row groups

        if region_key not in self.tested_regions:
            reward += 10  # 새 영역 탐색 보너스
            self.tested_regions.add(region_key)

        # 4. 효율성 (적은 테스트로 불량 발견)
        if not result.test_passed:
            efficiency_bonus = 1000 / max(1, self.total_tests)
            reward += efficiency_bonus

        # 5. 다양성 보너스
        if self._is_diverse_action(action):
            reward += 5

        self.total_tests += 1

        return reward

    def _is_diverse_action(self, action: np.ndarray) -> bool:
        """Check if action is diverse (not too repetitive)"""
        primitive, rank, bg, ba, row_start, row_end, pattern, repeat = action

        # 다른 primitive, 다른 패턴 사용 시 diverse
        return primitive not in [0, 1] or pattern not in [2, 3]


class DRAMCommandRLEnvironment(gym.Env):
    """
    Low-level DRAM Command RL Environment for Phase 1

    Goal: 불량 device에서 불량을 재현하는 DRAM command 시퀀스 발견
    """

    metadata = {'render_modes': ['human']}

    def __init__(self,
                 mbist_interface=None,
                 faulty_device_id: str = "CXL-SIM-001",
                 max_tests: int = 1000,
                 safety_mode: bool = True):
        """
        Initialize DRAM Command RL Environment

        Args:
            mbist_interface: MBISTInterface object (None for simulation)
            faulty_device_id: Device ID to test
            max_tests: Maximum tests per episode
            safety_mode: Use simulator instead of real hardware
        """
        super().__init__()

        self.mbist = mbist_interface
        self.device_id = faulty_device_id
        self.max_tests = max_tests
        self.safety_mode = safety_mode

        # Action space: [primitive, rank, bg, ba, row_start, row_end, pattern, repeat]
        self.action_space = gym.spaces.MultiDiscrete([
            12,   # primitives (DRAMPrimitive)
            4,    # ranks
            8,    # bank groups
            4,    # banks
            256,  # row_start (grouped: 262144 rows / 1024)
            256,  # row_end (grouped)
            8,    # patterns (DataPattern)
            10    # repeat count (1-10)
        ])

        # Observation space
        # State: [rank, bank_group, bank, row_group]
        self.observation_space = gym.spaces.Dict({
            'memory_map': gym.spaces.Box(
                low=0, high=1,
                shape=(4, 8, 4, 256),
                dtype=np.float32
            ),
            'fault_map': gym.spaces.Box(
                low=0, high=1,
                shape=(4, 8, 4, 256),
                dtype=np.float32
            ),
            'coverage': gym.spaces.Box(
                low=0, high=1,
                shape=(4, 8, 4, 256),
                dtype=np.float32
            ),
            'recent_commands': gym.spaces.Box(
                low=0, high=1,
                shape=(10, 8),  # Last 10 commands, 8 params each
                dtype=np.float32
            ),
            'metadata': gym.spaces.Box(
                low=0, high=np.inf,
                shape=(4,),  # total_tests, faults_found, coverage_ratio, confidence
                dtype=np.float32
            )
        })

        # State variables
        self.memory_map = None
        self.fault_map = None
        self.test_coverage = None
        self.confidence = None
        self.recent_commands = None
        self.command_history = []

        self.total_tests = 0
        self.faults_found = 0

        # Reward calculator
        self.reward_calculator = RewardCalculator()

        # Simulated fault map (for safety mode)
        self._init_simulated_faults()

    def _init_simulated_faults(self):
        """Initialize simulated fault patterns for testing"""
        # 시뮬레이션 모드에서만 사용
        self.sim_faults = []

        # 몇 가지 알려진 불량 패턴 추가
        # Pattern 1: Specific row in specific bank
        self.sim_faults.append({
            'type': 'stuck_at',
            'rank': 0,
            'bg': 0,
            'ba': 0,
            'row': 100 * 1024,  # Row group 100
            'detection_primitives': [DRAMPrimitive.WRITE_READ_ROW, DRAMPrimitive.WRITE_READ_BANK]
        })

        # Pattern 2: Row hammer victim
        self.sim_faults.append({
            'type': 'rowhammer',
            'rank': 0,
            'bg': 1,
            'ba': 2,
            'row': 150 * 1024,
            'detection_primitives': [DRAMPrimitive.ROW_HAMMER]
        })

        # Pattern 3: Retention failure
        self.sim_faults.append({
            'type': 'retention',
            'rank': 1,
            'bg': 3,
            'ba': 1,
            'row': 200 * 1024,
            'detection_primitives': [DRAMPrimitive.WRITE_DELAY_READ, DRAMPrimitive.REFRESH_STRESS]
        })

    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[Dict, Dict]:
        """Reset environment to initial state"""
        super().reset(seed=seed)

        # Reset state
        self.memory_map = np.zeros((4, 8, 4, 256), dtype=np.float32)
        self.fault_map = np.zeros((4, 8, 4, 256), dtype=np.float32)
        self.test_coverage = np.zeros((4, 8, 4, 256), dtype=np.int32)
        self.confidence = np.zeros((4, 8, 4, 256), dtype=np.float32)
        self.recent_commands = np.zeros((10, 8), dtype=np.float32)
        self.command_history = []

        self.total_tests = 0
        self.faults_found = 0

        # Reset reward calculator
        self.reward_calculator = RewardCalculator()

        observation = self._get_observation()
        info = self._get_info()

        return observation, info

    def step(self, action: np.ndarray) -> Tuple[Dict, float, bool, bool, Dict]:
        """Execute one step in the environment"""

        # Decode action
        primitive, rank, bg, ba, row_start, row_end, pattern, repeat = action
        primitive = DRAMPrimitive(primitive)
        pattern = DataPattern(pattern)

        # Ensure row_end >= row_start
        if row_end < row_start:
            row_start, row_end = row_end, row_start

        # Execute primitive
        result = self._execute_primitive(
            primitive=primitive,
            rank=rank,
            bank_group=bg,
            bank=ba,
            row_start=row_start * 1024,  # Ungroup to actual row number
            row_end=row_end * 1024,
            pattern=pattern,
            repeat=repeat + 1  # 1-10
        )

        # Calculate reward
        reward = self.reward_calculator.calculate(action, result)

        # Update state
        self._update_state(action, result)

        # Update command history
        self._update_command_history(action)

        # Termination
        terminated = (
            self.total_tests >= self.max_tests or
            self.faults_found >= 1  # Phase 1: 하나만 찾으면 성공
        )
        truncated = False

        self.total_tests += 1

        observation = self._get_observation()
        info = self._get_info()
        info['test_result'] = result.test_passed
        info['error_count'] = result.error_count
        info['command_sequence'] = self.command_history

        return observation, reward, terminated, truncated, info

    def _execute_primitive(self, primitive: DRAMPrimitive, rank: int, bank_group: int,
                          bank: int, row_start: int, row_end: int,
                          pattern: DataPattern, repeat: int) -> TestResult:
        """Execute DRAM primitive operation"""

        if self.safety_mode or self.mbist is None:
            # Simulation mode
            return self._simulate_primitive(primitive, rank, bank_group, bank,
                                           row_start, row_end, pattern, repeat)
        else:
            # Real hardware execution
            return self._execute_hardware_primitive(primitive, rank, bank_group, bank,
                                                   row_start, row_end, pattern, repeat)

    def _simulate_primitive(self, primitive: DRAMPrimitive, rank: int, bank_group: int,
                           bank: int, row_start: int, row_end: int,
                           pattern: DataPattern, repeat: int) -> TestResult:
        """Simulate primitive execution for testing"""

        # Check if this primitive can detect any simulated faults
        row_group_start = row_start // 1024
        row_group_end = row_end // 1024

        errors = []

        for fault in self.sim_faults:
            # Check if fault is in tested region
            if (fault['rank'] == rank and
                fault['bg'] == bank_group and
                fault['ba'] == bank):

                fault_row_group = fault['row'] // 1024

                if row_group_start <= fault_row_group <= row_group_end:
                    # Check if primitive can detect this fault
                    if primitive in fault['detection_primitives']:
                        # Detection probability based on repeat count
                        detection_prob = min(0.95, 0.5 + (repeat * 0.05))

                        if np.random.random() < detection_prob:
                            errors.append({
                                'rank': rank,
                                'bg': bank_group,
                                'ba': bank,
                                'row': fault['row'],
                                'type': fault['type']
                            })

        # Simulate duration
        duration = 0.01 + (row_group_end - row_group_start + 1) * 0.001 * repeat

        test_passed = len(errors) == 0

        if not test_passed:
            logger.info(f"✓ Simulated fault detected: {primitive.name} on "
                       f"rank={rank} bg={bank_group} ba={bank} rows={row_start}-{row_end}")

        return TestResult(
            test_passed=test_passed,
            error_addresses=errors,
            duration=duration,
            error_count=len(errors)
        )

    def _execute_hardware_primitive(self, primitive: DRAMPrimitive, rank: int,
                                    bank_group: int, bank: int, row_start: int,
                                    row_end: int, pattern: DataPattern,
                                    repeat: int) -> TestResult:
        """Execute primitive on real hardware via MBIST interface"""

        if self.mbist is None:
            raise RuntimeError("MBIST interface not initialized")

        start_time = time.time()

        try:
            # This will be implemented when MBIST C library wrapper is ready
            # For now, return a placeholder
            logger.warning("Hardware execution not yet implemented, using simulation")
            return self._simulate_primitive(primitive, rank, bank_group, bank,
                                           row_start, row_end, pattern, repeat)

        except Exception as e:
            logger.error(f"Hardware test failed: {e}")
            return TestResult(
                test_passed=True,  # Conservative: assume no fault if test fails
                error_addresses=[],
                duration=time.time() - start_time,
                error_count=0
            )

    def _update_state(self, action: np.ndarray, result: TestResult):
        """Update environment state based on test result"""

        primitive, rank, bg, ba, row_start, row_end, pattern, repeat = action

        # Update test coverage
        for row_group in range(row_start, min(row_end + 1, 256)):
            self.test_coverage[rank, bg, ba, row_group] += 1

        # Update fault map if errors found
        if not result.test_passed:
            for row_group in range(row_start, min(row_end + 1, 256)):
                self.fault_map[rank, bg, ba, row_group] = 1.0

            self.faults_found += result.error_count

        # Update memory map (normalized fault intensity)
        if result.error_count > 0:
            for row_group in range(row_start, min(row_end + 1, 256)):
                self.memory_map[rank, bg, ba, row_group] = min(1.0, result.error_count / 10.0)

        # Update confidence (higher confidence in well-tested areas)
        for row_group in range(row_start, min(row_end + 1, 256)):
            coverage = self.test_coverage[rank, bg, ba, row_group]
            self.confidence[rank, bg, ba, row_group] = min(1.0, coverage / 5.0)

    def _update_command_history(self, action: np.ndarray):
        """Update recent command history"""

        # Normalize action to [0, 1] range
        normalized_action = action.astype(np.float32) / np.array([12, 4, 8, 4, 256, 256, 8, 10])

        # Shift history and add new command
        self.recent_commands = np.roll(self.recent_commands, shift=1, axis=0)
        self.recent_commands[0] = normalized_action

        # Store full history
        self.command_history.append(action.tolist())

    def _get_observation(self) -> Dict:
        """Get current observation"""

        # Normalize coverage
        coverage_norm = np.clip(self.test_coverage / 10.0, 0, 1).astype(np.float32)

        # Metadata: [total_tests, faults_found, coverage_ratio, avg_confidence]
        total_coverage = np.sum(self.test_coverage > 0)
        total_regions = 4 * 8 * 4 * 256
        coverage_ratio = total_coverage / total_regions
        avg_confidence = np.mean(self.confidence)

        metadata = np.array([
            self.total_tests / self.max_tests,  # Normalized
            min(1.0, self.faults_found / 10.0),  # Normalized
            coverage_ratio,
            avg_confidence
        ], dtype=np.float32)

        return {
            'memory_map': self.memory_map,
            'fault_map': self.fault_map,
            'coverage': coverage_norm,
            'recent_commands': self.recent_commands,
            'metadata': metadata
        }

    def _get_info(self) -> Dict:
        """Get additional information"""

        return {
            'total_tests': self.total_tests,
            'faults_found': self.faults_found,
            'coverage': np.sum(self.test_coverage > 0),
            'total_regions': 4 * 8 * 4 * 256,
            'device_id': self.device_id
        }

    def render(self, mode='human'):
        """Render environment state"""

        if mode == 'human':
            print(f"\n{'='*60}")
            print(f"Device: {self.device_id}")
            print(f"Tests: {self.total_tests}/{self.max_tests}")
            print(f"Faults Found: {self.faults_found}")
            print(f"Coverage: {np.sum(self.test_coverage > 0)}/{4 * 8 * 4 * 256} regions")
            print(f"Avg Confidence: {np.mean(self.confidence):.3f}")

            if self.command_history:
                print(f"\nLast Command:")
                last_cmd = self.command_history[-1]
                primitive = DRAMPrimitive(last_cmd[0])
                print(f"  Primitive: {primitive.name}")
                print(f"  Location: rank={last_cmd[1]} bg={last_cmd[2]} ba={last_cmd[3]}")
                print(f"  Rows: {last_cmd[4]}-{last_cmd[5]}")
                print(f"  Pattern: {DataPattern(last_cmd[6]).name}")
                print(f"  Repeat: {last_cmd[7] + 1}")

            print(f"{'='*60}\n")


# Helper function for creating environment
def make_env(safety_mode: bool = True, max_tests: int = 1000,
             mbist_interface=None, device_id: str = "CXL-SIM-001") -> DRAMCommandRLEnvironment:
    """
    Factory function to create RL environment

    Args:
        safety_mode: Use simulator (True) or real hardware (False)
        max_tests: Maximum tests per episode
        mbist_interface: MBIST hardware interface (None for simulation)
        device_id: Device ID to test

    Returns:
        DRAMCommandRLEnvironment instance
    """
    return DRAMCommandRLEnvironment(
        mbist_interface=mbist_interface,
        faulty_device_id=device_id,
        max_tests=max_tests,
        safety_mode=safety_mode
    )
