"""
CXL Memory Fault Detection Reinforcement Learning Environment
"""

import gymnasium as gym
import numpy as np
import subprocess
import os
import time
import json
from typing import Dict, List, Tuple, Any, Optional
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class MemoryTestAction(Enum):
    """Available MBIST test actions"""
    PPR = 0x00
    WRC = 0x01           # Write, Read and Compare
    CHECKERBOARD = 0x02
    MARCH_C_PLUS = 0x03
    MATS = 0x04
    MATS_PLUS = 0x05
    WALKING_1S = 0x06
    WALKING_0S = 0x07
    MARCH_X = 0x08
    MARCH_Y = 0x09
    MARCHING_1_0 = 0x0a
    TPH = 0x0c
    WRITE_READ_1BL16 = 0x0e
    MRW_MRR = 0x0f


class DataPattern(Enum):
    """Data patterns for testing"""
    PRBS = 0x4
    PATTERN_55H = 0x10
    PATTERN_AAH = 0x20
    ALL_0 = 0x40
    ALL_1 = 0x80
    CHECKERBOARD = 0x100


class CXLMemoryRLEnvironment(gym.Env):
    """
    Reinforcement Learning Environment for CXL Memory Fault Detection
    
    State: Current memory test results, failure patterns, test history
    Action: Select next test algorithm and memory region
    Reward: Based on new fault discovery and test efficiency
    """
    
    def __init__(self, 
                 mbist_binary_path: str = "/home/dhkang/data2/mbist_sample_code-gen2_es/bin/mbist_smbus.exe",
                 memory_size: int = 0x100000000,  # 4GB
                 max_episode_steps: int = 1000,
                 safety_mode: bool = True):
        
        super().__init__()
        
        self.mbist_binary_path = mbist_binary_path
        self.memory_size = memory_size
        self.max_episode_steps = max_episode_steps
        self.safety_mode = safety_mode
        
        # Environment state
        self.current_step = 0
        self.total_faults_found = 0
        self.test_history = []
        self.memory_map = np.zeros((64, 64), dtype=np.float32)  # 64x64 grid representing memory regions
        self.fault_map = np.zeros((64, 64), dtype=np.int32)     # Known fault locations
        self.test_coverage = np.zeros((64, 64), dtype=np.int32) # Test coverage map
        
        # Action space: [algorithm_id, pattern_id, start_region, end_region]
        self.action_space = gym.spaces.MultiDiscrete([
            len(MemoryTestAction),  # Algorithm selection
            len(DataPattern),       # Pattern selection  
            64,                     # Start region (row)
            64                      # End region (row)
        ])
        
        # Observation space: memory map + fault map + coverage + metadata
        self.observation_space = gym.spaces.Box(
            low=0.0,
            high=1.0,
            shape=(64, 64, 4),  # memory_map, fault_map, coverage, confidence
            dtype=np.float32
        )
        
        # Reward parameters
        self.reward_params = {
            'new_fault_reward': 100.0,
            'coverage_reward': 1.0,
            'efficiency_bonus': 10.0,
            'redundant_test_penalty': -5.0,
            'time_penalty': -0.1,
            'safety_violation_penalty': -50.0
        }
        
        # Safety limits
        self.safety_limits = {
            'max_test_duration': 60,  # seconds
            'min_cooldown_time': 2,   # seconds between tests
            'max_consecutive_fails': 10,
            'max_region_size': 0x10000000  # 256MB max test region
        }
        
        self.last_test_time = 0
        self.consecutive_fails = 0
        
    def reset(self, seed: Optional[int] = None) -> Tuple[np.ndarray, Dict]:
        """Reset the environment to initial state"""
        super().reset(seed=seed)
        
        self.current_step = 0
        self.total_faults_found = 0
        self.test_history.clear()
        self.memory_map.fill(0.0)
        self.fault_map.fill(0)
        self.test_coverage.fill(0)
        self.consecutive_fails = 0
        
        # Initialize with some random fault patterns (simulation)
        if self.safety_mode:  # Only in simulation mode
            num_faults = np.random.randint(1, 10)
            for _ in range(num_faults):
                fault_row = np.random.randint(0, 64)
                fault_col = np.random.randint(0, 64)
                self.fault_map[fault_row, fault_col] = 1
        
        observation = self._get_observation()
        info = self._get_info()
        
        return observation, info
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Execute one step in the environment"""
        
        algorithm_id, pattern_id, start_region, end_region = action
        
        # Ensure end_region >= start_region
        if end_region < start_region:
            start_region, end_region = end_region, start_region
        
        # Safety checks
        if self.safety_mode:
            current_time = time.time()
            if current_time - self.last_test_time < self.safety_limits['min_cooldown_time']:
                time.sleep(self.safety_limits['min_cooldown_time'])
            
            if self.consecutive_fails >= self.safety_limits['max_consecutive_fails']:
                logger.warning("Too many consecutive failures, forced cooldown")
                time.sleep(self.safety_limits['min_cooldown_time'] * 5)
                self.consecutive_fails = 0
        
        # Execute test
        test_result = self._execute_mbist_test(algorithm_id, pattern_id, start_region, end_region)
        
        # Calculate reward
        reward = self._calculate_reward(test_result, start_region, end_region)
        
        # Update state
        self._update_state(test_result, start_region, end_region)
        
        # Check termination conditions
        terminated = self.current_step >= self.max_episode_steps
        truncated = False
        
        self.current_step += 1
        self.last_test_time = time.time()
        
        observation = self._get_observation()
        info = self._get_info()
        info['test_result'] = test_result
        
        return observation, reward, terminated, truncated, info
    
    def _execute_mbist_test(self, algorithm_id: int, pattern_id: int, 
                          start_region: int, end_region: int) -> Dict:
        """Execute MBIST test using the binary"""
        
        # Convert regions to memory addresses
        start_addr = (start_region * self.memory_size) // 64
        end_addr = ((end_region + 1) * self.memory_size) // 64 - 1
        
        # Ensure valid address range
        start_addr = max(0, start_addr)
        end_addr = min(self.memory_size - 1, end_addr)
        
        test_start_time = time.time()
        
        try:
            # Construct command based on available algorithms
            algorithm_map = {
                0: 0x00, 1: 0x01, 2: 0x02, 3: 0x03, 4: 0x04,
                5: 0x05, 6: 0x06, 7: 0x07, 8: 0x08, 9: 0x09,
                10: 0x0a, 11: 0x0c, 12: 0x0e, 13: 0x0f
            }
            
            pattern_map = {
                0: 0x4, 1: 0x10, 2: 0x20, 3: 0x40, 4: 0x80, 5: 0x100
            }
            
            cmd = [
                self.mbist_binary_path,
                "-i", hex(algorithm_map.get(algorithm_id, 0x01)),
                "-p", hex(pattern_map.get(pattern_id, 0x10)),
                "-s", hex(start_addr),
                "-e", hex(end_addr)
            ]
            
            # In safety mode, use simulator instead of real hardware
            if self.safety_mode:
                result = self._simulate_test(algorithm_id, pattern_id, start_region, end_region)
            else:
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=self.safety_limits['max_test_duration']
                )
                result = self._parse_mbist_output(result.stdout, result.stderr)
            
            test_duration = time.time() - test_start_time
            result['duration'] = test_duration
            result['algorithm'] = algorithm_id
            result['pattern'] = pattern_id
            result['start_region'] = start_region
            result['end_region'] = end_region
            
            return result
            
        except subprocess.TimeoutExpired:
            logger.error("MBIST test timeout")
            self.consecutive_fails += 1
            return {
                'status': 'timeout',
                'faults_found': 0,
                'duration': self.safety_limits['max_test_duration'],
                'error': 'Test timeout'
            }
        except Exception as e:
            logger.error(f"MBIST test failed: {e}")
            self.consecutive_fails += 1
            return {
                'status': 'error',
                'faults_found': 0,
                'duration': 0,
                'error': str(e)
            }
    
    def _simulate_test(self, algorithm_id: int, pattern_id: int, 
                      start_region: int, end_region: int) -> Dict:
        """Simulate test execution for safety"""
        
        # Simulate test based on known fault map
        faults_in_region = 0
        for row in range(start_region, min(end_region + 1, 64)):
            for col in range(64):
                if self.fault_map[row, col] == 1:
                    # Algorithm effectiveness varies
                    detection_probability = self._get_detection_probability(algorithm_id, pattern_id)
                    if np.random.random() < detection_probability:
                        faults_in_region += 1
        
        # Simulate some false positives and noise
        if faults_in_region == 0 and np.random.random() < 0.1:  # 10% false positive rate
            faults_in_region = np.random.randint(1, 3)
        
        # Simulate test duration
        region_size = end_region - start_region + 1
        base_duration = 0.1 + (region_size * 0.01)  # Larger regions take longer
        duration = base_duration + np.random.normal(0, 0.02)
        
        return {
            'status': 'pass' if faults_in_region == 0 else 'fail',
            'faults_found': faults_in_region,
            'duration': max(0.01, duration),
            'simulated': True
        }
    
    def _get_detection_probability(self, algorithm_id: int, pattern_id: int) -> float:
        """Get fault detection probability for algorithm/pattern combination"""
        
        # Different algorithms have different effectiveness
        algorithm_effectiveness = {
            0: 0.3,   # PPR
            1: 0.7,   # WRC
            2: 0.8,   # Checkerboard
            3: 0.9,   # March C+
            4: 0.6,   # MATS
            5: 0.7,   # MATS+
            6: 0.75,  # Walking 1s
            7: 0.75,  # Walking 0s
            8: 0.85,  # March X
            9: 0.87,  # March Y
            10: 0.9,  # Marching 1/0
            11: 0.6,  # TPH
            12: 0.5,  # Write Read 1BL16
            13: 0.4   # MRW MRR
        }
        
        # Pattern effectiveness varies
        pattern_effectiveness = {
            0: 0.8,   # PRBS
            1: 0.7,   # 55h
            2: 0.7,   # AAh
            3: 0.6,   # All 0
            4: 0.6,   # All 1
            5: 0.9    # Checkerboard
        }
        
        base_prob = algorithm_effectiveness.get(algorithm_id, 0.5)
        pattern_bonus = pattern_effectiveness.get(pattern_id, 0.5) * 0.2
        
        return min(0.95, base_prob + pattern_bonus)
    
    def _parse_mbist_output(self, stdout: str, stderr: str) -> Dict:
        """Parse MBIST binary output"""
        
        if stderr:
            return {
                'status': 'error',
                'faults_found': 0,
                'error': stderr
            }
        
        # Parse stdout for test results
        faults_found = 0
        if "FAIL" in stdout.upper():
            # Count error patterns in output
            faults_found = stdout.upper().count("ERROR") + stdout.upper().count("FAIL")
            status = 'fail'
        else:
            status = 'pass'
        
        return {
            'status': status,
            'faults_found': faults_found,
            'output': stdout
        }
    
    def _calculate_reward(self, test_result: Dict, start_region: int, end_region: int) -> float:
        """Calculate reward based on test result"""
        
        reward = 0.0
        
        # Reward for finding new faults
        faults_found = test_result.get('faults_found', 0)
        if faults_found > 0:
            reward += self.reward_params['new_fault_reward'] * faults_found
            self.consecutive_fails = 0
        else:
            # Only count as consecutive fail if it's an actual error, not just no faults found
            if test_result.get('status') in ['error', 'timeout']:
                self.consecutive_fails += 1
        
        # Coverage reward
        region_size = end_region - start_region + 1
        coverage_bonus = self.reward_params['coverage_reward'] * region_size
        reward += coverage_bonus
        
        # Efficiency bonus (more faults found per time)
        duration = test_result.get('duration', 1.0)
        if faults_found > 0 and duration > 0:
            efficiency = faults_found / duration
            reward += self.reward_params['efficiency_bonus'] * efficiency
        
        # Penalty for redundant testing
        if self._is_redundant_test(start_region, end_region):
            reward += self.reward_params['redundant_test_penalty']
        
        # Time penalty
        reward += self.reward_params['time_penalty'] * duration
        
        # Safety violations
        if test_result.get('status') == 'error' or test_result.get('status') == 'timeout':
            reward += self.reward_params['safety_violation_penalty']
        
        return reward
    
    def _is_redundant_test(self, start_region: int, end_region: int) -> bool:
        """Check if this test region has been heavily tested already"""
        
        total_coverage = 0
        region_size = end_region - start_region + 1
        
        for row in range(start_region, min(end_region + 1, 64)):
            total_coverage += np.sum(self.test_coverage[row, :])
        
        avg_coverage = total_coverage / (region_size * 64)
        return avg_coverage > 5  # Consider redundant if tested more than 5 times on average
    
    def _update_state(self, test_result: Dict, start_region: int, end_region: int):
        """Update environment state based on test result"""
        
        # Update test coverage
        for row in range(start_region, min(end_region + 1, 64)):
            self.test_coverage[row, :] += 1
        
        # Update memory map with test results
        faults_found = test_result.get('faults_found', 0)
        if faults_found > 0:
            for row in range(start_region, min(end_region + 1, 64)):
                self.memory_map[row, :] = np.maximum(
                    self.memory_map[row, :], 
                    faults_found / 10.0  # Normalize fault intensity
                )
            self.total_faults_found += faults_found
        
        # Record test in history
        self.test_history.append({
            'step': self.current_step,
            'algorithm': test_result.get('algorithm'),
            'pattern': test_result.get('pattern'),
            'start_region': start_region,
            'end_region': end_region,
            'faults_found': faults_found,
            'duration': test_result.get('duration', 0),
            'status': test_result.get('status')
        })
    
    def _get_observation(self) -> np.ndarray:
        """Get current observation"""
        
        # Normalize maps
        memory_map_norm = np.clip(self.memory_map, 0, 1)
        fault_map_norm = self.fault_map.astype(np.float32)
        coverage_norm = np.clip(self.test_coverage / 10.0, 0, 1)  # Normalize coverage
        
        # Confidence map (higher confidence in well-tested areas)
        confidence_map = np.tanh(self.test_coverage / 5.0).astype(np.float32)
        
        # Stack all maps
        observation = np.stack([
            memory_map_norm,
            fault_map_norm,
            coverage_norm,
            confidence_map
        ], axis=2)
        
        return observation
    
    def _get_info(self) -> Dict:
        """Get additional information"""
        
        return {
            'step': self.current_step,
            'total_faults_found': self.total_faults_found,
            'test_coverage': np.mean(self.test_coverage),
            'consecutive_fails': self.consecutive_fails,
            'total_tests': len(self.test_history)
        }
    
    def render(self, mode: str = 'human'):
        """Render the environment state"""
        
        if mode == 'human':
            print(f"Step: {self.current_step}")
            print(f"Total faults found: {self.total_faults_found}")
            print(f"Average coverage: {np.mean(self.test_coverage):.2f}")
            print(f"Recent tests: {len(self.test_history[-10:])}")
            
        return self._get_observation()