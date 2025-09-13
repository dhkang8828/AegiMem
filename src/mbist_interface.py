"""
Montage MBIST Engine Interface for CXL Type3 Memory Testing
"""

import subprocess
import struct
import time
from typing import Dict, List, Tuple, Optional
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class TestPattern(Enum):
    """Test patterns supported by MBIST engine"""
    MARCH_C_MINUS = "march_c_minus"
    CHECKERBOARD = "checkerboard"
    WALKING_ONES = "walking_ones"
    WALKING_ZEROS = "walking_zeros"
    ADDRESS_IN_ADDRESS = "address_in_address"
    RANDOM_PATTERN = "random_pattern"


class TestResult(Enum):
    """Test result status"""
    PASS = 0
    FAIL = 1
    ERROR = -1


class MBISTInterface:
    """Interface wrapper for Montage MBIST Engine"""
    
    def __init__(self, device_id: int = 0, simulator_path: str = "./mbist_simulator"):
        self.device_id = device_id
        self.simulator_path = simulator_path
        self.test_history: List[Dict] = []
        self.safety_limits = {
            'max_consecutive_tests': 1000,
            'max_test_duration': 300,  # seconds
            'cooldown_time': 1.0,  # seconds between tests
        }
        
    def initialize(self) -> bool:
        """Initialize MBIST engine connection"""
        try:
            result = subprocess.run(
                [self.simulator_path, "-h"], 
                capture_output=True, 
                text=True, 
                timeout=10
            )
            logger.info(f"MBIST simulator initialized: {result.stdout}")
            return result.returncode == 0
        except Exception as e:
            logger.error(f"Failed to initialize MBIST: {e}")
            return False
    
    def run_test(self, pattern: TestPattern, start_addr: int, end_addr: int) -> Dict:
        """
        Execute memory test with specified pattern
        
        Args:
            pattern: Test pattern to use
            start_addr: Starting memory address
            end_addr: Ending memory address
            
        Returns:
            Dictionary containing test results
        """
        test_start_time = time.time()
        
        # Safety check
        if len(self.test_history) > self.safety_limits['max_consecutive_tests']:
            logger.warning("Max consecutive tests reached, forcing cooldown")
            time.sleep(self.safety_limits['cooldown_time'] * 10)
        
        try:
            # Execute test command
            cmd = [
                self.simulator_path,
                "-i", hex(self.device_id),
                "-p", hex(start_addr),
                "-e", hex(end_addr),
                "-t", pattern.value
            ]
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=self.safety_limits['max_test_duration']
            )
            
            test_duration = time.time() - test_start_time
            
            # Parse result
            test_result = self._parse_test_result(result.stdout, result.stderr)
            
            # Record test in history
            test_record = {
                'timestamp': test_start_time,
                'pattern': pattern.value,
                'start_addr': start_addr,
                'end_addr': end_addr,
                'duration': test_duration,
                'result': test_result,
                'stdout': result.stdout,
                'stderr': result.stderr
            }
            
            self.test_history.append(test_record)
            
            # Safety cooldown
            time.sleep(self.safety_limits['cooldown_time'])
            
            return test_record
            
        except subprocess.TimeoutExpired:
            logger.error(f"Test timeout after {self.safety_limits['max_test_duration']} seconds")
            return self._create_error_result(pattern, start_addr, end_addr, "TIMEOUT")
        except Exception as e:
            logger.error(f"Test execution failed: {e}")
            return self._create_error_result(pattern, start_addr, end_addr, str(e))
    
    def _parse_test_result(self, stdout: str, stderr: str) -> TestResult:
        """Parse MBIST test output to determine result"""
        if stderr:
            logger.warning(f"Test stderr: {stderr}")
            return TestResult.ERROR
        
        # Basic pattern matching for result
        if "PASS" in stdout.upper():
            return TestResult.PASS
        elif "FAIL" in stdout.upper():
            return TestResult.FAIL
        else:
            return TestResult.ERROR
    
    def _create_error_result(self, pattern: TestPattern, start_addr: int, 
                           end_addr: int, error_msg: str) -> Dict:
        """Create error result record"""
        return {
            'timestamp': time.time(),
            'pattern': pattern.value,
            'start_addr': start_addr,
            'end_addr': end_addr,
            'duration': 0,
            'result': TestResult.ERROR,
            'stdout': '',
            'stderr': error_msg
        }
    
    def get_memory_layout(self) -> Dict:
        """Get memory layout information"""
        # Placeholder - would query actual hardware
        return {
            'total_size': 0x100000000,  # 4GB
            'bank_size': 0x10000000,    # 256MB per bank
            'num_banks': 16,
            'page_size': 0x1000,        # 4KB pages
        }
    
    def reset_device(self) -> bool:
        """Reset MBIST device"""
        try:
            result = subprocess.run(
                [self.simulator_path, "-r", hex(self.device_id)],
                capture_output=True,
                text=True,
                timeout=30
            )
            return result.returncode == 0
        except Exception as e:
            logger.error(f"Device reset failed: {e}")
            return False
    
    def get_test_statistics(self) -> Dict:
        """Get statistics from test history"""
        if not self.test_history:
            return {}
        
        total_tests = len(self.test_history)
        pass_count = sum(1 for t in self.test_history if t['result'] == TestResult.PASS)
        fail_count = sum(1 for t in self.test_history if t['result'] == TestResult.FAIL)
        error_count = sum(1 for t in self.test_history if t['result'] == TestResult.ERROR)
        
        return {
            'total_tests': total_tests,
            'pass_count': pass_count,
            'fail_count': fail_count,
            'error_count': error_count,
            'pass_rate': pass_count / total_tests if total_tests > 0 else 0,
            'fail_rate': fail_count / total_tests if total_tests > 0 else 0,
            'avg_test_duration': sum(t['duration'] for t in self.test_history) / total_tests
        }