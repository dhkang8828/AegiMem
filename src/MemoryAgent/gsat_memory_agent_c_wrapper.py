"""
Python ctypes wrapper for GSAT-like Memory Agent library

This module provides a Python interface to the StressAppTest-style
Memory Agent library with 64-action space (4 operations × 16 patterns).

Based on Google StressAppTest: https://github.com/stressapptest/stressapptest
"""

from ctypes import *
import os
from typing import Optional, Tuple
from dataclasses import dataclass


# Find library path
def _find_library():
    """Find libgsat_memory_agent.so"""
    possible_paths = [
        "./c_library/libgsat_memory_agent.so",
        "./libgsat_memory_agent.so",
        "/usr/local/lib/libgsat_memory_agent.so",
    ]

    for path in possible_paths:
        if os.path.exists(path):
            return path

    raise FileNotFoundError(
        "libgsat_memory_agent.so not found. Please build it first:\n"
        "  cd src/MemoryAgent/c_library && gcc -shared -fPIC -O3 "
        "-o libgsat_memory_agent.so gsat_like_memory_agent.c -lpthread"
    )


# C structure definitions
class CEInfo(Structure):
    """Correctable Error Information (matches C struct)"""
    _fields_ = [
        ("volatile_count", c_int),
        ("persistent_count", c_int),
        ("total_count", c_int),
        ("temperature", c_int),
        ("health_status", c_int),
    ]

    def __repr__(self):
        return (f"CEInfo(volatile={self.volatile_count}, "
                f"persistent={self.persistent_count}, "
                f"total={self.total_count}, "
                f"temp={self.temperature}°C)")


class ActionResult(Structure):
    """Action execution result (matches C struct)"""
    _fields_ = [
        ("success", c_int),
        ("error_message", c_char * 256),
        ("ce_info", CEInfo),
    ]


class OperationType:
    """
    GSAT-style Memory test operation types

    Based on StressAppTest operations for maximum stress.
    """
    OP_FILL   = 0  # Thermal stress (Solid write)
    OP_INVERT = 1  # Switching noise (Write -> Invert -> Write) [RECOMMENDED]
    OP_COPY   = 2  # Bandwidth saturation (Memcpy)
    OP_CHECK  = 3  # Read disturb (Read only)

    @staticmethod
    def name(op_type: int) -> str:
        names = {
            0: "FILL",
            1: "INVERT",
            2: "COPY",
            3: "CHECK",
        }
        return names.get(op_type, f"UNKNOWN({op_type})")


# Key patterns from StressAppTest
KEY_PATTERNS = [
    0x00,  # All zeros
    0xFF,  # All ones
    0x55,  # 01010101 - Checkerboard A
    0xAA,  # 10101010 - Checkerboard B
    0xF0,  # 11110000
    0x0F,  # 00001111
    0xCC,  # 11001100
    0x33,  # 00110011
    0x01,  # Walking ones start
    0x80,  # Walking ones end
    0x16,  # 8b10b low transition density
    0xB5,  # 8b10b high transition density
    0x4A,  # Checker pattern
    0x57,  # Edge case 1
    0x02,  # Edge case 2
    0xFD,  # Edge case 3
]


@dataclass
class CEInfoPython:
    """Python-friendly CE information"""
    volatile_count: int
    persistent_count: int
    total_count: int
    temperature: int
    health_status: int

    def has_errors(self) -> bool:
        return self.total_count > 0

    @classmethod
    def from_c_struct(cls, c_info: CEInfo):
        return cls(
            volatile_count=c_info.volatile_count,
            persistent_count=c_info.persistent_count,
            total_count=c_info.total_count,
            temperature=c_info.temperature,
            health_status=c_info.health_status
        )


class GSATMemoryAgentC:
    """
    Python wrapper for GSAT-like Memory Agent library

    Action Space: 64 actions (4 operations × 16 key patterns)

    Usage:
        agent = GSATMemoryAgentC()
        agent.init("/dev/dax0.0", memory_size_mb=1024)

        # Execute high-priority action (INVERT with 0x55)
        ce_info, success = agent.execute_action(17)
        if ce_info.has_errors():
            print(f"CE detected! Total: {ce_info.total_count}")

        agent.cleanup()

    High-Priority Actions for CE Detection:
        action=17: INVERT with 0x55 (01010101) - Best for CE
        action=19: INVERT with 0xAA (10101010) - Best for CE
        action=16: INVERT with 0xFF (11111111)
        action=16: INVERT with 0x00 (00000000)
    """

    def __init__(self, library_path: Optional[str] = None):
        """
        Initialize wrapper

        Args:
            library_path: Path to libgsat_memory_agent.so (auto-detect if None)
        """
        if library_path is None:
            library_path = _find_library()

        self.lib = CDLL(library_path)
        self._setup_function_signatures()
        self._initialized = False

    def _setup_function_signatures(self):
        """Setup C function signatures"""

        # int ma_init(const char* devdax_path, size_t memory_size_mb)
        self.lib.ma_init.argtypes = [c_char_p, c_size_t]
        self.lib.ma_init.restype = c_int

        # int ma_execute_action(int action, ActionResult* result)
        self.lib.ma_execute_action.argtypes = [c_int, POINTER(ActionResult)]
        self.lib.ma_execute_action.restype = c_int

        # int ma_get_ce_info(CEInfo* ce_info)
        self.lib.ma_get_ce_info.argtypes = [POINTER(CEInfo)]
        self.lib.ma_get_ce_info.restype = c_int

        # int ma_reset_baseline(void)
        self.lib.ma_reset_baseline.argtypes = []
        self.lib.ma_reset_baseline.restype = c_int

        # void ma_cleanup(void)
        self.lib.ma_cleanup.argtypes = []
        self.lib.ma_cleanup.restype = None

        # const char* ma_get_error(void)
        self.lib.ma_get_error.argtypes = []
        self.lib.ma_get_error.restype = c_char_p

        # int ma_is_initialized(void)
        self.lib.ma_is_initialized.argtypes = []
        self.lib.ma_is_initialized.restype = c_int

    def init(self, devdax_path: str = "/dev/dax0.0",
             memory_size_mb: int = 1024) -> bool:
        """
        Initialize GSAT Memory Agent

        Args:
            devdax_path: Path to devdax device
            memory_size_mb: Memory size in MB (entire memory will be tested)

        Returns:
            True if successful, False otherwise
        """
        result = self.lib.ma_init(
            devdax_path.encode('utf-8'),
            memory_size_mb
        )

        if result == 0:
            self._initialized = True
            return True
        else:
            error_msg = self.get_error()
            raise RuntimeError(f"Failed to initialize GSAT Memory Agent: {error_msg}")

    def execute_action(self, action: int) -> Tuple[CEInfoPython, bool]:
        """
        Execute memory stress test action

        Args:
            action: Action ID (0-63)
                    = operation_id * 16 + pattern_id

        Returns:
            (ce_info, success)

        Raises:
            RuntimeError: If not initialized or execution fails
        """
        if not self._initialized:
            raise RuntimeError("GSAT Memory Agent not initialized. Call init() first.")

        if action < 0 or action >= 64:
            raise ValueError(f"Invalid action: {action} (must be 0-63)")

        result = ActionResult()
        ret = self.lib.ma_execute_action(action, byref(result))

        if ret == 0 and result.success:
            ce_info = CEInfoPython.from_c_struct(result.ce_info)
            return ce_info, True
        else:
            error_msg = result.error_message.decode('utf-8') if result.error_message else "Unknown error"
            raise RuntimeError(f"Action execution failed: {error_msg}")

    def get_ce_info(self) -> CEInfoPython:
        """
        Get current CE information

        Returns:
            CE information

        Raises:
            RuntimeError: If not initialized or call fails
        """
        if not self._initialized:
            raise RuntimeError("GSAT Memory Agent not initialized")

        ce_info = CEInfo()
        ret = self.lib.ma_get_ce_info(byref(ce_info))

        if ret == 0:
            return CEInfoPython.from_c_struct(ce_info)
        else:
            error_msg = self.get_error()
            raise RuntimeError(f"Failed to get CE info: {error_msg}")

    def reset_baseline(self) -> bool:
        """
        Reset CE baseline

        Returns:
            True if successful, False otherwise
        """
        if not self._initialized:
            raise RuntimeError("GSAT Memory Agent not initialized")

        ret = self.lib.ma_reset_baseline()
        return ret == 0

    def cleanup(self):
        """Cleanup and release resources"""
        if self._initialized:
            self.lib.ma_cleanup()
            self._initialized = False

    def get_error(self) -> str:
        """Get last error message"""
        error_ptr = self.lib.ma_get_error()
        if error_ptr:
            return error_ptr.decode('utf-8')
        return ""

    def decode_action(self, action: int) -> Tuple[int, int, int]:
        """
        Decode action into operation type, pattern index, and pattern byte

        Args:
            action: Action ID (0-63)

        Returns:
            (operation_type, pattern_index, pattern_byte)
        """
        if action < 0 or action >= 64:
            raise ValueError(f"Invalid action: {action} (must be 0-63)")

        operation = action // 16
        pattern_idx = action % 16
        pattern_byte = KEY_PATTERNS[pattern_idx]

        return operation, pattern_idx, pattern_byte

    def is_initialized(self) -> bool:
        """Check if GSAT Memory Agent is initialized"""
        return self.lib.ma_is_initialized() == 1

    def get_action_description(self, action: int) -> str:
        """
        Get human-readable description of action

        Args:
            action: Action ID (0-63)

        Returns:
            Description string
        """
        op, pat_idx, pat_byte = self.decode_action(action)
        op_name = OperationType.name(op)
        return f"Action {action}: {op_name} with pattern 0x{pat_byte:02X}"

    @staticmethod
    def get_high_priority_actions():
        """
        Get list of high-priority actions for CE detection

        Returns:
            List of (action_id, description) tuples
        """
        high_priority = [
            (17, "INVERT with 0x55 (01010101) - Checkerboard A"),
            (19, "INVERT with 0xAA (10101010) - Checkerboard B"),
            (17, "INVERT with 0xFF (11111111) - All ones"),
            (16, "INVERT with 0x00 (00000000) - All zeros"),
            (20, "INVERT with 0xF0 (11110000)"),
            (21, "INVERT with 0x0F (00001111)"),
            (1, "FILL with 0xFF - Maximum thermal stress"),
            (3, "FILL with 0xAA - Thermal + pattern stress"),
        ]
        return high_priority

    def __enter__(self):
        """Context manager entry"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.cleanup()


if __name__ == "__main__":
    # Test GSAT Memory Agent wrapper
    print("="*60)
    print("Testing GSAT Memory Agent C Wrapper")
    print("="*60)

    agent = GSATMemoryAgentC()

    print("\n1. Check initialized state:")
    print(f"   Initialized: {agent.is_initialized()}")

    print("\n2. Decode actions:")
    test_actions = [0, 17, 19, 32, 63]
    for action in test_actions:
        op, pat_idx, pat_byte = agent.decode_action(action)
        op_name = OperationType.name(op)
        print(f"   Action {action:2d}: {op_name:7s} with pattern[{pat_idx:2d}] = 0x{pat_byte:02X}")

    print("\n3. High-priority actions for CE detection:")
    for action, desc in agent.get_high_priority_actions():
        print(f"   Action {action:2d}: {desc}")

    print("\n" + "="*60)
    print("Wrapper test completed!")
    print("="*60)
    print("\nNote: Full test requires initialization with devdax device.")
    print("To test on real hardware:")
    print("  agent.init('/dev/dax0.0', memory_size_mb=1024)")
    print("  ce_info, success = agent.execute_action(17)  # INVERT with 0x55")
    print("  agent.cleanup()")
