"""
Python ctypes wrapper for C Memory Agent library

This module provides a Python interface to the C-based Memory Agent library.
"""

from ctypes import *
import os
from typing import Optional, Tuple
from dataclasses import dataclass


# Find library path
def _find_library():
    """Find libmemory_agent.so"""
    possible_paths = [
        "./src/c_library/libmemory_agent.so",
        "./libmemory_agent.so",
        "/usr/local/lib/libmemory_agent.so",
    ]

    for path in possible_paths:
        if os.path.exists(path):
            return path

    raise FileNotFoundError(
        "libmemory_agent.so not found. Please build it first:\n"
        "  cd src/c_library && make"
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
        ("ce_info", CEInfo),
        ("error_message", c_char * 256),
    ]


class OperationType:
    """Memory test operation types (enum)"""
    WR_ASC_ASC = 0      # [^(W pat), ^(R pat)]
    WR_DESC_DESC = 1    # [v(W pat), v(R pat)]
    WR_ASC_DESC = 2     # [^(W pat), v(R pat)]
    WR_DESC_ASC = 3     # [v(W pat), ^(R pat)]
    WR_DESC_SINGLE = 4  # [v(W pat, R pat)]
    WR_ASC_SINGLE = 5   # [^(W pat, R pat)]

    @staticmethod
    def name(op_type: int) -> str:
        names = {
            0: "WR_ASC_ASC",
            1: "WR_DESC_DESC",
            2: "WR_ASC_DESC",
            3: "WR_DESC_ASC",
            4: "WR_DESC_SINGLE",
            5: "WR_ASC_SINGLE",
        }
        return names.get(op_type, f"UNKNOWN({op_type})")


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


class MemoryAgentC:
    """
    Python wrapper for C Memory Agent library

    Usage:
        agent = MemoryAgentC()
        agent.init("/dev/dax0.0", memory_size_mb=1024)

        # Execute action
        ce_info, success = agent.execute_action(512)
        if ce_info.has_errors():
            print(f"CE detected! Total: {ce_info.total_count}")

        agent.cleanup()
    """

    def __init__(self, library_path: Optional[str] = None):
        """
        Initialize wrapper

        Args:
            library_path: Path to libmemory_agent.so (auto-detect if None)
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

        # int ma_decode_action(int action, OperationType* operation, uint8_t* pattern)
        self.lib.ma_decode_action.argtypes = [c_int, POINTER(c_int), POINTER(c_uint8)]
        self.lib.ma_decode_action.restype = c_int

        # int ma_is_initialized(void)
        self.lib.ma_is_initialized.argtypes = []
        self.lib.ma_is_initialized.restype = c_int

    def init(self, devdax_path: str = "/dev/dax0.0",
             memory_size_mb: int = 1024) -> bool:
        """
        Initialize Memory Agent

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
            raise RuntimeError(f"Failed to initialize Memory Agent: {error_msg}")

    def execute_action(self, action: int) -> Tuple[CEInfoPython, bool]:
        """
        Execute memory test action

        Args:
            action: Action ID (0-1535)

        Returns:
            (ce_info, success)

        Raises:
            RuntimeError: If not initialized or execution fails
        """
        if not self._initialized:
            raise RuntimeError("Memory Agent not initialized. Call init() first.")

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
            raise RuntimeError("Memory Agent not initialized")

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
            raise RuntimeError("Memory Agent not initialized")

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

    def decode_action(self, action: int) -> Tuple[int, int]:
        """
        Decode action into operation type and pattern

        Args:
            action: Action ID (0-1535)

        Returns:
            (operation_type, pattern_byte)
        """
        operation = c_int()
        pattern = c_uint8()

        ret = self.lib.ma_decode_action(action, byref(operation), byref(pattern))

        if ret == 0:
            return operation.value, pattern.value
        else:
            raise ValueError(f"Invalid action: {action}")

    def is_initialized(self) -> bool:
        """Check if Memory Agent is initialized"""
        return self.lib.ma_is_initialized() == 1

    def __enter__(self):
        """Context manager entry"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.cleanup()


if __name__ == "__main__":
    # Test Memory Agent wrapper
    print("Testing Memory Agent C Wrapper...")

    agent = MemoryAgentC()

    print("\n1. Check initialized state:")
    print(f"   Initialized: {agent.is_initialized()}")

    print("\n2. Decode action 512:")
    op, pat = agent.decode_action(512)
    print(f"   Operation: {OperationType.name(op)} ({op})")
    print(f"   Pattern: 0x{pat:02X}")

    print("\n3. Decode action 1024:")
    op, pat = agent.decode_action(1024)
    print(f"   Operation: {OperationType.name(op)} ({op})")
    print(f"   Pattern: 0x{pat:02X}")

    print("\nWrapper test completed!")
    print("\nNote: Full test requires initialization with devdax device.")
    print("To test on real hardware:")
    print("  agent.init('/dev/dax0.0', memory_size_mb=1024)")
    print("  ce_info, success = agent.execute_action(512)")
    print("  agent.cleanup()")
