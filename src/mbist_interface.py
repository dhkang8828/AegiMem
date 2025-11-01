"""
Python interface to MBIST C library
Provides low-level DRAM command primitives for RL Agent
"""

import ctypes
import time
from enum import IntEnum
from typing import Optional, List, Dict


class PatternType(IntEnum):
    """Data pattern types"""
    FIXED_00 = 0
    FIXED_FF = 1
    FIXED_55 = 2
    FIXED_AA = 3
    CHECKERBOARD = 4
    PRBS = 5
    WALKING_1S = 6
    WALKING_0S = 7


class TestResult(ctypes.Structure):
    """Test result structure"""
    _fields_ = [
        ("test_result", ctypes.c_uint32),  # 0: pass, 1: fail
        ("error_count", ctypes.c_uint32),
    ]


class ErrorAddress(ctypes.Structure):
    """Error address structure"""
    _fields_ = [
        ("bg", ctypes.c_uint64, 3),
        ("ba", ctypes.c_uint64, 2),
        ("rank", ctypes.c_uint64, 2),
        ("column", ctypes.c_uint64, 9),
        ("row", ctypes.c_uint64, 18),
        ("cid", ctypes.c_uint64, 4),
    ]


class MBISTInterface:
    """Python interface to MBIST C library"""

    def __init__(self, lib_path: str = "/home/dhkang/data3/mbist_sample_code-gen2_es/bin/mbist_smbus"):
        """
        Initialize MBIST interface

        Args:
            lib_path: Path to MBIST shared library
        """
        # Load library (note: executable, not .so for now)
        # In production, we'd build a shared library
        self.lib_path = lib_path
        self._setup_mode = False

        # For testing without actual library
        self.mock_mode = not self._try_load_library()

        if not self.mock_mode:
            self._setup_functions()
            self._initialize()

    def _try_load_library(self) -> bool:
        """Try to load the library, return True if successful"""
        try:
            # TODO: Build shared library version
            # self.lib = ctypes.CDLL(self.lib_path)
            return False  # For now, use mock mode
        except Exception as e:
            print(f"Warning: Could not load MBIST library: {e}")
            print("Running in MOCK mode")
            return False

    def _setup_functions(self):
        """Setup ctypes function signatures"""

        # Sequence control
        self.lib.mt_rl_begin_sequence.argtypes = [ctypes.c_uint8]
        self.lib.mt_rl_begin_sequence.restype = None

        self.lib.mt_rl_end_sequence.argtypes = [ctypes.c_uint8]
        self.lib.mt_rl_end_sequence.restype = None

        # Atomic commands
        self.lib.mt_rl_send_activate.argtypes = [
            ctypes.c_uint8,  # rank
            ctypes.c_uint8,  # bg
            ctypes.c_uint8,  # ba
            ctypes.c_uint32  # row
        ]
        self.lib.mt_rl_send_activate.restype = ctypes.c_int

        self.lib.mt_rl_send_write.argtypes = [
            ctypes.c_uint8,  # rank
            ctypes.c_uint8,  # bg
            ctypes.c_uint8,  # ba
            ctypes.c_uint32, # row
            ctypes.c_uint16, # col
            ctypes.c_uint8   # pattern
        ]
        self.lib.mt_rl_send_write.restype = ctypes.c_int

        self.lib.mt_rl_send_read.argtypes = [
            ctypes.c_uint8,  # rank
            ctypes.c_uint8,  # bg
            ctypes.c_uint8,  # ba
            ctypes.c_uint32, # row
            ctypes.c_uint16  # col
        ]
        self.lib.mt_rl_send_read.restype = ctypes.c_int

        self.lib.mt_rl_send_precharge.argtypes = [
            ctypes.c_uint8,  # rank
            ctypes.c_uint8,  # bg
            ctypes.c_uint8,  # ba
            ctypes.c_uint8   # all_banks
        ]
        self.lib.mt_rl_send_precharge.restype = ctypes.c_int

        # Result checking
        self.lib.mt_rl_get_test_result.argtypes = [ctypes.c_uint8]
        self.lib.mt_rl_get_test_result.restype = ctypes.c_int

        self.lib.mt_rl_get_error_count.argtypes = [ctypes.c_uint8]
        self.lib.mt_rl_get_error_count.restype = ctypes.c_int

    def _initialize(self):
        """Initialize MBIST hardware"""
        # Initialization would be done here
        pass

    # ========================================
    # Sequence Mode Control
    # ========================================

    def begin_sequence(self, channel: int = 0):
        """Start command sequence mode"""
        if self.mock_mode:
            print(f"[MOCK] Begin sequence on channel {channel}")
        else:
            self.lib.mt_rl_begin_sequence(channel)

    def end_sequence(self, channel: int = 0):
        """End sequence and execute"""
        if self.mock_mode:
            print(f"[MOCK] Execute sequence on channel {channel}")
            time.sleep(0.01)  # Simulate execution time
        else:
            self.lib.mt_rl_end_sequence(channel)

    # ========================================
    # Low-level DRAM Commands
    # ========================================

    def send_activate(self, rank: int, bg: int, ba: int, row: int) -> int:
        """Send ACTIVATE command"""
        if self.mock_mode:
            print(f"[MOCK] ACT: rank={rank}, bg={bg}, ba={ba}, row={row}")
            return 0
        else:
            return self.lib.mt_rl_send_activate(rank, bg, ba, row)

    def send_write(self, rank: int, bg: int, ba: int, row: int, col: int,
                   pattern: PatternType = PatternType.FIXED_AA) -> int:
        """Send WRITE command"""
        if self.mock_mode:
            print(f"[MOCK] WR: rank={rank}, bg={bg}, ba={ba}, row={row}, col={col}, pattern={pattern.name}")
            return 0
        else:
            return self.lib.mt_rl_send_write(rank, bg, ba, row, col, pattern)

    def send_read(self, rank: int, bg: int, ba: int, row: int, col: int) -> int:
        """Send READ command"""
        if self.mock_mode:
            print(f"[MOCK] RD: rank={rank}, bg={bg}, ba={ba}, row={row}, col={col}")
            return 0
        else:
            return self.lib.mt_rl_send_read(rank, bg, ba, row, col)

    def send_precharge(self, rank: int, bg: int, ba: int, all_banks: bool = False) -> int:
        """Send PRECHARGE command"""
        if self.mock_mode:
            cmd_type = "PREab" if all_banks else "PREsb"
            print(f"[MOCK] {cmd_type}: rank={rank}, bg={bg}, ba={ba}")
            return 0
        else:
            return self.lib.mt_rl_send_precharge(rank, bg, ba, 1 if all_banks else 0)

    def send_refresh(self, rank: int, bg: int, ba: int, all_banks: bool = True) -> int:
        """Send REFRESH command"""
        if self.mock_mode:
            cmd_type = "REFab" if all_banks else "REFsb"
            print(f"[MOCK] {cmd_type}: rank={rank}, bg={bg}, ba={ba}")
            return 0
        else:
            return self.lib.mt_rl_send_refresh(rank, bg, ba, 1 if all_banks else 0)

    # ========================================
    # Result Checking
    # ========================================

    def get_test_result(self, channel: int = 0) -> int:
        """
        Get test result

        Returns:
            0: PASS
            1: FAIL
            -1: ERROR
        """
        if self.mock_mode:
            # Simulate random result
            import random
            return random.choice([0, 0, 0, 1])  # Mostly PASS
        else:
            return self.lib.mt_rl_get_test_result(channel)

    def get_error_count(self, channel: int = 0) -> int:
        """Get number of error addresses"""
        if self.mock_mode:
            return 0
        else:
            return self.lib.mt_rl_get_error_count(channel)

    def get_ce_count(self) -> int:
        """
        Get CE count from Mailbox get-health-info command

        This will be implemented by user to query CXL device mailbox
        """
        # TODO: Implement mailbox command
        # For now, return mock value
        return 0


# ========================================
# High-level Primitives
# ========================================

class DRAMPrimitives:
    """High-level DRAM operations for RL"""

    def __init__(self, mbist: MBISTInterface):
        self.mbist = mbist

    def write_read_cell(self, rank: int, bg: int, ba: int, row: int, col: int,
                       pattern: PatternType = PatternType.FIXED_AA) -> int:
        """Write and read a single cell"""
        self.mbist.begin_sequence()
        self.mbist.send_activate(rank, bg, ba, row)
        self.mbist.send_write(rank, bg, ba, row, col, pattern)
        self.mbist.send_precharge(rank, bg, ba, all_banks=False)
        self.mbist.send_activate(rank, bg, ba, row)
        self.mbist.send_read(rank, bg, ba, row, col)
        self.mbist.send_precharge(rank, bg, ba, all_banks=False)
        self.mbist.end_sequence()

        return self.mbist.get_test_result()

    def write_read_row(self, rank: int, bg: int, ba: int, row: int,
                      pattern: PatternType = PatternType.FIXED_AA) -> int:
        """Write and read entire row"""
        self.mbist.begin_sequence()

        # ACTIVATE
        self.mbist.send_activate(rank, bg, ba, row)

        # WRITE all columns (burst length 16, so 2048/16 = 128 bursts)
        for col in range(0, 2048, 16):
            self.mbist.send_write(rank, bg, ba, row, col, pattern)

        # PRECHARGE
        self.mbist.send_precharge(rank, bg, ba, all_banks=False)

        # ACTIVATE again
        self.mbist.send_activate(rank, bg, ba, row)

        # READ and verify
        for col in range(0, 2048, 16):
            self.mbist.send_read(rank, bg, ba, row, col)

        # PRECHARGE
        self.mbist.send_precharge(rank, bg, ba, all_banks=False)

        self.mbist.end_sequence()

        return self.mbist.get_test_result()

    def row_hammer(self, rank: int, bg: int, ba: int, target_row: int,
                  count: int = 10000) -> Dict:
        """
        Row hammer attack

        Args:
            rank, bg, ba: Target bank
            target_row: Row to hammer
            count: Number of hammer iterations

        Returns:
            Dict with victim row test results
        """
        # Hammer phase (in batches to avoid SRAM overflow)
        batch_size = 500
        num_batches = count // batch_size

        print(f"Row hammer: {num_batches} batches of {batch_size} iterations")

        for i in range(num_batches):
            self.mbist.begin_sequence()
            for _ in range(batch_size):
                self.mbist.send_activate(rank, bg, ba, target_row)
                self.mbist.send_precharge(rank, bg, ba, all_banks=False)
            self.mbist.end_sequence()

            if i % 10 == 0:
                print(f"  Completed {i}/{num_batches} batches")

        # Check victim rows
        results = self._check_victim_rows(rank, bg, ba, target_row)

        return results

    def _check_victim_rows(self, rank: int, bg: int, ba: int, target_row: int) -> Dict:
        """Check rows adjacent to hammered row"""
        results = {
            'target_row': target_row,
            'victims': []
        }

        for victim_row in [target_row - 1, target_row + 1]:
            if 0 <= victim_row < 262144:
                result = self.write_read_row(rank, bg, ba, victim_row)
                results['victims'].append({
                    'row': victim_row,
                    'test_result': result,
                    'passed': (result == 0)
                })

        return results

    def custom_sequence(self, commands: List[Dict]) -> int:
        """
        Execute custom command sequence

        Args:
            commands: List of command dictionaries
                Each dict should have:
                - type: 'ACT', 'WR', 'RD', 'PRE', 'DELAY'
                - rank, bg, ba, row, col (as needed)
                - pattern (for WR)
                - delay_sec (for DELAY)

        Example:
            commands = [
                {'type': 'ACT', 'rank': 0, 'bg': 0, 'ba': 0, 'row': 100},
                {'type': 'WR', 'rank': 0, 'bg': 0, 'ba': 0, 'row': 100, 'col': 0, 'pattern': PatternType.FIXED_AA},
                {'type': 'DELAY', 'delay_sec': 0.001},
                {'type': 'RD', 'rank': 0, 'bg': 0, 'ba': 0, 'row': 100, 'col': 0},
                {'type': 'PRE', 'rank': 0, 'bg': 0, 'ba': 0},
            ]
        """
        self.mbist.begin_sequence()

        for cmd in commands:
            cmd_type = cmd['type']

            if cmd_type == 'ACT':
                self.mbist.send_activate(cmd['rank'], cmd['bg'], cmd['ba'], cmd['row'])

            elif cmd_type == 'WR':
                self.mbist.send_write(cmd['rank'], cmd['bg'], cmd['ba'],
                                     cmd['row'], cmd['col'], cmd.get('pattern', PatternType.FIXED_AA))

            elif cmd_type == 'RD':
                self.mbist.send_read(cmd['rank'], cmd['bg'], cmd['ba'],
                                    cmd['row'], cmd['col'])

            elif cmd_type == 'PRE':
                self.mbist.send_precharge(cmd['rank'], cmd['bg'], cmd['ba'],
                                         cmd.get('all_banks', False))

            elif cmd_type == 'DELAY':
                # End current batch, delay, start new batch
                self.mbist.end_sequence()
                time.sleep(cmd['delay_sec'])
                self.mbist.begin_sequence()

        self.mbist.end_sequence()
        return self.mbist.get_test_result()
