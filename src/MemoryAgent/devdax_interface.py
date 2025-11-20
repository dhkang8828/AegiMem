"""
DevDax Interface for CXL Memory Access

This module provides a Python interface for accessing CXL memory via devdax (/dev/dax*)
with ECC enabled, supporting fault detection through CE (Correctable Error) count monitoring.
"""

import os
import mmap
import struct
from typing import Dict, Optional, Tuple
from enum import IntEnum


class DevDaxInterface:
    """
    Interface for CXL memory access via Linux devdax (/dev/dax*)

    Provides:
    - Direct memory read/write with ECC ON
    - CE count monitoring for fault detection
    - DPA and DRAM address-based access
    """

    def __init__(self, device_path: str, dpa_translator, ce_monitor,
                 mock_mode: bool = False, mock_verbose: bool = False):
        """
        Initialize DevDax interface

        Args:
            device_path: Path to devdax device (e.g., "/dev/dax0.0")
            dpa_translator: DPATranslator instance for address conversion
            ce_monitor: CECountMonitor instance for fault detection
            mock_mode: If True, simulate operations without real hardware
            mock_verbose: If True, print mock operation details
        """
        self.device_path = device_path
        self.translator = dpa_translator
        self.ce_monitor = ce_monitor
        self.mock_mode = mock_mode
        self.mock_verbose = mock_verbose

        self.fd = None
        self.mmap_obj = None
        self.device_size = 0

        if not mock_mode:
            self._open_device()

    def _open_device(self):
        """Open devdax device and get size"""
        try:
            self.fd = os.open(self.device_path, os.O_RDWR | os.O_SYNC)

            # Get device size
            self.device_size = os.lseek(self.fd, 0, os.SEEK_END)
            os.lseek(self.fd, 0, os.SEEK_SET)

            print(f"Opened {self.device_path}: {self.device_size / (1024**3):.2f} GB")

        except Exception as e:
            raise RuntimeError(f"Failed to open devdax device {self.device_path}: {e}")

    def close(self):
        """Close devdax device"""
        if self.mmap_obj:
            self.mmap_obj.close()
            self.mmap_obj = None

        if self.fd:
            os.close(self.fd)
            self.fd = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def write_dpa(self, dpa: int, data: bytes) -> int:
        """
        Write data to DPA address

        Args:
            dpa: Device Physical Address
            data: Data bytes to write (should be cache line aligned, typically 64B)

        Returns:
            Number of bytes written
        """
        if self.mock_mode:
            if self.mock_verbose:
                print(f"[MOCK] Write DPA 0x{dpa:x}: {len(data)} bytes")
            return len(data)

        # Check alignment
        if dpa % 64 != 0:
            print(f"Warning: DPA 0x{dpa:x} not cache line aligned")

        if dpa + len(data) > self.device_size:
            raise ValueError(f"Write beyond device size: DPA 0x{dpa:x} + {len(data)}")

        # Direct write
        os.lseek(self.fd, dpa, os.SEEK_SET)
        bytes_written = os.write(self.fd, data)

        return bytes_written

    def read_dpa(self, dpa: int, length: int) -> bytes:
        """
        Read data from DPA address

        Args:
            dpa: Device Physical Address
            length: Number of bytes to read

        Returns:
            Data bytes read
        """
        if self.mock_mode:
            if self.mock_verbose:
                print(f"[MOCK] Read DPA 0x{dpa:x}: {length} bytes")
            return b'\x00' * length

        # Check alignment
        if dpa % 64 != 0:
            print(f"Warning: DPA 0x{dpa:x} not cache line aligned")

        if dpa + length > self.device_size:
            raise ValueError(f"Read beyond device size: DPA 0x{dpa:x} + {length}")

        # Direct read
        os.lseek(self.fd, dpa, os.SEEK_SET)
        data = os.read(self.fd, length)

        return data

    def write_dram_cell(self, rank: int, bg: int, ba: int, row: int, col: int,
                       data: bytes, dimm: int = 0, subch: int = 0) -> int:
        """
        Write to specific DRAM cell (via reverse DPA translation)

        Args:
            rank, bg, ba, row, col: DRAM address components
            data: Data bytes to write
            dimm: DIMM number (default: 0)
            subch: Subchannel number (default: 0)

        Returns:
            Number of bytes written
        """
        # Translate DRAM address to DPA
        dpa = self.translator.dram_to_dpa(rank, bg, ba, row, col, dimm, subch)

        if self.mock_verbose:
            print(f"[DRAM Write] rank={rank} bg={bg} ba={ba} row=0x{row:x} col=0x{col:x} → DPA=0x{dpa:x}")

        return self.write_dpa(dpa, data)

    def read_dram_cell(self, rank: int, bg: int, ba: int, row: int, col: int,
                      length: int, dimm: int = 0, subch: int = 0) -> bytes:
        """
        Read from specific DRAM cell (via reverse DPA translation)

        Args:
            rank, bg, ba, row, col: DRAM address components
            length: Number of bytes to read
            dimm: DIMM number (default: 0)
            subch: Subchannel number (default: 0)

        Returns:
            Data bytes read
        """
        # Translate DRAM address to DPA
        dpa = self.translator.dram_to_dpa(rank, bg, ba, row, col, dimm, subch)

        if self.mock_verbose:
            print(f"[DRAM Read] rank={rank} bg={bg} ba={ba} row=0x{row:x} col=0x{col:x} → DPA=0x{dpa:x}")

        return self.read_dpa(dpa, length)

    def execute_pattern_test(self, operation_type: int, pattern_byte: int,
                            start_dram: Dict, end_dram: Dict,
                            step: int = 1) -> int:
        """
        Execute pattern test over DRAM address range

        This is the core function for Phase1 environment's _execute_action()

        Args:
            operation_type: One of 6 operation types (0-5)
            pattern_byte: Data pattern (0x00 - 0xFF)
            start_dram: Start DRAM address dict {'rank', 'bg', 'ba', 'row', 'col'}
            end_dram: End DRAM address dict
            step: Step size for row/column (default: 1)

        Returns:
            CE count delta (number of faults detected)
        """
        # Get baseline CE count
        ce_before = self.ce_monitor.get_ce_count()

        # Execute pattern based on operation type
        if operation_type == 0:  # WR_ASC_ASC: [^(W pat), ^(R pat)]
            self._execute_wr_asc_asc(pattern_byte, start_dram, end_dram, step)
        elif operation_type == 1:  # WR_DESC_DESC: [v(W pat), v(R pat)]
            self._execute_wr_desc_desc(pattern_byte, start_dram, end_dram, step)
        elif operation_type == 2:  # WR_ASC_DESC: [^(W pat), v(R pat)]
            self._execute_wr_asc_desc(pattern_byte, start_dram, end_dram, step)
        elif operation_type == 3:  # WR_DESC_ASC: [v(W pat), ^(R pat)]
            self._execute_wr_desc_asc(pattern_byte, start_dram, end_dram, step)
        elif operation_type == 4:  # WR_ASC_SINGLE: [^(W pat, R pat)]
            self._execute_wr_single_asc(pattern_byte, start_dram, end_dram, step)
        elif operation_type == 5:  # WR_DESC_SINGLE: [v(W pat, R pat)]
            self._execute_wr_single_desc(pattern_byte, start_dram, end_dram, step)
        else:
            raise ValueError(f"Invalid operation type: {operation_type}")

        # Get CE count after test
        ce_after = self.ce_monitor.get_ce_count()

        # Return CE delta (faults detected)
        ce_delta = ce_after - ce_before

        if self.mock_verbose:
            print(f"[TEST] CE delta: {ce_delta} (before={ce_before}, after={ce_after})")

        return ce_delta

    def _execute_wr_asc_asc(self, pattern: int, start: Dict, end: Dict, step: int):
        """Write ascending, Read ascending"""
        data = bytes([pattern] * 64)  # 64-byte cache line

        # Write phase (ascending)
        for row in range(start['row'], end['row'] + 1, step):
            for col in range(start['col'], end['col'] + 1, step):
                self.write_dram_cell(
                    start['rank'], start['bg'], start['ba'], row, col, data
                )

        # Read phase (ascending)
        for row in range(start['row'], end['row'] + 1, step):
            for col in range(start['col'], end['col'] + 1, step):
                self.read_dram_cell(
                    start['rank'], start['bg'], start['ba'], row, col, 64
                )

    def _execute_wr_desc_desc(self, pattern: int, start: Dict, end: Dict, step: int):
        """Write descending, Read descending"""
        data = bytes([pattern] * 64)

        # Write phase (descending)
        for row in range(end['row'], start['row'] - 1, -step):
            for col in range(end['col'], start['col'] - 1, -step):
                self.write_dram_cell(
                    start['rank'], start['bg'], start['ba'], row, col, data
                )

        # Read phase (descending)
        for row in range(end['row'], start['row'] - 1, -step):
            for col in range(end['col'], start['col'] - 1, -step):
                self.read_dram_cell(
                    start['rank'], start['bg'], start['ba'], row, col, 64
                )

    def _execute_wr_asc_desc(self, pattern: int, start: Dict, end: Dict, step: int):
        """Write ascending, Read descending"""
        data = bytes([pattern] * 64)

        # Write phase (ascending)
        for row in range(start['row'], end['row'] + 1, step):
            for col in range(start['col'], end['col'] + 1, step):
                self.write_dram_cell(
                    start['rank'], start['bg'], start['ba'], row, col, data
                )

        # Read phase (descending)
        for row in range(end['row'], start['row'] - 1, -step):
            for col in range(end['col'], start['col'] - 1, -step):
                self.read_dram_cell(
                    start['rank'], start['bg'], start['ba'], row, col, 64
                )

    def _execute_wr_desc_asc(self, pattern: int, start: Dict, end: Dict, step: int):
        """Write descending, Read ascending"""
        data = bytes([pattern] * 64)

        # Write phase (descending)
        for row in range(end['row'], start['row'] - 1, -step):
            for col in range(end['col'], start['col'] - 1, -step):
                self.write_dram_cell(
                    start['rank'], start['bg'], start['ba'], row, col, data
                )

        # Read phase (ascending)
        for row in range(start['row'], end['row'] + 1, step):
            for col in range(start['col'], end['col'] + 1, step):
                self.read_dram_cell(
                    start['rank'], start['bg'], start['ba'], row, col, 64
                )

    def _execute_wr_single_asc(self, pattern: int, start: Dict, end: Dict, step: int):
        """Write and Read in single pass, ascending"""
        data = bytes([pattern] * 64)

        for row in range(start['row'], end['row'] + 1, step):
            for col in range(start['col'], end['col'] + 1, step):
                self.write_dram_cell(
                    start['rank'], start['bg'], start['ba'], row, col, data
                )
                self.read_dram_cell(
                    start['rank'], start['bg'], start['ba'], row, col, 64
                )

    def _execute_wr_single_desc(self, pattern: int, start: Dict, end: Dict, step: int):
        """Write and Read in single pass, descending"""
        data = bytes([pattern] * 64)

        for row in range(end['row'], start['row'] - 1, -step):
            for col in range(end['col'], start['col'] - 1, -step):
                self.write_dram_cell(
                    start['rank'], start['bg'], start['ba'], row, col, data
                )
                self.read_dram_cell(
                    start['rank'], start['bg'], start['ba'], row, col, 64
                )
