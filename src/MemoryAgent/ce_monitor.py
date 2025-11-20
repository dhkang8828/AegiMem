"""
CE (Correctable Error) Monitor for CXL Memory

This module monitors correctable errors from various kernel interfaces:
1. CXL device sysfs (/sys/bus/cxl/devices/)
2. EDAC (Error Detection and Correction) subsystem
3. RAS daemon logs (rasdaemon)
4. PCIe AER (Advanced Error Reporting)

CE counts are critical for:
- Detecting weak memory cells before they become uncorrectable
- Providing feedback to RL agent about test effectiveness
- Identifying failure patterns for reward calculation
"""

import os
import re
import time
import subprocess
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path


@dataclass
class CEEvent:
    """Single Correctable Error event"""
    timestamp: float
    dpa: Optional[int] = None  # Device Physical Address (if available)
    source: str = ""  # 'cxl', 'edac', 'ras', 'aer'
    count: int = 1
    device: str = ""  # Device identifier (e.g., 'mem0', 'dax0.0')
    details: Dict = None  # Additional error details

    def __post_init__(self):
        if self.details is None:
            self.details = {}


class CEMonitor:
    """
    Monitor Correctable Errors from multiple kernel sources

    Usage:
        monitor = CEMonitor(device_path="/dev/dax0.0")
        monitor.start_monitoring()

        # After test
        new_errors = monitor.get_ce_delta()
        total_errors = monitor.get_total_ce_count()
    """

    def __init__(self, device_path: str = "/dev/dax0.0", cxl_device: str = "mem0"):
        """
        Initialize CE monitor

        Args:
            device_path: Path to devdax device (e.g., /dev/dax0.0)
            cxl_device: CXL device name in sysfs (e.g., mem0, mem1)
        """
        self.device_path = device_path
        self.cxl_device = cxl_device

        # CE count tracking
        self.baseline_counts: Dict[str, int] = {}
        self.current_counts: Dict[str, int] = {}
        self.ce_events: List[CEEvent] = []

        # Monitoring state
        self.monitoring_active = False
        self.last_update_time = 0.0

        # Kernel interface paths
        self.cxl_sysfs_path = f"/sys/bus/cxl/devices/{cxl_device}"
        self.edac_path = "/sys/devices/system/edac"

        print(f"[CE Monitor] Initialized for device: {device_path} (CXL: {cxl_device})")

    def start_monitoring(self):
        """Start CE monitoring - establishes baseline counts"""
        self.monitoring_active = True
        self.last_update_time = time.time()

        # Collect baseline counts from all sources
        self.baseline_counts = self._collect_all_ce_counts()
        self.current_counts = self.baseline_counts.copy()

        print(f"[CE Monitor] Started monitoring. Baseline CE counts: {self.baseline_counts}")

    def stop_monitoring(self):
        """Stop CE monitoring"""
        self.monitoring_active = False
        print(f"[CE Monitor] Stopped monitoring")

    def update(self) -> int:
        """
        Update CE counts from all sources

        Returns:
            Number of new CE events detected since last update
        """
        if not self.monitoring_active:
            return 0

        new_counts = self._collect_all_ce_counts()
        delta_count = 0

        # Calculate deltas from previous update and record new CE events
        for source, count in new_counts.items():
            # Compare to previous count, not baseline
            prev_count = self.current_counts.get(source, 0)
            new_errors = count - prev_count

            if new_errors > 0:
                # Record CE event
                event = CEEvent(
                    timestamp=time.time(),
                    source=source,
                    count=new_errors,
                    device=self.cxl_device
                )
                self.ce_events.append(event)
                delta_count += new_errors

        self.current_counts = new_counts
        self.last_update_time = time.time()

        return delta_count

    def get_ce_delta(self) -> int:
        """
        Get total CE count increase since monitoring started

        Returns:
            Total number of new correctable errors
        """
        total_delta = 0
        for source, current in self.current_counts.items():
            baseline = self.baseline_counts.get(source, 0)
            total_delta += max(0, current - baseline)
        return total_delta

    def get_total_ce_count(self) -> int:
        """
        Get current total CE count across all sources

        Returns:
            Current total CE count
        """
        return sum(self.current_counts.values())

    def get_ce_by_source(self) -> Dict[str, int]:
        """
        Get CE counts broken down by source

        Returns:
            Dictionary mapping source name to CE count delta
        """
        result = {}
        for source, current in self.current_counts.items():
            baseline = self.baseline_counts.get(source, 0)
            delta = current - baseline
            if delta > 0:
                result[source] = delta
        return result

    def reset_baseline(self):
        """Reset baseline to current counts (useful between test iterations)"""
        self.baseline_counts = self.current_counts.copy()
        self.ce_events.clear()
        print(f"[CE Monitor] Reset baseline. New baseline: {self.baseline_counts}")

    # ========== Internal Collection Methods ==========

    def _collect_all_ce_counts(self) -> Dict[str, int]:
        """Collect CE counts from all available sources"""
        counts = {}

        # Source 1: CXL device sysfs
        cxl_count = self._read_cxl_sysfs_ce()
        if cxl_count is not None:
            counts['cxl_sysfs'] = cxl_count

        # Source 2: EDAC subsystem
        edac_count = self._read_edac_ce()
        if edac_count is not None:
            counts['edac'] = edac_count

        # Source 3: RAS daemon (if available)
        ras_count = self._read_ras_ce()
        if ras_count is not None:
            counts['ras'] = ras_count

        # Source 4: MCE (Machine Check Exception) logs
        mce_count = self._read_mce_ce()
        if mce_count is not None:
            counts['mce'] = mce_count

        return counts

    def _read_cxl_sysfs_ce(self) -> Optional[int]:
        """
        Read CE count from CXL device sysfs

        Typical paths:
        - /sys/bus/cxl/devices/mem0/error_count
        - /sys/bus/cxl/devices/mem0/ras/correctable_error_count
        """
        possible_paths = [
            f"{self.cxl_sysfs_path}/error_count",
            f"{self.cxl_sysfs_path}/ras/correctable_error_count",
            f"{self.cxl_sysfs_path}/errors/ce_count",
        ]

        for path in possible_paths:
            try:
                with open(path, 'r') as f:
                    content = f.read().strip()
                    return int(content)
            except (FileNotFoundError, ValueError, PermissionError):
                continue

        return None

    def _read_edac_ce(self) -> Optional[int]:
        """
        Read CE count from EDAC subsystem

        EDAC (Error Detection and Correction) reports memory errors
        Path: /sys/devices/system/edac/mc/mc*/ce_count
        """
        try:
            edac_mc_path = Path(self.edac_path) / "mc"
            if not edac_mc_path.exists():
                return None

            total_ce = 0
            # Sum CE counts from all memory controllers
            for mc_dir in edac_mc_path.glob("mc*"):
                ce_file = mc_dir / "ce_count"
                if ce_file.exists():
                    with open(ce_file, 'r') as f:
                        count = int(f.read().strip())
                        total_ce += count

            return total_ce if total_ce > 0 else None

        except (FileNotFoundError, ValueError, PermissionError):
            return None

    def _read_ras_ce(self) -> Optional[int]:
        """
        Read CE count from RAS daemon logs

        RAS daemon (rasdaemon) logs memory errors to database
        Requires rasdaemon to be installed and running
        """
        try:
            # Query rasdaemon database for memory correctable errors
            result = subprocess.run(
                ['ras-mc-ctl', '--error-count'],
                capture_output=True,
                text=True,
                timeout=2
            )

            if result.returncode == 0:
                # Parse output for CE count
                match = re.search(r'CE:\s*(\d+)', result.stdout)
                if match:
                    return int(match.group(1))

            return None

        except (FileNotFoundError, subprocess.TimeoutExpired, ValueError):
            return None

    def _read_mce_ce(self) -> Optional[int]:
        """
        Read CE count from MCE (Machine Check Exception) logs

        MCE logs are in /dev/mcelog or dmesg
        Note: Most modern systems use EDAC instead of mcelog
        """
        try:
            # Try reading from dmesg for corrected memory errors
            result = subprocess.run(
                ['dmesg', '-T'],
                capture_output=True,
                text=True,
                timeout=2
            )

            if result.returncode == 0:
                # Count lines with corrected memory error messages
                ce_patterns = [
                    r'mce:.*corrected',
                    r'EDAC.*CE',
                    r'Hardware Error.*corrected',
                ]

                count = 0
                for pattern in ce_patterns:
                    matches = re.findall(pattern, result.stdout, re.IGNORECASE)
                    count += len(matches)

                return count if count > 0 else None

            return None

        except (FileNotFoundError, subprocess.TimeoutExpired, ValueError):
            return None

    # ========== Advanced Features ==========

    def check_ce_at_dpa(self, dpa: int, tolerance: int = 0x1000) -> bool:
        """
        Check if there are recent CE events near a specific DPA

        Args:
            dpa: Device Physical Address to check
            tolerance: Address range tolerance (default: 4KB)

        Returns:
            True if recent CE events detected near this DPA

        Note: DPA association requires kernel support for error address reporting
        """
        # Most kernel interfaces don't report DPA for CE events
        # This would require extended error reporting (CPER, UEFI, etc.)
        # For now, return False - implement when DPA reporting is available
        return False

    def get_ce_rate(self, window_seconds: float = 60.0) -> float:
        """
        Calculate CE rate (errors per second) over recent time window

        Args:
            window_seconds: Time window for rate calculation

        Returns:
            CE events per second
        """
        if not self.ce_events:
            return 0.0

        current_time = time.time()
        cutoff_time = current_time - window_seconds

        recent_events = [e for e in self.ce_events if e.timestamp >= cutoff_time]
        total_errors = sum(e.count for e in recent_events)

        if len(recent_events) == 0:
            return 0.0

        time_span = current_time - recent_events[0].timestamp
        if time_span > 0:
            return total_errors / time_span

        return 0.0

    def export_ce_history(self) -> List[Dict]:
        """
        Export CE event history for analysis

        Returns:
            List of CE events as dictionaries
        """
        return [
            {
                'timestamp': event.timestamp,
                'dpa': event.dpa,
                'source': event.source,
                'count': event.count,
                'device': event.device,
                'details': event.details
            }
            for event in self.ce_events
        ]

    def summary(self) -> str:
        """
        Generate human-readable summary of CE monitoring

        Returns:
            Summary string
        """
        delta = self.get_ce_delta()
        total = self.get_total_ce_count()
        by_source = self.get_ce_by_source()
        rate = self.get_ce_rate()

        summary = f"""
CE Monitor Summary
==================
Device: {self.device_path} ({self.cxl_device})
Monitoring: {'Active' if self.monitoring_active else 'Inactive'}

Total CE Count: {total}
New CEs (since baseline): {delta}
CE Rate: {rate:.2f} errors/sec

CE by Source:
"""
        for source, count in by_source.items():
            summary += f"  {source}: {count}\n"

        return summary


class MockCEMonitor(CEMonitor):
    """
    Mock CE Monitor for testing without actual hardware

    Simulates CE events based on memory access patterns
    """

    def __init__(self, device_path: str = "/dev/dax0.0", failure_rate: float = 0.01):
        """
        Initialize mock CE monitor

        Args:
            device_path: Path to devdax device
            failure_rate: Probability of CE event on memory access (0.0-1.0)
        """
        super().__init__(device_path, "mock_mem0")
        self.failure_rate = failure_rate
        self.simulated_ce_count = 0
        self._pending_dpa_events = []  # Track (count, dpa) pairs
        print(f"[Mock CE Monitor] Initialized with failure_rate={failure_rate}")

    def _collect_all_ce_counts(self) -> Dict[str, int]:
        """Return simulated CE counts"""
        return {'mock': self.simulated_ce_count}

    def inject_ce(self, count: int = 1, dpa: Optional[int] = None):
        """
        Inject simulated CE events

        Args:
            count: Number of CE events to inject
            dpa: DPA address associated with error (optional)

        Note: Call update() after injecting to have events recorded
        """
        self.simulated_ce_count += count
        # Store DPA for next update (if needed for future enhancement)
        self._last_injected_dpa = dpa

        print(f"[Mock CE Monitor] Injected {count} CE event(s)" +
              (f" at DPA 0x{dpa:X}" if dpa else ""))
