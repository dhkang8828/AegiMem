#!/usr/bin/env python3
"""
DPA to DRAM Address Mapping Collector for GNR-CRB Board

This script collects DPA to DRAM address mapping data using umxc tool.
It should be run on the GNR-CRB board where umxc is installed.

Usage:
    python3 dpa_mapping_collector.py --output mapping_data.csv
    python3 dpa_mapping_collector.py --start 0x0 --end 0x100000000 --step 0x1000
"""

import subprocess
import re
import csv
import json
import argparse
from typing import Dict, List, Optional
import sys


class DPAMappingCollector:
    """Collects DPA to DRAM address mapping using umxc tool"""

    def __init__(self, umxc_path: str = "umxc"):
        """
        Initialize the collector

        Args:
            umxc_path: Path to umxc executable (default: "umxc" assumes it's in PATH)
        """
        self.umxc_path = umxc_path
        self.mapping_data = []

    def translate_dpa(self, dpa: int) -> Optional[Dict]:
        """
        Translate DPA to DRAM address using umxc ei -t command

        Args:
            dpa: Device Physical Address (integer)

        Returns:
            Dictionary containing mapping or None if failed
            {
                'dpa': int,
                'subch': int,
                'dimm': int,
                'rank': int,
                'bg': int,
                'ba': int,
                'row': int,
                'col': int
            }
        """
        try:
            # Run umxc ei -t [DPA]
            cmd = [self.umxc_path, "ei", "-t", f"0x{dpa:x}"]
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=5
            )

            if result.returncode != 0:
                print(f"Error running umxc for DPA 0x{dpa:x}: {result.stderr}", file=sys.stderr)
                return None

            # Parse output
            # Actual format (multiline):
            #   dpa       = 0x0
            #   subch     = 0
            #   dimm      = 0
            #   rank      = 0
            #   bg        = 0
            #   ba        = 0
            #   row       = 0x0
            #   col       = 0x0
            output = result.stdout

            # Extract each field individually
            def extract_field(field_name, is_hex=False):
                pattern = rf'{field_name}\s*=\s*(0x[0-9a-fA-F]+|\d+)'
                match = re.search(pattern, output)
                if not match:
                    return None
                value_str = match.group(1)
                if is_hex or value_str.startswith('0x'):
                    return int(value_str, 16)
                else:
                    return int(value_str)

            # Extract all fields
            parsed_dpa = extract_field('dpa', is_hex=True)
            subch = extract_field('subch')
            dimm = extract_field('dimm')
            rank = extract_field('rank')
            bg = extract_field('bg')
            ba = extract_field('ba')
            row = extract_field('row', is_hex=True)
            col = extract_field('col', is_hex=True)

            # Check if all fields were parsed successfully
            if any(v is None for v in [parsed_dpa, subch, dimm, rank, bg, ba, row, col]):
                print(f"Failed to parse umxc output for DPA 0x{dpa:x}", file=sys.stderr)
                print(f"Output was:\n{output}", file=sys.stderr)
                return None

            return {
                'dpa': parsed_dpa,
                'subch': subch,
                'dimm': dimm,
                'rank': rank,
                'bg': bg,
                'ba': ba,
                'row': row,
                'col': col
            }

        except subprocess.TimeoutExpired:
            print(f"Timeout running umxc for DPA 0x{dpa:x}", file=sys.stderr)
            return None
        except Exception as e:
            print(f"Exception translating DPA 0x{dpa:x}: {e}", file=sys.stderr)
            return None

    def collect_samples(self, start_dpa: int, end_dpa: int, step: int,
                       verbose: bool = True) -> List[Dict]:
        """
        Collect mapping samples over a range of DPA addresses

        Args:
            start_dpa: Starting DPA address
            end_dpa: Ending DPA address (exclusive)
            step: Step size between samples
            verbose: Print progress

        Returns:
            List of mapping dictionaries
        """
        samples = []
        total_samples = (end_dpa - start_dpa) // step

        for i, dpa in enumerate(range(start_dpa, end_dpa, step)):
            # Print progress more frequently (every 10 samples)
            if verbose and i % 10 == 0:
                percentage = (i / total_samples * 100) if total_samples > 0 else 0
                print(f"Progress: {i}/{total_samples} ({percentage:.1f}%) - DPA: 0x{dpa:x}", flush=True)

            mapping = self.translate_dpa(dpa)
            if mapping:
                samples.append(mapping)
                self.mapping_data.append(mapping)
            else:
                if verbose:
                    print(f"Warning: Failed to parse DPA 0x{dpa:x}", file=sys.stderr, flush=True)

        if verbose:
            print(f"Collection complete: {len(samples)}/{total_samples} samples successfully collected")

        return samples

    def save_csv(self, filename: str):
        """Save mapping data to CSV file"""
        if not self.mapping_data:
            print("No data to save", file=sys.stderr)
            return

        fieldnames = ['dpa', 'subch', 'dimm', 'rank', 'bg', 'ba', 'row', 'col']

        with open(filename, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(self.mapping_data)

        print(f"Saved {len(self.mapping_data)} samples to {filename}")

    def save_json(self, filename: str):
        """Save mapping data to JSON file"""
        if not self.mapping_data:
            print("No data to save", file=sys.stderr)
            return

        with open(filename, 'w') as f:
            json.dump(self.mapping_data, f, indent=2)

        print(f"Saved {len(self.mapping_data)} samples to {filename}")


def main():
    parser = argparse.ArgumentParser(
        description='Collect DPA to DRAM address mapping data using umxc tool'
    )
    parser.add_argument('--umxc', default='umxc',
                       help='Path to umxc executable (default: umxc)')
    parser.add_argument('--start', default='0x0',
                       help='Start DPA address (default: 0x0)')
    parser.add_argument('--end', default='0x100000',
                       help='End DPA address (default: 0x100000 = 1MB)')
    parser.add_argument('--step', default='0x40',
                       help='Step size (default: 0x40 = 64B)')
    parser.add_argument('--output', default='dpa_mapping.csv',
                       help='Output CSV filename (default: dpa_mapping.csv)')
    parser.add_argument('--json',
                       help='Also save as JSON file')
    parser.add_argument('--quiet', action='store_true',
                       help='Suppress progress output')

    args = parser.parse_args()

    # Parse hex addresses
    start_dpa = int(args.start, 16) if args.start.startswith('0x') else int(args.start)
    end_dpa = int(args.end, 16) if args.end.startswith('0x') else int(args.end)
    step = int(args.step, 16) if args.step.startswith('0x') else int(args.step)

    print(f"DPA Mapping Collector")
    print(f"=====================")
    print(f"DPA Range: 0x{start_dpa:x} - 0x{end_dpa:x}")
    print(f"Step: 0x{step:x} ({step} bytes)")
    print(f"Expected samples: {(end_dpa - start_dpa) // step}")
    print()

    # Create collector and run
    collector = DPAMappingCollector(umxc_path=args.umxc)
    collector.collect_samples(start_dpa, end_dpa, step, verbose=not args.quiet)

    # Save results
    collector.save_csv(args.output)
    if args.json:
        collector.save_json(args.json)

    print("\nDone!")


if __name__ == '__main__':
    main()
