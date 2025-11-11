#!/usr/bin/env python3
"""
DPA to DRAM Address Mapping Visualizer

Visualizes and analyzes DPA to DRAM address mapping data.
Can run on both GNR-CRB board and development machine.

Usage:
    python3 dpa_mapping_visualizer.py mapping_data.csv
    python3 dpa_mapping_visualizer.py mapping_data.csv --plot
    python3 dpa_mapping_visualizer.py mapping_data.csv --analyze
"""

import csv
import argparse
import sys
from typing import List, Dict
from collections import defaultdict


class DPAMappingVisualizer:
    """Visualizes and analyzes DPA to DRAM mapping data"""

    def __init__(self, csv_file: str):
        """Load mapping data from CSV file"""
        self.data = []
        self.csv_file = csv_file

        with open(csv_file, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                self.data.append({
                    'dpa': int(row['dpa']),
                    'subch': int(row['subch']),
                    'dimm': int(row['dimm']),
                    'rank': int(row['rank']),
                    'bg': int(row['bg']),
                    'ba': int(row['ba']),
                    'row': int(row['row']),
                    'col': int(row['col'])
                })

        print(f"Loaded {len(self.data)} mapping samples from {csv_file}")

    def print_table(self, max_rows: int = 50):
        """Print mapping data as formatted table"""
        print("\nDPA to DRAM Address Mapping Table")
        print("=" * 100)
        print(f"{'DPA':>12} | {'SubCh':>5} | {'DIMM':>4} | {'Rank':>4} | {'BG':>2} | {'BA':>2} | {'Row':>8} | {'Col':>6}")
        print("-" * 100)

        for i, entry in enumerate(self.data):
            if i >= max_rows and len(self.data) > max_rows:
                print(f"... ({len(self.data) - max_rows} more rows)")
                break

            print(f"0x{entry['dpa']:010x} | "
                  f"{entry['subch']:>5} | "
                  f"{entry['dimm']:>4} | "
                  f"{entry['rank']:>4} | "
                  f"{entry['bg']:>2} | "
                  f"{entry['ba']:>2} | "
                  f"0x{entry['row']:06x} | "
                  f"0x{entry['col']:04x}")

        print("=" * 100)

    def analyze_patterns(self):
        """Analyze mapping patterns and print insights"""
        print("\nMapping Pattern Analysis")
        print("=" * 80)

        if len(self.data) < 2:
            print("Not enough data for analysis")
            return

        # Analyze DPA increments
        print("\n1. DPA Increment Analysis")
        print("-" * 40)
        increments = defaultdict(list)

        for i in range(1, len(self.data)):
            prev = self.data[i-1]
            curr = self.data[i]
            delta_dpa = curr['dpa'] - prev['dpa']

            # Check what changed
            changes = []
            if curr['dimm'] != prev['dimm']:
                changes.append(f"DIMM {prev['dimm']}→{curr['dimm']}")
            if curr['rank'] != prev['rank']:
                changes.append(f"Rank {prev['rank']}→{curr['rank']}")
            if curr['bg'] != prev['bg']:
                changes.append(f"BG {prev['bg']}→{curr['bg']}")
            if curr['ba'] != prev['ba']:
                changes.append(f"BA {prev['ba']}→{curr['ba']}")
            if curr['row'] != prev['row']:
                changes.append(f"Row 0x{prev['row']:x}→0x{curr['row']:x}")
            if curr['col'] != prev['col']:
                changes.append(f"Col 0x{prev['col']:x}→0x{curr['col']:x}")

            if changes:
                key = ', '.join(changes)
                increments[key].append(delta_dpa)

        # Print increment patterns
        for change, dpas in sorted(increments.items()):
            unique_dpas = set(dpas)
            if len(unique_dpas) == 1:
                dpa_val = list(unique_dpas)[0]
                print(f"  {change:50} → DPA +0x{dpa_val:x} ({dpa_val} bytes)")
            else:
                print(f"  {change:50} → Multiple DPAs: {[hex(d) for d in sorted(unique_dpas)]}")

        # Analyze interleaving
        print("\n2. Interleaving Pattern")
        print("-" * 40)
        subch_values = set(entry['subch'] for entry in self.data)
        dimm_values = set(entry['dimm'] for entry in self.data)
        rank_values = set(entry['rank'] for entry in self.data)
        bg_values = set(entry['bg'] for entry in self.data)
        ba_values = set(entry['ba'] for entry in self.data)

        print(f"  Subchannels: {sorted(subch_values)} (count: {len(subch_values)})")
        print(f"  DIMMs: {sorted(dimm_values)} (count: {len(dimm_values)})")
        print(f"  Ranks: {sorted(rank_values)} (count: {len(rank_values)})")
        print(f"  Bank Groups: {sorted(bg_values)} (count: {len(bg_values)})")
        print(f"  Banks: {sorted(ba_values)} (count: {len(ba_values)})")

        # Analyze address ranges
        print("\n3. Address Ranges")
        print("-" * 40)
        if self.data:
            max_row = max(entry['row'] for entry in self.data)
            max_col = max(entry['col'] for entry in self.data)
            max_dpa = max(entry['dpa'] for entry in self.data)

            print(f"  Max DPA: 0x{max_dpa:x} ({max_dpa / (1024**3):.2f} GB)")
            print(f"  Max Row: 0x{max_row:x} ({max_row})")
            print(f"  Max Col: 0x{max_col:x} ({max_col})")

        # Try to infer mapping formula
        print("\n4. Inferred Mapping Rules")
        print("-" * 40)
        self._infer_mapping_rules()

        print("=" * 80)

    def _infer_mapping_rules(self):
        """Attempt to infer the DPA to DRAM mapping formula"""

        if len(self.data) < 10:
            print("  Not enough data to infer mapping rules")
            return

        # Find the granularity of each field
        print("  Analyzing bit-level mapping...")

        # Look for the smallest DPA change for each field
        dimm_change_dpa = None
        col_change_dpa = None
        row_change_dpa = None
        ba_change_dpa = None
        bg_change_dpa = None

        for i in range(1, len(self.data)):
            prev = self.data[i-1]
            curr = self.data[i]
            delta_dpa = curr['dpa'] - prev['dpa']

            # DIMM change (and nothing else)
            if (curr['dimm'] != prev['dimm'] and
                curr['col'] == prev['col'] and
                curr['row'] == prev['row'] and
                curr['ba'] == prev['ba'] and
                curr['bg'] == prev['bg']):
                if dimm_change_dpa is None or delta_dpa < dimm_change_dpa:
                    dimm_change_dpa = delta_dpa

            # Column change (and nothing else in higher order)
            if (curr['col'] != prev['col'] and
                curr['row'] == prev['row'] and
                curr['ba'] == prev['ba'] and
                curr['bg'] == prev['bg']):
                if col_change_dpa is None or delta_dpa < col_change_dpa:
                    col_change_dpa = delta_dpa

            # Row change
            if (curr['row'] != prev['row'] and
                curr['ba'] == prev['ba'] and
                curr['bg'] == prev['bg']):
                if row_change_dpa is None or delta_dpa < row_change_dpa:
                    row_change_dpa = delta_dpa

            # BA change
            if (curr['ba'] != prev['ba'] and
                curr['bg'] == prev['bg']):
                if ba_change_dpa is None or delta_dpa < ba_change_dpa:
                    ba_change_dpa = delta_dpa

            # BG change
            if curr['bg'] != prev['bg']:
                if bg_change_dpa is None or delta_dpa < bg_change_dpa:
                    bg_change_dpa = delta_dpa

        # Print findings
        if dimm_change_dpa:
            print(f"  DIMM interleaving granularity: {dimm_change_dpa} bytes (0x{dimm_change_dpa:x})")
        if col_change_dpa:
            print(f"  Column increment: {col_change_dpa} bytes (0x{col_change_dpa:x})")
        if row_change_dpa:
            print(f"  Row increment: {row_change_dpa} bytes (0x{row_change_dpa:x})")
        if ba_change_dpa:
            print(f"  Bank increment: {ba_change_dpa} bytes (0x{ba_change_dpa:x})")
        if bg_change_dpa:
            print(f"  Bank Group increment: {bg_change_dpa} bytes (0x{bg_change_dpa:x})")

    def plot_mapping(self):
        """Create visualization plots"""
        try:
            import matplotlib.pyplot as plt
            import numpy as np
        except ImportError:
            print("matplotlib not available. Install with: pip install matplotlib")
            return

        fig, axes = plt.subplots(3, 2, figsize=(15, 12))
        fig.suptitle('DPA to DRAM Address Mapping Analysis', fontsize=16)

        dpas = [e['dpa'] for e in self.data]
        dpas_mb = [dpa / (1024**2) for dpa in dpas]  # Convert to MB for readability

        # Plot 1: DIMM vs DPA
        axes[0, 0].scatter(dpas_mb, [e['dimm'] for e in self.data], s=1)
        axes[0, 0].set_xlabel('DPA (MB)')
        axes[0, 0].set_ylabel('DIMM')
        axes[0, 0].set_title('DIMM Interleaving')
        axes[0, 0].grid(True, alpha=0.3)

        # Plot 2: Rank vs DPA
        axes[0, 1].scatter(dpas_mb, [e['rank'] for e in self.data], s=1)
        axes[0, 1].set_xlabel('DPA (MB)')
        axes[0, 1].set_ylabel('Rank')
        axes[0, 1].set_title('Rank Distribution')
        axes[0, 1].grid(True, alpha=0.3)

        # Plot 3: Bank Group vs DPA
        axes[1, 0].scatter(dpas_mb, [e['bg'] for e in self.data], s=1)
        axes[1, 0].set_xlabel('DPA (MB)')
        axes[1, 0].set_ylabel('Bank Group')
        axes[1, 0].set_title('Bank Group Interleaving')
        axes[1, 0].grid(True, alpha=0.3)

        # Plot 4: Bank vs DPA
        axes[1, 1].scatter(dpas_mb, [e['ba'] for e in self.data], s=1)
        axes[1, 1].set_xlabel('DPA (MB)')
        axes[1, 1].set_ylabel('Bank Address')
        axes[1, 1].set_title('Bank Address Distribution')
        axes[1, 1].grid(True, alpha=0.3)

        # Plot 5: Row vs DPA
        axes[2, 0].scatter(dpas_mb, [e['row'] for e in self.data], s=1)
        axes[2, 0].set_xlabel('DPA (MB)')
        axes[2, 0].set_ylabel('Row Address')
        axes[2, 0].set_title('Row Address Progression')
        axes[2, 0].grid(True, alpha=0.3)

        # Plot 6: Column vs DPA
        axes[2, 1].scatter(dpas_mb, [e['col'] for e in self.data], s=1)
        axes[2, 1].set_xlabel('DPA (MB)')
        axes[2, 1].set_ylabel('Column Address')
        axes[2, 1].set_title('Column Address Progression')
        axes[2, 1].grid(True, alpha=0.3)

        plt.tight_layout()

        output_file = self.csv_file.replace('.csv', '_plot.png')
        plt.savefig(output_file, dpi=150)
        print(f"\nPlot saved to: {output_file}")

        try:
            plt.show()
        except:
            print("(Display not available, plot saved to file)")


def main():
    parser = argparse.ArgumentParser(
        description='Visualize DPA to DRAM address mapping data'
    )
    parser.add_argument('csv_file', help='CSV file containing mapping data')
    parser.add_argument('--table', action='store_true',
                       help='Print data as table (default: True)')
    parser.add_argument('--analyze', action='store_true',
                       help='Analyze mapping patterns')
    parser.add_argument('--plot', action='store_true',
                       help='Create visualization plots (requires matplotlib)')
    parser.add_argument('--max-rows', type=int, default=50,
                       help='Maximum rows to display in table (default: 50)')

    args = parser.parse_args()

    # If no specific action, do all
    if not (args.table or args.analyze or args.plot):
        args.table = True
        args.analyze = True

    try:
        visualizer = DPAMappingVisualizer(args.csv_file)

        if args.table:
            visualizer.print_table(max_rows=args.max_rows)

        if args.analyze:
            visualizer.analyze_patterns()

        if args.plot:
            visualizer.plot_mapping()

    except FileNotFoundError:
        print(f"Error: File not found: {args.csv_file}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()
