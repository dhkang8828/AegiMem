#!/usr/bin/env python3
"""
DPA Translation Validation Tool

Validates DPA-to-DRAM address translation by comparing:
- Our implementation (dpa_translator.py)
- Ground truth from umxc tool (umxc ei -t [DPA])

Usage:
    python3 validate_dpa_translation.py --samples 1000
    python3 validate_dpa_translation.py --samples 10000 --output validation_results.csv
"""

import sys
import subprocess
import random
import re
import argparse
import csv
from typing import Dict, Optional, Tuple, List

sys.path.insert(0, '/home/dhkang/cxl_memory_rl_project/src')
from dpa_translator import DPATranslator


class UmxcInterface:
    """Interface to umxc tool for DPA translation"""

    @staticmethod
    def translate_dpa(dpa: int) -> Optional[Dict]:
        """
        Translate DPA using umxc tool

        Args:
            dpa: Device Physical Address (must be 64B aligned)

        Returns:
            Dictionary with DRAM address components or None if failed
        """
        try:
            # Execute umxc ei -t [DPA]
            cmd = ['umxc', 'ei', '-t', f'0x{dpa:x}']
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=5
            )

            if result.returncode != 0:
                print(f"[ERROR] umxc failed for DPA 0x{dpa:X}: {result.stderr}")
                return None

            # Parse umxc output
            return UmxcInterface._parse_umxc_output(result.stdout, dpa)

        except subprocess.TimeoutExpired:
            print(f"[ERROR] umxc timeout for DPA 0x{dpa:X}")
            return None
        except FileNotFoundError:
            print("[ERROR] umxc tool not found. Make sure it's installed and in PATH.")
            return None
        except Exception as e:
            print(f"[ERROR] Failed to run umxc: {e}")
            return None

    @staticmethod
    def _parse_umxc_output(output: str, dpa: int) -> Optional[Dict]:
        """
        Parse umxc output to extract DRAM address components

        Expected format (actual umxc ei -t output):
        ```
        Select MXC_GEN2_SOC-PCI 0000:0c:00.0 (...)
        Adress Decode (Request Code F1h)
          Device Physical Address = 0x0
        Successful
          dpa       = 0x0
          subch     = 0
          dimm      = 0
          rank      = 0
          bg        = 0
          ba        = 0
          row       = 0x0
          col       = 0x0
        ```
        """
        # Check if command was successful
        if 'Successful' not in output:
            print(f"[ERROR] umxc command failed for DPA 0x{dpa:X}")
            return None

        # Parse each field individually using line-by-line format
        result = {}

        # Extract each field value
        patterns = {
            'subch': r'^\s*subch\s*=\s*(\d+)',
            'dimm': r'^\s*dimm\s*=\s*(\d+)',
            'rank': r'^\s*rank\s*=\s*(\d+)',
            'bg': r'^\s*bg\s*=\s*(\d+)',
            'ba': r'^\s*ba\s*=\s*(\d+)',
            'row': r'^\s*row\s*=\s*(0x[0-9a-fA-F]+|\d+)',
            'col': r'^\s*col\s*=\s*(0x[0-9a-fA-F]+|\d+)',
        }

        for field, pattern in patterns.items():
            match = re.search(pattern, output, re.MULTILINE | re.IGNORECASE)
            if match:
                value_str = match.group(1).strip()
                result[field] = int(value_str, 0)  # Handles both hex (0x...) and decimal
            else:
                print(f"[WARNING] Could not find '{field}' in umxc output for DPA 0x{dpa:X}")
                return None

        # Verify we got all required fields
        required_fields = ['subch', 'dimm', 'rank', 'bg', 'ba', 'row', 'col']
        if not all(field in result for field in required_fields):
            print(f"[WARNING] Missing fields in umxc output for DPA 0x{dpa:X}")
            print(f"  Found: {list(result.keys())}")
            print(f"  Output: {output[:500]}")
            return None

        return result


class DPAValidator:
    """Validates DPA translation accuracy"""

    def __init__(self, max_dpa: int = 0x1FFFFFFFF):
        """
        Initialize validator

        Args:
            max_dpa: Maximum DPA value (default: 0x1FFFFFFFF for 128GB)
        """
        self.max_dpa = max_dpa
        self.translator = DPATranslator(mock_mode=True)
        self.results: List[Dict] = []

        print(f"[Validator] Initialized with max_dpa=0x{max_dpa:X}")

    def generate_random_dpas(self, count: int, seed: Optional[int] = None) -> List[int]:
        """
        Generate random DPAs (64B aligned)

        Args:
            count: Number of DPAs to generate
            seed: Random seed for reproducibility

        Returns:
            List of DPA values
        """
        if seed is not None:
            random.seed(seed)

        dpas = []
        cache_line_size = 0x40  # 64 bytes

        for _ in range(count):
            # Generate random DPA aligned to 64B
            num_cache_lines = self.max_dpa // cache_line_size
            random_line = random.randint(0, num_cache_lines - 1)
            dpa = random_line * cache_line_size
            dpas.append(dpa)

        return dpas

    def validate_single_dpa(self, dpa: int) -> Tuple[bool, Optional[Dict]]:
        """
        Validate translation for a single DPA

        Args:
            dpa: Device Physical Address

        Returns:
            (match: bool, details: Dict with comparison results)
        """
        # Get our translation
        our_result = self.translator.dpa_to_dram(dpa)

        # Get umxc translation
        umxc_result = UmxcInterface.translate_dpa(dpa)

        if umxc_result is None:
            return False, {
                'dpa': dpa,
                'error': 'umxc_failed',
                'our_result': our_result,
                'umxc_result': None
            }

        # Compare results
        match = all([
            our_result['subch'] == umxc_result['subch'],
            our_result['dimm'] == umxc_result['dimm'],
            our_result['rank'] == umxc_result['rank'],
            our_result['bg'] == umxc_result['bg'],
            our_result['ba'] == umxc_result['ba'],
            our_result['row'] == umxc_result['row'],
            our_result['col'] == umxc_result['col']
        ])

        details = {
            'dpa': dpa,
            'match': match,
            'our_result': our_result,
            'umxc_result': umxc_result
        }

        if not match:
            # Find which fields differ
            details['differences'] = []
            for field in ['subch', 'dimm', 'rank', 'bg', 'ba', 'row', 'col']:
                if our_result[field] != umxc_result[field]:
                    details['differences'].append({
                        'field': field,
                        'ours': our_result[field],
                        'umxc': umxc_result[field]
                    })

        return match, details

    def validate_multiple(self, dpas: List[int], verbose: bool = True) -> Dict:
        """
        Validate translation for multiple DPAs

        Args:
            dpas: List of DPA values to validate
            verbose: Print progress

        Returns:
            Summary statistics
        """
        total = len(dpas)
        correct = 0
        failed = 0
        mismatches = []

        print(f"\n{'='*60}")
        print(f"Validating {total} DPA translations")
        print(f"{'='*60}\n")

        for i, dpa in enumerate(dpas):
            if verbose and (i + 1) % 100 == 0:
                print(f"Progress: {i+1}/{total} ({(i+1)*100//total}%)")

            match, details = self.validate_single_dpa(dpa)

            if details.get('error') == 'umxc_failed':
                failed += 1
            elif match:
                correct += 1
            else:
                mismatches.append(details)

            self.results.append(details)

        # Calculate statistics
        accuracy = (correct / (total - failed) * 100) if (total - failed) > 0 else 0.0

        summary = {
            'total': total,
            'correct': correct,
            'mismatches': len(mismatches),
            'failed': failed,
            'accuracy': accuracy
        }

        return summary, mismatches

    def print_summary(self, summary: Dict, mismatches: List[Dict]):
        """Print validation summary"""
        print(f"\n{'='*60}")
        print("Validation Results")
        print(f"{'='*60}")
        print(f"Total samples:     {summary['total']}")
        print(f"Correct:           {summary['correct']}")
        print(f"Mismatches:        {summary['mismatches']}")
        print(f"umxc failures:     {summary['failed']}")
        print(f"Accuracy:          {summary['accuracy']:.2f}%")
        print(f"{'='*60}\n")

        if mismatches:
            print(f"{'='*60}")
            print(f"Mismatch Details (showing first 10)")
            print(f"{'='*60}")

            for i, mismatch in enumerate(mismatches[:10]):
                dpa = mismatch['dpa']
                our = mismatch['our_result']
                umxc = mismatch['umxc_result']

                print(f"\nMismatch #{i+1}: DPA 0x{dpa:X}")
                print(f"  Our result:  subch={our['subch']}, dimm={our['dimm']}, "
                      f"rank={our['rank']}, bg={our['bg']}, ba={our['ba']}, "
                      f"row=0x{our['row']:X}, col=0x{our['col']:X}")
                print(f"  umxc result: subch={umxc['subch']}, dimm={umxc['dimm']}, "
                      f"rank={umxc['rank']}, bg={umxc['bg']}, ba={umxc['ba']}, "
                      f"row=0x{umxc['row']:X}, col=0x{umxc['col']:X}")

                if 'differences' in mismatch:
                    print(f"  Differences: ", end='')
                    for diff in mismatch['differences']:
                        print(f"{diff['field']}(ours={diff['ours']}, umxc={diff['umxc']}) ", end='')
                    print()

            if len(mismatches) > 10:
                print(f"\n... and {len(mismatches) - 10} more mismatches")

    def export_results(self, filename: str):
        """Export results to CSV file"""
        with open(filename, 'w', newline='') as f:
            writer = csv.writer(f)

            # Header
            writer.writerow([
                'DPA', 'Match', 'Error',
                'Our_SubCh', 'Our_DIMM', 'Our_Rank', 'Our_BG', 'Our_BA', 'Our_Row', 'Our_Col',
                'Umxc_SubCh', 'Umxc_DIMM', 'Umxc_Rank', 'Umxc_BG', 'Umxc_BA', 'Umxc_Row', 'Umxc_Col'
            ])

            # Data
            for result in self.results:
                dpa = result['dpa']
                match = result.get('match', False)
                error = result.get('error', '')

                our = result.get('our_result', {})
                umxc = result.get('umxc_result', {})

                writer.writerow([
                    f'0x{dpa:X}',
                    'MATCH' if match else 'MISMATCH',
                    error,
                    our.get('subch', ''),
                    our.get('dimm', ''),
                    our.get('rank', ''),
                    our.get('bg', ''),
                    our.get('ba', ''),
                    f"0x{our.get('row', 0):X}" if our.get('row') is not None else '',
                    f"0x{our.get('col', 0):X}" if our.get('col') is not None else '',
                    umxc.get('subch', '') if umxc else '',
                    umxc.get('dimm', '') if umxc else '',
                    umxc.get('rank', '') if umxc else '',
                    umxc.get('bg', '') if umxc else '',
                    umxc.get('ba', '') if umxc else '',
                    f"0x{umxc.get('row', 0):X}" if umxc and umxc.get('row') is not None else '',
                    f"0x{umxc.get('col', 0):X}" if umxc and umxc.get('col') is not None else ''
                ])

        print(f"\n[Export] Results saved to {filename}")


def main():
    parser = argparse.ArgumentParser(
        description='Validate DPA-to-DRAM address translation against umxc'
    )
    parser.add_argument(
        '--samples', '-n',
        type=int,
        default=1000,
        help='Number of random DPA samples to test (default: 1000)'
    )
    parser.add_argument(
        '--max-dpa',
        type=lambda x: int(x, 0),
        default=0x1FFFFFFFF,
        help='Maximum DPA value (default: 0x1FFFFFFFF for 128GB)'
    )
    parser.add_argument(
        '--output', '-o',
        type=str,
        help='Output CSV file for detailed results'
    )
    parser.add_argument(
        '--seed',
        type=int,
        help='Random seed for reproducibility'
    )
    parser.add_argument(
        '--quiet', '-q',
        action='store_true',
        help='Suppress progress messages'
    )

    args = parser.parse_args()

    # Create validator
    validator = DPAValidator(max_dpa=args.max_dpa)

    # Generate random DPAs
    print(f"Generating {args.samples} random DPAs...")
    dpas = validator.generate_random_dpas(args.samples, seed=args.seed)
    print(f"Generated {len(dpas)} DPAs (range: 0x0 - 0x{args.max_dpa:X})")

    # Validate
    summary, mismatches = validator.validate_multiple(dpas, verbose=not args.quiet)

    # Print summary
    validator.print_summary(summary, mismatches)

    # Export if requested
    if args.output:
        validator.export_results(args.output)

    # Exit code
    if summary['accuracy'] == 100.0:
        print("✓ All translations match umxc!")
        sys.exit(0)
    else:
        print(f"✗ Accuracy: {summary['accuracy']:.2f}%")
        sys.exit(1)


if __name__ == '__main__':
    main()
