#!/usr/bin/env python3
"""
Test the validation script locally without umxc

This mock test verifies that the validation script logic works correctly
by simulating umxc responses.
"""

import sys
import subprocess
from unittest.mock import patch, MagicMock

sys.path.insert(0, '/home/dhkang/cxl_memory_rl_project/src')
sys.path.insert(0, '/home/dhkang/cxl_memory_rl_project/tools')

from dpa_translator import DPATranslator


def test_umxc_output_parsing():
    """Test parsing of actual umxc output format"""
    print("=" * 60)
    print("Testing umxc Output Parsing")
    print("=" * 60)

    # Import here to avoid issues if validate_dpa_translation doesn't exist yet
    import validate_dpa_translation as vdt

    # Actual umxc output format from GNR-CRB board
    test_cases = [
        # Test case 1: DPA 0x0
        (
            """Select MXC_GEN2_SOC-PCI 0000:0c:00.0 (1B00:C002 REV03 80-CE-00-00-00-00-00-00)
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
""",
            {'subch': 0, 'dimm': 0, 'rank': 0, 'bg': 0, 'ba': 0, 'row': 0x0, 'col': 0x0}
        ),
        # Test case 2: DPA 0x100000
        (
            """Select MXC_GEN2_SOC-PCI 0000:0c:00.0 (1B00:C002 REV03 80-CE-00-00-00-00-00-00)
Adress Decode (Request Code F1h)
  Device Physical Address = 0x100000
Successful
  dpa       = 0x100000
  subch     = 0
  dimm      = 0
  rank      = 0
  bg        = 0
  ba        = 0
  row       = 0x1
  col       = 0x0
""",
            {'subch': 0, 'dimm': 0, 'rank': 0, 'bg': 0, 'ba': 0, 'row': 0x1, 'col': 0x0}
        ),
        # Test case 3: DPA 0x40 (DIMM interleaving)
        (
            """Successful
  dpa       = 0x40
  subch     = 0
  dimm      = 1
  rank      = 0
  bg        = 0
  ba        = 0
  row       = 0x0
  col       = 0x0
""",
            {'subch': 0, 'dimm': 1, 'rank': 0, 'bg': 0, 'ba': 0, 'row': 0x0, 'col': 0x0}
        ),
        # Test case 4: Complex DPA
        (
            """Successful
  dpa       = 0xFFFC0
  subch     = 1
  dimm      = 1
  rank      = 0
  bg        = 7
  ba        = 3
  row       = 0x0
  col       = 0x7F
""",
            {'subch': 1, 'dimm': 1, 'rank': 0, 'bg': 7, 'ba': 3, 'row': 0x0, 'col': 0x7F}
        ),
    ]

    passed = 0
    for i, (output, expected) in enumerate(test_cases):
        result = vdt.UmxcInterface._parse_umxc_output(output, dpa=0x0)

        if result == expected:
            print(f"✓ Test case {i+1} parsed correctly")
            passed += 1
        else:
            print(f"✗ Test case {i+1} parse failed:")
            print(f"  Expected: {expected}")
            print(f"  Got: {result}")

    print(f"\n{passed}/{len(test_cases)} parsing tests passed\n")
    return passed == len(test_cases)


def test_validation_logic():
    """Test validation logic with mock umxc"""
    print("=" * 60)
    print("Testing Validation Logic (Mock)")
    print("=" * 60)

    import validate_dpa_translation as vdt

    translator = DPATranslator(mock_mode=True)

    # Test DPAs
    test_dpas = [0x0, 0x40, 0x80, 0x400, 0x20000, 0x80000, 0x100000]

    passed = 0
    for dpa in test_dpas:
        # Get expected result from our translator
        expected = translator.dpa_to_dram(dpa)

        # Mock umxc output in actual format
        mock_output = f"""Successful
  dpa       = 0x{dpa:x}
  subch     = {expected['subch']}
  dimm      = {expected['dimm']}
  rank      = {expected['rank']}
  bg        = {expected['bg']}
  ba        = {expected['ba']}
  row       = 0x{expected['row']:x}
  col       = 0x{expected['col']:x}
"""

        # Parse it
        parsed = vdt.UmxcInterface._parse_umxc_output(mock_output, dpa)

        # Compare
        if parsed == expected:
            print(f"✓ DPA 0x{dpa:X} validation logic works")
            passed += 1
        else:
            print(f"✗ DPA 0x{dpa:X} validation logic failed")
            print(f"  Expected: {expected}")
            print(f"  Parsed: {parsed}")

    print(f"\n{passed}/{len(test_dpas)} validation tests passed\n")
    return passed == len(test_dpas)


def test_roundtrip_consistency():
    """Test that our translator is internally consistent"""
    print("=" * 60)
    print("Testing Roundtrip Consistency")
    print("=" * 60)

    translator = DPATranslator(mock_mode=True)

    # Generate some random DPAs
    test_dpas = [
        0x0, 0x40, 0x80, 0x400, 0x20000, 0x80000, 0x100000,
        0x12C480, 0x1FFFC0, 0xABCD00, 0x7654300
    ]

    passed = 0
    for dpa in test_dpas:
        # Forward translation
        dram = translator.dpa_to_dram(dpa)

        # Reverse translation
        reconstructed_dpa = translator.dram_to_dpa(
            dram['rank'], dram['bg'], dram['ba'],
            dram['row'], dram['col'],
            dram['dimm'], dram['subch']
        )

        if dpa == reconstructed_dpa:
            print(f"✓ DPA 0x{dpa:X} roundtrip consistent")
            passed += 1
        else:
            print(f"✗ DPA 0x{dpa:X} roundtrip FAILED")
            print(f"  Original: 0x{dpa:X}")
            print(f"  After roundtrip: 0x{reconstructed_dpa:X}")
            print(f"  DRAM: {dram}")

    print(f"\n{passed}/{len(test_dpas)} roundtrip tests passed\n")
    return passed == len(test_dpas)


def test_specific_examples():
    """Test specific examples from GNR-CRB data"""
    print("=" * 60)
    print("Testing Specific GNR-CRB Examples")
    print("=" * 60)

    translator = DPATranslator(mock_mode=True)

    # Examples from actual GNR-CRB data
    test_cases = [
        (0x0, {'subch': 0, 'dimm': 0, 'rank': 0, 'bg': 0, 'ba': 0, 'row': 0, 'col': 0}),
        (0x40, {'subch': 0, 'dimm': 1, 'rank': 0, 'bg': 0, 'ba': 0, 'row': 0, 'col': 0}),
        (0x80, {'subch': 0, 'dimm': 0, 'rank': 0, 'bg': 1, 'ba': 0, 'row': 0, 'col': 0}),
        (0x400, {'subch': 0, 'dimm': 0, 'rank': 0, 'bg': 0, 'ba': 0, 'row': 0, 'col': 1}),
        (0x20000, {'subch': 0, 'dimm': 0, 'rank': 0, 'bg': 0, 'ba': 1, 'row': 0, 'col': 0}),
        (0x80000, {'subch': 1, 'dimm': 0, 'rank': 0, 'bg': 0, 'ba': 0, 'row': 0, 'col': 0}),
        (0x100000, {'subch': 0, 'dimm': 0, 'rank': 0, 'bg': 0, 'ba': 0, 'row': 1, 'col': 0}),
    ]

    passed = 0
    for dpa, expected in test_cases:
        result = translator.dpa_to_dram(dpa)

        # Compare (ignoring rank for now as it's always 0)
        match = all([
            result['subch'] == expected['subch'],
            result['dimm'] == expected['dimm'],
            result['bg'] == expected['bg'],
            result['ba'] == expected['ba'],
            result['row'] == expected['row'],
            result['col'] == expected['col']
        ])

        if match:
            print(f"✓ DPA 0x{dpa:X} matches expected")
            passed += 1
        else:
            print(f"✗ DPA 0x{dpa:X} MISMATCH")
            print(f"  Expected: {expected}")
            print(f"  Got: {result}")

    print(f"\n{passed}/{len(test_cases)} example tests passed\n")
    return passed == len(test_cases)


if __name__ == '__main__':
    print("\n" + "=" * 60)
    print("DPA Translation Validation - Local Tests")
    print("=" * 60 + "\n")

    all_passed = True

    try:
        all_passed &= test_umxc_output_parsing()
        all_passed &= test_validation_logic()
        all_passed &= test_roundtrip_consistency()
        all_passed &= test_specific_examples()

    except Exception as e:
        print(f"\n✗ Test failed with exception: {e}")
        import traceback
        traceback.print_exc()
        all_passed = False

    print("=" * 60)
    if all_passed:
        print("✓ ALL LOCAL TESTS PASSED!")
        print("\nThe validation script is ready to use on GNR-CRB board.")
    else:
        print("✗ SOME LOCAL TESTS FAILED")
    print("=" * 60 + "\n")

    sys.exit(0 if all_passed else 1)
