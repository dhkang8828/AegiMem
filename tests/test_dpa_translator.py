#!/usr/bin/env python3
"""
Test DPA Translator with actual mapping rules
"""

import sys
sys.path.insert(0, '/home/dhkang/cxl_memory_rl_project/src')

from dpa_translator import DPATranslator


def test_forward_translation():
    """Test DPA → DRAM translation"""
    print("=" * 60)
    print("Testing Forward Translation (DPA → DRAM)")
    print("=" * 60)

    translator = DPATranslator(mock_mode=True)

    test_cases = [
        (0x0, {'dimm': 0, 'bg': 0, 'ba': 0, 'row': 0, 'col': 0, 'subch': 0}),
        (0x40, {'dimm': 1, 'bg': 0, 'ba': 0, 'row': 0, 'col': 0, 'subch': 0}),
        (0x80, {'dimm': 0, 'bg': 1, 'ba': 0, 'row': 0, 'col': 0, 'subch': 0}),
        (0x400, {'dimm': 0, 'bg': 0, 'ba': 0, 'row': 0, 'col': 0x10, 'subch': 0}),  # col encoded
        (0x20000, {'dimm': 0, 'bg': 0, 'ba': 1, 'row': 0, 'col': 0, 'subch': 0}),
        (0x80000, {'dimm': 0, 'bg': 0, 'ba': 0, 'row': 0, 'col': 0, 'subch': 1}),
        (0x100000, {'dimm': 0, 'bg': 0, 'ba': 0, 'row': 1, 'col': 0, 'subch': 0}),
    ]

    passed = 0
    for dpa, expected in test_cases:
        result = translator.dpa_to_dram(dpa)
        match = all(result[k] == expected[k] for k in expected.keys())
        status = "✓" if match else "✗"

        print(f"\n{status} DPA 0x{dpa:X}")
        print(f"  Expected: {expected}")
        print(f"  Got:      {{dimm: {result['dimm']}, bg: {result['bg']}, ba: {result['ba']}, row: {result['row']}, col: {result['col']}, subch: {result['subch']}}}")

        if match:
            passed += 1

    print(f"\n{passed}/{len(test_cases)} tests passed")
    return passed == len(test_cases)


def test_reverse_translation():
    """Test DRAM → DPA translation"""
    print("\n" + "=" * 60)
    print("Testing Reverse Translation (DRAM → DPA)")
    print("=" * 60)

    translator = DPATranslator(mock_mode=True)

    test_cases = [
        ((0, 0, 0, 0, 0, 0, 0), 0x0),           # rank, bg, ba, row, col, dimm, subch
        ((0, 0, 0, 0, 0, 1, 0), 0x40),          # dimm=1
        ((0, 1, 0, 0, 0, 0, 0), 0x80),          # bg=1
        ((0, 0, 0, 0, 0x10, 0, 0), 0x400),      # col=0x10 (encoded)
        ((0, 0, 1, 0, 0, 0, 0), 0x20000),       # ba=1
        ((0, 0, 0, 0, 0, 0, 1), 0x80000),       # subch=1
        ((0, 0, 0, 1, 0, 0, 0), 0x100000),      # row=1
    ]

    passed = 0
    for (rank, bg, ba, row, col, dimm, subch), expected_dpa in test_cases:
        result_dpa = translator.dram_to_dpa(rank, bg, ba, row, col, dimm, subch)
        match = result_dpa == expected_dpa
        status = "✓" if match else "✗"

        print(f"\n{status} DRAM(dimm={dimm}, bg={bg}, ba={ba}, row={row}, col={col}, subch={subch})")
        print(f"  Expected DPA: 0x{expected_dpa:X}")
        print(f"  Got DPA:      0x{result_dpa:X}")

        if match:
            passed += 1

    print(f"\n{passed}/{len(test_cases)} tests passed")
    return passed == len(test_cases)


def test_roundtrip():
    """Test DPA → DRAM → DPA roundtrip"""
    print("\n" + "=" * 60)
    print("Testing Roundtrip (DPA → DRAM → DPA)")
    print("=" * 60)

    translator = DPATranslator(mock_mode=True)

    test_dpas = [0x0, 0x40, 0x80, 0x400, 0x20000, 0x80000, 0x100000, 0x1FFFC0]

    passed = 0
    for original_dpa in test_dpas:
        # Forward
        dram = translator.dpa_to_dram(original_dpa)

        # Reverse
        calculated_dpa = translator.dram_to_dpa(
            dram['rank'], dram['bg'], dram['ba'],
            dram['row'], dram['col'],
            dram['dimm'], dram['subch']
        )

        match = original_dpa == calculated_dpa
        status = "✓" if match else "✗"

        print(f"\n{status} DPA 0x{original_dpa:X} → DRAM → DPA 0x{calculated_dpa:X}")
        if match:
            passed += 1
        else:
            print(f"  DRAM: {dram}")

    print(f"\n{passed}/{len(test_dpas)} tests passed")
    return passed == len(test_dpas)


def test_specific_examples():
    """Test specific examples from GNR-CRB data"""
    print("\n" + "=" * 60)
    print("Testing Specific Examples from GNR-CRB")
    print("=" * 60)

    translator = DPATranslator(mock_mode=True)

    # From user's data (col is encoded: every 0x10 in col = 1KB in DPA)
    examples = [
        {
            'dpa': 0x0FFFC0,
            'expected': {'subch': 1, 'dimm': 1, 'rank': 0, 'bg': 7, 'ba': 3, 'row': 0, 'col': 0x7F0}
        },
        {
            'dpa': 0x100000,
            'expected': {'subch': 0, 'dimm': 0, 'rank': 0, 'bg': 0, 'ba': 0, 'row': 1, 'col': 0}
        },
        {
            'dpa': 0x7FFC0,
            'expected': {'subch': 0, 'dimm': 1, 'rank': 0, 'bg': 7, 'ba': 3, 'row': 0, 'col': 0x7F0}
        },
        {
            'dpa': 0x80000,
            'expected': {'subch': 1, 'dimm': 0, 'rank': 0, 'bg': 0, 'ba': 0, 'row': 0, 'col': 0}
        },
    ]

    passed = 0
    for example in examples:
        dpa = example['dpa']
        expected = example['expected']

        result = translator.dpa_to_dram(dpa)

        match = all(result[k] == expected[k] for k in expected.keys() if k != 'rank')
        status = "✓" if match else "✗"

        print(f"\n{status} DPA 0x{dpa:X}")
        print(f"  Expected: subch={expected['subch']}, dimm={expected['dimm']}, bg={expected['bg']}, ba={expected['ba']}, row={expected['row']}, col=0x{expected['col']:X}")
        print(f"  Got:      subch={result['subch']}, dimm={result['dimm']}, bg={result['bg']}, ba={result['ba']}, row={result['row']}, col=0x{result['col']:X}")

        if match:
            passed += 1

    print(f"\n{passed}/{len(examples)} tests passed")
    return passed == len(examples)


if __name__ == '__main__':
    print("\n" + "=" * 60)
    print("DPA Translator Test Suite")
    print("128GB CMM-D Device")
    print("=" * 60 + "\n")

    all_passed = True

    all_passed &= test_forward_translation()
    all_passed &= test_reverse_translation()
    all_passed &= test_roundtrip()
    all_passed &= test_specific_examples()

    print("\n" + "=" * 60)
    if all_passed:
        print("✓ ALL TESTS PASSED!")
    else:
        print("✗ SOME TESTS FAILED")
    print("=" * 60 + "\n")

    sys.exit(0 if all_passed else 1)
