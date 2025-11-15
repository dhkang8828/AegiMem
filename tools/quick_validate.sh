#!/bin/bash
#
# Quick DPA Translation Validation
# Run this on GNR-CRB board to quickly validate DPA translation accuracy
#

set -e

echo "=========================================="
echo "DPA Translation Quick Validation"
echo "=========================================="
echo ""

# Check if umxc is available
if ! command -v umxc &> /dev/null; then
    echo "ERROR: umxc command not found"
    echo "Please make sure umxc is installed and in PATH"
    exit 1
fi

echo "✓ umxc found: $(which umxc)"
echo ""

# Test a few specific DPAs manually
echo "Testing specific DPAs with umxc..."
echo ""

test_dpas=(
    "0x0"
    "0x40"
    "0x80"
    "0x400"
    "0x20000"
    "0x80000"
    "0x100000"
)

for dpa in "${test_dpas[@]}"; do
    echo "  Testing DPA $dpa..."
    umxc ei -t $dpa > /tmp/umxc_test_$dpa.txt 2>&1
    if grep -q "Successful" /tmp/umxc_test_$dpa.txt; then
        echo "    ✓ Success"
    else
        echo "    ✗ Failed"
        cat /tmp/umxc_test_$dpa.txt
    fi
done

echo ""
echo "=========================================="
echo "Running Python validation script..."
echo "=========================================="
echo ""

# Run Python validation with 100 samples first
python3 validate_dpa_translation.py --samples 100

RESULT=$?

echo ""
echo "=========================================="
if [ $RESULT -eq 0 ]; then
    echo "✓ Quick validation PASSED"
    echo ""
    echo "Next steps:"
    echo "  1. Run larger validation:"
    echo "     python3 validate_dpa_translation.py --samples 10000 --output results.csv"
    echo ""
    echo "  2. Review results.csv if there are any mismatches"
else
    echo "✗ Quick validation FAILED"
    echo ""
    echo "Please check the error messages above"
fi
echo "=========================================="

exit $RESULT
