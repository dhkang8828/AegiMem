# DPA Translation Validation

## Overview

`validate_dpa_translation.py` validates the accuracy of our DPA-to-DRAM address translation by comparing against the ground truth from the `umxc` tool.

## Prerequisites

1. **GNR-CRB board** with CXL Type3 memory installed
2. **umxc tool** installed and accessible in PATH
3. **Python 3.x** with access to `src/dpa_translator.py`

## Usage

### Basic Usage (1000 samples)

```bash
python3 validate_dpa_translation.py --samples 1000
```

### Large-scale validation (10,000 samples)

```bash
python3 validate_dpa_translation.py --samples 10000
```

### Save results to CSV

```bash
python3 validate_dpa_translation.py --samples 5000 --output validation_results.csv
```

### Reproducible validation (with seed)

```bash
python3 validate_dpa_translation.py --samples 1000 --seed 42
```

### Custom DPA range

```bash
# For 64GB device (0x0 - 0xFFFFFFFFF)
python3 validate_dpa_translation.py --samples 1000 --max-dpa 0xFFFFFFFFF

# For 128GB device (default: 0x0 - 0x1FFFFFFFF)
python3 validate_dpa_translation.py --samples 1000
```

### Quiet mode (suppress progress)

```bash
python3 validate_dpa_translation.py --samples 10000 --quiet
```

## Command-line Options

| Option | Description | Default |
|--------|-------------|---------|
| `--samples`, `-n` | Number of random DPA samples to test | 1000 |
| `--max-dpa` | Maximum DPA value (hex or decimal) | 0x1FFFFFFFF |
| `--output`, `-o` | Output CSV file for detailed results | None |
| `--seed` | Random seed for reproducibility | None |
| `--quiet`, `-q` | Suppress progress messages | False |

## Output

### Console Output

```
============================================================
Validating 1000 DPA translations
============================================================

Progress: 100/1000 (10%)
Progress: 200/1000 (20%)
...
Progress: 1000/1000 (100%)

============================================================
Validation Results
============================================================
Total samples:     1000
Correct:           1000
Mismatches:        0
umxc failures:     0
Accuracy:          100.00%
============================================================

✓ All translations match umxc!
```

### CSV Output (with --output)

The CSV file contains detailed results for each DPA:

| DPA | Match | Error | Our_SubCh | Our_DIMM | Our_BG | ... | Umxc_SubCh | Umxc_DIMM | Umxc_BG | ... |
|-----|-------|-------|-----------|----------|--------|-----|------------|-----------|---------|-----|
| 0x0 | MATCH | | 0 | 0 | 0 | ... | 0 | 0 | 0 | ... |
| 0x40 | MATCH | | 0 | 1 | 0 | ... | 0 | 1 | 0 | ... |
| ... | ... | ... | ... | ... | ... | ... | ... | ... | ... | ... |

## Mismatch Analysis

If mismatches are detected, the tool will print details:

```
============================================================
Mismatch Details (showing first 10)
============================================================

Mismatch #1: DPA 0x123400
  Our result:  subch=0, dimm=1, rank=0, bg=4, ba=2, row=0x12, col=0x34
  umxc result: subch=0, dimm=1, rank=0, bg=4, ba=2, row=0x12, col=0x35
  Differences: col(ours=52, umxc=53)

...
```

## Expected Results

For the 128GB CMM-D device with correct implementation:
- **Accuracy: 100.00%**
- **Mismatches: 0**

If accuracy is less than 100%, review the mismatches to identify:
1. Systematic errors in translation formula
2. Edge cases at boundaries
3. Interleaving rule misunderstandings

## umxc Output Format

The validation tool expects umxc output in the following format:

```
Select MXC_GEN2_SOC-PCI 0000:0c:00.0 (1B00:C002 REV03 80-CE-00-00-00-00-00-00)
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

The parser looks for the "Successful" keyword and then extracts each field using regex patterns.

## Troubleshooting

### Error: "umxc tool not found"

Make sure `umxc` is installed and in PATH:

```bash
which umxc
umxc --help
```

### Error: "umxc failed for DPA 0x..."

Check if the DPA is valid:

```bash
# Test umxc manually
umxc ei -t 0x100000
```

Verify the output contains "Successful" keyword.

### Parsing errors

If umxc output format is different from expected, the parser will print:
```
[WARNING] Could not find 'field_name' in umxc output for DPA 0x...
```

Check the actual umxc output and update the regex patterns in `_parse_umxc_output()` if needed.

### Testing the parser locally

Before running on GNR-CRB, test the parser with actual umxc output:

```bash
# Save umxc output to file
umxc ei -t 0x12345 > test_output.txt

# Test parsing
python3 -c "
from validate_dpa_translation import UmxcInterface
with open('test_output.txt', 'r') as f:
    output = f.read()
result = UmxcInterface._parse_umxc_output(output, 0x12345)
print(result)
"
```

## Example Workflow

```bash
# Step 1: Quick test (100 samples)
python3 validate_dpa_translation.py --samples 100

# Step 2: If successful, larger test (10K samples)
python3 validate_dpa_translation.py --samples 10000 --output validation_10k.csv

# Step 3: Analyze CSV if there are mismatches
# Review validation_10k.csv for patterns in mismatches

# Step 4: Full validation (100K samples)
python3 validate_dpa_translation.py --samples 100000 --output validation_full.csv --quiet
```

## Integration with CI/CD

```bash
#!/bin/bash
# validate_ci.sh - Run DPA translation validation

set -e

echo "Running DPA translation validation..."
python3 validate_dpa_translation.py --samples 10000 --seed 12345

if [ $? -eq 0 ]; then
    echo "✓ DPA translation validation PASSED"
    exit 0
else
    echo "✗ DPA translation validation FAILED"
    exit 1
fi
```

## Performance

Approximate validation times (GNR-CRB board):
- 100 samples: ~5-10 seconds
- 1,000 samples: ~30-60 seconds
- 10,000 samples: ~5-10 minutes
- 100,000 samples: ~50-100 minutes

Each umxc call takes approximately 50ms, so validation time scales linearly with sample count.
