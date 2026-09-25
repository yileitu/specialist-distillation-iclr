# NUM_TASKS Number Extraction and Validation Strategy

## Overview

This document describes the strategy for extracting and validating numerical answers from model-generated responses for NUM_TASKS (numerical property prediction tasks).

## NUM_TASKS Definition

According to `smol/util.py`, NUM_TASKS include:
- `property_prediction-lipo` (Lipophilicity prediction)
- `property_prediction-esol` (Aqueous solubility prediction)

These tasks require extracting continuous numerical values (integers or decimals) from model outputs.

## Processing Pipeline

### 1. Number Extraction Strategy

When the `extracted_core_answer` field contains an empty string for any generation, the system attempts to extract numbers from the corresponding `generated` text using a **three-tier fallback strategy**:

#### Priority 1: LaTeX \boxed{} Pattern
```
Pattern: \boxed{number}
```
- **Target**: The last `\boxed{}` command containing a number
- **Example**: `\boxed{2.95}` → extracts `2.95`
- **Rationale**: Models often wrap final answers in `\boxed{}` following mathematical convention
- **Constraint**: Must contain a valid number, not text (e.g., `\boxed{approximately 3}` would fail)
- **Special Handling**: If content contains `infinite`, `infinity`, `inf`, `undefined`, or `undef`, extracts `2147483647` (max int32 as placeholder)
  - `\boxed{infinite}` → extracts `2147483647`
  - `\boxed{undefined}` → extracts `2147483647`
  - `\boxed{The solubility is infinite}` → extracts `2147483647`

#### Priority 2: Markdown Bold Pattern
```
Pattern: **number**
```
- **Target**: The last number wrapped in double asterisks (Markdown bold)
- **Example**: `The answer is **0.48**` → extracts `0.48`
- **Rationale**: Models frequently emphasize numerical answers using bold formatting
- **Constraint**: Must contain a valid number, not descriptive text
- **Special Handling**: If content contains `infinite`, `infinity`, `inf`, `undefined`, or `undef`, extracts `2147483647` (max int32 as placeholder)
  - `**infinite**` → extracts `2147483647`
  - `**undefined**` → extracts `2147483647`
  - `The logarithmic solubility is **infinity**` → extracts `2147483647`

#### Priority 3: Last Number in Text
```
Pattern: Any numerical value
```
- **Target**: The very last number appearing anywhere in the generated text
- **Example**: `...estimated at 1.5 units.` → extracts `1.5`
- **Rationale**: When structured formatting is absent, assume the final number is the answer
- **Risk**: May capture incidental numbers (e.g., references, dates)
- **Note**: This priority does NOT handle `infinite`/`undefined` keywords (only Priority 1 & 2 do)

### 2. Number Format Support

The extraction supports:
- **Integers**: `42`, `-7`
- **Decimals with varying precision**: `3.14`, `0.001`, `-2.718281828`
- **Leading zeros**: `0.5`
- **Negative numbers**: `-1.23`
- **Special keywords** (Priority 1 & 2 only): `infinite`, `infinity`, `inf`, `undefined`, `undef` → mapped to `2147483647`

### 3. Fallback Behavior

If all three strategies fail to find a valid number:
- The `extracted_core_answer` remains as an empty string `""`
- The corresponding `extracted_core_answer_correctness` is set to `False`

## Answer Validation

### Correctness Criterion

An extracted answer is considered correct if:

```
|extracted_value - ground_truth| ≤ tolerance
```

**Default Tolerance**: ±1.0 (absolute difference, configurable via `--tolerance` parameter)

### Examples

| Ground Truth | Extracted | Difference | Correct? |
|--------------|-----------|------------|----------|
| 0.26         | 0.0       | 0.26       | ✓ True   |
| 0.7          | 2.95      | 2.25       | ✗ False  |
| -0.18        | -0.5      | 0.32       | ✓ True   |
| 2.5          | 3.5       | 1.0        | ✓ True   |
| 2.5          | 3.51      | 1.01       | ✗ False  |

### Rationale for Default ±1.0 Tolerance

1. **Measurement Uncertainty**: Chemical property predictions inherently have experimental uncertainty
2. **Model Approximation**: Language models may provide rounded or approximate values
3. **Unit Consistency**: Ensures minor unit conversion issues don't incorrectly mark answers as wrong
4. **Task Difficulty**: Continuous value prediction is more challenging than classification

**Configurable Tolerance**: You can adjust the tolerance value based on your specific requirements:
- **Stricter**: Use `--tolerance 0.1` or `--tolerance 0.5` for more precise validation
- **Looser**: Use `--tolerance 2.0` or higher for more lenient validation
- **Task-specific**: Different chemical properties may require different tolerance levels

## Output Format

### Added Fields

The script adds the `extracted_core_answer_correctness` field to each data point:

```json
{
  "task": "property_prediction-esol",
  "ground_truth": "0.26",
  "extracted_core_answer": {
    "generation1": "0",
    "generation2": "2.95",
    "generation3": ""
  },
  "extracted_core_answer_correctness": {
    "generation1": true,
    "generation2": false,
    "generation3": false
  }
}
```

### Output File Naming

Processed files are saved with a suffix indicating tolerance value (default: `_with_extracted_numbers_tol{tolerance}.jsonl`):

```
Original: inference_split1000.jsonl
Output (tolerance=1.0):   inference_split1000_with_extracted_numbers_tol1.0.jsonl
Output (tolerance=0.5):   inference_split1000_with_extracted_numbers_tol0.5.jsonl
Output (custom suffix):   inference_split1000_custom_suffix.jsonl
```

## Usage

### Command Line

```bash
python smol/validate_answer/extract_numbers.py \
  /path/to/inference_split1000.jsonl \
  [--tolerance TOLERANCE] \
  [--output-suffix SUFFIX]
```

### Parameters

- `input_file`: Path to input JSONL file (required)
- `--tolerance`: Tolerance for answer validation (default: 1.0)
- `--output-suffix`: Custom suffix for output filename (default: `_with_extracted_numbers_tol{tolerance}`)

### Examples

**Basic usage (default tolerance=1.0):**
```bash
python smol/validate_answer/extract_numbers.py \
  smol/inference/50k_all_subtasks/Intern-S1-mini_smol_instruct_all_no_cot_b4_t16_epoch1/presence0.0_freq0.0_repetition1.0_loop5-200_rep3/property_prediction-esol/inference_split1000.jsonl
```
Output: `inference_split1000_with_extracted_numbers_tol1.0.jsonl`

**Custom tolerance (±0.5):**
```bash
python smol/validate_answer/extract_numbers.py \
  inference_split1000.jsonl \
  --tolerance 0.5
```
Output: `inference_split1000_with_extracted_numbers_tol0.5.jsonl`

**Stricter validation (±0.1):**
```bash
python smol/validate_answer/extract_numbers.py \
  inference_split1000.jsonl \
  --tolerance 0.1
```
Output: `inference_split1000_with_extracted_numbers_tol0.1.jsonl`

**Custom suffix:**
```bash
python smol/validate_answer/extract_numbers.py \
  inference_split1000.jsonl \
  --tolerance 1.0 \
  --output-suffix _validated
```
Output: `inference_split1000_validated.jsonl`

## Implementation Details

### Script: `extract_numbers.py`

**Key Functions**:

1. `extract_number_from_text(text: str) -> Optional[str]`
   - Implements the three-tier extraction strategy
   - Returns extracted number as string or None

2. `extract_pure_number(text: str) -> Optional[str]`
   - Helper to extract clean numbers from mixed text
   - Handles edge cases like "approximately 3" → extracts "3"

3. `is_answer_correct(extracted: str, ground_truth: str, tolerance: float = 1.0) -> bool`
   - Validates extracted answer against ground truth
   - Configurable tolerance (default: 1.0)

4. `process_jsonl_file(input_path: str, output_suffix: str) -> str`
   - Main processing pipeline
   - Handles file I/O and progress reporting

### Task Filtering

The script **only processes lines where** `task ∈ NUM_TASKS`. Other tasks are written to output unchanged.

## Statistics Output

The script reports:
- Total lines processed
- Number of values successfully extracted
- Number of correct answers (with tolerance value)
- Output file location

Example:
```
Processing complete:
  Total lines processed: 1000
  Numbers extracted: 234
  Correct answers (tolerance=±1.0): 187
  Output saved to: inference_split1000_with_extracted_numbers_tol1.0.jsonl
```

## Edge Cases

### Empty Generated Text
- If `generated[generationN]` is empty or missing → `extracted_core_answer[generationN]` remains `""`

### Non-numeric Content in \boxed{}
- Example: `\boxed{approximately zero}` → Falls back to Priority 2 or 3
- Example: `\boxed{infinite}` → Extracts `2147483647` (special handling)
- Example: `\boxed{undefined}` → Extracts `2147483647` (special handling)

### Multiple Numbers in \boxed{}
- Example: `\boxed{1.5 to 2.5}` → Extracts the last number (`2.5`)

### Infinite/Undefined Values
- **Priority 1 & 2**: Keywords like `infinite`, `infinity`, `inf`, `undefined`, `undef` are mapped to `2147483647`
  - `\boxed{infinite}` → `2147483647`
  - `**infinity**` → `2147483647`
  - `\boxed{The answer is undefined}` → `2147483647`
- **Priority 3**: These keywords are NOT handled; only actual numbers are extracted
- **Rationale**: `2147483647` (max int32) serves as a sentinel value indicating unbounded/undefined results
- **Validation**: With default tolerance ±1.0, these will likely be marked incorrect unless ground truth is also very large

### Scientific Notation
- Currently **not supported**: `1.23e-5` would extract `5` (the last number)
- Future enhancement opportunity

## Related Files

- **Task Definitions**: `smol/util.py` (defines `NUM_TASKS`)
- **Inference Results**: `smol/inference/*/property_prediction-{esol,lipo}/inference_split*.jsonl`
- **Extraction Script**: `smol/validate_answer/extract_numbers.py`

## Future Enhancements

1. **Scientific Notation Support**: Handle `1e-3`, `2.5E+2` formats
2. **Unit Extraction**: Parse and normalize units (e.g., "mg/L" vs "g/L")
3. **Confidence Scores**: Extract model confidence when available
4. **Multi-value Extraction**: Handle ranges like "1.5 to 2.0"
5. **Relative Tolerance**: Support percentage-based tolerance (e.g., ±10% of ground truth)

---

**Last Updated**: 2026-01-16  
**Author**: Automated extraction pipeline for NUM_TASKS evaluation
