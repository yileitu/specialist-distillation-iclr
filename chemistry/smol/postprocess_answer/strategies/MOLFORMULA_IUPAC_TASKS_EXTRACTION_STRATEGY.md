# MOLFORMULA_TASKS & IUPAC_TASKS Extraction and Validation Strategy

## Overview

This document describes the strategy for extracting and validating molecular formulas and IUPAC names from model-generated responses for MOLFORMULA_TASKS and IUPAC_TASKS.

## Task Definitions

According to `smol/util.py`, these tasks include:

### MOLFORMULA_TASKS
- `name_conversion-s2f` (SMILES to molecular formula)
- `name_conversion-i2f` (IUPAC name to molecular formula)

These tasks require extracting molecular formulas like "C6H12O6" from model outputs.

### IUPAC_TASKS
- `name_conversion-s2i` (SMILES to IUPAC name)

These tasks require extracting IUPAC chemical names like "ethanol" or "2-methylpropane" from model outputs.

## Processing Pipeline

### 1. Text Extraction Strategy

When the `extracted_core_answer` field contains an empty string for any generation, the system attempts to extract text strings from the corresponding `generated` text using a **three-tier fallback strategy**:

#### Priority 1: XML Tags (`<MOLFORMULA>` or `<IUPAC>`)

**For MOLFORMULA_TASKS:**
```xml
<MOLFORMULA>C6H12O6</MOLFORMULA>
```

**For IUPAC_TASKS:**
```xml
<IUPAC>ethanol</IUPAC>
```

- **Target**: Content within the last tag of the appropriate type
- **Tag Selection**: Automatically determined based on task type
- **Examples**:
  - `<MOLFORMULA>H2O</MOLFORMULA>` → extracts `H2O`
  - `<IUPAC>2-methylpropane</IUPAC>` → extracts `2-methylpropane`
  - Multiple tags: `<MOLFORMULA>CH4</MOLFORMULA>...<MOLFORMULA>C2H6</MOLFORMULA>` → extracts `C2H6` (last one)
- **Processing**: Strips leading/trailing whitespace
- **Rationale**: Explicit tags provide the clearest signal of the model's answer

#### Priority 2: `\boxed{}` Pattern
```latex
\boxed{C6H12O6}
```

- **Target**: Content within the last `\boxed{}` LaTeX command
- **Examples**:
  - `\boxed{C6H12O6}` → extracts `C6H12O6`
  - `\boxed{ethanol}` → extracts `ethanol`
  - `\boxed{2-methyl-3-pentanone}` → extracts `2-methyl-3-pentanone`
- **Processing**: Strips leading/trailing whitespace
- **Rationale**: Models often use `\boxed{}` for final answers following mathematical convention

#### Priority 3: `**text**` (Markdown Bold)
```markdown
**C6H12O6**
```

- **Target**: Content within the last `**...**` (markdown bold)
- **Examples**:
  - `The formula is **C6H12O6**` → extracts `C6H12O6`
  - `**ethanol**` → extracts `ethanol`
  - `Answer: **2,3-dimethylbutane**` → extracts `2,3-dimethylbutane`
- **Processing**: Strips leading/trailing whitespace
- **Rationale**: Bold formatting often emphasizes the final answer

### 2. Fallback Behavior

If all three strategies fail to extract text:
- The `extracted_core_answer` remains as an empty string `""`
- The corresponding `extracted_core_answer_correctness` is set to `False`

## Answer Validation

### Correctness Criterion

**This follows the official LLM4Chem evaluation methods from the SMol dataset.**

#### For MOLFORMULA_TASKS:
```
parse_and_compare_elements(normalize(extracted_formula), normalize(ground_truth_formula))
```

#### For IUPAC_TASKS:
```
split_sort_and_compare(normalize(extracted_name), normalize(ground_truth_name))
```

### Method 1: Element Match (MOLFORMULA_TASKS)

**Evaluation Process**:
1. **Preprocessing** (optional, enabled by default): Normalize molecular formula
   - Remove all whitespace
   - Remove special characters except valid formula characters
   - Example: `C5 H8 O2.` → `C5H8O2`

2. **Parse both formulas** using `parse_molecule()` from LLM4Chem reference code
   - Parse molecular formula into element count dictionary
   - Handle parentheses: `Ca(OH)2` → `{'Ca': 1, 'O': 2, 'H': 2}`
   - Handle charges: `NH4+` → `{'N': 1, 'H': 4, '+': 1}`

3. **Compare element dictionaries** for exact match
   - Order-independent comparison
   - Returns True only if all elements and counts match exactly

**Example**:
```python
parse_molecule("C5H8O2")  # -> {'C': 5, 'H': 8, 'O': 2}
parse_molecule("H8C5O2")  # -> {'C': 5, 'H': 8, 'O': 2}
# Dictionaries are equal → True (order doesn't matter)
```

#### Parsing Algorithm

The `parse_molecule()` function uses a **stack-based recursive algorithm**:

1. **Match patterns**: atom, opening bracket, closing bracket
2. **Atom matched**: Add to current stack level with count
3. **Opening bracket**: Push new stack level
4. **Closing bracket**: Pop stack level, multiply counts, add to parent level
5. **Charges**: Parse `+`, `-` with optional count

**Example**: Parsing `"Ca(OH)2"`
```
Initial: stack = [{}]
'Ca'   → stack = [{'Ca': 1}]
'('    → stack = [{'Ca': 1}, {}]
'O'    → stack = [{'Ca': 1}, {'O': 1}]
'H'    → stack = [{'Ca': 1}, {'O': 1, 'H': 1}]
')2'   → Pop: multiply by 2, add to parent
         stack = [{'Ca': 1, 'O': 2, 'H': 2}]
Result: {'Ca': 1, 'O': 2, 'H': 2}
```

### Method 2: Split Match (IUPAC_TASKS)

**Evaluation Process**:
1. **Preprocessing** (optional, enabled by default): Normalize IUPAC name
   - Convert to lowercase
   - Remove extra whitespace
   - Remove special characters except: letters, numbers, hyphens, commas, separator
   - Example: `Ethanol.` → `ethanol`

2. **Split by separator** (default: `;`)
   - Multiple IUPAC names can be separated by semicolons
   - Example: `"ethanol;methanol"` → `["ethanol", "methanol"]`

3. **Sort the split parts**
   - Create sorted tuple for order-independent comparison
   - Example: `["ethanol", "methanol"]` → `("ethanol", "methanol")`

4. **Compare sorted tuples** for exact match
   - Returns True only if sorted parts match exactly

**Example**:
```python
"ethanol;methanol".split(';')     # -> ["ethanol", "methanol"]
sorted(["ethanol", "methanol"])   # -> ("ethanol", "methanol")

"methanol;ethanol".split(';')     # -> ["methanol", "ethanol"]  
sorted(["methanol", "ethanol"])   # -> ("ethanol", "methanol")

# Both produce the same sorted tuple → True (order doesn't matter)
```

### Normalization (Optional but Recommended)

**Why Normalization?**

Model outputs may have formatting variations:
- Case differences: `"Ethanol"` vs `"ethanol"`
- Whitespace: `"C5 H8 O2"` vs `"C5H8O2"`
- Punctuation: `"Ethanol."` vs `"ethanol"`

Normalization makes evaluation more robust while maintaining chemical accuracy.

#### MOLFORMULA Normalization

`normalize_molformula()` function:

```python
normalize_molformula("C5 H8 O2")   # -> "C5H8O2" (removes whitespace)
normalize_molformula("C5H8O2.")    # -> "C5H8O2" (removes punctuation)
normalize_molformula("Ca(OH)2")    # -> "Ca(OH)2" (preserves parentheses)
```

**Operations**:
1. Remove all whitespace
2. Remove special characters
3. **Preserve**: Letters, numbers, parentheses `()[]{}`, plus/minus `+-`

#### IUPAC Normalization

`normalize_iupac_name()` function:

```python
normalize_iupac_name("Ethanol  ")           # -> "ethanol"
normalize_iupac_name("Ethanol.")            # -> "ethanol"
normalize_iupac_name("2-Propanol")          # -> "2-propanol"
normalize_iupac_name("ethanol; methanol")   # -> "ethanol; methanol"
```

**Operations**:
1. Convert to lowercase
2. Remove extra whitespace
3. Remove special characters
4. **Preserve**: Letters, numbers, hyphens `-`, commas `,`, separator `;`

#### Command-Line Control

```bash
# Enable normalization (default, recommended)
python postprocess_molformula_iupac_tasks.py inference.jsonl
python postprocess_molformula_iupac_tasks.py inference.jsonl --normalize

# Disable normalization (strict matching, matches LLM4Chem original code exactly)
python postprocess_molformula_iupac_tasks.py inference.jsonl --no-normalize
```

### Why Element Match & Split Match?

**Comparison with Old Method (Levenshtein Similarity)**:

| Aspect | Old Method (Levenshtein) | New Method (LLM4Chem) |
|--------|--------------------------|----------------------|
| Evaluation | Fuzzy string similarity ≥ 0.97 | Exact chemical/semantic match |
| MOLFORMULA | Character similarity | Element count comparison |
| IUPAC | Character similarity | Split & sort comparison |
| Order sensitivity | Sensitive | **Insensitive** ✓ |
| Alignment with official | ✗ Different | **✓ Exact match** |
| False positives | Possible (fuzzy) | Minimal (exact) |
| False negatives | Possible (threshold) | Minimal (with normalization) |
| Threshold required | Yes (e.g., 0.97) | **No** |

### Examples

#### MOLFORMULA Tasks

| Ground Truth | Extracted | Normalized | Element Dict | Match? |
|--------------|-----------|------------|--------------|--------|
| C5H8O2 | C5H8O2 | C5H8O2 / C5H8O2 | {'C':5,'H':8,'O':2} / {'C':5,'H':8,'O':2} | ✓ True |
| C5H8O2 | H8C5O2 | C5H8O2 / H8C5O2 | {'C':5,'H':8,'O':2} / {'C':5,'H':8,'O':2} | ✓ True |
| C5H8O2 | C5 H8 O2 | C5H8O2 / C5H8O2 | Same / Same | ✓ True |
| C5H8O2 | C5H8O2. | C5H8O2 / C5H8O2 | Same / Same | ✓ True |
| Ca(OH)2 | CaO2H2 | Ca(OH)2 / CaO2H2 | {'Ca':1,'O':2,'H':2} / {'Ca':1,'O':2,'H':2} | ✓ True |
| Ca(OH)2 | CaH2O2 | Ca(OH)2 / CaH2O2 | {'Ca':1,'O':2,'H':2} / {'Ca':1,'H':2,'O':2} | ✓ True |
| C5H8O2 | C5H8O | C5H8O2 / C5H8O | {'C':5,'H':8,'O':2} / {'C':5,'H':8,'O':1} | ✗ False |
| C5H8O2 | "" | C5H8O2 / N/A | Valid / N/A | ✗ False |
| C5H8O2 | invalid | C5H8O2 / N/A | Valid / None (error) | ✗ False |

**Note**: Order doesn't matter - `C5H8O2` == `H8C5O2` == `O2H8C5` (all parse to same element dictionary)

#### IUPAC Tasks

| Ground Truth | Extracted | Normalized | Split & Sort | Match? |
|--------------|-----------|------------|--------------|--------|
| ethanol | ethanol | ethanol / ethanol | ("ethanol",) / ("ethanol",) | ✓ True |
| ethanol | Ethanol | ethanol / ethanol | Same / Same | ✓ True |
| ethanol | Ethanol. | ethanol / ethanol | Same / Same | ✓ True |
| 2-propanol | 2-Propanol | 2-propanol / 2-propanol | ("2-propanol",) / ("2-propanol",) | ✓ True |
| ethanol;methanol | methanol;ethanol | Same / Same | ("ethanol","methanol") / ("ethanol","methanol") | ✓ True |
| ethanol;methanol | ethanol;methanol | Same / Same | Same / Same | ✓ True |
| ethanol | ethanol  | ethanol / ethanol | Same / Same | ✓ True |
| ethanol | methanol | ethanol / methanol | ("ethanol",) / ("methanol",) | ✗ False |
| ethanol | "" | ethanol / N/A | Valid / N/A | ✗ False |

**Note**: Order doesn't matter - `ethanol;methanol` == `methanol;ethanol` (both sort to same tuple)

### Normalization Comparison

#### MOLFORMULA Tasks

| Ground Truth | Extracted | --normalize (default) | --no-normalize |
|--------------|-----------|----------------------|----------------|
| C5H8O2 | C5H8O2 | ✓ True | ✓ True |
| C5H8O2 | H8C5O2 | ✓ True | ✓ True |
| C5H8O2 | C5 H8 O2 | ✓ True | ✗ False |
| C5H8O2 | C5H8O2. | ✓ True | ✗ False |
| Ca(OH)2 | CaH2O2 | ✓ True | ✓ True |

#### IUPAC Tasks

| Ground Truth | Extracted | --normalize (default) | --no-normalize |
|--------------|-----------|----------------------|----------------|
| ethanol | ethanol | ✓ True | ✓ True |
| ethanol | Ethanol | ✓ True | ✗ False |
| ethanol | Ethanol. | ✓ True | ✗ False |
| ethanol;methanol | methanol;ethanol | ✓ True | ✓ True |
| 2-propanol | 2-Propanol | ✓ True | ✗ False |
| ethanol | ethanol  | ✓ True | ✗ False |

### Error Handling

- **Empty string**: Directly marked as `False`, no processing attempted
- **Invalid molecular formula**: If `parse_molecule()` fails, marked as `False`
- **Parsing exception**: Any error during parsing → `False`

## Output Format

### Added Fields

The script adds the `extracted_core_answer_correctness` field to each data point:

```json
{
  "task": "name_conversion-s2f",
  "ground_truth": "C6H12O6",
  "extracted_core_answer": {
    "generation1": "C6H12O6",
    "generation2": "H12C6O6",
    "generation3": ""
  },
  "extracted_core_answer_correctness": {
    "generation1": true,
    "generation2": true,
    "generation3": false
  }
}
```

**Note**: `generation2` is also correct because element counts match (order-independent).

### Output File Naming

Processed files are saved with `_text_postprocessed` suffix:

```
Original: inference_split1000.jsonl
Output:   inference_split1000_text_postprocessed.jsonl
Log:      inference_split1000_text_postprocessed_log.txt
```

## Usage

### Command Line

```bash
python smol/postprocess_answer/postprocess_molformula_iupac_tasks.py \
  /path/to/inference_split1000.jsonl \
  [--output-suffix SUFFIX] \
  [--normalize | --no-normalize]
```

### Parameters

- `input_file`: Path to input JSONL file (required)
- `--output-suffix`: Custom suffix for output filename (default: `_text_postprocessed`)
- `--normalize`: Enable normalization before comparison (default: enabled, recommended)
- `--no-normalize`: Disable normalization (strict matching, matches LLM4Chem exactly)

### Examples

**Basic usage (default, normalization enabled):**
```bash
python smol/postprocess_answer/postprocess_molformula_iupac_tasks.py \
  smol/inference/*/name_conversion-s2f/inference_split1000.jsonl
```
Output: `inference_split1000_text_postprocessed.jsonl`

**Explicit normalization enable:**
```bash
python smol/postprocess_answer/postprocess_molformula_iupac_tasks.py \
  inference_split1000.jsonl \
  --normalize
```

**Disable normalization (strict matching):**
```bash
python smol/postprocess_answer/postprocess_molformula_iupac_tasks.py \
  inference_split1000.jsonl \
  --no-normalize
```
Use this only for:
- Exact match with LLM4Chem original code
- Debugging format issues
- Studying model's format output capability

**Custom suffix:**
```bash
python smol/postprocess_answer/postprocess_molformula_iupac_tasks.py \
  inference_split1000.jsonl \
  --output-suffix _validated
```
Output: `inference_split1000_validated.jsonl`

**Process all MOLFORMULA and IUPAC tasks:**
```bash
for task in name_conversion-{s2f,i2f,s2i}; do
  python smol/postprocess_answer/postprocess_molformula_iupac_tasks.py \
    smol/inference/*/${task}/inference_split*.jsonl
done
```

## Implementation Details

### Script: `postprocess_molformula_iupac_tasks.py`

**Key Functions**:

1. `extract_text_from_generation(text: str, task: str) -> Optional[str]`
   - Implements the three-tier extraction strategy
   - Returns text string or None

2. `extract_from_xml_tags(text: str, task: str) -> Optional[str]`
   - Priority 1: Extracts from `<MOLFORMULA>` or `<IUPAC>` tags based on task

3. `extract_from_boxed(text: str) -> Optional[str]`
   - Priority 2: Extracts from `\boxed{}` pattern

4. `extract_from_bold_markers(text: str) -> Optional[str]`
   - Priority 3: Extracts from `**text**` pattern

5. `parse_molecule(molecular_formula: str) -> dict`
   - **From LLM4Chem reference code** (`metrics.py:270-321`)
   - Parses molecular formula into element count dictionary
   - Handles parentheses, brackets, and charges
   - Stack-based recursive algorithm

6. `normalize_molformula(formula: str) -> str`
   - Normalizes molecular formula (removes whitespace, special chars)
   - Preserves valid formula characters

7. `normalize_iupac_name(name: str, separator: str = ';') -> str`
   - Normalizes IUPAC name (lowercase, removes punctuation)
   - Preserves hyphens, numbers, separator

8. `is_molformula_correct(extracted: str, ground_truth: str, normalize: bool = True) -> bool`
   - **Based on LLM4Chem** `count_element_match()` (`metrics.py:324-351`)
   - Optionally normalizes both formulas
   - Parses to element dictionaries
   - Compares for exact match

9. `is_iupac_correct(extracted: str, ground_truth: str, separator: str = ';', normalize: bool = True) -> bool`
   - **Based on LLM4Chem** `judge_string_split_match()` (`metrics.py:256-267`)
   - Optionally normalizes both names
   - Splits by separator and sorts
   - Compares sorted tuples for exact match

10. `is_text_correct(extracted: str, ground_truth: str, task: str, normalize: bool = True) -> bool`
    - Routes to appropriate validation method based on task type
    - Returns True if correct according to task-specific method

### Task Filtering

The script processes **both MOLFORMULA_TASKS and IUPAC_TASKS**:
- `name_conversion-s2f`
- `name_conversion-i2f`
- `name_conversion-s2i`

Other tasks are written to output unchanged.

## Statistics Output

The script reports:
- Total lines processed
- Number of text answers successfully extracted
- Number of correct answers
- Overall accuracy (correct / total lines)
- Breakdown by extraction strategy
- Normalization setting status
- **Per-task statistics** (extracted, correct, accuracy for each task)
- Output file location
- Log file location

Example:
```
======================================================================
TEXT_TASKS (MOLFORMULA + IUPAC) Postprocessing Report
======================================================================
Timestamp: 2026-01-21 15:30:45
Input file: inference_split1000.jsonl
Output file: inference_split1000_text_postprocessed.jsonl
Evaluation method: LLM4Chem (element_match for MOLFORMULA, split_match for IUPAC)
Normalization: Enabled

Processing complete:
  Total lines processed: 1500
  Text answers extracted: 4200
  Correct answers: 3654
  Accuracy: 87.00%

Extraction strategies used:
    xml_tags: 950
    boxed: 1300
    bold_markers: 1950
    not_extracted: 300

Per-task statistics:
  name_conversion-s2f:
    Extracted: 1450
    Correct: 1276
    Accuracy: 88.00%
  name_conversion-i2f:
    Extracted: 1400
    Correct: 1204
    Accuracy: 86.00%
  name_conversion-s2i:
    Extracted: 1350
    Correct: 1174
    Accuracy: 87.00%
======================================================================

Log saved to: inference_split1000_text_postprocessed_log.txt
```

## Edge Cases

### Empty Generated Text
- If `generated[generationN]` is empty or missing → `extracted_core_answer[generationN]` remains `""`

### MOLFORMULA Edge Cases
- **Parentheses**: `Ca(OH)2` ✓ Supported
- **Nested parentheses**: `Fe2(SO4)3` ✓ Supported
- **Charges**: `NH4+`, `SO4-2` ✓ Supported
- **Invalid format**: Parsing fails → `False`
- **Whitespace**: `C5 H8 O2` → normalized to `C5H8O2` (with normalization)

### IUPAC Edge Cases
- **Single name**: `ethanol` ✓ Works normally
- **Multiple names**: `ethanol;methanol` ✓ Order-independent
- **Case variations**: `Ethanol`, `ETHANOL` → normalized to `ethanol` (with normalization)
- **Whitespace**: Leading/trailing spaces removed (with normalization)
- **Empty parts**: `ethanol;` → May cause issues

## Known Limitations

1. **No Chemical Validation**: The script does not verify if:
   - Molecular formulas are chemically valid
   - IUPAC names follow proper nomenclature rules

2. **Synonym Issues**: Different valid IUPAC names for the same compound may not match:
   - `ethanol` vs `ethyl alcohol` → Different strings, won't match
   - `propan-2-ol` vs `isopropanol` → Different strings, won't match

3. **Stereochemistry**: Limited handling of stereochemical descriptors:
   - `(R)`, `(S)`, `(E)`, `(Z)` notations removed by normalization
   - May not distinguish all stereoisomers correctly

4. **Language Variations**: Systematic names in different languages will not match

5. **Normalization Trade-offs**:
   - **With normalization (default)**: More robust, may accept format variations
   - **Without normalization**: Strict, matches LLM4Chem exactly, may reject valid answers with format differences

## Future Enhancements

1. **Chemical Validation**: Use RDKit to validate molecular formulas
2. **IUPAC Synonym Matching**: Use chemical databases to match synonyms
3. **Confidence Scores**: Report confidence based on extraction strategy
4. **Batch Analysis**: Report distribution of matches to help tune settings

## Code References

### From LLM4Chem (`reference_code/LLM4Chem/`)

1. **`compute_metrics.py:98-101`** - Task routing
   - Line 99: MOLFORMULA uses `element_match`
   - Line 101: IUPAC uses `split_match`

2. **`metrics.py:270-321`** - `parse_molecule()`
   - Parses molecular formula into element counts
   - Handles parentheses, brackets, and charges
   - Stack-based recursive algorithm

3. **`metrics.py:324-351`** - `count_element_match()`
   - Compares molecular formulas by element dictionaries
   - Core logic for MOLFORMULA evaluation

4. **`metrics.py:256-267`** - `judge_string_split_match()`
   - Splits strings by separator and compares sorted tuples
   - Core logic for IUPAC evaluation

## Related Files

- **Task Definitions**: `smol/util.py` (defines `MOLFORMULA_TASKS` and `IUPAC_TASKS`)
- **Inference Results**: `smol/inference/*/name_conversion-{s2f,i2f,s2i}/inference_split*.jsonl`
- **Extraction Script**: `smol/postprocess_answer/postprocess_molformula_iupac_tasks.py`
- **Other Task Scripts**:
  - `postprocess_num_tasks.py` (numerical tasks)
  - `postprocess_bool_tasks.py` (boolean tasks)
  - `postprocess_smiles_tasks.py` (SMILES tasks)

---

**Last Updated**: 2026-01-21  
**Author**: LLM4Chem evaluation method integration  
**Reference Code**: LLM4Chem/SMol official evaluation code
