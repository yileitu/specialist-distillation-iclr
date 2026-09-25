# SMILES_TASKS Extraction and Validation Strategy

## Overview

This document describes the strategy for extracting and validating SMILES (Simplified Molecular-Input Line-Entry System) strings from model-generated responses for SMILES_TASKS (molecule structure tasks).

## SMILES_TASKS Definition

According to `smol/util.py`, SMILES_TASKS include:
- `retrosynthesis` (Predict reactants from products)
- `name_conversion-i2s` (IUPAC name to SMILES conversion)
- `molecule_generation` (Generate molecule from description)
- `forward_synthesis` (Predict products from reactants)

These tasks require extracting valid SMILES strings representing molecular structures from model outputs.

## Processing Pipeline

### 1. SMILES Extraction Strategy

When the `extracted_core_answer` field contains an empty string for any generation, the system attempts to extract SMILES strings from the corresponding `generated` text using a **three-tier fallback strategy**:

#### Priority 1: `<SMILES></SMILES>` Tags
```xml
<SMILES>CCO</SMILES>
```
- **Target**: Content within the last `<SMILES></SMILES>` tag
- **Examples**:
  - `<SMILES>CCO</SMILES>` → extracts `CCO`
  - `<SMILES>c1ccccc1</SMILES>` → extracts `c1ccccc1`
  - Multiple tags: `<SMILES>C</SMILES>...<SMILES>CCO</SMILES>` → extracts `CCO` (last one)
- **Processing**: Strips leading/trailing whitespace
- **Rationale**: Explicit SMILES tags provide the clearest signal of the model's answer

#### Priority 2: `\boxed{}` Pattern
```latex
\boxed{CCO}
```
- **Target**: Content within the last `\boxed{}` LaTeX command
- **Examples**:
  - `\boxed{CCO}` → extracts `CCO`
  - `\boxed{C1=CC=CC=C1}` → extracts `C1=CC=CC=C1`
  - Multiple boxed: `\boxed{C}...\boxed{CCO}` → extracts `CCO` (last one)
- **Processing**: Strips leading/trailing whitespace
- **Rationale**: Models often use `\boxed{}` for final answers following mathematical convention

#### Priority 3: Task-Specific Extraction Functions

When explicit tags/patterns are absent, the script uses specialized extraction functions from `util.py`:

##### **For `retrosynthesis` task:**
Function: `extract_retrosynthesis(text)`
- **Strategy**:
  1. Check if entire text matches SMILES pattern
  2. Search for keywords: "Reactant SMILES:", "Reactants:", "predicted reactants", etc.
  3. Extract SMILES immediately after these keywords
  4. Use regex pattern to identify valid SMILES characters
- **Example**:
  ```
  "Reactant SMILES: CC(C)O.CCO"
  → extracts "CC(C)O.CCO"
  ```

##### **For `forward_synthesis` task:**
Function: `extract_forward_synthesis(text)`
- **Strategy**:
  1. Check if entire text matches SMILES pattern
  2. Search for "Predicted product SMILES:" keyword
  3. Look for SMILES after the last colon `:` in text
  4. Extract SMILES from beginning of text if it matches pattern
- **Example**:
  ```
  "Predicted product SMILES: CC(=O)O"
  → extracts "CC(=O)O"
  ```

##### **For `molecule_generation` task:**
Function: `extract_molecule_generation(text)`
- **Strategy**:
  1. Check if first line is a complete SMILES
  2. Search for keywords: "SMILES:", "molecule is:", "SMILES representation", etc.
  3. Extract SMILES after these keywords
  4. Use regex pattern to identify valid SMILES
- **Example**:
  ```
  "SMILES: c1ccccc1
  This represents benzene."
  → extracts "c1ccccc1"
  ```

##### **For `name_conversion-i2s` task:**
- Uses `extract_molecule_generation()` as fallback
- Handles IUPAC name to SMILES conversion

### 2. SMILES Pattern

Task-specific extraction functions use this regex pattern to identify SMILES:

```regex
(\[|]|\[[^\]]+]|Br?|Cl?|H|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|;|=|#|-|\+|\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])+
```

**Captures**:
- Atoms: `C`, `N`, `O`, `S`, `P`, `H`, `F`, `Cl`, `Br`, `I`
- Aromatic atoms: `c`, `n`, `o`, `s`, `p`
- Brackets: `[`, `]`, `[NH3+]`
- Bonds: `-`, `=`, `#`, `/`, `\`
- Rings: Digits `0-9`
- Branches: `(`, `)`
- Separators: `.` (for multiple molecules)
- Stereochemistry: `@`, `@@`
- And more

### 3. Fallback Behavior

If all three strategies fail to extract a valid SMILES:
- The `extracted_core_answer` remains as an empty string `""`
- The corresponding `extracted_core_answer_correctness` is set to `False`

## Answer Validation

### Correctness Criterion

An extracted SMILES is considered correct if:

```
InChI(canonicalize(preprocess(extracted_smiles))) == InChI(canonicalize(preprocess(ground_truth_smiles)))
```

**This follows the official LLM4Chem evaluation method from the SMol dataset.**

**Validation Process**:
1. **Preprocessing** (LLM4Chem method): Replace semicolons with dots
   - Both `;` and `.` can separate multiple molecules in SMILES
   - `.` is the standard separator, so we normalize `;` → `.`
   - Example: `CC(C)O;CCO` → `CC(C)O.CCO`
2. Both extracted and ground truth SMILES are canonicalized using `canonicalize_molecule_smiles()` from `util.py`
3. Canonicalization includes:
   - Removing atom mapping
   - Standardizing representation
   - Kekulization
   - Sorting molecule fragments (for multi-component SMILES like `A.B`)
   - Handling stereochemistry
4. Convert both canonical SMILES to InChI (International Chemical Identifier) using RDKit
5. Compare InChI identifiers for exact match → True if identical

### Why InChI Comparison?

**InChI (International Chemical Identifier)** is the gold standard for determining chemical equivalence:

1. **More robust than SMILES comparison**: Different canonical SMILES can represent the same molecule
2. **Standardized by IUPAC**: InChI is an internationally recognized chemical identifier
3. **Handles tautomers and resonance**: Better captures chemical equivalence
4. **Used by official SMol evaluation**: This is how the benchmark creators evaluate submissions

**Examples of why InChI is better**:
```
CCO vs OCC:
  - Direct SMILES: May differ depending on canonicalization algorithm
  - InChI: Always produces identical InChI for same molecule
  
c1ccccc1 vs C1=CC=CC=C1:
  - Direct SMILES: Different representations (aromatic vs Kekulé)
  - InChI: Identical InChI identifier
```

### Why Canonicalization First?

SMILES strings can represent the same molecule in multiple ways:

```
CCO = OCC = C(C)O  (all represent ethanol)
c1ccccc1 = C1=CC=CC=C1  (benzene, aromatic vs. Kekulé form)
```

Canonicalization converts all equivalent representations to a single standard form before InChI conversion, which:
- Cleans up malformed SMILES
- Removes atom mapping
- Standardizes representation
- Handles multi-molecule ordering (e.g., `A.B` vs `B.A`)

### Examples

| Ground Truth | Extracted | Preprocessed | Canonical | InChI Comparison | Correct? |
|--------------|-----------|--------------|-----------|------------------|----------|
| CCO          | CCO       | CCO / CCO    | CCO / CCO | Same             | ✓ True   |
| CCO          | OCC       | CCO / OCC    | CCO / CCO | Same             | ✓ True   |
| c1ccccc1     | C1=CC=CC=C1 | (no change) | c1ccccc1 / c1ccccc1 | Same    | ✓ True   |
| CCO.CC       | CC.CCO    | (no change)  | CC.CCO / CC.CCO | Same        | ✓ True   |
| CCO.CC       | CCO;CC    | CCO.CC / CCO.CC | CC.CCO / CC.CCO | Same     | ✓ True   |
| CC(C)O;CCO   | CCO.CC(C)O | Both→dots   | CC.CCO.CC(C)O / CC.CCO.CC(C)O | Same | ✓ True |
| CCO          | CC        | (no change)  | CCO / CC  | Different        | ✗ False  |
| CCO          | invalid   | (no change)  | CCO / None | Valid / None    | ✗ False  |
| CCO          | ""        | N/A          | N/A       | N/A              | ✗ False  |

**Notes**: 
- **Preprocessing**: Semicolons (`;`) are replaced with dots (`.`) before canonicalization
- **Multi-molecule SMILES**: The InChI comparison handles different orderings by comparing sorted sets
- **Semicolon normalization**: Both `A;B` and `A.B` are treated as equivalent after preprocessing

### Error Handling

- **Empty string**: Directly marked as `False`, no canonicalization attempted
- **Invalid SMILES**: If `canonicalize_molecule_smiles()` returns `None`, marked as `False`
- **Canonicalization exception**: Any error during canonicalization → `False`

## Output Format

### Added Fields

The script adds the `extracted_core_answer_correctness` field to each data point:

```json
{
  "task": "retrosynthesis",
  "ground_truth": "CC(C)O.CCO",
  "extracted_core_answer": {
    "generation1": "CC(C)O.CCO",
    "generation2": "CCO.CC(C)O",
    "generation3": ""
  },
  "extracted_core_answer_correctness": {
    "generation1": true,
    "generation2": true,
    "generation3": false
  }
}
```

**Note**: `generation2` is also correct because canonicalization sorts fragments: both `CC(C)O.CCO` and `CCO.CC(C)O` canonicalize to the same form.

### Output File Naming

Processed files are saved with `_smiles_postprocessed` suffix:

```
Original: inference_split1000.jsonl
Output:   inference_split1000_smiles_postprocessed.jsonl
```

## Usage

### Command Line

```bash
python smol/postprocess_answer/postprocess_smiles_tasks.py \
  /path/to/inference_split1000.jsonl \
  [--output-suffix SUFFIX]
```

### Parameters

- `input_file`: Path to input JSONL file (required)
- `--output-suffix`: Custom suffix for output filename (default: `_smiles_postprocessed`)

### Examples

**Basic usage:**
```bash
python smol/postprocess_answer/postprocess_smiles_tasks.py \
  smol/inference/50k_all_subtasks/Intern-S1-mini_smol_instruct_all_no_cot_b4_t16_epoch1/\
presence0.0_freq0.0_repetition1.0_loop5-200_rep3/retrosynthesis/\
inference_split1000.jsonl
```
Output: `inference_split1000_smiles_postprocessed.jsonl`

**Custom suffix:**
```bash
python smol/postprocess_answer/postprocess_smiles_tasks.py \
  inference_split1000.jsonl \
  --output-suffix _validated
```
Output: `inference_split1000_validated.jsonl`

**Process all SMILES_TASKS:**
```bash
for task in retrosynthesis forward_synthesis molecule_generation name_conversion-i2s; do
  python smol/postprocess_answer/postprocess_smiles_tasks.py \
    smol/inference/*/${task}/inference_split*.jsonl
done
```

## Implementation Details

### Script: `postprocess_smiles_tasks.py`

**Key Functions**:

1. `extract_smiles_from_text(text: str, task: str) -> Optional[str]`
   - Implements the three-tier extraction strategy
   - Returns SMILES string or None

2. `extract_from_smiles_tags(text: str) -> Optional[str]`
   - Priority 1: Extracts from `<SMILES></SMILES>` tags

3. `extract_from_boxed(text: str) -> Optional[str]`
   - Priority 2: Extracts from `\boxed{}` pattern

4. `extract_task_specific(text: str, task: str) -> Optional[str]`
   - Priority 3: Calls appropriate task-specific function
   - Dispatches to `extract_retrosynthesis`, `extract_forward_synthesis`, or `extract_molecule_generation`

5. `get_molecule_id(smiles: str, remove_duplicate: bool = True) -> Optional[tuple]`
   - **From LLM4Chem reference code**: Converts SMILES to InChI identifier
   - For multi-molecule SMILES, splits by '.', converts each to InChI, returns sorted tuple
   - Enables robust chemical equivalence comparison

6. `is_smiles_correct(extracted: str, ground_truth: str) -> bool`
   - **Uses LLM4Chem evaluation method**: InChI-based comparison
   - First canonicalizes both SMILES to clean them up
   - Converts to InChI identifiers using `get_molecule_id()`
   - Returns True only if InChI identifiers match exactly

7. `canonicalize_molecule_smiles(smiles: str, ...) -> Optional[str]`
   - From `util.py`: Canonicalizes SMILES using RDKit
   - Handles multi-component SMILES (e.g., `A.B.C`)
   - Sorts fragments for consistent ordering
   - Prepares SMILES for InChI conversion

### Task Filtering

The script **only processes lines where** `task ∈ SMILES_TASKS`. Other tasks are written to output unchanged.

## Statistics Output

The script reports:
- Total lines processed
- Number of SMILES successfully extracted
- Number of correct answers
- Canonicalization errors (if any)
- Breakdown by extraction strategy
- Output file location

Example:
```
Counting lines in inference_split1000.jsonl...
Processing: 100%|██████████| 888/888 [00:45<00:00, 19.5lines/s]

Processing complete:
  Total lines processed: 888
  SMILES extracted: 2450
  Correct answers: 1876
  Canonicalization errors: 23

Extraction strategies used:
    smiles_tags: 420
    boxed: 380
    task_specific: 1650
    not_extracted: 214

  Output saved to: inference_split1000_smiles_postprocessed.jsonl
```

## Edge Cases

### Empty Generated Text
- If `generated[generationN]` is empty or missing → `extracted_core_answer[generationN]` remains `""`

### Invalid SMILES
- If extracted string is not a valid SMILES → `canonicalize_molecule_smiles()` returns `None` → marked as incorrect

### Multiple Molecules (Dot-Separated)
- SMILES with `.` separator represent multiple molecules: `CC(C)O.CCO`
- Canonicalization **sorts** fragments: `A.B` and `B.A` → both become `A.B` (alphabetically)
- Therefore, `CC(C)O.CCO` and `CCO.CC(C)O` are considered equivalent

### Stereochemistry
- Canonicalization preserves stereochemistry: `C[C@H](O)C` ≠ `C[C@@H](O)C`
- Models must predict correct stereochemistry for a match

### Task-Specific Extraction Failures
- If task-specific function returns empty string `""` → tries next strategy or remains empty

### Whitespace in Extracted SMILES
- Leading/trailing whitespace is stripped from Priority 1 & 2
- Task-specific functions handle whitespace internally

## Known Limitations

1. **Canonicalization Dependency**: Requires RDKit library; errors can occur with:
   - Malformed SMILES
   - Very large molecules
   - Unusual chemical features

2. **Task-Specific Heuristics**: Priority 3 functions use keyword matching and may:
   - Miss SMILES in unusual formats
   - Extract incorrect substrings if text structure is unexpected

3. **No Partial Credit**: Only exact canonical match → True; close but not identical → False

4. **Performance**: Canonicalization can be slow for complex molecules

## Dependencies

**Required Python packages**:
- `rdkit`: For SMILES canonicalization and validation
- `rdchiral`: For chirality handling
- `tqdm`: For progress bar
- Standard library: `json`, `re`, `argparse`, `pathlib`, `typing`, `sys`

**Install RDKit**:
```bash
conda install -c conda-forge rdkit
# or
pip install rdkit
```

## Future Enhancements

1. **Parallel Processing**: Use multiprocessing for faster canonicalization

2. **Fuzzy Matching**: Consider SMILES "close enough" if they differ by minor features

3. **Stereochemistry Tolerance**: Option to ignore stereochemistry for certain tasks

4. **Improved Extraction**: Machine learning-based SMILES extraction instead of regex

5. **Validation Metrics**: Report Tanimoto similarity or other chemical similarity metrics

## Related Files

- **Task Definitions**: `smol/util.py` (defines `SMILES_TASKS`)
- **Canonicalization**: `smol/util.py::canonicalize_molecule_smiles()`
- **Task-Specific Extractors**: `smol/util.py` (`extract_retrosynthesis`, `extract_forward_synthesis`, `extract_molecule_generation`)
- **Inference Results**: `smol/inference/*/{ retrosynthesis,forward_synthesis,molecule_generation,name_conversion-i2s}/inference_split*.jsonl`
- **Extraction Script**: `smol/postprocess_answer/postprocess_smiles_tasks.py`

---

**Last Updated**: 2026-01-16  
**Author**: Automated extraction pipeline for SMILES_TASKS evaluation
