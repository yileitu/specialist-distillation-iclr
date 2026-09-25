# BOOL_TASKS Boolean Extraction and Validation Strategy

## Overview

This document describes the strategy for extracting and validating boolean (Yes/No) answers from model-generated responses for BOOL_TASKS (binary classification tasks).

## BOOL_TASKS Definition

According to `smol/util.py`, BOOL_TASKS include:
- `property_prediction-sider` (Side effects prediction)
- `property_prediction-hiv` (HIV inhibition prediction)
- `property_prediction-clintox` (Clinical toxicity prediction)
- `property_prediction-bbbp` (Blood-brain barrier permeability prediction)

These tasks require extracting binary boolean values (Yes/No, True/False, etc.) from model outputs.

## Processing Pipeline

### 1. Boolean Extraction Strategy

When the `extracted_core_answer` field contains an empty string for any generation, the system attempts to extract boolean values from the corresponding `generated` text using a **six-tier fallback strategy**:

#### Priority 1: `<BOOLEAN></BOOLEAN>` Tags
```xml
<BOOLEAN>Yes</BOOLEAN>
```
- **Target**: Content within the last `<BOOLEAN></BOOLEAN>` tag
- **Examples**: 
  - `<BOOLEAN>Yes</BOOLEAN>` → extracts `Yes`
  - `<BOOLEAN>No</BOOLEAN>` → extracts `No`
  - `<BOOLEAN>True</BOOLEAN>` → extracts `Yes`
  - `<BOOLEAN>False</BOOLEAN>` → extracts `No`
- **Rationale**: Explicit boolean tags provide the clearest signal of the model's answer
- **Parsing**: Converts various boolean representations (yes/no, true/false, correct/incorrect, positive/negative) to Yes/No

#### Priority 2: Markdown Bold `**bool**` Pattern
```markdown
**Yes**
```
- **Target**: Content within the last `**...**` (markdown bold) that represents a boolean
- **Examples**:
  - `The answer is **Yes**` → extracts `Yes`
  - `**No**, this molecule does not...` → extracts `No`
  - `**True**` → extracts `Yes`
  - `**False**` → extracts `No`
- **Rationale**: Models often emphasize boolean answers using bold formatting
- **Parsing**: Only considers bold text that can be parsed as boolean; ignores non-boolean bold text

#### Priority 3: After Answer Markers
```
Answer: Yes
```
- **Target**: Text immediately after keywords like "Answer:", "Final answer:", "Conclusion:", "Result:", etc.
- **Examples**:
  - `Answer: Yes` → keyword counting in "Yes"
  - `Final answer: The molecule is not inhibitory` → keyword counting finds "not" → `No`
  - `Therefore: This compound shows positive activity` → keyword counting finds "positive" → `Yes`
- **Method**: **Keyword Counting** (see section below)
- **Rationale**: Answer markers indicate the model is providing its final conclusion

#### Priority 4: Last Paragraph Analysis
- **Target**: The last non-empty paragraph (text segments separated by `\n\n`)
- **Examples**:
  - Last paragraph: "In conclusion, this molecule is inhibitory." → keyword counting → `Yes`
  - Last paragraph: "Therefore, the compound is not toxic." → keyword counting → `No`
- **Method**: **Keyword Counting** (see section below)
- **Rationale**: Conclusions are typically placed at the end of generated text

#### Priority 5: After `</think>` Tag
```
</think> The answer is Yes.
```
- **Target**: Text after the closing `</think>` tag
- **Examples**:
  - `</think>\nYes, the molecule is active.` → keyword counting → `Yes`
  - `</think> No` → keyword counting → `No`
- **Method**: **Keyword Counting** (see section below)
- **Rationale**: Text after thinking tags often contains the final answer
- **Note**: Only processes text after `</think>`, not within `<think>...</think>`

#### Priority 6: Full Text Analysis
- **Target**: The entire generated text
- **Method**: **Keyword Counting** (see section below)
- **Rationale**: Last resort when no structured formatting is present
- **Risk**: May be less accurate due to considering all mentions, including hypothetical or negated statements

### 2. Keyword Counting Method

For strategies 3-6, boolean extraction uses keyword counting:

**YES Keywords** (indicating True/Positive):
- `yes`, `true`, `correct`, `positive`, `active`, `inhibitory`
- `toxic`, `permeable`, `soluble`, `effective`

**NO Keywords** (indicating False/Negative):
- `no`, `false`, `incorrect`, `negative`, `inactive`, `non-inhibitory`
- `non-toxic`, `not permeable`, `insoluble`, `ineffective`, `not`

**Counting Logic**:
1. Count occurrences of YES keywords → `yes_count`
2. Count occurrences of NO keywords → `no_count`
3. Decision:
   - If `yes_count > no_count` → Extract `Yes` (True)
   - If `no_count > yes_count` → Extract `No` (False)
   - If `yes_count == no_count` → **Inconclusive**, proceed to next strategy

**Example**:
```
Text: "This molecule shows positive inhibitory activity. It is active against HIV."
YES keywords found: "positive" (1), "inhibitory" (1), "active" (1) = 3
NO keywords found: none = 0
Result: yes_count (3) > no_count (0) → Extract "Yes"
```

### 3. Fallback Behavior

If all six strategies fail to extract a conclusive boolean:
- The `extracted_core_answer` remains as an empty string `""`
- The corresponding `extracted_core_answer_correctness` is set to `False`

## Answer Validation

### Correctness Criterion

An extracted answer is considered correct if:

```
extracted_answer.lower() == ground_truth.lower()
```

Where both `extracted_answer` and `ground_truth` are strings: `"Yes"` or `"No"`

### Examples

| Ground Truth | Extracted | Correct? |
|--------------|-----------|----------|
| Yes          | Yes       | ✓ True   |
| Yes          | No        | ✗ False  |
| No           | No        | ✓ True   |
| No           | Yes       | ✗ False  |
| Yes          | ""        | ✗ False  |

## Output Format

### Added Fields

The script adds the `extracted_core_answer_correctness` field to each data point:

```json
{
  "task": "property_prediction-hiv",
  "ground_truth": "No",
  "extracted_core_answer": {
    "generation1": "No",
    "generation2": "Yes",
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

Processed files are saved with `_bool_postprocessed` suffix:

```
Original: inference_split1000.jsonl
Output:   inference_split1000_bool_postprocessed.jsonl
```

## Usage

### Command Line

```bash
python smol/postprocess_answer/postprocess_bool_tasks.py \
  /path/to/inference_split1000.jsonl \
  [--output-suffix SUFFIX]
```

### Parameters

- `input_file`: Path to input JSONL file (required)
- `--output-suffix`: Custom suffix for output filename (default: `_bool_postprocessed`)

### Examples

**Basic usage:**
```bash
python smol/postprocess_answer/postprocess_bool_tasks.py \
  smol/inference/50k_all_subtasks/Intern-S1-mini_smol_instruct_all_no_cot_b4_t16_epoch1/\
presence0.0_freq0.0_repetition1.0_loop5-200_rep3/property_prediction-hiv/\
inference_split1000.jsonl
```
Output: `inference_split1000_bool_postprocessed.jsonl`

**Custom suffix:**
```bash
python smol/postprocess_answer/postprocess_bool_tasks.py \
  inference_split1000.jsonl \
  --output-suffix _validated
```
Output: `inference_split1000_validated.jsonl`

**Process all BOOL_TASKS:**
```bash
for task in property_prediction-{sider,hiv,clintox,bbbp}; do
  python smol/postprocess_answer/postprocess_bool_tasks.py \
    smol/inference/*//${task}/inference_split*.jsonl
done
```

## Implementation Details

### Script: `postprocess_bool_tasks.py`

**Key Functions**:

1. `extract_boolean_from_text(text: str) -> Optional[bool]`
   - Implements the six-tier extraction strategy
   - Returns True/False or None

2. `extract_from_boolean_tags(text: str) -> Optional[bool]`
   - Priority 1: Extracts from `<BOOLEAN></BOOLEAN>` tags

3. `extract_from_bold_markers(text: str) -> Optional[bool]`
   - Priority 2: Extracts from `**bool**` markers

4. `extract_after_answer_marker(text: str) -> Optional[bool]`
   - Priority 3: Keyword counting after "Answer:" markers

5. `extract_from_last_paragraph(text: str) -> Optional[bool]`
   - Priority 4: Keyword counting in last paragraph

6. `extract_after_think_tag(text: str) -> Optional[bool]`
   - Priority 5: Keyword counting after `</think>`

7. `extract_from_full_text(text: str) -> Optional[bool]`
   - Priority 6: Keyword counting in entire text

8. `count_keywords(text: str) -> Optional[bool]`
   - Core keyword counting logic
   - Returns True/False/None based on yes/no keyword frequency

9. `parse_boolean_text(text: str) -> Optional[bool]`
   - Converts various boolean representations to True/False

10. `is_answer_correct(extracted: str, ground_truth: str) -> bool`
    - Validates extracted answer against ground truth

### Task Filtering

The script **only processes lines where** `task ∈ BOOL_TASKS`. Other tasks are written to output unchanged.

## Statistics Output

The script reports:
- Total lines processed
- Number of booleans successfully extracted
- Number of correct answers
- Breakdown by extraction strategy
- Output file location

Example:
```
Processing complete:
  Total lines processed: 888
  Booleans extracted: 2450
  Correct answers: 1987

Extraction strategies used:
    boolean_tags: 150
    bold_markers: 320
    answer_marker: 450
    last_paragraph: 680
    after_think: 420
    full_text: 430
    not_extracted: 214

  Output saved to: inference_split1000_bool_postprocessed.jsonl
```

## Edge Cases

### Empty Generated Text
- If `generated[generationN]` is empty or missing → `extracted_core_answer[generationN]` remains `""`

### Ambiguous Boolean Text
- `<BOOLEAN>maybe</BOOLEAN>` → Falls back to Priority 2-6
- `**possibly**` → Falls back to Priority 3-6

### Equal Keyword Counts
- If `yes_count == no_count` in Priority 3 → Proceeds to Priority 4
- If still equal in Priority 4 → Proceeds to Priority 5
- Continues until a strategy returns a conclusive answer

### Negation Handling
- The keyword "not" is counted as a NO keyword
- Examples:
  - "not inhibitory" → counts both "not" and "inhibitory" (may cancel out)
  - "not toxic" → counts both "not" (NO) and "toxic" (YES), may be inconclusive
- **Limitation**: Simple keyword counting may not handle complex negation correctly

### Multiple Boolean Tags
- `<BOOLEAN>No</BOOLEAN> ... <BOOLEAN>Yes</BOOLEAN>` → Takes last one (`Yes`)

### Mixed Format
- `<BOOLEAN>The answer is Yes</BOOLEAN>` → Parsed via keyword counting → `Yes`

## Known Limitations

1. **Negation Complexity**: Simple keyword counting may mishandle complex negations like "not ineffective" or "does not lack activity"

2. **Context Sensitivity**: Keyword counting doesn't understand context. Hypothetical statements ("If it were active...") may be counted incorrectly

3. **Task-Specific Keywords**: Some keywords (e.g., "toxic") are YES for toxicity tasks but might be NO for other contexts

4. **Ambiguous Text**: When YES and NO keywords are equal, the strategy may fail to extract an answer

## Future Enhancements

1. **Improved Negation Handling**: Use dependency parsing or negation scope detection

2. **Context-Aware Extraction**: Distinguish between hypothetical and factual statements

3. **Task-Specific Keyword Lists**: Customize keywords based on specific BOOL_TASK

4. **Confidence Scoring**: Report confidence based on keyword margin (yes_count - no_count)

5. **Regex Patterns for Common Phrases**: Detect phrases like "The answer is yes" more reliably

## Related Files

- **Task Definitions**: `smol/util.py` (defines `BOOL_TASKS`)
- **Inference Results**: `smol/inference/*/property_prediction-{sider,hiv,clintox,bbbp}/inference_split*.jsonl`
- **Extraction Script**: `smol/postprocess_answer/postprocess_bool_tasks.py`
- **NUM_TASKS Strategy**: `smol/postprocess_answer/strategies/NUM_TASKS_EXTRACTION_STRATEGY.md`

---

**Last Updated**: 2026-01-16  
**Author**: Automated extraction pipeline for BOOL_TASKS evaluation
