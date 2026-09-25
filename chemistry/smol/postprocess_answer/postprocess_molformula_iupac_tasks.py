#!/usr/bin/env python3
"""
Extract and validate text answers for MOLFORMULA_TASKS and IUPAC_TASKS.

This script processes inference JSONL files to:
1. Extract molecular formula or IUPAC names from generated text using multiple fallback strategies
2. Validate extracted answers against ground truth using LLM4Chem evaluation methods:
   - MOLFORMULA: element_match (compare element counts, order-independent)
   - IUPAC: split_match (split by ';' and compare sorted)
"""

import json
import re
import argparse
from pathlib import Path
from typing import Dict, Any, Optional
import sys
from tqdm import tqdm
from datetime import datetime
from collections import defaultdict

# Add parent directory to path to import util
sys.path.append(str(Path(__file__).parent.parent))
from util import MOLFORMULA_TASKS, IUPAC_TASKS


# Combine both task types for processing
TEXT_TASKS = MOLFORMULA_TASKS + IUPAC_TASKS


def parse_molecule(molecular_formula: str) -> dict:
    """
    Parse molecular formula into element counts dictionary.
    
    This function is adapted from LLM4Chem reference code (metrics.py:270-321).
    
    Args:
        molecular_formula: Molecular formula string (e.g., "C5H8O2", "Ca(OH)2")
        
    Returns:
        Dictionary mapping element symbols to their counts
        
    Examples:
        - "C5H8O2" -> {'C': 5, 'H': 8, 'O': 2}
        - "Ca(OH)2" -> {'Ca': 1, 'O': 2, 'H': 2}
        - "H2O" -> {'H': 2, 'O': 1}
        
    Raises:
        ValueError: If molecular formula is not valid
    """
    valid = re.match(r'([A-Za-z]\d*)+([\+\-]\d*)*$', molecular_formula)
    if valid is None:
        raise ValueError("Molecular formula \"%s\" is not valid." % molecular_formula)

    stack = [defaultdict(int)]

    def _parse_formula(formula, _stack):
        # Set remainder equal to 'None'
        r = None

        # Regular expression matching for each of the three cases:
        atom = re.match(r'([A-Z][a-z]?)(\d+)?', formula)
        opening = re.match(r'[\(\[\{]', formula)
        closing = re.match(r'[\)\]\}](\d+)?', formula)

        # If atom is identified:
        if atom:
            r = formula[len(atom.group()):]
            _stack[-1][atom.group(1)] += int(atom.group(2) or 1)

        # If opening brackets encountered:
        elif opening:
            r = formula[len(opening.group()):]  # remainder after opening brackets
            _stack.append(defaultdict(int)) 

        # If closing brackets encountered:
        elif closing:
            r = formula[len(closing.group()):]  # remainder after closing brackets
            for (k, v) in _stack.pop().items():
                _stack[-1][k] += v * int(closing.group(1) or 1)

        # If anything remains, process remainders recursively:
        if r:
            _parse_formula(r, _stack)

        return dict(_stack[0])
    
    result = _parse_formula(molecular_formula, stack)

    # Handle charges (e.g., "NH4+", "SO4-2")
    charge = re.search(r'[\+\-]\d*', molecular_formula)
    if charge is not None:
        charge_str = charge.group()
        charge_type = charge_str[0]
        if len(charge_str) == 1:
            charge_num = 1
        else:
            charge_num = int(charge_str[1:])
        result[charge_type] = charge_num

    return result


def normalize_molformula(formula: str) -> str:
    """
    Normalize molecular formula for more robust comparison.
    
    Normalizations applied:
    1. Remove all whitespace (molecular formulas shouldn't have spaces)
    2. Remove special characters except valid molecular formula characters
    3. Strip leading/trailing whitespace
    
    Args:
        formula: Molecular formula to normalize
        
    Returns:
        Normalized molecular formula
        
    Examples:
        - "C5 H8 O2" -> "C5H8O2"
        - "C5H8O2." -> "C5H8O2"
        - "Ca(OH)2" -> "Ca(OH)2" (parentheses preserved)
    """
    if not formula:
        return ""
    
    # Remove leading/trailing whitespace first
    formula = formula.strip()
    
    # Keep only valid molecular formula characters:
    # - Uppercase and lowercase letters (A-Z, a-z)
    # - Numbers (0-9)
    # - Parentheses, brackets, braces: ( ) [ ] { }
    # - Plus and minus for charges: + -
    allowed_chars = set('ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789()[]{}+-')
    formula = ''.join(c for c in formula if c in allowed_chars)
    
    return formula


def is_molformula_correct(extracted: str, ground_truth: str, normalize: bool = True) -> bool:
    """
    Check if extracted molecular formula matches ground truth using element_match.
    
    This follows the LLM4Chem evaluation method (metrics.py:324-351, count_element_match).
    Compares element counts, so order doesn't matter: "C5H8O2" == "H8C5O2"
    
    With normalize=True (recommended), removes whitespace before comparison.
    
    Args:
        extracted: Extracted molecular formula
        ground_truth: Ground truth molecular formula
        normalize: Whether to normalize before comparison (default: True)
        
    Returns:
        True if element counts match, False otherwise
        
    Examples:
        - "C5H8O2" vs "H8C5O2" -> True (same elements, different order)
        - "C5 H8 O2" vs "C5H8O2" -> True (with normalize=True, whitespace removed)
        - "C5H8O2" vs "C5H8O2" -> True (exact match)
        - "C5H8O2" vs "C5H8O" -> False (different element counts)
    """
    if not extracted or extracted == "":
        return False
    
    try:
        # Optionally normalize both formulas
        if normalize:
            extracted = normalize_molformula(extracted)
            ground_truth = normalize_molformula(ground_truth)
        
        # Parse both formulas into element count dictionaries
        extracted_elements = parse_molecule(extracted)
        ground_truth_elements = parse_molecule(ground_truth)
        
        # Compare element dictionaries
        return extracted_elements == ground_truth_elements
        
    except Exception:
        # If parsing fails, consider it incorrect
        return False


def normalize_iupac_name(name: str, separator: str = ';') -> str:
    """
    Normalize IUPAC name for more robust comparison.
    
    Normalizations applied:
    1. Convert to lowercase (handles case differences)
    2. Remove leading/trailing whitespace
    3. Normalize internal whitespace to single space
    4. Remove special characters and punctuation (except separator and hyphens/numbers)
    
    Args:
        name: IUPAC name to normalize
        separator: Separator to preserve (default: ';')
        
    Returns:
        Normalized IUPAC name
        
    Examples:
        - "Ethanol." -> "ethanol"
        - "2-propanol, methanol" -> "2propanol methanol" (comma removed)
        - "ethanol; methanol" -> "ethanol; methanol" (separator preserved)
    """
    if not name:
        return ""
    
    # Convert to lowercase
    name = name.lower()
    
    # Remove leading/trailing whitespace
    name = name.strip()
    
    # Remove special characters and punctuation, but keep:
    # - Letters (a-z)
    # - Numbers (0-9)
    # - Hyphens (-) - common in IUPAC names like "2-propanol"
    # - Whitespace (will be normalized later)
    # - The separator character
    allowed_chars = set('abcdefghijklmnopqrstuvwxyz0123456789-, ' + separator)
    name = ''.join(c for c in name if c in allowed_chars)
    
    # Normalize internal whitespace to single space
    name = ' '.join(name.split())
    
    return name


def is_iupac_correct(extracted: str, ground_truth: str, separator: str = ';', 
                     normalize: bool = True) -> bool:
    """
    Check if extracted IUPAC name matches ground truth using split_match.
    
    This follows the LLM4Chem evaluation method (metrics.py:256-267, judge_string_split_match).
    Splits by separator and compares as sorted tuples, so order doesn't matter.
    
    With normalize=True (recommended), applies normalization before comparison:
    - Convert to lowercase
    - Remove extra whitespace
    - Remove special characters/punctuation (except separator and hyphens)
    This makes the comparison more robust while maintaining chemical accuracy.
    
    Args:
        extracted: Extracted IUPAC name(s)
        ground_truth: Ground truth IUPAC name(s)
        separator: Separator for multiple names (default: ';')
        normalize: Whether to normalize before comparison (default: True)
        
    Returns:
        True if sorted split parts match, False otherwise
        
    Examples:
        - "ethanol;methanol" vs "methanol;ethanol" -> True (same, different order)
        - "Ethanol." vs "ethanol" -> True (punctuation removed with normalize=True)
        - "2-propanol" vs "2-Propanol" -> True (case-insensitive with normalize=True)
        - "ethanol" vs "methanol" -> False (different)
    """
    if not extracted or extracted == "":
        return False
    
    try:
        # Optionally normalize both strings (pass separator to preserve it)
        if normalize:
            extracted = normalize_iupac_name(extracted, separator)
            ground_truth = normalize_iupac_name(ground_truth, separator)
        
        # Split and sort both strings
        extracted_parts = tuple(sorted(extracted.split(separator)))
        ground_truth_parts = tuple(sorted(ground_truth.split(separator)))
        
        # Compare sorted tuples
        return extracted_parts == ground_truth_parts
        
    except Exception:
        # If any error occurs, consider it incorrect
        return False


def extract_from_xml_tags(text: str, task: str) -> Optional[str]:
    """
    Extract content from <MOLFORMULA></MOLFORMULA> or <IUPAC></IUPAC> tags.
    
    Args:
        text: Generated text to search
        task: Task name to determine which tag to look for
        
    Returns:
        Extracted string (last tag, stripped), or None if not found
    """
    # Determine which tag to look for based on task
    if task in MOLFORMULA_TASKS:
        tag_name = 'MOLFORMULA'
    elif task in IUPAC_TASKS:
        tag_name = 'IUPAC'
    else:
        return None
    
    pattern = f'<{tag_name}>(.*?)</{tag_name}>'
    matches = re.findall(pattern, text, re.IGNORECASE | re.DOTALL)
    
    if matches:
        # Take the last match and strip whitespace
        return matches[-1].strip()
    
    return None


def extract_from_boxed(text: str) -> Optional[str]:
    """
    Extract content from \boxed{} pattern.
    
    Args:
        text: Generated text to search
        
    Returns:
        Extracted string (last boxed content, stripped), or None if not found
    """
    pattern = r'\\boxed\{([^}]+)\}'
    matches = re.findall(pattern, text)
    
    if matches:
        # Take the last match and strip whitespace
        return matches[-1].strip()
    
    return None


def extract_from_bold_markers(text: str) -> Optional[str]:
    """
    Extract content from **text** (markdown bold).
    
    Args:
        text: Generated text to search
        
    Returns:
        Extracted string (last bold content, stripped), or None if not found
    """
    pattern = r'\*\*([^*]+)\*\*'
    matches = re.findall(pattern, text)
    
    if matches:
        # Take the last match and strip whitespace
        return matches[-1].strip()
    
    return None


def extract_text_from_generation(text: str, task: str) -> Optional[str]:
    """
    Extract molecular formula or IUPAC name from text using multiple fallback strategies.
    
    Strategy priority:
    1. <MOLFORMULA></MOLFORMULA> or <IUPAC></IUPAC> tags (last one)
    2. \boxed{} pattern (last one)
    3. **text** pattern (last one)
    
    Args:
        text: Generated text to extract from
        task: Task name for tag selection
        
    Returns:
        Extracted string or None if not found
    """
    if not text or not isinstance(text, str):
        return None
    
    # Strategy 1: XML tags (MOLFORMULA or IUPAC)
    result = extract_from_xml_tags(text, task)
    if result:
        return result
    
    # Strategy 2: \boxed{} pattern
    result = extract_from_boxed(text)
    if result:
        return result
    
    # Strategy 3: **text** pattern
    result = extract_from_bold_markers(text)
    if result:
        return result
    
    return None


def is_text_correct(extracted: str, ground_truth: str, task: str, normalize: bool = True) -> bool:
    """
    Check if extracted text matches ground truth using task-appropriate method.
    
    This follows the LLM4Chem evaluation methods:
    - MOLFORMULA tasks: element_match (compare element counts)
    - IUPAC tasks: split_match (split by ';' and compare sorted)
    
    Args:
        extracted: Extracted text string
        ground_truth: Ground truth text string
        task: Task name to determine evaluation method
        normalize: Whether to apply normalization before comparison (default: True)
        
    Returns:
        True if correct according to task-specific method, False otherwise
    """
    if not extracted or extracted == "":
        return False
    
    # Route to appropriate validation method based on task
    if task in MOLFORMULA_TASKS:
        return is_molformula_correct(extracted, ground_truth, normalize=normalize)
    elif task in IUPAC_TASKS:
        return is_iupac_correct(extracted, ground_truth, normalize=normalize)
    else:
        # Unknown task, return False
        return False


def process_jsonl_file(input_path: str, output_suffix: str = "_text_postprocessed", 
                       normalize: bool = True) -> str:
    """
    Process a JSONL file to extract text answers and validate them.
    
    Uses LLM4Chem evaluation methods:
    - MOLFORMULA tasks: element_match (compare element counts)
    - IUPAC tasks: split_match (split by ';' and compare sorted)
    
    Args:
        input_path: Path to input JSONL file
        output_suffix: Suffix to add to output filename
        normalize: Whether to apply normalization before comparison (default: True)
        
    Returns:
        Path to output file
    """
    input_file = Path(input_path)
    
    if not input_file.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")
    
    # Create output path
    output_path = input_file.parent / f"{input_file.stem}{output_suffix}{input_file.suffix}"
    
    # Statistics
    total_lines = 0
    extracted_count = 0
    correct_count = 0
    strategy_counts = {
        'xml_tags': 0,
        'boxed': 0,
        'bold_markers': 0,
        'not_extracted': 0
    }
    task_counts = {}
    
    # Count total lines for progress bar
    print(f"Counting lines in {input_file.name}...")
    with open(input_file, 'r', encoding='utf-8') as f:
        total_file_lines = sum(1 for line in f if line.strip())
    
    with open(input_file, 'r', encoding='utf-8') as f_in, \
         open(output_path, 'w', encoding='utf-8') as f_out:
        
        # Create progress bar
        pbar = tqdm(total=total_file_lines, desc="Processing", unit="lines")
        
        for line in f_in:
            line = line.strip()
            if not line:
                continue
            
            total_lines += 1
            pbar.update(1)
            data = json.loads(line)
            
            # Only process TEXT_TASKS (MOLFORMULA_TASKS + IUPAC_TASKS)
            task = data.get('task', '')
            if task not in TEXT_TASKS:
                # Write unchanged for non-TEXT_TASKS
                f_out.write(json.dumps(data, ensure_ascii=False) + '\n')
                continue
            
            # Track task counts
            if task not in task_counts:
                task_counts[task] = {'extracted': 0, 'correct': 0}
            
            # Get fields
            extracted_core_answer = data.get('extracted_core_answer', {})
            generated = data.get('generated', {})
            ground_truth = data.get('ground_truth', '')
            
            # Process each generation
            for gen_key in extracted_core_answer.keys():
                # If extracted answer is empty, try to extract from generated text
                if not extracted_core_answer[gen_key] or extracted_core_answer[gen_key] == "":
                    gen_text = generated.get(gen_key, '')
                    
                    # Try each strategy and track which one worked
                    extracted_text = None
                    strategy_used = 'not_extracted'
                    
                    # Try strategies in order
                    extracted_text = extract_from_xml_tags(gen_text, task)
                    if extracted_text:
                        strategy_used = 'xml_tags'
                    else:
                        extracted_text = extract_from_boxed(gen_text)
                        if extracted_text:
                            strategy_used = 'boxed'
                        else:
                            extracted_text = extract_from_bold_markers(gen_text)
                            if extracted_text:
                                strategy_used = 'bold_markers'
                    
                    if extracted_text:
                        extracted_core_answer[gen_key] = extracted_text
                        extracted_count += 1
                        task_counts[task]['extracted'] += 1
                        strategy_counts[strategy_used] += 1
                    else:
                        # Keep as empty string if nothing found
                        extracted_core_answer[gen_key] = ""
                        strategy_counts['not_extracted'] += 1
            
            # Add correctness field using task-specific validation
            extracted_core_answer_correctness = {}
            for gen_key, extracted_value in extracted_core_answer.items():
                is_correct = is_text_correct(extracted_value, ground_truth, task, normalize=normalize)
                extracted_core_answer_correctness[gen_key] = is_correct
                if is_correct:
                    correct_count += 1
                    task_counts[task]['correct'] += 1
            
            # Update data
            data['extracted_core_answer'] = extracted_core_answer
            data['extracted_core_answer_correctness'] = extracted_core_answer_correctness
            
            # Write to output
            f_out.write(json.dumps(data, ensure_ascii=False) + '\n')
        
        pbar.close()
    
    # Prepare statistics report
    report_lines = []
    report_lines.append("=" * 70)
    report_lines.append("TEXT_TASKS (MOLFORMULA + IUPAC) Postprocessing Report")
    report_lines.append("=" * 70)
    report_lines.append(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append(f"Input file: {input_path}")
    report_lines.append(f"Output file: {output_path}")
    report_lines.append(f"Evaluation method: LLM4Chem (element_match for MOLFORMULA, split_match for IUPAC)")
    report_lines.append(f"Normalization: {'Enabled' if normalize else 'Disabled'}")
    report_lines.append("")
    report_lines.append("Processing complete:")
    report_lines.append(f"  Total lines processed: {total_lines}")
    report_lines.append(f"  Text answers extracted: {extracted_count}")
    report_lines.append(f"  Correct answers: {correct_count}")
    if total_lines > 0:
        accuracy = (correct_count / total_lines) * 100
        report_lines.append(f"  Accuracy: {accuracy:.2f}%")
    report_lines.append("")
    report_lines.append("Extraction strategies used:")
    for strategy, count in strategy_counts.items():
        if count > 0:
            report_lines.append(f"    {strategy}: {count}")
    
    if task_counts:
        report_lines.append("")
        report_lines.append("Per-task statistics:")
        for task, counts in sorted(task_counts.items()):
            report_lines.append(f"  {task}:")
            report_lines.append(f"    Extracted: {counts['extracted']}")
            report_lines.append(f"    Correct: {counts['correct']}")
            if counts['extracted'] > 0:
                task_acc = (counts['correct'] / counts['extracted']) * 100
                report_lines.append(f"    Accuracy: {task_acc:.2f}%")
    
    report_lines.append("=" * 70)
    
    # Print to console
    print("\n" + "\n".join(report_lines))
    
    # Save to log file
    log_path = output_path.parent / f"{output_path.stem}_log.txt"
    with open(log_path, 'w', encoding='utf-8') as log_file:
        log_file.write("\n".join(report_lines) + "\n")
    
    print(f"\nLog saved to: {log_path}")
    
    return str(output_path)


def main():
    parser = argparse.ArgumentParser(
        description="Extract and validate text answers for MOLFORMULA_TASKS and IUPAC_TASKS using LLM4Chem methods"
    )
    parser.add_argument(
        'input_file',
        type=str,
        help='Path to input JSONL file'
    )
    parser.add_argument(
        '--output-suffix',
        type=str,
        default='_text_postprocessed',
        help='Suffix for output filename (default: _text_postprocessed)'
    )
    parser.add_argument(
        '--normalize',
        action='store_true',
        default=True,
        help='Enable normalization before comparison (default: True)'
    )
    parser.add_argument(
        '--no-normalize',
        dest='normalize',
        action='store_false',
        help='Disable normalization (strict comparison)'
    )
    
    args = parser.parse_args()
    
    try:
        output_path = process_jsonl_file(args.input_file, args.output_suffix, args.normalize)
        print(f"\nSuccess! Processed file saved to:\n{output_path}")
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
