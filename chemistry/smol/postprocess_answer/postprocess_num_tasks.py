#!/usr/bin/env python3
"""
Extract and validate numerical answers for NUM_TASKS (property prediction tasks).

This script processes inference JSONL files to:
1. Extract numerical values from generated text using multiple fallback strategies
2. Validate extracted answers against ground truth with tolerance
"""

import json
import re
import argparse
from pathlib import Path
from typing import Dict, Any, Optional
import sys
from tqdm import tqdm
from datetime import datetime

# Add parent directory to path to import util
sys.path.append(str(Path(__file__).parent.parent))
from util import NUM_TASKS


def extract_number_from_text(text: str) -> Optional[str]:
    """
    Extract numerical value from text using multiple fallback strategies.
    
    Strategy priority:
    1. Last \boxed{} containing a number (or "infinite"/"undefined" → 2147483647)
    2. Last **num** pattern (or "infinite"/"undefined" → 2147483647)
    3. Last number in the entire text
    
    Special handling in Priority 1 & 2:
    - If content contains "infinite", "infinity", "inf", "undefined": return "2147483647"
    
    Args:
        text: Generated text to extract number from
        
    Returns:
        Extracted number as string, or None if no number found
    """
    if not text or not isinstance(text, str):
        return None
    
    # Strategy 1: Find last \boxed{number}
    # Pattern matches \boxed{...} and captures content
    boxed_pattern = r'\\boxed\{([^}]+)\}'
    boxed_matches = re.findall(boxed_pattern, text)
    
    if boxed_matches:
        # Check from last to first for a valid number or infinite/undefined
        for match in reversed(boxed_matches):
            # Extract number from the match (could have text around it)
            # This handles both numbers and special keywords like "infinite"
            number_in_boxed = extract_pure_number(match)
            if number_in_boxed:
                return number_in_boxed
    
    # Strategy 2: Find last **number** pattern
    # Pattern matches **...** and captures content
    bold_pattern = r'\*\*([^*]+)\*\*'
    bold_matches = re.findall(bold_pattern, text)
    
    if bold_matches:
        # Check from last to first for a valid number or infinite/undefined
        for match in reversed(bold_matches):
            # This handles both numbers and special keywords like "infinite"
            number_in_bold = extract_pure_number(match)
            if number_in_bold:
                return number_in_bold
    
    # Strategy 3: Find last number in entire text
    # Pattern matches integers and decimals (positive or negative)
    # Note: Priority 3 does NOT handle infinite/undefined keywords
    # Support forms like "0.91" and ".91" (and optional trailing dot like "1.")
    number_pattern = r'-?(?:\d+(?:\.\d*)?|\.\d+)'
    number_matches = re.findall(number_pattern, text)
    
    if number_matches:
        # Return the last number found
        return number_matches[-1]
    
    return None


def extract_pure_number(text: str) -> Optional[str]:
    """
    Extract a pure number from text, ignoring surrounding text.
    Special handling: "infinite" or "undefined" → "2147483647"
    
    Args:
        text: Text that might contain a number
        
    Returns:
        Number as string, or None if no valid number found
    """
    text_stripped = text.strip().lower()
    
    # Special case: check for "infinite" or "undefined" keywords
    # These indicate the model thinks the value is unbounded
    infinite_keywords = ['infinite', 'infinity', 'inf', 'undefined', 'undef']
    for keyword in infinite_keywords:
        if keyword in text_stripped:
            # Return max int32 value as a placeholder for infinite/undefined
            return "2147483647"
    
    # Pattern to match a number (integer or decimal, positive or negative)
    # Support forms like "0.91" and ".91" (and optional trailing dot like "1.")
    number_pattern = r'-?(?:\d+(?:\.\d*)?|\.\d+)'
    matches = re.findall(number_pattern, text.strip())
    
    if matches:
        # If there's only one match and it's a significant part of the text, return it
        if len(matches) == 1:
            return matches[0]
        # If multiple matches, prefer the longest one or the last one
        return matches[-1]
    
    return None


def is_answer_correct(extracted: str, ground_truth: str, tolerance: float = 1.0) -> bool:
    """
    Check if extracted answer is correct within tolerance.
    
    Args:
        extracted: Extracted numerical answer as string
        ground_truth: Ground truth value as string
        tolerance: Acceptable difference (default: ±1.0)
        
    Returns:
        True if |extracted - ground_truth| <= tolerance, False otherwise
    """
    try:
        extracted_val = float(extracted)
        gt_val = float(ground_truth)
        return abs(extracted_val - gt_val) <= tolerance
    except (ValueError, TypeError):
        return False


def process_jsonl_file(input_path: str, output_suffix: str = "_with_extracted_numbers", tolerance: float = 1.0) -> str:
    """
    Process a JSONL file to extract numbers and validate answers.
    
    Args:
        input_path: Path to input JSONL file
        output_suffix: Suffix to add to output filename
        tolerance: Acceptable difference for answer validation (default: 1.0)
        
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
            
            # Only process NUM_TASKS
            task = data.get('task', '')
            if task not in NUM_TASKS:
                # Write unchanged for non-NUM_TASKS
                f_out.write(json.dumps(data, ensure_ascii=False) + '\n')
                continue
            
            # Get fields
            extracted_core_answer = data.get('extracted_core_answer', {})
            generated = data.get('generated', {})
            ground_truth = data.get('ground_truth', '')
            
            # Process each generation
            for gen_key in extracted_core_answer.keys():
                # If extracted answer is empty, try to extract from generated text
                if not extracted_core_answer[gen_key] or extracted_core_answer[gen_key] == "":
                    gen_text = generated.get(gen_key, '')
                    extracted_value = extract_number_from_text(gen_text)
                    
                    if extracted_value:
                        extracted_core_answer[gen_key] = extracted_value
                        extracted_count += 1
                    else:
                        # Keep as empty string if nothing found
                        extracted_core_answer[gen_key] = ""
            
            # Add correctness field
            extracted_core_answer_correctness = {}
            for gen_key, extracted_value in extracted_core_answer.items():
                if extracted_value and extracted_value != "":
                    is_correct = is_answer_correct(extracted_value, ground_truth, tolerance)
                    extracted_core_answer_correctness[gen_key] = is_correct
                    if is_correct:
                        correct_count += 1
                else:
                    extracted_core_answer_correctness[gen_key] = False
            
            # Update data
            data['extracted_core_answer'] = extracted_core_answer
            data['extracted_core_answer_correctness'] = extracted_core_answer_correctness
            
            # Write to output
            f_out.write(json.dumps(data, ensure_ascii=False) + '\n')
        
        pbar.close()
    
    # Prepare statistics report
    report_lines = []
    report_lines.append("=" * 70)
    report_lines.append("NUM_TASKS Postprocessing Report")
    report_lines.append("=" * 70)
    report_lines.append(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append(f"Input file: {input_path}")
    report_lines.append(f"Output file: {output_path}")
    report_lines.append(f"Tolerance: ±{tolerance}")
    report_lines.append("")
    report_lines.append("Processing complete:")
    report_lines.append(f"  Total lines processed: {total_lines}")
    report_lines.append(f"  Numbers extracted: {extracted_count}")
    report_lines.append(f"  Correct answers (tolerance=±{tolerance}): {correct_count}")
    if total_lines > 0:
        accuracy = (correct_count / total_lines) * 100
        report_lines.append(f"  Accuracy: {accuracy:.2f}%")
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
        description="Extract and validate numerical answers for NUM_TASKS"
    )
    parser.add_argument(
        'input_file',
        type=str,
        help='Path to input JSONL file'
    )
    parser.add_argument(
        '--tolerance',
        type=float,
        default=1.0,
        help='Tolerance for answer validation (default: 1.0, meaning ±1.0 is acceptable)'
    )
    parser.add_argument(
        '--output-suffix',
        type=str,
        default=None,
        help='Suffix for output filename (default: _num_postprocessed_tolerance{tolerance})'
    )
    
    args = parser.parse_args()
    
    # Generate default suffix with tolerance value if not provided
    if args.output_suffix is None:
        args.output_suffix = f'_num_postprocessed_tolerance{args.tolerance}'
    
    try:
        output_path = process_jsonl_file(args.input_file, args.output_suffix, args.tolerance)
        print(f"\nSuccess! Processed file saved to:\n{output_path}")
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()
