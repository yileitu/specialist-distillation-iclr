#!/usr/bin/env python3
"""
Extract and validate boolean answers for BOOL_TASKS (binary classification tasks).

This script processes inference JSONL files to:
1. Extract boolean values (Yes/No, True/False, etc.) from generated text using multiple fallback strategies
2. Validate extracted answers against ground truth
"""

import json
import re
import argparse
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
import sys
from tqdm import tqdm
from datetime import datetime

# Add parent directory to path to import util
sys.path.append(str(Path(__file__).parent.parent))
from util import BOOL_TASKS


# Keywords for YES/TRUE and NO/FALSE
YES_KEYWORDS = [
    'yes', 'true', 'correct', 'positive', 'active', 'inhibitory',
    'toxic', 'permeable', 'soluble', 'effective'
]

NO_KEYWORDS = [
    'no', 'false', 'incorrect', 'negative', 'inactive', 'non-inhibitory',
    'non-toxic', 'not permeable', 'insoluble', 'ineffective', 'not'
]

ANSWER_MARKERS = [
    'answer', 'final answer', 'conclusion', 'result', 
    'therefore', 'thus', 'in summary'
]


def extract_from_boolean_tags(text: str) -> Optional[bool]:
    """
    Extract boolean value from <BOOLEAN></BOOLEAN> tags.
    
    Args:
        text: Generated text to search
        
    Returns:
        True/False if found and parseable, None otherwise
    """
    pattern = r'<BOOLEAN>(.*?)</BOOLEAN>'
    matches = re.findall(pattern, text, re.IGNORECASE | re.DOTALL)
    
    if matches:
        # Take the last match
        last_match = matches[-1].strip().lower()
        return parse_boolean_text(last_match)
    
    return None


def extract_from_bold_markers(text: str) -> Optional[bool]:
    """
    Extract boolean value from **text** (markdown bold).
    Only considers bold text that contains explicit boolean keywords.
    
    Args:
        text: Generated text to search
        
    Returns:
        True/False if found and parseable, None otherwise
    """
    pattern = r'\*\*([^*]+)\*\*'
    matches = re.findall(pattern, text)
    
    if matches:
        # Take the last match that looks like a boolean
        for match in reversed(matches):
            match_lower = match.strip().lower()
            
            # Only consider if it's a clear boolean keyword
            clear_yes = ['yes', 'true', 'correct', 'positive', 'active']
            clear_no = ['no', 'false', 'incorrect', 'negative', 'inactive']
            
            if match_lower in clear_yes:
                return True
            elif match_lower in clear_no:
                return False
            # For other text, try parsing but be strict
            elif len(match_lower.split()) <= 2:  # Only short phrases
                result = parse_boolean_text(match.strip())
                if result is not None:
                    return result
    
    return None


def extract_after_answer_marker(text: str) -> Optional[bool]:
    """
    Extract boolean value after "Answer:" or similar markers.
    
    Args:
        text: Generated text to search
        
    Returns:
        True/False based on keyword counting, None if equal or not found
    """
    text_lower = text.lower()
    
    # Find the position of the last answer marker
    last_marker_pos = -1
    for marker in ANSWER_MARKERS:
        pos = text_lower.rfind(marker)
        if pos > last_marker_pos:
            last_marker_pos = pos
    
    if last_marker_pos >= 0:
        # Extract text after the marker
        after_marker = text[last_marker_pos:]
        return count_keywords(after_marker)
    
    return None


def extract_from_last_paragraph(text: str) -> Optional[bool]:
    """
    Extract boolean value from the last paragraph (split by \n\n).
    
    Args:
        text: Generated text to search
        
    Returns:
        True/False based on keyword counting, None if equal or not found
    """
    paragraphs = text.split('\n\n')
    
    if paragraphs:
        # Analyze the last non-empty paragraph
        for paragraph in reversed(paragraphs):
            paragraph = paragraph.strip()
            if paragraph:
                result = count_keywords(paragraph)
                if result is not None:
                    return result
    
    return None


def extract_after_think_tag(text: str) -> Optional[bool]:
    """
    Extract boolean value after </think> tag.
    
    Args:
        text: Generated text to search
        
    Returns:
        True/False based on keyword counting, None if equal or not found
    """
    think_close_pattern = r'</think>'
    match = re.search(think_close_pattern, text, re.IGNORECASE)
    
    if match:
        # Extract text after </think>
        after_think = text[match.end():]
        return count_keywords(after_think)
    
    return None


def extract_from_full_text(text: str) -> Optional[bool]:
    """
    Extract boolean value from the entire text using keyword counting.
    
    Args:
        text: Generated text to search
        
    Returns:
        True/False based on keyword counting, None if equal
    """
    return count_keywords(text)


def parse_boolean_text(text: str) -> Optional[bool]:
    """
    Parse a short text snippet to boolean.
    
    Args:
        text: Short text that should represent a boolean value
        
    Returns:
        True for yes-like values, False for no-like values, None if unclear
    """
    text_lower = text.lower().strip()
    
    # Direct matches
    if text_lower in ['yes', 'true', '1', 'correct', 'positive']:
        return True
    if text_lower in ['no', 'false', '0', 'incorrect', 'negative']:
        return False
    
    # Check for keywords
    return count_keywords(text)


def count_keywords(text: str) -> Optional[bool]:
    """
    Count YES and NO keywords in text and return majority.
    
    Args:
        text: Text to analyze
        
    Returns:
        True if yes_count > no_count, False if no_count > yes_count, None if equal
    """
    text_lower = text.lower()
    
    yes_count = 0
    no_count = 0
    
    # Count YES keywords
    for keyword in YES_KEYWORDS:
        yes_count += text_lower.count(keyword)
    
    # Count NO keywords
    for keyword in NO_KEYWORDS:
        no_count += text_lower.count(keyword)
    
    if yes_count > no_count:
        return True
    elif no_count > yes_count:
        return False
    else:
        return None  # Equal counts - inconclusive


def extract_boolean_from_text(text: str) -> Optional[bool]:
    """
    Extract boolean value from text using multiple fallback strategies.
    
    Strategy priority:
    1. <BOOLEAN></BOOLEAN> tags (last one)
    2. **bool** markers (last one with valid boolean)
    3. After "Answer:" or similar markers (keyword counting)
    4. Last paragraph (split by \\n\\n, keyword counting)
    5. After </think> tag (keyword counting)
    6. Full text analysis (keyword counting)
    
    Args:
        text: Generated text to extract boolean from
        
    Returns:
        True/False if extracted, None if no clear answer
    """
    if not text or not isinstance(text, str):
        return None
    
    # Strategy 1: <BOOLEAN></BOOLEAN> tags
    result = extract_from_boolean_tags(text)
    if result is not None:
        return result
    
    # Strategy 2: **bool** markers
    result = extract_from_bold_markers(text)
    if result is not None:
        return result
    
    # Strategy 3: After "Answer:" markers
    result = extract_after_answer_marker(text)
    if result is not None:
        return result
    
    # Strategy 4: Last paragraph
    result = extract_from_last_paragraph(text)
    if result is not None:
        return result
    
    # Strategy 5: After </think> tag
    result = extract_after_think_tag(text)
    if result is not None:
        return result
    
    # Strategy 6: Full text analysis
    result = extract_from_full_text(text)
    if result is not None:
        return result
    
    return None


def boolean_to_string(value: Optional[bool]) -> str:
    """
    Convert boolean to "Yes"/"No" string format.
    
    Args:
        value: Boolean value or None
        
    Returns:
        "Yes", "No", or "" (empty string)
    """
    if value is True:
        return "Yes"
    elif value is False:
        return "No"
    else:
        return ""


def is_answer_correct(extracted: str, ground_truth: str) -> bool:
    """
    Check if extracted answer matches ground truth.
    
    Args:
        extracted: Extracted answer ("Yes" or "No")
        ground_truth: Ground truth value ("Yes" or "No")
        
    Returns:
        True if match, False otherwise
    """
    if not extracted or extracted == "":
        return False
    
    return extracted.lower() == ground_truth.lower()


def process_jsonl_file(input_path: str, output_suffix: str = "_bool_postprocessed") -> str:
    """
    Process a JSONL file to extract boolean answers and validate them.
    
    Args:
        input_path: Path to input JSONL file
        output_suffix: Suffix to add to output filename
        
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
        'boolean_tags': 0,
        'bold_markers': 0,
        'answer_marker': 0,
        'last_paragraph': 0,
        'after_think': 0,
        'full_text': 0,
        'not_extracted': 0
    }
    
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
            
            # Only process BOOL_TASKS
            task = data.get('task', '')
            if task not in BOOL_TASKS:
                # Write unchanged for non-BOOL_TASKS
                f_out.write(json.dumps(data, ensure_ascii=False) + '\n')
                continue
            
            # Get fields
            extracted_core_answer = data.get('extracted_core_answer', {})
            generated = data.get('generated', {})
            ground_truth = data.get('ground_truth', '')
            
            # Process each generation
            for gen_key in extracted_core_answer.keys():
                # # If extracted answer is empty, try to extract from generated text
                # if not extracted_core_answer[gen_key] or extracted_core_answer[gen_key] == "":
                gen_text = generated.get(gen_key, '')
                
                # Try each strategy and track which one worked
                extracted_bool = None
                strategy_used = 'not_extracted'
                
                # Try strategies in order
                extracted_bool = extract_from_boolean_tags(gen_text)
                if extracted_bool is not None:
                    strategy_used = 'boolean_tags'
                else:
                    extracted_bool = extract_from_bold_markers(gen_text)
                    if extracted_bool is not None:
                        strategy_used = 'bold_markers'
                    else:
                        extracted_bool = extract_after_answer_marker(gen_text)
                        if extracted_bool is not None:
                            strategy_used = 'answer_marker'
                        else:
                            extracted_bool = extract_from_last_paragraph(gen_text)
                            if extracted_bool is not None:
                                strategy_used = 'last_paragraph'
                            else:
                                extracted_bool = extract_after_think_tag(gen_text)
                                if extracted_bool is not None:
                                    strategy_used = 'after_think'
                                # else:
                                #     extracted_bool = extract_from_full_text(gen_text)
                                #     if extracted_bool is not None:
                                #         strategy_used = 'full_text'
                    
                    # Convert to string and update
                    extracted_value = boolean_to_string(extracted_bool)
                    
                    if extracted_value:
                        extracted_core_answer[gen_key] = extracted_value
                        extracted_count += 1
                        strategy_counts[strategy_used] += 1
                    else:
                        extracted_core_answer[gen_key] = ""
                        strategy_counts['not_extracted'] += 1
            
            # Add correctness field
            extracted_core_answer_correctness = {}
            for gen_key, extracted_value in extracted_core_answer.items():
                is_correct = is_answer_correct(extracted_value, ground_truth)
                extracted_core_answer_correctness[gen_key] = is_correct
                if is_correct:
                    correct_count += 1
            
            # Update data
            data['extracted_core_answer'] = extracted_core_answer
            data['extracted_core_answer_correctness'] = extracted_core_answer_correctness
            
            # Write to output
            f_out.write(json.dumps(data, ensure_ascii=False) + '\n')
        
        pbar.close()
    
    # Prepare statistics report
    report_lines = []
    report_lines.append("=" * 70)
    report_lines.append("BOOL_TASKS Postprocessing Report")
    report_lines.append("=" * 70)
    report_lines.append(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append(f"Input file: {input_path}")
    report_lines.append(f"Output file: {output_path}")
    report_lines.append("")
    report_lines.append("Processing complete:")
    report_lines.append(f"  Total lines processed: {total_lines}")
    report_lines.append(f"  Booleans extracted: {extracted_count}")
    report_lines.append(f"  Correct answers: {correct_count}")
    if total_lines > 0:
        accuracy = (correct_count / total_lines) * 100
        report_lines.append(f"  Accuracy: {accuracy:.2f}%")
    report_lines.append("")
    report_lines.append("Extraction strategies used:")
    for strategy, count in strategy_counts.items():
        if count > 0:
            report_lines.append(f"    {strategy}: {count}")
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
        description="Extract and validate boolean answers for BOOL_TASKS"
    )
    parser.add_argument(
        'input_file',
        type=str,
        help='Path to input JSONL file'
    )
    parser.add_argument(
        '--output-suffix',
        type=str,
        default='_bool_postprocessed',
        help='Suffix for output filename (default: _bool_postprocessed)'
    )
    
    args = parser.parse_args()
    
    try:
        output_path = process_jsonl_file(args.input_file, args.output_suffix)
        print(f"\nSuccess! Processed file saved to:\n{output_path}")
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()
