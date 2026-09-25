#!/usr/bin/env python3
"""
Extract and validate molecule captioning answers using METEOR score.

This script processes inference JSONL files to:
1. Extract text after </think> tag from generated text
2. Calculate METEOR score between extracted text and ground truth
3. Validate if METEOR score exceeds threshold
"""

import json
import re
import argparse
from pathlib import Path
from typing import Optional
import sys
from tqdm import tqdm
from datetime import datetime
import os

# Import METEOR score
try:
    from nltk.translate.meteor_score import meteor_score
    from nltk import word_tokenize
    import nltk
    # Ensure required NLTK data is downloaded
    try:
        nltk.data.find('tokenizers/punkt_tab')
    except LookupError:
        print("Downloading required NLTK data...")
        nltk.download('punkt', quiet=True)
        nltk.download('wordnet', quiet=True)
except ImportError:
    print("Error: Please install nltk: pip install nltk")
    sys.exit(1)

# Add parent directory to path to import util
sys.path.append(str(Path(__file__).parent.parent))
from util import ALL_TASKS


# Task name
MOLECULE_CAPTIONING_TASK = "molecule_captioning"


def extract_text_after_think(text: str) -> Optional[str]:
    """
    Extract text after </think> tag.
    
    Args:
        text: Generated text to search
        
    Returns:
        Text after </think> tag if found, None otherwise
    """
    if not text or not isinstance(text, str):
        return None
    
    # Search for </think> tag (case insensitive)
    think_close_pattern = r'</think>'
    match = re.search(think_close_pattern, text, re.IGNORECASE)
    
    if match:
        # Extract text after </think>
        after_think = text[match.end():].strip()
        return after_think if after_think else None
    
    return None


def calculate_meteor_score(hypothesis: str, reference: str) -> float:
    """
    Calculate METEOR score between hypothesis and reference.
    
    Args:
        hypothesis: Generated text
        reference: Ground truth text
        
    Returns:
        METEOR score (0.0 to 1.0)
    """
    try:
        # Tokenize
        hypothesis_tokens = word_tokenize(hypothesis.lower())
        reference_tokens = word_tokenize(reference.lower())
        
        # Calculate METEOR score
        score = meteor_score([reference_tokens], hypothesis_tokens)
        return score
    except Exception as e:
        print(f"Warning: Error calculating METEOR score: {e}")
        return 0.0


def process_jsonl_file(
    input_path: str,
    meteor_threshold: float = 0.4,
    output_suffix: str = "_molecule_captioning_postprocessed",
    enable_thinking: bool = True,
) -> str:
    """
    Process a JSONL file to extract molecule captioning answers and validate with METEOR score.
    
    Args:
        input_path: Path to input JSONL file
        meteor_threshold: Threshold for METEOR score (default: 0.4)
        output_suffix: Suffix to add to output filename
        
    Returns:
        Path to output file
    """
    input_file = Path(input_path)
    
    if not input_file.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")
    
    # Create output path with threshold in filename
    threshold_str = f"_threshold{meteor_threshold:.2f}"
    output_path = input_file.parent / f"{input_file.stem}{output_suffix}{threshold_str}{input_file.suffix}"
    
    # Statistics
    total_lines = 0
    task_lines = 0
    generations_processed = 0
    generations_with_think = 0
    generations_without_think = 0
    generations_correct = 0
    meteor_scores = []
    
    # Count total lines for progress bar
    print(f"Counting lines in {input_file.name}...")
    with open(input_file, 'r', encoding='utf-8') as f:
        total_file_lines = sum(1 for line in f if line.strip())
    
    print(f"Processing {input_file.name}...")
    print(f"METEOR threshold: {meteor_threshold}")
    print(f"Enable thinking: {enable_thinking}")
    print("")
    
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
            
            # Only process molecule_captioning task
            task = data.get('task', '')
            if task != MOLECULE_CAPTIONING_TASK:
                # Write unchanged for other tasks
                f_out.write(json.dumps(data, ensure_ascii=False) + '\n')
                continue
            
            task_lines += 1
            
            # Get fields
            generated = data.get('generated', {})
            ground_truth = data.get('ground_truth', '')
            
            # Initialize result dictionaries
            extracted_core_answer = {}
            extracted_core_answer_correctness = {}
            
            # Process each generation
            for gen_key in sorted(generated.keys()):
                generations_processed += 1
                gen_text = generated.get(gen_key, '')

                if enable_thinking:
                    # Extract text after </think>
                    after_think = extract_text_after_think(gen_text)

                    if after_think is None:
                        # No </think> tag found
                        generations_without_think += 1
                        extracted_core_answer[gen_key] = None
                        extracted_core_answer_correctness[gen_key] = False
                        continue

                    # </think> tag found, calculate METEOR score
                    generations_with_think += 1
                    hypothesis = after_think
                else:
                    # Use entire generated text to calculate METEOR score
                    hypothesis = gen_text.strip() if isinstance(gen_text, str) else ""
                    if not hypothesis:
                        extracted_core_answer[gen_key] = None
                        extracted_core_answer_correctness[gen_key] = False
                        continue

                meteor = calculate_meteor_score(hypothesis, ground_truth)
                meteor_scores.append(meteor)

                # Store METEOR score in extracted_core_answer
                extracted_core_answer[gen_key] = meteor

                # Check if METEOR score exceeds threshold
                is_correct = meteor >= meteor_threshold
                extracted_core_answer_correctness[gen_key] = is_correct

                if is_correct:
                    generations_correct += 1
            
            # Update data
            data['extracted_core_answer'] = extracted_core_answer
            data['extracted_core_answer_correctness'] = extracted_core_answer_correctness
            
            # Write to output
            f_out.write(json.dumps(data, ensure_ascii=False) + '\n')
        
        pbar.close()
    
    # Calculate statistics
    avg_meteor = sum(meteor_scores) / len(meteor_scores) if meteor_scores else 0.0
    max_meteor = max(meteor_scores) if meteor_scores else 0.0
    min_meteor = min(meteor_scores) if meteor_scores else 0.0
    
    # Prepare statistics report
    report_lines = []
    report_lines.append("=" * 70)
    report_lines.append("Molecule Captioning Postprocessing Report")
    report_lines.append("=" * 70)
    report_lines.append(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append(f"Input file: {input_path}")
    report_lines.append(f"Output file: {output_path}")
    report_lines.append(f"METEOR threshold: {meteor_threshold}")
    report_lines.append(f"Enable thinking: {enable_thinking}")
    report_lines.append("")
    report_lines.append("Processing complete:")
    report_lines.append(f"  Total lines processed: {total_lines}")
    report_lines.append(f"  Molecule captioning task lines: {task_lines}")
    report_lines.append(f"  Total generations processed: {generations_processed}")
    report_lines.append("")
    if enable_thinking:
        report_lines.append("Generation statistics:")
        report_lines.append(f"  Generations with </think> tag: {generations_with_think}")
        report_lines.append(f"  Generations without </think> tag: {generations_without_think}")
    report_lines.append(f"  Generations meeting threshold: {generations_correct}")
    
    if generations_processed > 0:
        accuracy = (generations_correct / generations_processed) * 100
        report_lines.append(f"  Accuracy (threshold={meteor_threshold}): {accuracy:.2f}%")
    
    if meteor_scores:
        report_lines.append("")
        stats_scope = "for generations with </think>" if enable_thinking else "for all generations"
        report_lines.append(f"METEOR score statistics ({stats_scope}):")
        report_lines.append(f"  Average METEOR: {avg_meteor:.4f}")
        report_lines.append(f"  Max METEOR: {max_meteor:.4f}")
        report_lines.append(f"  Min METEOR: {min_meteor:.4f}")
        report_lines.append(f"  Total scores calculated: {len(meteor_scores)}")
    
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
        description="Extract and validate molecule captioning answers using METEOR score"
    )
    parser.add_argument(
        'input_file',
        type=str,
        help='Path to input JSONL file'
    )
    parser.add_argument(
        '--meteor-threshold',
        type=float,
        default=0.4,
        help='METEOR score threshold for correctness (default: 0.4)'
    )
    parser.add_argument(
        '--output-suffix',
        type=str,
        default='_molecule_captioning_postprocessed',
        help='Suffix for output filename (default: _molecule_captioning_postprocessed)'
    )
    parser.add_argument(
        '--enable-thinking',
        action=argparse.BooleanOptionalAction,
        default=True,
        help='Whether to extract text after </think> before scoring (default: True). Use --no-enable-thinking to score the full generation.'
    )
    
    args = parser.parse_args()
    
    # Validate threshold
    if not 0.0 <= args.meteor_threshold <= 1.0:
        print("Error: meteor-threshold must be between 0.0 and 1.0", file=sys.stderr)
        sys.exit(1)
    
    try:
        output_path = process_jsonl_file(
            args.input_file,
            meteor_threshold=args.meteor_threshold,
            output_suffix=args.output_suffix,
            enable_thinking=args.enable_thinking,
        )
        print(f"\nSuccess! Processed file saved to:\n{output_path}")
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
