#!/usr/bin/env python3
"""
Extract and validate SMILES answers for SMILES_TASKS (molecule structure tasks).

This script processes inference JSONL files to:
1. Extract SMILES strings from generated text using multiple fallback strategies
2. Canonicalize extracted SMILES and validate against ground truth using InChI comparison
   (following the LLM4Chem evaluation method)
"""

import json
import re
import argparse
from pathlib import Path
from typing import Dict, Any, Optional
import sys
from tqdm import tqdm
from datetime import datetime

# RDKit for SMILES validation and InChI conversion
from rdkit import Chem, RDLogger

# Disable RDKit warnings
RDLogger.DisableLog('rdApp.*')

# Add parent directory to path to import util
sys.path.append(str(Path(__file__).parent.parent))
from util import (
    SMILES_TASKS,
    canonicalize_molecule_smiles,
    extract_forward_synthesis,
    extract_retrosynthesis,
    extract_molecule_generation
)


def get_molecule_id(smiles: str, remove_duplicate: bool = True) -> Optional[tuple]:
    """
    Convert SMILES to InChI identifier for robust chemical equivalence comparison.
    
    This function follows the LLM4Chem evaluation method:
    - For multi-molecule SMILES (e.g., "A.B.C"), splits by '.', converts each part 
      to InChI, and returns a sorted tuple of InChI identifiers
    - For single molecule SMILES, returns a tuple with one InChI
    
    Args:
        smiles: SMILES string (can contain '.' for multiple molecules)
                Note: Semicolons should be replaced with dots before calling this function
        remove_duplicate: If True, handles multi-molecule SMILES by splitting and 
                         creating set of InChI identifiers
        
    Returns:
        Tuple of InChI identifier(s), or None if conversion fails
        
    Examples:
        - "CCO" -> ("InChI=1S/C2H6O/c1-2-3/h3H,2H2,1H3",)
        - "CCO.CC" -> ("InChI=1S/C2H6O/c1-2-3/h3H,2H2,1H3", "InChI=1S/C2H6/c1-2/h1-2H3")
    """
    try:
        if remove_duplicate:
            # Handle multi-molecule SMILES (e.g., "A.B.C")
            if ';' in smiles:
                # Semicolons should have been replaced with dots in preprocessing
                # If we see one here, it's an error
                return None
                
            all_inchi = set()
            for part in smiles.split('.'):
                inchi = get_molecule_id(part, remove_duplicate=False)
                if inchi is None:
                    return None
                all_inchi.add(inchi)
            
            # Return sorted tuple for consistent comparison
            return tuple(sorted(all_inchi))
        else:
            # Single molecule: convert to InChI
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return None
            return Chem.MolToInchi(mol)
    except Exception:
        return None


def extract_from_smiles_tags(text: str) -> Optional[str]:
    """
    Extract SMILES from <SMILES></SMILES> tags.
    
    Args:
        text: Generated text to search
        
    Returns:
        SMILES string (last tag, stripped), or None if not found
    """
    pattern = r'<SMILES>(.*?)</SMILES>'
    matches = re.findall(pattern, text, re.IGNORECASE | re.DOTALL)
    
    if matches:
        # Take the last match and strip whitespace
        return matches[-1].strip()
    
    return None


def extract_from_boxed(text: str) -> Optional[str]:
    """
    Extract SMILES from \boxed{} pattern.
    
    Args:
        text: Generated text to search
        
    Returns:
        SMILES string (last boxed content, stripped), or None if not found
    """
    pattern = r'\\boxed\{([^}]+)\}'
    matches = re.findall(pattern, text)
    
    if matches:
        # Take the last match and strip whitespace
        return matches[-1].strip()
    
    return None


def extract_task_specific(text: str, task: str) -> Optional[str]:
    """
    Extract SMILES using task-specific extraction functions.
    
    Args:
        text: Generated text to search
        task: Task name (e.g., "retrosynthesis", "forward_synthesis", etc.)
        
    Returns:
        SMILES string or empty string
    """
    if task == "retrosynthesis":
        result = extract_retrosynthesis(text)
    elif task == "forward_synthesis":
        result = extract_forward_synthesis(text)
    elif task == "molecule_generation":
        result = extract_molecule_generation(text)
    elif task == "name_conversion-i2s":
        # For name conversion, try generic extraction methods
        result = extract_molecule_generation(text)
    else:
        result = ""
    
    # Return None if empty string, otherwise return the result
    return result if result else None


def extract_smiles_from_text(text: str, task: str) -> Optional[str]:
    """
    Extract SMILES from text using multiple fallback strategies.
    
    Strategy priority:
    1. <SMILES></SMILES> tags (last one)
    2. \boxed{} pattern (last one)
    3. Task-specific extraction functions
    
    Args:
        text: Generated text to extract SMILES from
        task: Task name for task-specific extraction
        
    Returns:
        SMILES string or None if not found
    """
    if not text or not isinstance(text, str):
        return None
    
    # Strategy 1: <SMILES></SMILES> tags
    result = extract_from_smiles_tags(text)
    if result:
        return result
    
    # Strategy 2: \boxed{} pattern
    result = extract_from_boxed(text)
    if result:
        return result
    
    # Strategy 3: Task-specific extraction
    result = extract_task_specific(text, task)
    if result:
        return result
    
    return None


def is_smiles_correct(extracted: str, ground_truth: str) -> bool:
    """
    Check if extracted SMILES matches ground truth using InChI comparison.
    
    This follows the LLM4Chem evaluation method from the SMol dataset official code.
    Instead of comparing canonical SMILES strings directly, we convert both SMILES
    to InChI (International Chemical Identifier) and compare those. This is more
    robust for determining chemical equivalence.
    
    For multi-molecule SMILES (e.g., "A.B.C"), the function splits by '.', converts
    each molecule to InChI, and compares the sorted sets of InChI identifiers.
    
    Args:
        extracted: Extracted SMILES string
        ground_truth: Ground truth SMILES string
        
    Returns:
        True if InChI identifiers match, False otherwise
        
    Examples:
        - "CCO" vs "OCC" -> True (same molecule, different representation)
        - "CCO.CC" vs "CC.CCO" -> True (same molecules, different order)
        - "CCO;CC" vs "CC.CCO" -> True (semicolon converted to dot)
        - "CCO" vs "CC" -> False (different molecules)
    """
    if not extracted or extracted == "":
        return False
    
    try:
        # Preprocessing: Replace semicolons with dots (LLM4Chem method)
        # Both ';' and '.' can separate multiple molecules, but '.' is standard
        extracted = extracted.replace(';', '.')
        ground_truth = ground_truth.replace(';', '.')
        
        # First try to canonicalize to clean up the SMILES
        # This helps handle malformed SMILES before InChI conversion
        canonical_extracted = canonicalize_molecule_smiles(extracted)
        canonical_ground_truth = canonicalize_molecule_smiles(ground_truth)
        
        # Check if canonicalization succeeded
        if canonical_extracted is None or canonical_ground_truth is None:
            return False
        
        # Convert to InChI for comparison (LLM4Chem method)
        extracted_inchi = get_molecule_id(canonical_extracted)
        ground_truth_inchi = get_molecule_id(canonical_ground_truth)
        
        # Check if InChI conversion succeeded
        if extracted_inchi is None or ground_truth_inchi is None:
            return False
        
        # Compare InChI identifiers
        return extracted_inchi == ground_truth_inchi
    
    except Exception:
        # If any error occurs, consider it incorrect
        return False


def process_jsonl_file(input_path: str, output_suffix: str = "_smiles_postprocessed") -> str:
    """
    Process a JSONL file to extract SMILES and validate them.
    
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
    canonicalization_errors = 0
    strategy_counts = {
        'smiles_tags': 0,
        'boxed': 0,
        'task_specific': 0,
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
            
            # Only process SMILES_TASKS
            task = data.get('task', '')
            if task not in SMILES_TASKS:
                # Write unchanged for non-SMILES_TASKS
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
                    
                    # Try each strategy and track which one worked
                    extracted_smiles = None
                    strategy_used = 'not_extracted'
                    
                    # Try strategies in order
                    extracted_smiles = extract_from_smiles_tags(gen_text)
                    if extracted_smiles:
                        strategy_used = 'smiles_tags'
                    else:
                        extracted_smiles = extract_from_boxed(gen_text)
                        if extracted_smiles:
                            strategy_used = 'boxed'
                        else:
                            extracted_smiles = extract_task_specific(gen_text, task)
                            if extracted_smiles:
                                strategy_used = 'task_specific'
                    
                    if extracted_smiles:
                        extracted_core_answer[gen_key] = extracted_smiles
                        extracted_count += 1
                        strategy_counts[strategy_used] += 1
                    else:
                        # Keep as empty string if nothing found
                        extracted_core_answer[gen_key] = ""
                        strategy_counts['not_extracted'] += 1
            
            # Add correctness field
            extracted_core_answer_correctness = {}
            for gen_key, extracted_value in extracted_core_answer.items():
                if extracted_value and extracted_value != "":
                    try:
                        is_correct = is_smiles_correct(extracted_value, ground_truth)
                        extracted_core_answer_correctness[gen_key] = is_correct
                        if is_correct:
                            correct_count += 1
                    except Exception:
                        # If comparison fails, mark as incorrect
                        extracted_core_answer_correctness[gen_key] = False
                        canonicalization_errors += 1
                else:
                    # Empty string -> False
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
    report_lines.append("SMILES_TASKS Postprocessing Report")
    report_lines.append("=" * 70)
    report_lines.append(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append(f"Input file: {input_path}")
    report_lines.append(f"Output file: {output_path}")
    report_lines.append("")
    report_lines.append("Processing complete:")
    report_lines.append(f"  Total lines processed: {total_lines}")
    report_lines.append(f"  SMILES extracted: {extracted_count}")
    report_lines.append(f"  Correct answers: {correct_count}")
    if total_lines > 0:
        accuracy = (correct_count / total_lines) * 100
        report_lines.append(f"  Accuracy: {accuracy:.2f}%")
    if canonicalization_errors > 0:
        report_lines.append(f"  Canonicalization errors: {canonicalization_errors}")
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
        description="Extract and validate SMILES answers for SMILES_TASKS"
    )
    parser.add_argument(
        'input_file',
        type=str,
        help='Path to input JSONL file'
    )
    parser.add_argument(
        '--output-suffix',
        type=str,
        default='_smiles_postprocessed',
        help='Suffix for output filename (default: _smiles_postprocessed)'
    )
    
    args = parser.parse_args()
    
    try:
        output_path = process_jsonl_file(args.input_file, args.output_suffix)
        print(f"\nSuccess! Processed file saved to:\n{output_path}")
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
