#!/usr/bin/env python3
"""
Find records with the same sample_id in two JSONL files and compare them.

Each output record contains three fields:
- sample_id: The shared sample ID.
- from_file1: The complete record from the first file.
- from_file2: The complete record from the second file.

This format makes differences between matching records easy to inspect.
"""

import argparse
import json
import os
from typing import Set, List, Dict

def read_sample_ids(jsonl_path: str) -> Set[str]:
    """Read every sample_id from a JSONL file."""
    sample_ids = set()
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                try:
                    data = json.loads(line)
                    if 'sample_id' in data:
                        sample_ids.add(data['sample_id'])
                except json.JSONDecodeError as e:
                    print(f"Warning: Skipping invalid JSON line: {e}")
    return sample_ids

def read_jsonl_data(jsonl_path: str) -> List[Dict]:
    """Read all records from a JSONL file."""
    data_list = []
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                try:
                    data = json.loads(line)
                    data_list.append(data)
                except json.JSONDecodeError as e:
                    print(f"Warning: Skipping invalid JSON line: {e}")
    return data_list

def read_jsonl_as_dict(jsonl_path: str) -> Dict[str, Dict]:
    """Read JSONL records into a mapping keyed by sample_id."""
    data_dict = {}
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                try:
                    data = json.loads(line)
                    if 'sample_id' in data:
                        sample_id = data['sample_id']
                        # Keep the last record when sample_id is duplicated
                        data_dict[sample_id] = data
                except json.JSONDecodeError as e:
                    print(f"Warning: Skipping invalid JSON line: {e}")
    return data_dict

def extract_common_samples(file1_path: str, file2_path: str, output_path: str):
    """Merge records with the same sample_id from two files."""
    
    print(f"Reading first file: {file1_path}")
    data_dict_1 = read_jsonl_as_dict(file1_path)
    print(f"First file contains {len(data_dict_1)} unique sample_id values")
    
    print(f"Reading second file: {file2_path}")
    data_dict_2 = read_jsonl_as_dict(file2_path)
    print(f"Second file contains {len(data_dict_2)} unique sample_id values")
    
    # Find shared sample_id values
    sample_ids_1 = set(data_dict_1.keys())
    sample_ids_2 = set(data_dict_2.keys())
    common_sample_ids = sample_ids_1.intersection(sample_ids_2)
    print(f"Found {len(common_sample_ids)} shared sample_id values")
    
    if len(common_sample_ids) == 0:
        print("No shared sample_id values found; stopping.")
        return
    
    # Merge data from the two files
    print("Merging records...")
    merged_data = []
    for sample_id in sorted(common_sample_ids):  # Sort for deterministic output
        merged_item = {
            "sample_id": sample_id,
            "from_file1": data_dict_1[sample_id],
            "from_file2": data_dict_2[sample_id]
        }
        merged_data.append(merged_item)
    
    print(f"Merged {len(merged_data)} records")
    
    # Save to a new file
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        for data in merged_data:
            f.write(json.dumps(data, ensure_ascii=False) + '\n')
    
    print(f"Results saved to: {output_path}")
    print(f"Saved {len(merged_data)} records")

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Merge records that share a sample_id across two JSONL files."
    )
    parser.add_argument("file1", help="Path to the first JSONL file")
    parser.add_argument("file2", help="Path to the second JSONL file")
    parser.add_argument("output_file", help="Path for the merged JSONL output")
    args = parser.parse_args()

    print("=" * 80)
    print("Comparing shared sample IDs")
    print("=" * 80)
    print(f"First file: {args.file1}")
    print(f"Second file: {args.file2}")
    print(f"Output: {args.output_file}")
    print("=" * 80)

    extract_common_samples(args.file1, args.file2, args.output_file)
    print("Processing complete.")


if __name__ == "__main__":
    main()
