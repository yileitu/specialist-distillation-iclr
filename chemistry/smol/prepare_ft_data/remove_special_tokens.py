#!/usr/bin/env python3
"""
Remove special tokens from assistant responses in JSONL files.

This script processes JSONL files to remove special tokens from assistant content,
including model-specific tokens like <|im_start|>, <|im_end|>, and domain-specific
tokens like <SMILES>, <IUPAC>, <FASTA>, etc.
"""

import json
import argparse
from pathlib import Path
from typing import Dict, Any, List
from tqdm import tqdm
from datetime import datetime


# Special tokens to remove (based on InternLM tokenizer)
SPECIAL_TOKENS = [
    # Chat tokens
    "<|endoftext|>",
    "<|im_start|>",
    "<|im_end|>",
    
    # Object reference tokens
    "<|object_ref_start|>",
    "<|object_ref_end|>",
    "<|box_start|>",
    "<|box_end|>",
    "<|quad_start|>",
    "<|quad_end|>",
    
    # Vision tokens
    "<|vision_start|>",
    "<|vision_end|>",
    "<|vision_pad|>",
    "<|image_pad|>",
    "<|video_pad|>",
    
    # Tool tokens
    "<tool_call>",
    "</tool_call>",
    "<tool_response>",
    "</tool_response>",
    
    # Fill-in-middle tokens
    "<|fim_prefix|>",
    "<|fim_middle|>",
    "<|fim_suffix|>",
    "<|fim_pad|>",
    
    # Code tokens
    "<|repo_name|>",
    "<|file_sep|>",
    
    # Domain-specific tokens
    "<SELFIES>",
    "</SELFIES>",
    "<FASTA>",
    "</FASTA>",
    
    # Image context tokens
    "<IMG_CONTEXT>",
    "<image>",
    "</image>",
    "<img>",
    "</img>",
    "<quad>",
    "</quad>",
    "<ref>",
    "</ref>",
    "<box>",
    "</box>",
    
    # Action tokens
    "<|action_start|>",
    "<|action_end|>",
    "<|interpreter|>",
    "<|plugin|>",
    
    # Video token
    "<video>",
    "</video>",
]


def remove_special_tokens(text: str, tokens: List[str]) -> str:
    """
    Remove all special tokens from the text.
    
    Args:
        text: Input text containing special tokens
        tokens: List of special tokens to remove
        
    Returns:
        Text with all special tokens removed
    """
    cleaned_text = text
    for token in tokens:
        cleaned_text = cleaned_text.replace(token, "")
    return cleaned_text


def process_jsonl_file(
    input_path: Path,
    output_path: Path,
    tokens_to_remove: List[str]
) -> Dict[str, Any]:
    """
    Process JSONL file and remove special tokens from assistant responses.
    
    Args:
        input_path: Path to input JSONL file
        output_path: Path to output JSONL file
        tokens_to_remove: List of special tokens to remove
        
    Returns:
        Dictionary with processing statistics
    """
    stats = {
        'total_lines': 0,
        'assistant_messages': 0,
        'modified_messages': 0,
        'total_tokens_removed': 0
    }
    
    with open(input_path, 'r', encoding='utf-8') as infile, \
         open(output_path, 'w', encoding='utf-8') as outfile:
        
        for line in tqdm(infile, desc="Processing"):
            stats['total_lines'] += 1
            
            try:
                data = json.loads(line.strip())
                
                # Check if this is a conversation format with messages
                if 'messages' in data:
                    for message in data['messages']:
                        if message.get('role') == 'assistant' and 'content' in message:
                            stats['assistant_messages'] += 1
                            original_content = message['content']
                            cleaned_content = remove_special_tokens(original_content, tokens_to_remove)
                            
                            if original_content != cleaned_content:
                                stats['modified_messages'] += 1
                                # Count how many tokens were removed
                                for token in tokens_to_remove:
                                    stats['total_tokens_removed'] += original_content.count(token)
                                
                                message['content'] = cleaned_content
                
                # Check if this is a direct format with role and content
                elif data.get('role') == 'assistant' and 'content' in data:
                    stats['assistant_messages'] += 1
                    original_content = data['content']
                    cleaned_content = remove_special_tokens(original_content, tokens_to_remove)
                    
                    if original_content != cleaned_content:
                        stats['modified_messages'] += 1
                        # Count how many tokens were removed
                        for token in tokens_to_remove:
                            stats['total_tokens_removed'] += original_content.count(token)
                        
                        data['content'] = cleaned_content
                
                # Write the processed line
                outfile.write(json.dumps(data, ensure_ascii=False) + '\n')
                
            except json.JSONDecodeError as e:
                print(f"Warning: Failed to parse line {stats['total_lines']}: {e}")
                # Write the original line if parsing fails
                outfile.write(line)
    
    return stats


def main():
    parser = argparse.ArgumentParser(
        description="Remove special tokens from assistant responses in JSONL files"
    )
    parser.add_argument(
        '--input',
        type=str,
        required=True,
        help='Path to input JSONL file'
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Path to output JSONL file (default: input_file with _cleaned suffix)'
    )
    parser.add_argument(
        '--custom-tokens',
        type=str,
        nargs='+',
        default=None,
        help='Additional custom tokens to remove (space-separated)'
    )
    parser.add_argument(
        '--only-custom',
        action='store_true',
        help='Only remove custom tokens (ignore default token list)'
    )
    
    args = parser.parse_args()
    
    # Setup paths
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: Input file not found: {input_path}")
        return 1
    
    if args.output:
        output_path = Path(args.output)
    else:
        # Create output filename with _cleaned suffix
        output_path = input_path.parent / f"{input_path.stem}_special_tokens_removed{input_path.suffix}"
    
    # Determine which tokens to remove
    if args.only_custom:
        tokens_to_remove = args.custom_tokens or []
    else:
        tokens_to_remove = SPECIAL_TOKENS.copy()
        if args.custom_tokens:
            tokens_to_remove.extend(args.custom_tokens)
    
    if not tokens_to_remove:
        print("Error: No tokens specified for removal")
        return 1
    
    # Process the file
    print(f"Input file: {input_path}")
    print(f"Output file: {output_path}")
    print(f"Tokens to remove: {len(tokens_to_remove)} tokens")
    print("-" * 80)
    
    start_time = datetime.now()
    stats = process_jsonl_file(input_path, output_path, tokens_to_remove)
    end_time = datetime.now()
    
    # Print statistics
    print("-" * 80)
    print("Processing complete!")
    print(f"Total lines processed: {stats['total_lines']}")
    print(f"Assistant messages found: {stats['assistant_messages']}")
    print(f"Messages modified: {stats['modified_messages']}")
    print(f"Total token occurrences removed: {stats['total_tokens_removed']}")
    print(f"Processing time: {end_time - start_time}")
    print(f"Output saved to: {output_path}")
    
    return 0


if __name__ == "__main__":
    exit(main())
