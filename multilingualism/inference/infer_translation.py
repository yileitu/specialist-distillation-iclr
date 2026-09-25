#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Multilingual translation inference
Run inference on translation datasets with vLLM.
Supports early termination when model output enters a repetitive pattern.

Loop detection uses the vLLM V1 LogitsProcessor API.
"""

import os
import sys
import json
import random
import argparse
from pathlib import Path
from typing import List, Dict, Any, Optional
import re
import torch
from vllm import LLM, SamplingParams
from vllm.v1.sample.logits_processor import LogitsProcessor, BatchUpdate, MoveDirectionality
from vllm.config import VllmConfig

# Import shared utilities.
sys.path.append(str(Path(__file__).parent.parent))
from util.util_func import get_available_gpu_count

TRANSLATION_SYSTEM_PROMPT = "You are a professional multilingual translator. Translate the given text accurately and fluently. Preserve the original meaning, tone, and style. Please put your thinking process within <think> </think> tags."

# Portable default output path.
DEFAULT_OUTPUT_DIR = str(Path(__file__).resolve().parent / "output")


def process_dict_updates(
    req_entries: Dict[int, Any],
    batch_update: Optional[BatchUpdate],
    new_state,
) -> bool:
    """
    Update the sparse dictionary state for a LogitsProcessor.
    Based on the upstream vLLM implementation.
    """
    if not batch_update:
        return False

    updated = False
    for index, params, prompt_tok_ids, output_tok_ids in batch_update.added:
        state = new_state(params, prompt_tok_ids, output_tok_ids)
        if state is not None:
            req_entries[index] = state
            updated = True
        elif req_entries.pop(index, None) is not None:
            updated = True

    if req_entries:
        # Remove completed requests.
        for index in batch_update.removed:
            if req_entries.pop(index, None):
                updated = True

        # Move requests (one-way a->b or swap a<->b).
        for a_index, b_index, direct in batch_update.moved:
            a_entry = req_entries.pop(a_index, None)
            b_entry = req_entries.pop(b_index, None)
            if a_entry is not None:
                req_entries[b_index] = a_entry
                updated = True
            if b_entry is not None:
                updated = True
                if direct == MoveDirectionality.SWAP:
                    req_entries[a_index] = b_entry

    return updated


class RepetitionStoppingLogitsProcessor(LogitsProcessor):
    """
    Dynamic loop-detection logits processor (vLLM V1 API).
    
    Detect consecutive repeated patterns at the end of a sequence and force EOS at the configured threshold.
    
    The detector checks every pattern length from min_pattern_len through max_pattern_len.
    - Example: min=5, max=50, threshold=3
    - Pattern lengths 5, 6, 7, ..., 50 are checked.
    - Generation stops when any pattern repeats three times consecutively.
    
    Examples:
    - [..., A, B, C, A, B, C, A, B, C] triggers (pattern length=3, three consecutive copies)
    - [..., X, Y, Z, ..., X, Y, Z, ..., X, Y, Z, ...] triggers (pattern length=N, three consecutive copies)
    
    Parameters are passed through SamplingParams.extra_args:
    - enable_loop_detection: bool - whether to enable loop detection
    - min_pattern_len: int - minimum pattern length (default: 5, to avoid short-phrase matches)
    - max_pattern_len: int - maximum pattern length (default: 100)
    - max_repeat_times: int - maximum allowed consecutive repetitions (default: 3)
    """
    
    # Default parameters.
    DEFAULT_MIN_PATTERN_LEN = 5
    DEFAULT_MAX_PATTERN_LEN = 100
    DEFAULT_MAX_REPEAT_TIMES = 3
    
    @classmethod
    def validate_params(cls, params: SamplingParams):
        """Validate custom SamplingParams values"""
        if params.extra_args:
            min_pattern_len = params.extra_args.get("min_pattern_len")
            if min_pattern_len is not None and (not isinstance(min_pattern_len, int) or min_pattern_len < 1):
                raise ValueError(f"min_pattern_len must be a positive integer, got {min_pattern_len}")
            
            max_pattern_len = params.extra_args.get("max_pattern_len")
            if max_pattern_len is not None and (not isinstance(max_pattern_len, int) or max_pattern_len < 1):
                raise ValueError(f"max_pattern_len must be a positive integer, got {max_pattern_len}")
            
            max_repeat_times = params.extra_args.get("max_repeat_times")
            if max_repeat_times is not None and (not isinstance(max_repeat_times, int) or max_repeat_times < 2):
                raise ValueError(f"max_repeat_times must be >= 2, got {max_repeat_times}")
    
    def __init__(self, vllm_config: "VllmConfig", device: torch.device, is_pin_memory: bool):
        """Initialize the logits processor"""
        self.device = device
        self.pin_memory = is_pin_memory
        
        # Read the EOS token ID from the vLLM configuration.
        try:
            self.eos_token_id = vllm_config.model_config.hf_config.eos_token_id
            if isinstance(self.eos_token_id, list):
                self.eos_token_id = self.eos_token_id[0]
        except:
            self.eos_token_id = 2  # Fallback value.
        
        # Sparse dictionary: batch_index -> [output_tok_ids_ref, min_pattern_len, max_pattern_len, max_repeat_times, force_stop_flag]
        self.req_info: Dict[int, list] = {}
        
        # Debug mode, controlled by an environment variable.
        self.debug = os.getenv("LOOP_DETECTION_DEBUG", "0") == "1"
        
        print(f"[Loop Detection] RepetitionStoppingLogitsProcessor initialized, device={device}, eos_token_id={self.eos_token_id}, debug={self.debug}")
    
    def is_argmax_invariant(self) -> bool:
        """This processor can change argmax and is therefore not argmax invariant"""
        return False
    
    def _add_request(
        self,
        params: SamplingParams,
        prompt_tok_ids: Optional[List[int]],
        output_tok_ids: List[int]
    ) -> Optional[list]:
        """
        Create state for a new request.
        
        Returns:
            [output_tok_ids_ref, min_pattern_len, max_pattern_len, max_repeat_times, force_stop_flag] or None
        """
        if not params.extra_args:
            return None
        
        enable = params.extra_args.get("enable_loop_detection", False)
        if not enable:
            return None
        
        min_pattern_len = params.extra_args.get("min_pattern_len", self.DEFAULT_MIN_PATTERN_LEN)
        max_pattern_len = params.extra_args.get("max_pattern_len", self.DEFAULT_MAX_PATTERN_LEN)
        max_repeat_times = params.extra_args.get("max_repeat_times", self.DEFAULT_MAX_REPEAT_TIMES)
        
        # Return [output token IDs, minimum length, maximum length, repetition limit, force-stop flag].
        return [output_tok_ids, min_pattern_len, max_pattern_len, max_repeat_times, False]
    
    def update_state(self, batch_update: Optional[BatchUpdate]) -> None:
        """Update internal state"""
        old_count = len(self.req_info)
        process_dict_updates(self.req_info, batch_update, self._add_request)
        new_count = len(self.req_info)
        
        # Log only when the tracked request count changes.
        # if old_count != new_count:
        #     print(f"[Loop Detection] update_state: tracked requests {old_count} -> {new_count}")
    
    def _check_consecutive_repeat(self, output_tok_ids: List[int], pattern_len: int, max_repeat_times: int) -> int:
        """
        Count consecutive copies of a pattern at the end of the sequence.
        
        Args:
            output_tok_ids: Generated token sequence.
            pattern_len: Pattern length.
            max_repeat_times: Repetition limit used for early stopping.
        
        Returns:
            Number of consecutive repetitions (>=1).
        """
        seq_len = len(output_tok_ids)
        
        # The sequence is too short to contain the pattern.
        if seq_len < pattern_len:
            return 0
        
        # Read the pattern at the end of the sequence.
        pattern = tuple(output_tok_ids[-pattern_len:])
        consecutive_count = 1  # Count the final occurrence.
        
        # Check preceding consecutive occurrences.
        for i in range(1, max_repeat_times):
            start_idx = seq_len - (i + 1) * pattern_len
            end_idx = seq_len - i * pattern_len
            
            # Stop before crossing the sequence boundary.
            if start_idx < 0:
                break
            
            # Read the preceding pattern.
            prev_pattern = tuple(output_tok_ids[start_idx:end_idx])
            
            if prev_pattern == pattern:
                consecutive_count += 1
            else:
                # Stop at the first non-matching pattern.
                break
        
        return consecutive_count
    
    def _check_ngram_repeat(self, state: list) -> bool:
        """
        Detect a consecutive repeated pattern of any configured length at the end of a sequence.
        
        Check every pattern length from min_pattern_len through max_pattern_len.
        Stop when a pattern of any checked length reaches the repetition threshold.
        
        Args:
            state: [output_tok_ids, min_pattern_len, max_pattern_len, max_repeat_times, force_stop_flag]
        
        Returns:
            Whether generation must be stopped.
        """
        output_tok_ids, min_pattern_len, max_pattern_len, max_repeat_times, force_stop_flag = state
        
        # Return immediately when the request is already marked for stopping.
        if force_stop_flag:
            return True
        
        seq_len = len(output_tok_ids)
        
        # The sequence is too short to check the minimum pattern.
        if seq_len < min_pattern_len * max_repeat_times:
            return False
        
        # Bound the maximum pattern length by sequence length / max_repeat_times.
        effective_max_len = min(max_pattern_len, seq_len // max_repeat_times)
        
        # Check pattern lengths from shortest to longest.
        for pattern_len in range(min_pattern_len, effective_max_len + 1):
            consecutive_count = self._check_consecutive_repeat(output_tok_ids, pattern_len, max_repeat_times)
            
            # Stop when this pattern length reaches the repetition threshold.
            if consecutive_count >= max_repeat_times:
                state[4] = True  # force_stop_flag = True
                
                if self.debug:
                    pattern = tuple(output_tok_ids[-pattern_len:])
                    print(f"[Loop Detection] detected a consecutive repeated pattern (length={pattern_len}), repeated {consecutive_count} times >= threshold {max_repeat_times}, sequence length={seq_len}")
                    print(f"  first five tokens of the repeated pattern: {pattern[:min(5, len(pattern))]}...")
                
                return True
        
        return False
    
    def apply(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Apply n-gram repetition detection to a batch of logits.
        
        Args:
            logits: Tensor with shape (num_requests, vocab_size).
        
        Returns:
            Modified logits tensor.
        """
        if not self.req_info:
            return logits
        
        # Check every tracked request for forced stopping.
        for batch_idx, state in self.req_info.items():
            if batch_idx >= logits.shape[0]:
                continue
            
            if self._check_ngram_repeat(state):
                # Force EOS when repetition is detected.
                logits[batch_idx, :] = -float("inf")
                logits[batch_idx, self.eos_token_id] = 0
        
        return logits


def load_test_data(data_file: str) -> List[Dict[str, Any]]:
    """
    Load test data.
    
    Args:
        data_file: Path to a JSONL data file.
    
    Returns:
        Loaded test records.
    """
    data = []
    data_path = Path(data_file)
    if data_path.exists():
        with open(data_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    data.append(json.loads(line))
    else:
        raise FileNotFoundError(f"Data file not found: {data_file}")
    return data


def build_sample_id(sample: Dict[str, Any], task_name: str, sample_idx: int) -> str:
    """Build a stable sample ID for checkpoint matching"""
    return sample.get("uuid", f"{task_name}.{sample_idx}")


def chunk_list(items: List[Any], chunk_size: int):
    """Yield fixed-size chunks from a list"""
    for i in range(0, len(items), chunk_size):
        yield items[i:i + chunk_size]


def load_existing_results(output_file: Path) -> Dict[str, Any]:
    """Load saved results for checkpoint resume"""
    if not output_file.exists():
        return {}
    existing = {}
    with open(output_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
                sid = record.get("sample_id")
                if sid is not None:
                    existing[sid] = record
            except json.JSONDecodeError:
                continue
    return existing


def save_results_atomic(output_file: Path, results_map: Dict[str, Any]):
    """Atomically write results to avoid partial files"""
    output_file.parent.mkdir(parents=True, exist_ok=True)
    tmp_file = output_file.with_suffix(output_file.suffix + ".tmp")
    sorted_results = sorted(results_map.values(), key=lambda x: x.get("sample_idx", 0))
    with open(tmp_file, "w", encoding="utf-8") as f:
        for result in sorted_results:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")
    tmp_file.replace(output_file)


def extract_translation(generated_text: str) -> str:
    """
    Extract the translation from generated text.
    Remove content inside <think> tags and return the remaining text.
    """
    cleaned = re.sub(r'<think>.*?</think>', '', generated_text, flags=re.DOTALL)
    cleaned = cleaned.strip()
    # Keep only the final paragraph.
    cleaned = cleaned.split('\n')[-1]
    return cleaned.strip()


def build_messages(
    user_content: str,
    enable_thinking: bool = True,
    answer_before_thinking: bool = False,
    oracle_think: bool = False,
    ground_truth_answer: str = "",
) -> List[Dict[str, Any]]:
    """
    Build chat-formatted messages.
    
    Args:
        user_content: User translation request.
        enable_thinking: Whether reasoning mode is enabled.
        answer_before_thinking: Whether to output the answer before reasoning.
        oracle_think: Whether to generate a chain of thought from a query and reference answer.
        ground_truth_answer: Reference answer used only in oracle-think mode.
    
    Returns:
        Chat messages.
    """
    if oracle_think:
        user_prompt = (
            f"{user_content}\n\n"
            f"Reference Ground-truth Translation:\n{ground_truth_answer}\n\n"
            "Please generate a detailed reasoning process that can derive the provided reference translation. "
            "Put all reasoning inside <think> </think> tags. "
            "After reasoning, provide the final translation to the original request."
        )
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a professional multilingual translator. "
                    "You will be given a translation request and its ground-truth translation. "
                    "Your task is to generate a coherent reasoning chain that supports the provided reference translation. "
                    "All reasoning must be inside <think> </think> tags. "
                    "After reasoning, output the final translation for the original request."
                ),
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_prompt}
                ]
            }
        ]
        return messages

    if answer_before_thinking:
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a professional multilingual translator. Translate the given text accurately and fluently. "
                    "Preserve the original meaning, tone, and style. "
                    "First provide the translation directly, without any explanation or intermediate steps. "
                    "After that, provide your detailed thinking process inside <think> </think> tags."
                ),
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_content}
                ]
            }
        ]
    elif enable_thinking:
        messages = [
            {
                "role": "system",
                "content": TRANSLATION_SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_content}
                ]
            }
        ]
    else:
        messages = [
            {
                "role": "system",
                "content": "You are a professional multilingual translator. Translate the given text accurately and fluently. Preserve the original meaning, tone, and style. Provide only the translation without any explanation.",
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_content}
                ]
            }
        ]

    return messages


def run_inference(
    model_path: str,
    data_file: str,
    output_dir: str,
    max_new_tokens: int = 2048,
    enable_thinking: bool = True,
    tensor_parallel_size: int = 1,
    gpu_memory_utilization: float = 0.9,
    seed: int = 42,
    num_generations: int = 3,
    temperature: float = 0.8,
    presence_penalty: float = 0.0,
    frequency_penalty: float = 0.0,
    repetition_penalty: float = 1.0,
    enable_loop_detection: bool = False,
    min_pattern_len: int = 5,
    max_pattern_len: int = 100,
    max_repeat_times: int = 3,
    batch_size: int = 32,
    answer_before_thinking: bool = False,
    max_model_len: int = 65536,
    oracle_think: bool = False,
):
    """
    Run inference.
    
    Args:
        model_path: Model path.
        data_file: Path to a JSONL data file.
        output_dir: Output directory.
        max_new_tokens: Maximum number of generated tokens.
        enable_thinking: Whether reasoning mode is enabled.
        tensor_parallel_size: Tensor-parallel size.
        gpu_memory_utilization: Fraction of GPU memory available to vLLM.
        seed: Random seed.
        num_generations: Independent responses per prompt.
        temperature: Sampling temperature.
        presence_penalty: Presence penalty.
        frequency_penalty: Frequency penalty.
        repetition_penalty: Repetition penalty.
        enable_loop_detection: whether to enable loop detection
        min_pattern_len: Minimum repeated-pattern length in tokens.
        max_pattern_len: Maximum repeated-pattern length in tokens.
        max_repeat_times: Consecutive repetition limit.
        batch_size: Records per checkpointed inference batch.
        answer_before_thinking: Whether to output the answer before reasoning.
        max_model_len: Maximum vLLM context length.
        oracle_think: Whether to generate a chain of thought from a query and reference answer.
    """
    random.seed(seed)
    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}")
    
    # Derive the task name from the input file name.
    task_name = Path(data_file).stem
    
    # Derive the model name from the model path.
    model_name = Path(model_path).name
    
    # The caller may organize output by model, parameters, task, and split.
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    output_file = output_path / f"inference_{task_name}.jsonl"
    
    print(f"=" * 60)
    print(f"Multilingual translation inference")
    print(f"=" * 60)
    print(f"Task: {task_name}")
    print(f"Model path: {model_path}")
    print(f"Model name: {model_name}")
    print(f"Data file: {data_file}")
    print(f"Output directory: {output_path}")
    print(f"Maximum generated tokens: {max_new_tokens}")
    print(f"Generations per sample: {num_generations}")
    print(f"Sampling parameters: temperature={temperature}, top_p=1.0, top_k=50")
    print(f"Reasoning mode: {'enabled' if enable_thinking else 'disabled'}")
    print(f"Oracle-think mode: {'enabled' if oracle_think else 'disabled'}")
    print(f"Loop detection: {'enabled (pattern length=' + str(min_pattern_len) + '-' + str(max_pattern_len) + ', repetition limit=' + str(max_repeat_times) + ')' if enable_loop_detection else 'disabled'}")
    print(f"Batch size: {batch_size}")
    print(f"=" * 60)
    
    # Load data.
    print("\nLoading data...")
    all_data = load_test_data(data_file)
    total_samples = len(all_data)
    print(f"Data loaded. Total samples: {total_samples}")

    # Load completed results for checkpoint resume.
    existing_results = load_existing_results(output_file)
    if existing_results:
        print(f"[Checkpoint] loaded saved results: {len(existing_results)} records; matching samples will be skipped.")

    indexed_data = list(enumerate(all_data))
    pending_data = [
        (idx, sample)
        for idx, sample in indexed_data
        if build_sample_id(sample, task_name, idx) not in existing_results
    ]
    print(f"Pending samples: {len(pending_data)}(completed: {len(existing_results)})")
    if not pending_data:
        print(f"All samples are complete. Results: {output_file}")
        # Print a few examples.
        sorted_results = sorted(existing_results.values(), key=lambda x: x.get("sample_idx", 0))
        print(f"\nExample results (first three):")
        for j, result in enumerate(sorted_results[:3]):
            print(f"--- Example {j+1} ---")
            print(f"Input: {result.get('user_content', '')[:150]}...")
            print(f"Reference: {result.get('gold_answer', '')[:150]}...")
            first_gen = result.get('generated', {}).get('generation1', '')
            print(f"Generation 1: {first_gen[:150]}...")
            first_translated = result.get('extracted_translation', {}).get('generation1', '')
            if first_translated:
                print(f"Translation 1: {first_translated[:150]}...")
        print(f"\n{'=' * 60}")
        print(f"Inference complete!")
        print(f"{'=' * 60}")
        return
    
    # Prepare logits processors for LLM initialization.
    custom_logits_processors = None
    if enable_loop_detection:
        # Register the LogitsProcessor class through the V1 API.
        custom_logits_processors = [RepetitionStoppingLogitsProcessor]
        print(f"[Loop Detection] RepetitionStoppingLogitsProcessor will be registered during LLM initialization")
    
    # Load the model.
    print("\nLoading model...")
    llm_kwargs = {
        "model": model_path,
        "trust_remote_code": True,
        "tensor_parallel_size": tensor_parallel_size,
        "gpu_memory_utilization": gpu_memory_utilization,
        "dtype": "bfloat16",
        "max_model_len": max_model_len,
    }
    
    # Pass logits processors at initialization when loop detection is enabled.
    if custom_logits_processors is not None:
        llm_kwargs["logits_processors"] = custom_logits_processors
    
    llm = LLM(**llm_kwargs)
    print(f"Model loaded!\n")
    
    # Resolve the EOS token ID when loop detection is enabled.
    if enable_loop_detection:
        tokenizer = llm.get_tokenizer()
        eos_token_id = tokenizer.eos_token_id
        if eos_token_id is None:
            eos_token_id = tokenizer.convert_tokens_to_ids(tokenizer.eos_token) if tokenizer.eos_token else 2
        print(f"[Loop Detection] EOS token ID: {eos_token_id}")
        # The EOS token ID is passed to each request through extra_args.
    
    # Build extra_args for the V1 LogitsProcessor.
    extra_args = None
    if enable_loop_detection:
        extra_args = {
            "enable_loop_detection": True,
            "min_pattern_len": min_pattern_len,
            "max_pattern_len": max_pattern_len,
            "max_repeat_times": max_repeat_times,
        }
        print(f"[Loop Detection] extra_args: {extra_args}")
    
    # Configure sampling.
    sampling_params = SamplingParams(
        n=num_generations,
        temperature=temperature,
        top_p=1.0,  # Recommended value.
        top_k=50,  # Recommended value.
        max_tokens=max_new_tokens,
        skip_special_tokens=True,
        presence_penalty=presence_penalty,
        frequency_penalty=frequency_penalty,
        repetition_penalty=repetition_penalty,
        extra_args=extra_args,  # V1 API passes custom parameters through extra_args.
    )
    
    # Run checkpointed inference in batches.
    print(f"Starting batched inference with {batch_size} records per batch...")
    chat_template_kwargs = {}
    if enable_thinking:
        chat_template_kwargs["enable_thinking"] = True
    else:
        chat_template_kwargs["enable_thinking"] = False

    for batch_idx, batch_items in enumerate(chunk_list(pending_data, batch_size), start=1):
        prompts = []
        for idx, sample in batch_items:
            messages = build_messages(
                sample["user_content"],
                enable_thinking,
                answer_before_thinking=answer_before_thinking,
                oracle_think=oracle_think,
                ground_truth_answer=sample.get("gold_answer", ""),
            )
            prompts.append(messages)

        outputs = llm.chat(
            messages=prompts,
            sampling_params=sampling_params,
            use_tqdm=False,
            chat_template_kwargs=chat_template_kwargs
        )

        for local_idx, ((idx, sample), output) in enumerate(zip(batch_items, outputs)):
            generated_dict = {}
            translated_dict = {}

            for gen_idx, gen_output in enumerate(output.outputs, start=1):
                gen_key = f"generation{gen_idx}"
                generated_text = gen_output.text
                generated_dict[gen_key] = generated_text
                translated_dict[gen_key] = extract_translation(generated_text)

            sample_id = build_sample_id(sample, task_name, idx)
            result = {
                "sample_idx": idx,
                "sample_id": sample_id,
                "user_content": sample["user_content"],
                "gold_answer": sample.get("gold_answer", ""),
                "prompt": prompts[local_idx] if prompts else [],
                "generated": generated_dict,
                "extracted_translation": translated_dict,
            }
            existing_results[sample_id] = result

        save_results_atomic(output_file, existing_results)
        print(f"[Checkpoint] batch {batch_idx}: completed {len(existing_results)}/{total_samples}; results saved to {output_file}")

    # Print the final summary and examples.
    sorted_results = sorted(existing_results.values(), key=lambda x: x.get("sample_idx", 0))
    print(f"Inference complete! Results saved to: {output_file}")
    print(f"\nExample results (first three):")
    for j, result in enumerate(sorted_results[:3]):
        print(f"--- Example {j+1} ---")
        print(f"Input: {result['user_content'][:150]}...")
        print(f"Reference: {result['gold_answer'][:150]}...")
        first_gen = result['generated'].get('generation1', '')
        print(f"Generation 1: {first_gen[:150]}...")
        first_translated = result['extracted_translation'].get('generation1', '')
        if first_translated:
            print(f"Translation 1: {first_translated[:150]}...")

    print(f"\n{'=' * 60}")
    print(f"Inference complete!")
    print(f"{'=' * 60}")
    

def main():
    parser = argparse.ArgumentParser(
        description='Multilingual translation inference with vLLM',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run translation inference.
  python infer_translation.py --data_file /path/to/translation_data.jsonl
  
  # Specify an output directory and tensor-parallel size.
  python infer_translation.py --data_file /path/to/translation_data.jsonl --output_dir /path/to/output --tensor_parallel_size 4
  
  # Enable loop detection.
  python infer_translation.py --data_file /path/to/translation_data.jsonl --enable_loop_detection
        """
    )
    
    parser.add_argument(
        '--model_path',
        type=str,
        required=True,
        help='Model directory'
    )
    
    parser.add_argument(
        '--data_file',
        type=str,
        required=True,
        help='Path to the input JSONL file'
    )
    
    parser.add_argument(
        '--output_dir',
        type=str,
        default=DEFAULT_OUTPUT_DIR,
        help=f'Output directory (default: {DEFAULT_OUTPUT_DIR})'
    )
    
    parser.add_argument(
        '--max_new_tokens',
        type=int,
        default=60000,
        help='Maximum number of generated tokens'
    )
    
    parser.add_argument(
        '--presence_penalty',
        type=float,
        default=0.0,
        help='Presence penalty (default: 0.0)'
    )
    
    parser.add_argument(
        '--frequency_penalty',
        type=float,
        default=0.0,
        help='Frequency penalty (default: 0.0)'
    )
        
    parser.add_argument(
        '--repetition_penalty',
        type=float,
        default=1.0,
        help='Repetition penalty (default: 1.0)'
    )  
    
    parser.add_argument(
        '--disable_thinking',
        action='store_true',
        help='Disable reasoning mode (enabled by default)'
    )
    
    parser.add_argument(
        '--answer_before_thinking',
        action='store_true',
        help='Output the final answer before reasoning; this overrides --disable_thinking'
    )

    parser.add_argument(
        '--oracle_think',
        action='store_true',
        help='Generate reasoning from the query and reference answer, followed by the final translation'
    )
    
    parser.add_argument(
        '--tensor_parallel_size',
        type=int,
        default=None,
        help='Tensor-parallel size (default: detect available devices)'
    )
    
    parser.add_argument(
        '--gpu_memory_utilization',
        type=float,
        default=0.9,
        help='GPU memory utilization (default: 0.9)'
    )
    
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed (default: 42)'
    )
    
    parser.add_argument(
        '--num_generations',
        type=int,
        default=3,
        help='Independent responses per prompt (default: 3)'
    )
    
    parser.add_argument(
        '--temperature',
        type=float,
        default=0.8,
        help='Sampling temperature (default: 0.8)'
    )
    
    parser.add_argument(
        '--enable_loop_detection',
        action='store_true',
        help='Stop generation when a repeated pattern is detected'
    )
    
    parser.add_argument(
        '--min_pattern_len',
        type=int,
        default=5,
        help='Minimum repeated-pattern length in tokens (default: 5)'
    )
    
    parser.add_argument(
        '--max_pattern_len',
        type=int,
        default=200,
        help='Maximum repeated-pattern length in tokens (default: 200)'
    )
    
    parser.add_argument(
        '--max_repeat_times',
        type=int,
        default=3,
        help='Consecutive repetition limit (default: 3)'
    )
    
    parser.add_argument(
        '--batch_size',
        type=int,
        default=32,
        help='Checkpointed inference batch size (default: 32)'
    )
    
    parser.add_argument(
        '--max_model_len',
        type=int,
        default=65536,
        help='Maximum vLLM context length (default: 65536)'
    )
    
    args = parser.parse_args()
    
    # Detect the tensor-parallel size when it is not specified.
    tensor_parallel_size = args.tensor_parallel_size
    if tensor_parallel_size is None:
        tensor_parallel_size = get_available_gpu_count()
    
    # Resolve reasoning mode from the command-line flags:
    # - By default, enable_thinking is the inverse of disable_thinking.
    # - answer_before_thinking always enables reasoning.
    enable_thinking = not args.disable_thinking
    if args.answer_before_thinking:
        enable_thinking = True
    if args.oracle_think:
        enable_thinking = True

    if args.answer_before_thinking and args.oracle_think:
        raise ValueError("--answer_before_thinking and --oracle_think are mutually exclusive")
    
    run_inference(
        model_path=args.model_path,
        data_file=args.data_file,
        output_dir=args.output_dir,
        max_new_tokens=args.max_new_tokens,
        enable_thinking=enable_thinking,
        tensor_parallel_size=tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        seed=args.seed,
        num_generations=args.num_generations,
        temperature=args.temperature,
        presence_penalty=args.presence_penalty,
        frequency_penalty=args.frequency_penalty,
        repetition_penalty=args.repetition_penalty,
        enable_loop_detection=args.enable_loop_detection,
        min_pattern_len=args.min_pattern_len,
        max_pattern_len=args.max_pattern_len,
        max_repeat_times=args.max_repeat_times,
        batch_size=args.batch_size,
        answer_before_thinking=args.answer_before_thinking,
        max_model_len=args.max_model_len,
        oracle_think=args.oracle_think,
    )


if __name__ == '__main__':
    main()
