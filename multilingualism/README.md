# Multilingualism experiments

This directory contains the Multilingualism track for *Training Specialist Models without Reasoning Trajectories for Domain Expert Distillation*. It covers low-resource translation data preparation, specialist adaptation, rationale and translation generation, spBLEU-based filtering, distillation-data assembly, and evaluation preparation.

For the cross-domain overview, see the [project README](../README.md).

## Languages and datasets

The reference experiments use English paired bidirectionally with eight languages, giving 16 translation directions:

| Code | Language |
| --- | --- |
| `bn` | Bengali |
| `cs` | Czech |
| `hu` | Hungarian |
| `sr` | Serbian |
| `sw` | Swahili |
| `te` | Telugu |
| `th` | Thai |
| `vi` | Vietnamese |

OPUS parallel data is used for specialist training. Flores-101 provides the main in-task evaluation set. Dataset files are not included in the repository.

## Directory layout

- `translation_data/`: language-pair extraction and conversion to inference records.
- `ft_scripts/`: ShareGPT dataset registration, FFT/LST YAML generation, and a local FFT launcher.
- `lora_scripts/`: LoRA configuration, local launch, and adapter-weight merging.
- `lst_scripts/`: two-stage layer-selective training launcher.
- `asft_scripts/`: ASFT configuration and local launcher for drift-control ablations.
- `inference/`: vLLM generation, split/batch helpers, spBLEU scoring, trajectory filtering, and ShareGPT merging.
- `util/`: aggregate spBLEU reporting and shared helpers.

## Input formats

Training utilities expect ShareGPT JSONL. Each record contains a `messages` list, and translation records may also carry `src_lg` and `trg_lg` language codes.

The inference utility expects one JSON object per line with:

- `user_content`: the translation instruction and source text.
- `gold_answer`: the reference translation.
- `uuid`: an optional stable identifier; a generated ID is added by the conversion utility.

## Workflow

### 1. Prepare direction-specific inference data

Convert a directory of ShareGPT JSONL files into `en2xx/` and `xx2en/` outputs:

```bash
python translation_data/process_rawfull_data.py \
  --input-dir <sharegpt-jsonl-dir> \
  --output-dir <inference-data-dir>
```

For a mixed JSONL source, `translation_data/8low_langs/test_data/extract_low_langs_jsonl.py` extracts selected English-centric pairs. The Swahili-specific helper is in `translation_data/en=sw/`.

### 2. Train an intermediate specialist

Install [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory), then inspect one of the local launchers:

```bash
bash ft_scripts/submit_ft.sh --help
bash lora_scripts/submit_lora.sh --help
bash lst_scripts/submit_lst.sh --help
bash asft_scripts/submit_asft.sh --help
```

A minimal FFT invocation is:

```bash
bash ft_scripts/submit_ft.sh \
  --model <origin-model-path> \
  --data <training.jsonl> \
  --output-dir <specialist-output-dir>
```

The launchers register the dataset, generate a YAML file, and call `llamafactory-cli` locally. They contain no scheduler submission or fixed hardware allocation; wrap them with the launcher appropriate for your environment if needed.

### 3. Generate translations and rationales

```bash
python inference/infer_translation.py \
  --model_path <specialist-model-path> \
  --data_file <inference.jsonl> \
  --output_dir <inference-output-dir>
```

Generation is resumable and supports multiple candidates, optional reasoning modes, answer-conditioned reasoning, repetition stopping, and explicit vLLM parallelism settings. Run `--help` for all controls.

For large files, `inference/batch_submit_tasks_with_params.sh` splits the JSONL and processes splits sequentially. `inference/batch_submit_8low_langs_all_models.sh` iterates over language directions and model paths without assuming a particular scheduler.

### 4. Score and filter trajectories

```bash
python inference/postprocess_translation.py \
  --input_path <inference-output-file-or-dir> \
  --top_ratio 0.2 \
  --num_workers <worker-count>
```

Postprocessing computes sentence-level spBLEU with the Flores tokenizer, validates or repairs `<think>...</think>` structure, selects the best valid generation per record, retains the requested top-scoring fraction, and emits ShareGPT data for final fine-tuning.

Use `inference/batch_postprocess_8low_langs.sh` to apply the same operation across a prepared language-direction tree. Use `inference/merge_ft_sharegpt.py` or `inference/merge_bidirectional_ft_sharegpt.py` to combine and reproducibly shuffle filtered files.

### 5. Train and evaluate the distilled model

Train the final model with FFT on the retained ShareGPT trajectories. For paper-style comparisons, keep the retained data size and final-model configuration fixed across specialist strategies. Evaluate both translation directions on the corresponding Flores-101 subsets and aggregate spBLEU with `util/mean_spbleu_subset.py`.

## Portability

Provide all dataset, model, and output locations for the current environment. Generated data, checkpoints, logs, and plots are excluded from version control. Review generated YAML files and set batch size, precision, sequence length, and distributed settings for your own software and hardware stack.
