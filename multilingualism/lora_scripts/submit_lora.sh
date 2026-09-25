#!/bin/bash
#
# General-purpose LLaMA-Factory LoRA launcher
#
# Usage:
#   bash submit_lora.sh --model /path/to/model --data /path/to/data.jsonl [OPTIONS]
#
# Examples:
#   # Basic usage
#   bash submit_lora.sh \
#     --model /path/to/model \
#     --data /path/to/aligned_ft_data.jsonl
#
#   # Specify additional options
#   bash submit_lora.sh \
#     --model /path/to/model \
#     --data /path/to/aligned_ft_data.jsonl \
#     --dataset-name my_dataset \
#     --job-name my_lora \
#     --epochs 3 \
#     --lr 1e-4 \
#     --lora-rank 128 \
#     --cutoff-len 4096
#

set -e

show_usage() {
    cat << EOF
Usage: $0 --model MODEL_PATH --data DATA_FILE [OPTIONS]

Required arguments:
  --model PATH              Path to the pretrained model
  --data PATH               Path to ShareGPT JSONL training data

Dataset options:
  --dataset-name NAME       Dataset name (default: derived from file name)
  --template NAME           Conversation template (default: qwen3)
                            qwen3: includes <think> tags
                            qwen3_nothink: excludes <think> tags

LoRA options:
  --lora-rank N             LoRA rank (default: 128)
  --lora-alpha N            LoRA alpha (default: 2 * lora_rank)
  --lora-target NAME        LoRA target modules (default: all linear layers)
  --lora-dropout FLOAT      LoRA dropout (default: 0.05)
  --rslora                  Enable RSLoRA (Rank-Stabilized LoRA)
  --dora                    Enable DoRA (Weight-Decomposed LoRA)

Training hyperparameters:
  --epochs N                Training epochs (default: 1)
  --lr FLOAT                Learning rate (default: 1e-4)
  --batch-size N            Batch size per device (default: 1)
  --grad-accum N            Gradient accumulation steps (default: 8)
  --cutoff-len N            Maximum sequence length (default: 4096)
  --warmup-ratio FLOAT      Warmup ratio (default: 0.03)
  --weight-decay FLOAT      Weight decay (default: 0.01)
  --lr-scheduler TYPE       Learning-rate scheduler (default: cosine)
  --val-size FLOAT          Validation split ratio (for example 0.05; disabled by default)

Optimization options:
  --flash-attn TYPE         Flash Attention (default: fa2)
                            Choices: disabled, sdpa, fa2
  --no-gradient-checkpoint  Disable gradient checkpointing
  --packing                 Enable dataset packing
  --no-plot-loss            Disable loss plotting
  --deepspeed PATH          DeepSpeed configuration path (disabled by default)

Logging and checkpoints:
  --logging-steps N         Logging interval in steps (default: 10)
  --save-steps N            Checkpoint interval in steps (default: 500)
  --save-total-limit N      Maximum retained checkpoints (default: 3)

Output options:
  --output-dir PATH         Output directory (default: generated automatically)
  --output-base DIR         Output root (default: lora_outputs)
  --job-name NAME           Run name (default: generated automatically)

Other:
  --help                    Show this help message

Examples:
  # Basic usage
  $0 --model /path/to/model --data /path/to/data.jsonl

  # Full configuration
  $0 \\
    --model /path/to/model \\
    --data /path/to/data.jsonl \\
    --dataset-name my_dataset \\
    --epochs 3 \\
    --lr 1e-4 \\
    --lora-rank 64 \\
    --cutoff-len 8192

EOF
}

# ============================================================
# Project paths
# ============================================================

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
LORA_SCRIPTS_DIR="${SCRIPT_DIR}"
FT_SCRIPTS_DIR="${PROJECT_ROOT}/ft_scripts"
DATA_PREPARE_SCRIPT="${FT_SCRIPTS_DIR}/data_prepare.py"
GENERATE_YAML_SCRIPT="${LORA_SCRIPTS_DIR}/generate_lora_yaml.py"

# Reuse the LLaMA-Factory dataset registry in ft_scripts.
LLAMA_FACTORY_DATA_DIR="${FT_SCRIPTS_DIR}"
DATASET_INFO_PATH="${LLAMA_FACTORY_DATA_DIR}/dataset_info.json"

# Optional environment activation. Leave both variables empty to use the current environment.
CONDA_ACTIVATE="${CONDA_ACTIVATE:-}"
CONDA_ENV="${CONDA_ENV:-}"

# ============================================================
# Default arguments
# ============================================================

# Required model and data paths
MODEL_PATH=""
DATA_FILE=""

# Dataset options
DATASET_NAME=""
TEMPLATE="qwen3"

# LoRA options
LORA_RANK=128
LORA_ALPHA=""  # defaults to 2 * lora_rank in the Python generator
LORA_TARGET="all"
LORA_DROPOUT=0.05
USE_RSLORA="false"
USE_DORA="false"

# Training hyperparameters
NUM_TRAIN_EPOCHS=1
LEARNING_RATE="1e-4"
PER_DEVICE_TRAIN_BATCH_SIZE=1
GRADIENT_ACCUMULATION_STEPS=8
CUTOFF_LEN=4096
WARMUP_RATIO=0.03
WEIGHT_DECAY=0.01
LR_SCHEDULER="cosine"
LR_SCHEDULER_KWARGS=""

# Validation
VAL_SIZE=""

# Optimization
FLASH_ATTN="fa2"
GRADIENT_CHECKPOINTING="true"
PACKING="false"
PLOT_LOSS="true"
DEEPSPEED_CONFIG=""

# Logging and checkpoints
LOGGING_STEPS=10
SAVE_STEPS=50000
SAVE_TOTAL_LIMIT=3

# Output options
OUTPUT_BASE_DIR="${PROJECT_ROOT}/lora_outputs"
JOB_NAME=""
OUTPUT_DIR=""

# ============================================================
# Argument parsing
# ============================================================

while [[ $# -gt 0 ]]; do
    case $1 in
        --model)
            MODEL_PATH="$2"
            shift 2
            ;;
        --data)
            DATA_FILE="$2"
            shift 2
            ;;
        --dataset-name)
            DATASET_NAME="$2"
            shift 2
            ;;
        --template)
            TEMPLATE="$2"
            shift 2
            ;;
        --lora-rank)
            LORA_RANK="$2"
            shift 2
            ;;
        --lora-alpha)
            LORA_ALPHA="$2"
            shift 2
            ;;
        --lora-target)
            LORA_TARGET="$2"
            shift 2
            ;;
        --lora-dropout)
            LORA_DROPOUT="$2"
            shift 2
            ;;
        --rslora)
            USE_RSLORA="true"
            shift
            ;;
        --dora)
            USE_DORA="true"
            shift
            ;;
        --epochs)
            NUM_TRAIN_EPOCHS="$2"
            shift 2
            ;;
        --lr)
            LEARNING_RATE="$2"
            shift 2
            ;;
        --batch-size)
            PER_DEVICE_TRAIN_BATCH_SIZE="$2"
            shift 2
            ;;
        --grad-accum)
            GRADIENT_ACCUMULATION_STEPS="$2"
            shift 2
            ;;
        --cutoff-len)
            CUTOFF_LEN="$2"
            shift 2
            ;;
        --warmup-ratio)
            WARMUP_RATIO="$2"
            shift 2
            ;;
        --weight-decay)
            WEIGHT_DECAY="$2"
            shift 2
            ;;
        --lr-scheduler)
            LR_SCHEDULER="$2"
            shift 2
            ;;
        --val-size)
            VAL_SIZE="$2"
            shift 2
            ;;
        --flash-attn)
            FLASH_ATTN="$2"
            shift 2
            ;;
        --no-gradient-checkpoint)
            GRADIENT_CHECKPOINTING="false"
            shift
            ;;
        --packing)
            PACKING="true"
            shift
            ;;
        --no-plot-loss)
            PLOT_LOSS="false"
            shift
            ;;
        --deepspeed)
            DEEPSPEED_CONFIG="$2"
            shift 2
            ;;
        --logging-steps)
            LOGGING_STEPS="$2"
            shift 2
            ;;
        --save-steps)
            SAVE_STEPS="$2"
            shift 2
            ;;
        --save-total-limit)
            SAVE_TOTAL_LIMIT="$2"
            shift 2
            ;;
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --output-base)
            OUTPUT_BASE_DIR="$2"
            shift 2
            ;;
        --job-name)
            JOB_NAME="$2"
            shift 2
            ;;
        --help)
            show_usage
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            show_usage
            exit 1
            ;;
    esac
done

# ============================================================
# Argument validation
# ============================================================

if [ -z "$MODEL_PATH" ]; then
    echo "Error: --model is required"
    show_usage
    exit 1
fi

if [ -z "$DATA_FILE" ]; then
    echo "Error: --data is required"
    show_usage
    exit 1
fi

if [ ! -d "$MODEL_PATH" ]; then
    echo "Error: model directory not found: $MODEL_PATH"
    exit 1
fi

if [ ! -f "$DATA_FILE" ]; then
    echo "Error: data file not found: $DATA_FILE"
    exit 1
fi

# ============================================================
# Generate names and paths
# ============================================================

TIMESTAMP=$(date +%Y%m%d_%H%M%S)

MODEL_NAME=$(basename "$MODEL_PATH")

if [ -z "$DATASET_NAME" ]; then
    DATA_BASENAME=$(basename "$DATA_FILE" | sed 's/\.jsonl$//' | sed 's/\.json$//')
    DATASET_NAME="${DATA_BASENAME}"
fi

CONFIG_DESC="${DATASET_NAME}_lora${LORA_RANK}_cutoff${CUTOFF_LEN}"

if [ -z "$OUTPUT_DIR" ]; then
    OUTPUT_DIR="${OUTPUT_BASE_DIR}/${MODEL_NAME}_template-${TEMPLATE}/${CONFIG_DESC}_${TIMESTAMP}"
fi

if [ -z "$JOB_NAME" ]; then
    JOB_NAME="lora_${MODEL_NAME}_${DATASET_NAME}_${TIMESTAMP}"
fi

YAML_OUTPUT_PATH="${OUTPUT_DIR}/training.yaml"

mkdir -p "$OUTPUT_DIR"

# ============================================================
# Print the configuration.
# ============================================================

echo "============================================================"
echo "LLaMA-Factory LoRA Fine-tuning configuration"
echo "============================================================"
echo "Timestamp: $TIMESTAMP"
echo ""
echo "Model:"
echo "  Model path: $MODEL_PATH"
echo "  Model name: $MODEL_NAME"
echo "  Conversation template: $TEMPLATE"
echo ""
echo "Dataset:"
echo "  Data file: $DATA_FILE"
echo "  Dataset name: $DATASET_NAME"
echo ""
echo "LoRA options:"
echo "  LoRA rank: $LORA_RANK"
if [ -n "$LORA_ALPHA" ]; then
    echo "  LoRA Alpha: $LORA_ALPHA"
else
    echo "  LoRA alpha: $((LORA_RANK * 2)) (default: 2*rank)"
fi
echo "  LoRA target: $LORA_TARGET"
echo "  LoRA Dropout: $LORA_DROPOUT"
echo "  RSLoRA: $USE_RSLORA"
echo "  DoRA: $USE_DORA"
echo ""
echo "Training hyperparameters:"
echo "  Training epochs: $NUM_TRAIN_EPOCHS"
echo "  Learning rate: $LEARNING_RATE"
echo "  Batch size: $PER_DEVICE_TRAIN_BATCH_SIZE (per device)"
echo "  Gradient accumulation: $GRADIENT_ACCUMULATION_STEPS"
echo "  Maximum length: $CUTOFF_LEN"
echo "  Warmup ratio: $WARMUP_RATIO"
echo "  Weight decay: $WEIGHT_DECAY"
echo "  Learning-rate scheduler: $LR_SCHEDULER"
if [ -n "$VAL_SIZE" ]; then
    echo "  Validation split: $VAL_SIZE"
fi
echo ""
echo "Optimization options:"
echo "  Flash Attention: $FLASH_ATTN"
echo "  Gradient checkpointing: $GRADIENT_CHECKPOINTING"
echo "  Dataset packing: $PACKING"
if [ -n "$DEEPSPEED_CONFIG" ]; then
    echo "  DeepSpeed: $DEEPSPEED_CONFIG"
else
    echo "  DeepSpeed: disabled"
fi
echo ""
echo "Output options:"
echo "  Run name: $JOB_NAME"
echo "  Output directory: $OUTPUT_DIR"
echo "  YAML configuration: $YAML_OUTPUT_PATH"
echo ""
echo "============================================================"
echo ""

# ============================================================
# Build the training script.
# ============================================================

TRAIN_SCRIPT="${OUTPUT_DIR}/run_training.sh"

cat > "$TRAIN_SCRIPT" << 'SCRIPT_END'
#!/bin/bash
set -e

# Activate an optional environment.
if [ -n "CONDA_ACTIVATE_PLACEHOLDER" ] && [ -n "CONDA_ENV_PLACEHOLDER" ]; then
    source CONDA_ACTIVATE_PLACEHOLDER CONDA_ENV_PLACEHOLDER
fi

# Set runtime environment variables.
export PYTORCH_ALLOC_CONF=expandable_segments:False
export HF_HUB_OFFLINE=1
export WANDB_DISABLED=True
export DISABLE_VERSION_CHECK=1
export NCCL_DEBUG=WARN

echo "============================================================"
echo "Step 1: Prepare data"
echo "============================================================"

mkdir -p "LLAMA_FACTORY_DATA_DIR_PLACEHOLDER"

if [ ! -f "DATASET_INFO_PATH_PLACEHOLDER" ] || [ ! -s "DATASET_INFO_PATH_PLACEHOLDER" ]; then
    echo "{}" > "DATASET_INFO_PATH_PLACEHOLDER"
    echo "Initialized dataset_info.json"
fi

python "DATA_PREPARE_SCRIPT_PLACEHOLDER" \
    --data_file "DATA_FILE_PLACEHOLDER" \
    --dataset_name "DATASET_NAME_PLACEHOLDER" \
    --dataset_info_path "DATASET_INFO_PATH_PLACEHOLDER"

echo ""
echo "============================================================"
echo "Step 2: Generate the training configuration"
echo "============================================================"

YAML_ARGS=(
    --model_path "MODEL_PATH_PLACEHOLDER"
    --dataset "DATASET_NAME_PLACEHOLDER"
    --dataset_dir "LLAMA_FACTORY_DATA_DIR_PLACEHOLDER"
    --output_dir "OUTPUT_DIR_PLACEHOLDER"
    --yaml_output_path "YAML_OUTPUT_PATH_PLACEHOLDER"
    --template "TEMPLATE_PLACEHOLDER"
    --num_train_epochs NUM_TRAIN_EPOCHS_PLACEHOLDER
    --learning_rate LEARNING_RATE_PLACEHOLDER
    --per_device_train_batch_size PER_DEVICE_TRAIN_BATCH_SIZE_PLACEHOLDER
    --gradient_accumulation_steps GRADIENT_ACCUMULATION_STEPS_PLACEHOLDER
    --cutoff_len CUTOFF_LEN_PLACEHOLDER
    --warmup_ratio WARMUP_RATIO_PLACEHOLDER
    --weight_decay WEIGHT_DECAY_PLACEHOLDER
    --lr_scheduler_type "LR_SCHEDULER_PLACEHOLDER"
    --flash_attn "FLASH_ATTN_PLACEHOLDER"
    --logging_steps LOGGING_STEPS_PLACEHOLDER
    --save_steps SAVE_STEPS_PLACEHOLDER
    --save_total_limit SAVE_TOTAL_LIMIT_PLACEHOLDER
    --lora_rank LORA_RANK_PLACEHOLDER
    --lora_target "LORA_TARGET_PLACEHOLDER"
    --lora_dropout LORA_DROPOUT_PLACEHOLDER
)

# Optional arguments.
LORA_ALPHA_FLAG
LR_SCHEDULER_KWARGS_FLAG
GRADIENT_CHECKPOINTING_FLAG
PACKING_FLAG
PLOT_LOSS_FLAG
VAL_SIZE_FLAG
DEEPSPEED_FLAG
USE_RSLORA_FLAG
USE_DORA_FLAG

python "GENERATE_YAML_SCRIPT_PLACEHOLDER" "${YAML_ARGS[@]}"

echo ""
echo "============================================================"
echo "Step 3: Start training"
echo "============================================================"

llamafactory-cli train "YAML_OUTPUT_PATH_PLACEHOLDER"

echo ""
echo "============================================================"
echo "Training complete!"
echo "============================================================"
echo "Model saved to: OUTPUT_DIR_PLACEHOLDER"

SCRIPT_END

# Replace placeholders.
sed -i "s|CONDA_ACTIVATE_PLACEHOLDER|$CONDA_ACTIVATE|g" "$TRAIN_SCRIPT"
sed -i "s|CONDA_ENV_PLACEHOLDER|$CONDA_ENV|g" "$TRAIN_SCRIPT"
sed -i "s|LLAMA_FACTORY_DATA_DIR_PLACEHOLDER|$LLAMA_FACTORY_DATA_DIR|g" "$TRAIN_SCRIPT"
sed -i "s|DATASET_INFO_PATH_PLACEHOLDER|$DATASET_INFO_PATH|g" "$TRAIN_SCRIPT"
sed -i "s|DATA_PREPARE_SCRIPT_PLACEHOLDER|$DATA_PREPARE_SCRIPT|g" "$TRAIN_SCRIPT"
sed -i "s|DATA_FILE_PLACEHOLDER|$DATA_FILE|g" "$TRAIN_SCRIPT"
sed -i "s|DATASET_NAME_PLACEHOLDER|$DATASET_NAME|g" "$TRAIN_SCRIPT"
sed -i "s|MODEL_PATH_PLACEHOLDER|$MODEL_PATH|g" "$TRAIN_SCRIPT"
sed -i "s|OUTPUT_DIR_PLACEHOLDER|$OUTPUT_DIR|g" "$TRAIN_SCRIPT"
sed -i "s|YAML_OUTPUT_PATH_PLACEHOLDER|$YAML_OUTPUT_PATH|g" "$TRAIN_SCRIPT"
sed -i "s|TEMPLATE_PLACEHOLDER|$TEMPLATE|g" "$TRAIN_SCRIPT"
sed -i "s|NUM_TRAIN_EPOCHS_PLACEHOLDER|$NUM_TRAIN_EPOCHS|g" "$TRAIN_SCRIPT"
sed -i "s|LEARNING_RATE_PLACEHOLDER|$LEARNING_RATE|g" "$TRAIN_SCRIPT"
sed -i "s|PER_DEVICE_TRAIN_BATCH_SIZE_PLACEHOLDER|$PER_DEVICE_TRAIN_BATCH_SIZE|g" "$TRAIN_SCRIPT"
sed -i "s|GRADIENT_ACCUMULATION_STEPS_PLACEHOLDER|$GRADIENT_ACCUMULATION_STEPS|g" "$TRAIN_SCRIPT"
sed -i "s|CUTOFF_LEN_PLACEHOLDER|$CUTOFF_LEN|g" "$TRAIN_SCRIPT"
sed -i "s|WARMUP_RATIO_PLACEHOLDER|$WARMUP_RATIO|g" "$TRAIN_SCRIPT"
sed -i "s|WEIGHT_DECAY_PLACEHOLDER|$WEIGHT_DECAY|g" "$TRAIN_SCRIPT"
sed -i "s|LR_SCHEDULER_PLACEHOLDER|$LR_SCHEDULER|g" "$TRAIN_SCRIPT"
sed -i "s|FLASH_ATTN_PLACEHOLDER|$FLASH_ATTN|g" "$TRAIN_SCRIPT"
sed -i "s|LOGGING_STEPS_PLACEHOLDER|$LOGGING_STEPS|g" "$TRAIN_SCRIPT"
sed -i "s|SAVE_STEPS_PLACEHOLDER|$SAVE_STEPS|g" "$TRAIN_SCRIPT"
sed -i "s|SAVE_TOTAL_LIMIT_PLACEHOLDER|$SAVE_TOTAL_LIMIT|g" "$TRAIN_SCRIPT"
sed -i "s|GENERATE_YAML_SCRIPT_PLACEHOLDER|$GENERATE_YAML_SCRIPT|g" "$TRAIN_SCRIPT"
sed -i "s|LORA_RANK_PLACEHOLDER|$LORA_RANK|g" "$TRAIN_SCRIPT"
sed -i "s|LORA_TARGET_PLACEHOLDER|$LORA_TARGET|g" "$TRAIN_SCRIPT"
sed -i "s|LORA_DROPOUT_PLACEHOLDER|$LORA_DROPOUT|g" "$TRAIN_SCRIPT"

# Replace optional flags.
if [ -n "$LORA_ALPHA" ]; then
    sed -i "s|LORA_ALPHA_FLAG|YAML_ARGS+=(--lora_alpha $LORA_ALPHA)|g" "$TRAIN_SCRIPT"
else
    sed -i "s|LORA_ALPHA_FLAG|# lora_alpha uses default (2 * lora_rank)|g" "$TRAIN_SCRIPT"
fi

if [ -n "$LR_SCHEDULER_KWARGS" ]; then
    sed -i "s|LR_SCHEDULER_KWARGS_FLAG|YAML_ARGS+=(--lr_scheduler_kwargs '$LR_SCHEDULER_KWARGS')|g" "$TRAIN_SCRIPT"
else
    sed -i "s|LR_SCHEDULER_KWARGS_FLAG|# no lr_scheduler_kwargs|g" "$TRAIN_SCRIPT"
fi

if [ "$GRADIENT_CHECKPOINTING" = "true" ]; then
    sed -i "s|GRADIENT_CHECKPOINTING_FLAG|YAML_ARGS+=(--gradient_checkpointing)|g" "$TRAIN_SCRIPT"
else
    sed -i "s|GRADIENT_CHECKPOINTING_FLAG|# gradient_checkpointing disabled|g" "$TRAIN_SCRIPT"
fi

if [ "$PACKING" = "true" ]; then
    sed -i "s|PACKING_FLAG|YAML_ARGS+=(--packing)|g" "$TRAIN_SCRIPT"
else
    sed -i "s|PACKING_FLAG|# packing disabled|g" "$TRAIN_SCRIPT"
fi

if [ "$PLOT_LOSS" = "true" ]; then
    sed -i "s|PLOT_LOSS_FLAG|YAML_ARGS+=(--plot_loss)|g" "$TRAIN_SCRIPT"
else
    sed -i "s|PLOT_LOSS_FLAG|# plot_loss disabled|g" "$TRAIN_SCRIPT"
fi

if [ -n "$VAL_SIZE" ]; then
    sed -i "s|VAL_SIZE_FLAG|YAML_ARGS+=(--val_size $VAL_SIZE)|g" "$TRAIN_SCRIPT"
else
    sed -i "s|VAL_SIZE_FLAG|# no validation set|g" "$TRAIN_SCRIPT"
fi

if [ -n "$DEEPSPEED_CONFIG" ]; then
    sed -i "s|DEEPSPEED_FLAG|YAML_ARGS+=(--deepspeed \"$DEEPSPEED_CONFIG\")|g" "$TRAIN_SCRIPT"
else
    sed -i "s|DEEPSPEED_FLAG|# deepspeed disabled|g" "$TRAIN_SCRIPT"
fi

if [ "$USE_RSLORA" = "true" ]; then
    sed -i "s|USE_RSLORA_FLAG|YAML_ARGS+=(--use_rslora)|g" "$TRAIN_SCRIPT"
else
    sed -i "s|USE_RSLORA_FLAG|# rslora disabled|g" "$TRAIN_SCRIPT"
fi

if [ "$USE_DORA" = "true" ]; then
    sed -i "s|USE_DORA_FLAG|YAML_ARGS+=(--use_dora)|g" "$TRAIN_SCRIPT"
else
    sed -i "s|USE_DORA_FLAG|# dora disabled|g" "$TRAIN_SCRIPT"
fi

chmod +x "$TRAIN_SCRIPT"

echo "Training script generated: $TRAIN_SCRIPT"
echo ""

# ============================================================
# Run locally in the current environment
# ============================================================

echo "Starting training: $TRAIN_SCRIPT"
bash "$TRAIN_SCRIPT"
