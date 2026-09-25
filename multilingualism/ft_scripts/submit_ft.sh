#!/bin/bash
#
# General-purpose LLaMA-Factory fine-tuning launcher
#
# Usage:
#   bash submit_ft.sh --model /path/to/model --data /path/to/data.jsonl [OPTIONS]
#
# Examples:
#   # Basic usage
#   bash submit_ft.sh \
#     --model /path/to/model \
#     --data /path/to/aligned_ft_data.jsonl
#
#   # Specify additional options
#   bash submit_ft.sh \
#     --model /path/to/model \
#     --data /path/to/aligned_ft_data.jsonl \
#     --dataset-name my_dataset \
#     --job-name my_finetune \
#     --epochs 3 \
#     --lr 1e-5 \
#     --batch-size 2 \
#     --grad-accum 8 \
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
  --template NAME           Conversation template (default: qwen3_nothink)
                            qwen3: includes <think> tags
                            qwen3_nothink: excludes <think> tags

Training hyperparameters:
  --epochs N                Training epochs (default: 1)
  --lr FLOAT                Learning rate (default: 1e-5)
  --batch-size N            Batch size per device (default: 1)
  --grad-accum N            Gradient accumulation steps (default: 16)
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

Logging and checkpoints:
  --logging-steps N         Logging interval in steps (default: 10)
  --save-steps N            Checkpoint interval in steps (default: 500)
  --save-total-limit N      Maximum retained checkpoints (default: 3)

Output options:
  --output-dir PATH         Output directory (default: generated automatically)
  --output-base DIR         Output root (default: ft_outputs)
  --job-name NAME           Run name (default: generated automatically)

Layer-selective tuning (optional):
  --trainable-layers N      Number of trainable bottom layers
  --top-trainable-layers N  Number of trainable top layers
  --specific-layers LIST    Specific trainable layers (comma-separated)
  --freeze-extra-modules L  Additional modules to freeze (comma-separated)

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
    --lr 1e-5 \\
    --cutoff-len 8192

EOF
}

# ============================================================
# Project paths
# ============================================================

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
FT_SCRIPTS_DIR="${SCRIPT_DIR}"
DATA_PREPARE_SCRIPT="${FT_SCRIPTS_DIR}/data_prepare.py"
GENERATE_YAML_SCRIPT="${FT_SCRIPTS_DIR}/generate_training_yaml.py"
DEEPSPEED_CONFIG="${FT_SCRIPTS_DIR}/ds_z3_config.json"

# LLaMA-Factory dataset registry.
LLAMA_FACTORY_DATA_DIR="${FT_SCRIPTS_DIR}"
DATASET_INFO_PATH="${LLAMA_FACTORY_DATA_DIR}/dataset_info.json"

# Optional environment activation.

CONDA_ACTIVATE="${CONDA_ACTIVATE:-}"
CONDA_ENV="${CONDA_ENV:-}"

# ============================================================
# Default arguments
# ============================================================

# Required model and data paths
MODEL_PATH=""
DATA_FILE=""

# Dataset options
DATASET_NAME=""  # derived from the file name when empty
TEMPLATE="qwen3"  # qwen3 (with <think>) or qwen3_nothink (without <think>)

# Training hyperparameters
NUM_TRAIN_EPOCHS=1
LEARNING_RATE="1e-5"
PER_DEVICE_TRAIN_BATCH_SIZE=1
GRADIENT_ACCUMULATION_STEPS=8
CUTOFF_LEN=4096
WARMUP_RATIO=0.03
WEIGHT_DECAY=0.01
LR_SCHEDULER="cosine_with_min_lr"
LR_SCHEDULER_KWARGS='{"min_lr": 2.0e-6}'

# Validation
VAL_SIZE=""  # for example, 0.05 means 5%

# Optimization
FLASH_ATTN="fa2"
GRADIENT_CHECKPOINTING="true"
PACKING="false"
PLOT_LOSS="true"

# Logging and checkpoints
LOGGING_STEPS=10
SAVE_STEPS=50000
SAVE_TOTAL_LIMIT=0

# Output options
OUTPUT_BASE_DIR="${PROJECT_ROOT}/ft_outputs"
JOB_NAME=""
OUTPUT_DIR=""

# Optional layer-selective tuning
TRAINABLE_LAYERS=""
TOP_TRAINABLE_LAYERS=""
SPECIFIC_LAYERS=""
FREEZE_EXTRA_MODULES=""

# ============================================================
# Argument parsing
# ============================================================



# Parse command-line arguments.
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
        --trainable-layers)
            TRAINABLE_LAYERS="$2"
            shift 2
            ;;
        --top-trainable-layers)
            TOP_TRAINABLE_LAYERS="$2"
            shift 2
            ;;
        --specific-layers)
            SPECIFIC_LAYERS="$2"
            shift 2
            ;;
        --freeze-extra-modules)
            FREEZE_EXTRA_MODULES="$2"
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

# Timestamp.
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# Model name.
MODEL_NAME=$(basename "$MODEL_PATH")

# Derive the dataset name from the file name when omitted.
if [ -z "$DATASET_NAME" ]; then
    DATA_BASENAME=$(basename "$DATA_FILE" | sed 's/\.jsonl$//' | sed 's/\.json$//')
    DATASET_NAME="${DATA_BASENAME}"
fi

# Build a description used in output and run names.
CONFIG_DESC="${DATASET_NAME}_cutoff${CUTOFF_LEN}"

# Generate the output directory when omitted.
if [ -z "$OUTPUT_DIR" ]; then
    OUTPUT_DIR="${OUTPUT_BASE_DIR}/${MODEL_NAME}_template-${TEMPLATE}/${CONFIG_DESC}_${TIMESTAMP}"
fi

# Generate the run name when omitted.
if [ -z "$JOB_NAME" ]; then
    JOB_NAME="ft_${MODEL_NAME}_${DATASET_NAME}_${TIMESTAMP}"
fi

# YAML output path.
YAML_OUTPUT_PATH="${OUTPUT_DIR}/training.yaml"

# Create the output directory.
mkdir -p "$OUTPUT_DIR"

# ============================================================
# Print the configuration.
# ============================================================

echo "============================================================"
echo "LLaMA-Factory Fine-tuning configuration"
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
echo ""
echo "Output options:"
echo "  Run name: $JOB_NAME"
echo "  Output directory: $OUTPUT_DIR"
echo "  YAML configuration: $YAML_OUTPUT_PATH"
echo ""

# Layer-selective tuning details
if [ -n "$TRAINABLE_LAYERS" ] || [ -n "$TOP_TRAINABLE_LAYERS" ] || [ -n "$SPECIFIC_LAYERS" ]; then
    echo "Layer-selective Tuning:"
    [ -n "$TRAINABLE_LAYERS" ] && echo "  Trainable bottom layers: $TRAINABLE_LAYERS"
    [ -n "$TOP_TRAINABLE_LAYERS" ] && echo "  Trainable top layers: $TOP_TRAINABLE_LAYERS"
    [ -n "$SPECIFIC_LAYERS" ] && echo "  Specific trainable layers: $SPECIFIC_LAYERS"
    [ -n "$FREEZE_EXTRA_MODULES" ] && echo "  Additional frozen modules: $FREEZE_EXTRA_MODULES"
    echo ""
fi

echo "============================================================"
echo ""

# ============================================================
# Build the training script.
# ============================================================

# Create the training script.
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

# Create the dataset directory.
mkdir -p "LLAMA_FACTORY_DATA_DIR_PLACEHOLDER"

# Initialize dataset_info.json.
# Create an empty registry when the file is missing or empty.
if [ ! -f "DATASET_INFO_PATH_PLACEHOLDER" ] || [ ! -s "DATASET_INFO_PATH_PLACEHOLDER" ]; then
    echo "{}" > "DATASET_INFO_PATH_PLACEHOLDER"
    echo "Initialized dataset_info.json"
fi

# Register the dataset.
python "DATA_PREPARE_SCRIPT_PLACEHOLDER" \
    --data_file "DATA_FILE_PLACEHOLDER" \
    --dataset_name "DATASET_NAME_PLACEHOLDER" \
    --dataset_info_path "DATASET_INFO_PATH_PLACEHOLDER"

echo ""
echo "============================================================"
echo "Step 2: Generate the training configuration"
echo "============================================================"

# Build generate_yaml arguments.
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
    --lr_scheduler_kwargs 'LR_SCHEDULER_KWARGS_PLACEHOLDER'
    --flash_attn "FLASH_ATTN_PLACEHOLDER"
    --deepspeed "DEEPSPEED_CONFIG_PLACEHOLDER"
    --logging_steps LOGGING_STEPS_PLACEHOLDER
    --save_steps SAVE_STEPS_PLACEHOLDER
    --save_total_limit SAVE_TOTAL_LIMIT_PLACEHOLDER
)

# Add optional arguments.
GRADIENT_CHECKPOINTING_FLAG
PACKING_FLAG
PLOT_LOSS_FLAG
VAL_SIZE_FLAG

# Layer-selective tuning arguments.
TRAINABLE_LAYERS_FLAG
TOP_TRAINABLE_LAYERS_FLAG
SPECIFIC_LAYERS_FLAG
FREEZE_EXTRA_MODULES_FLAG

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
sed -i "s|LR_SCHEDULER_KWARGS_PLACEHOLDER|$LR_SCHEDULER_KWARGS|g" "$TRAIN_SCRIPT"
sed -i "s|FLASH_ATTN_PLACEHOLDER|$FLASH_ATTN|g" "$TRAIN_SCRIPT"
sed -i "s|DEEPSPEED_CONFIG_PLACEHOLDER|$DEEPSPEED_CONFIG|g" "$TRAIN_SCRIPT"
sed -i "s|LOGGING_STEPS_PLACEHOLDER|$LOGGING_STEPS|g" "$TRAIN_SCRIPT"
sed -i "s|SAVE_STEPS_PLACEHOLDER|$SAVE_STEPS|g" "$TRAIN_SCRIPT"
sed -i "s|SAVE_TOTAL_LIMIT_PLACEHOLDER|$SAVE_TOTAL_LIMIT|g" "$TRAIN_SCRIPT"
sed -i "s|GENERATE_YAML_SCRIPT_PLACEHOLDER|$GENERATE_YAML_SCRIPT|g" "$TRAIN_SCRIPT"

# Replace optional flags.
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

# Layer-selective tuning flags.
# Replace longer placeholders first to avoid partial matches.
# TOP_TRAINABLE_LAYERS_FLAG contains TRAINABLE_LAYERS_FLAG and must be replaced first.
if [ -n "$TOP_TRAINABLE_LAYERS" ]; then
    sed -i "s|TOP_TRAINABLE_LAYERS_FLAG|YAML_ARGS+=(--top_trainable_layers $TOP_TRAINABLE_LAYERS)|g" "$TRAIN_SCRIPT"
else
    sed -i "s|TOP_TRAINABLE_LAYERS_FLAG|# no top_trainable_layers|g" "$TRAIN_SCRIPT"
fi

if [ -n "$FREEZE_EXTRA_MODULES" ]; then
    sed -i "s|FREEZE_EXTRA_MODULES_FLAG|YAML_ARGS+=(--freeze_extra_modules \"$FREEZE_EXTRA_MODULES\")|g" "$TRAIN_SCRIPT"
else
    sed -i "s|FREEZE_EXTRA_MODULES_FLAG|# no freeze_extra_modules|g" "$TRAIN_SCRIPT"
fi

if [ -n "$SPECIFIC_LAYERS" ]; then
    sed -i "s|SPECIFIC_LAYERS_FLAG|YAML_ARGS+=(--specific_layers \"$SPECIFIC_LAYERS\")|g" "$TRAIN_SCRIPT"
else
    sed -i "s|SPECIFIC_LAYERS_FLAG|# no specific_layers|g" "$TRAIN_SCRIPT"
fi

if [ -n "$TRAINABLE_LAYERS" ]; then
    sed -i "s|TRAINABLE_LAYERS_FLAG|YAML_ARGS+=(--trainable_layers $TRAINABLE_LAYERS)|g" "$TRAIN_SCRIPT"
else
    sed -i "s|TRAINABLE_LAYERS_FLAG|# no trainable_layers|g" "$TRAIN_SCRIPT"
fi

chmod +x "$TRAIN_SCRIPT"

echo "Training script generated: $TRAIN_SCRIPT"
echo ""

# ============================================================
# Run locally in the current environment
# ============================================================

echo "Starting training: $TRAIN_SCRIPT"
bash "$TRAIN_SCRIPT"
