#!/bin/bash

# GPU Configuration
# Usage:
#   Single GPU: bash run_rqmoe.sh [GPU_ID]           # e.g., bash run_rqmoe.sh 0
#   Multi GPU:  bash run_rqmoe.sh --gpus 0,1,2,3     # Use specific GPUs
#   Default:    bash run_rqmoe.sh                    # Use GPU 0

# Parse arguments
if [ "$1" == "--gpus" ] && [ -n "$2" ]; then
    GPUS="$2"
    if [[ "$GPUS" == *","* ]]; then
        USE_DDP=true
        echo "Multi-GPU DDP mode: Using GPUs ${GPUS}"
    else
        GPU_ID="$GPUS"
        DEVICE="cuda:${GPU_ID}"
        USE_DDP=false
        echo "Single GPU mode: Using device ${DEVICE}"
    fi
else
    GPU_ID=${1:-0}
    DEVICE="cuda:${GPU_ID}"
    GPUS="${GPU_ID}"
    USE_DDP=false
    echo "Single GPU mode: Using device ${DEVICE}"
fi

M=4
N=2
L=2
DATASET="bigann1M"
BATCH_SIZE=4096
K=256
H=256
MAX_EPOCHS=1000
RQ_BEAM_SIZE=1
NUM_TRAIN=500_000
NUM_VAL=10_000
LR=1e-3
LR_PATIENCE=10
DROPOUT=0.2
MODEL_PATH="checkpoint/model.pt"
CHECKPOINT_PATH="checkpoint/checkpoint.pt"

mkdir -p checkpoint

CHECKPOINT="${CHECKPOINT_PATH}"

TRAIN_CMD="python -u train_rqmoe.py \
    --K ${K} \
    --M ${M} \
    --N ${N} \
    --L ${L} \
    --H ${H} \
    --dropout ${DROPOUT} \
    --rq_beam_size ${RQ_BEAM_SIZE} \
    --lr ${LR} \
    --lr_patience ${LR_PATIENCE} \
    --batch_size ${BATCH_SIZE} \
    --max_epochs ${MAX_EPOCHS} \
    --dataset ${DATASET} \
    --model ${MODEL_PATH} \
    --nt ${NUM_TRAIN} \
    --nval ${NUM_VAL}"

if [ -n "${CHECKPOINT}" ]; then
    TRAIN_CMD="${TRAIN_CMD} --checkpoint ${CHECKPOINT}"
    if [ -f "${CHECKPOINT}" ]; then
        echo "Resuming from checkpoint: ${CHECKPOINT}"
    else
        echo "Starting new training, checkpoint will be saved to: ${CHECKPOINT}"
    fi
fi

if [ "$USE_DDP" = true ]; then
    TRAIN_CMD="${TRAIN_CMD} --gpus ${GPUS}"
else
    TRAIN_CMD="${TRAIN_CMD} --device ${DEVICE}"
fi

eval ${TRAIN_CMD}
