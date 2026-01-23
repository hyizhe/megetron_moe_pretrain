#!/usr/bin/env bash
# Direct-launch helper that bypasses torchrun rendezvous.
# Use this when torchrun rendezvous fails; both machines must be able to reach
# MASTER_ADDR:MASTER_PORT and have identical code and envs.
# Usage (run on master):
#   RANK=0 WORLD_SIZE=2 LOCAL_RANK=0 MASTER_ADDR=10.90.1.11 MASTER_PORT=29500 bash train_t5_direct.sh
# On worker:
#   RANK=1 WORLD_SIZE=2 LOCAL_RANK=0 MASTER_ADDR=10.90.1.11 MASTER_PORT=29500 bash train_t5_direct.sh

set -euo pipefail

RANK=${RANK:-0}
WORLD_SIZE=${WORLD_SIZE:-1}
LOCAL_RANK=${LOCAL_RANK:-0}
MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
MASTER_PORT=${MASTER_PORT:-29500}

export RANK WORLD_SIZE LOCAL_RANK MASTER_ADDR MASTER_PORT
# Also export legacy vars some code expects
export WORLD_SIZE
export RANK
export LOCAL_RANK
export LOCAL_WORLD_SIZE=1
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

# Useful NCCL debug / network options for multi-node debugging
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=1
# export NCCL_SOCKET_IFNAME=eth0   # uncomment & set to your interface if needed

echo "Starting direct launch: RANK=${RANK} WORLD_SIZE=${WORLD_SIZE} MASTER_ADDR=${MASTER_ADDR} MASTER_PORT=${MASTER_PORT} LOCAL_RANK=${LOCAL_RANK}"

# Run the training script directly. Megatron's init should pick up env vars and call init_process_group accordingly.
python3 -u pretrain_t5.py \
    --local-rank ${LOCAL_RANK} \
    --rank ${RANK} \
    --world-size ${WORLD_SIZE} \
    --master-addr ${MASTER_ADDR} \
    --master-port ${MASTER_PORT}
