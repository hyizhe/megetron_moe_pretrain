#!/usr/bin/env bash

# ========================== CONFIG ==========================
# 两台机器配置

MASTER_ADDR=10.90.1.11

MASTER_PORT=29500

NODE_RANK=${NODE_RANK:-0}


NNODES=3
# 每节点 GPU 数量，这里每台1张卡
NPROC_PER_NODE=1

# ========================== NCCL / GPU ==========================
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=1
export NCCL_SOCKET_IFNAME=enp6s18   
export NCCL_ASYNC_ERROR_HANDLING=1
export CUDA_DEVICE_MAX_CONNECTIONS=1

COMMON_ARGS="
--use-mcore-models
--transformer-impl local
--normalization LayerNorm
--no-persist-layer-norm

--no-gradient-accumulation-fusion
--no-bias-gelu-fusion
--no-bias-dropout-fusion
--no-bias-swiglu-fusion
--no-masked-softmax-fusion

--num-layers 32
--hidden-size 1024
--num-attention-heads 8
--seq-length 1024
--decoder-seq-length 1024
--max-position-embeddings 1024
--bf16

--micro-batch-size 4
--global-batch-size 12

--train-data-path /mnt/nfs/bert_data/bert_wikitext103/bert_wikitext103_text_document
--tokenizer-type GPT2BPETokenizer
--vocab-file /home/ubuntu/lzj/tokenizers/gpt2/vocab.json
--merge-file /home/ubuntu/lzj/tokenizers/gpt2/merges.txt

--optimizer adam
--lr 1e-4
--train-iters 300
--log-interval 10
--eval-interval 1000000000
--eval-iters 0

--tensor-model-parallel-size 1
--pipeline-model-parallel-size 1

--num-experts 9
--expert-model-parallel-size 3
--moe-router-topk 2
--moe-router-load-balancing-type aux_loss
--moe-aux-loss-coeff 0.1
--moe-token-dispatcher-type alltoall

--distributed-backend nccl
"


echo "Starting T5 + MoE training"
echo "NODE_RANK=$NODE_RANK, MASTER_ADDR=$MASTER_ADDR, MASTER_PORT=$MASTER_PORT"

torchrun \
  --nnodes=$NNODES \
  --nproc_per_node=$NPROC_PER_NODE \
  --node_rank=$NODE_RANK \
  --master_addr=$MASTER_ADDR \
  --master_port=$MASTER_PORT \
  pretrain_t5.py \
  $COMMON_ARGS
