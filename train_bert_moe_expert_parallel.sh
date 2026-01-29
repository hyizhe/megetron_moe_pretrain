#!/usr/bin/env bash

# ========================== CLUSTER CONFIG ==========================
MASTER_ADDR=10.90.1.11
MASTER_PORT=29500
NODE_RANK=${NODE_RANK:-0}

NNODES=2
NPROC_PER_NODE=1

# ========================== NCCL ==========================
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=1
export NCCL_SOCKET_IFNAME=enp6s18
export NCCL_ASYNC_ERROR_HANDLING=1
export CUDA_DEVICE_MAX_CONNECTIONS=1

# ========================== COMMON ARGS ==========================
COMMON_ARGS="
--use-mcore-models
--transformer-impl local
--normalization LayerNorm
--no-persist-layer-norm


--spec local


--no-gradient-accumulation-fusion
--no-bias-gelu-fusion
--no-bias-dropout-fusion
--no-bias-swiglu-fusion
--no-masked-softmax-fusion


--num-layers 32
--hidden-size 1024
--num-attention-heads 8
--seq-length 1024
--max-position-embeddings 1024

--micro-batch-size 8
--global-batch-size 16

--tokenizer-type BertWordPieceLowerCase
--vocab-file vocab.txt
--legacy-tokenizer
--optimizer adam
--lr 5e-5
--clip-grad 1.0
--bf16
--train-iters 2000
--log-interval 10
--eval-interval 500
--eval-iters 10
--tensor-model-parallel-size 1
--pipeline-model-parallel-size 1


--num-experts 8
--expert-model-parallel-size 2
--moe-router-topk 2
--moe-router-load-balancing-type aux_loss
--moe-aux-loss-coeff 0.01
--moe-token-dispatcher-type alltoall
--distributed-backend nccl
--train-data-path /mnt/nfs/bert_data/bert_wikitext103/bert_wikitext103_text_document
--bert-no-binary-head
--rerun-mode disabled
--save /home/ubuntu/hyz/ckpts/bert_moe_test
--save-interval 2000
"


echo "Starting BERT + MoE (HF tokenizer)"
echo "NODE_RANK=$NODE_RANK"

# ========================== RUN WITH NOHUP ==========================
# nohup torchrun \
#   --nnodes=$NNODES \
#   --nproc_per_node=$NPROC_PER_NODE \
#   --node_rank=$NODE_RANK \
#   --master_addr=$MASTER_ADDR \
#   --master_port=$MASTER_PORT \
#   pretrain_bert.py \
#   $COMMON_ARGS \
#   > train.log 2>&1 &
  
torchrun \
  --nnodes=$NNODES \
  --nproc_per_node=$NPROC_PER_NODE \
  --node_rank=$NODE_RANK \
  --master_addr=$MASTER_ADDR \
  --master_port=$MASTER_PORT \
  pretrain_bert.py \
  $COMMON_ARGS
# echo "Training started in background. Logs are in train.log"
