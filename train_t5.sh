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

--micro-batch-size 8
--global-batch-size 16

--train-data-path /mnt/nfs/bert_data/bert_wikitext103/bert_wikitext103_text_document
--tokenizer-type GPT2BPETokenizer
--vocab-file /home/ubuntu/lzj/tokenizers/gpt2/vocab.json
--merge-file /home/ubuntu/lzj/tokenizers/gpt2/merges.txt

--optimizer adam
--lr 1e-4
--train-iters 50
--log-interval 10
--eval-interval 1000000000
--eval-iters 0

--tensor-model-parallel-size 1
--pipeline-model-parallel-size 1

--distributed-backend nccl
"
# MoE / Expert Parallelism args (small-scale smoke-test defaults)
export CUDA_DEVICE_MAX_CONNECTIONS=1
MOE_ARGS="
--num-experts 4
--expert-model-parallel-size 1
--moe-router-topk 2
--moe-router-load-balancing-type aux_loss
--moe-aux-loss-coeff 1e-2
--moe-token-dispatcher-type alltoall
--sequence-parallel
"
torchrun \
    --nnodes=1 \
    --nproc_per_node=1 \
    --master_addr=127.0.0.1 \
    --master_port=29500 \
    pretrain_t5.py \
    $COMMON_ARGS \
    $MOE_ARGS