export VLLM_TORCH_COMPILE=0
# or any of these depending on your entrypoint:
export TORCH_COMPILE_DISABLE=1
export TORCHDYNAMO_DISABLE=1

CUDA_VISIBLE_DEVICES=0,1 \
vllm serve /home/test/weights/gpt-oss-120b \
    --host 0.0.0.0 \
    --port 8300 \
    --tensor-parallel-size 4 \
    --max-num-seqs 1 \
    --tokenizer /home/test/weights/gpt-oss-120b \
    --dtype bfloat16 \
    --max-model-len 4096 \
    --trust_remote_code \
    --gpu_memory_utilization 0.9 \
    --block_size 128 \
    --served-model-name gptoss \
    --max-num-batched-tokens 20000 \
    --max-num-seqs 128 2>&1 | tee api_server_gptoss.log
