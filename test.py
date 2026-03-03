import subprocess
import json
import os

def start_vllm():
    main_model_path = "/mnt/data/weights/openPangu-R-72B-2512"
    spec_model_path = "/mnt/data/weights/openPangu-R-72B-2512"

    served_model_name = "pangu"
    spec_cfg = {
        "method": "mtp",
        "model": spec_model_path,
        "num_speculative_tokens": 1
    }

    command = [
        "vllm", "serve", main_model_path,
        "--host", "0.0.0.0",
        "--port", "8300",
        "--dtype", "bfloat16",
        "--tensor-parallel-size", "4",
        "--max-num-seqs", "1",
        "--tokenizer", main_model_path,
        "--gpu-memory-utilization", "0.95",
        "--served-model-name", served_model_name,
        "--speculative-config", json.dumps(spec_cfg),
        "--max-model-len", "4096",
        "--trust-remote-code"
    ]

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
    env["VLLM_TORCH_COMPILE"] = "0"
    env["TORCH_COMPILE_DISABLE"] = "1"
    env["TORCHDYNAMO_DISABLE"] = "1"

    with open("vllm_serve.log", "w") as f:
        subprocess.run(command, env=env, stdout=f, stderr=subprocess.STDOUT)

if __name__ == "__main__":
    start_vllm()