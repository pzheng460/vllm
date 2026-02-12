"""End-to-end test: Parallel-SD vs MTP baseline on Pangu 72B.

Compares outputs at temperature=0 to verify correctness.
With top_k=1 (default), cache hit rate should be ~100%,
so Parallel-SD should produce identical outputs.
"""

import os
import json
import time

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
os.environ["VLLM_TORCH_COMPILE"] = "0"
os.environ["TORCH_COMPILE_DISABLE"] = "1"
os.environ["TORCHDYNAMO_DISABLE"] = "1"

MODEL_PATH = "/mnt/data/weights/openPangu-R-72B-2512"
TP_SIZE = 4
MAX_MODEL_LEN = 4096
GPU_MEM = 0.95
NUM_SPEC_TOKENS = 1  # Pangu has 1 MTP layer

PROMPTS = [
    "Hello, how are you today?",
    "What is the capital of France?",
    "Explain the theory of relativity in simple terms.",
    "Write a Python function to compute fibonacci numbers.",
]

from vllm import LLM, SamplingParams

sp = SamplingParams(temperature=0, max_tokens=100)

# ------------------------------------------------------------------
# Step 1: MTP baseline
# ------------------------------------------------------------------
print("=" * 60)
print("Step 1: MTP baseline")
print("=" * 60)

t0 = time.time()
llm_mtp = LLM(
    model=MODEL_PATH,
    speculative_config={
        "method": "mtp",
        "model": MODEL_PATH,
        "num_speculative_tokens": NUM_SPEC_TOKENS,
        "draft_tensor_parallel_size": 1,
    },
    dtype="bfloat16",
    tensor_parallel_size=TP_SIZE,
    max_num_seqs=1,
    max_model_len=MAX_MODEL_LEN,
    gpu_memory_utilization=GPU_MEM,
    trust_remote_code=True,
)
mtp_load_time = time.time() - t0
print(f"MTP model loaded in {mtp_load_time:.1f}s")

t0 = time.time()
mtp_outputs = llm_mtp.generate(PROMPTS, sp)
mtp_gen_time = time.time() - t0
print(f"MTP generation done in {mtp_gen_time:.1f}s")

mtp_texts = [o.outputs[0].text for o in mtp_outputs]
for i, text in enumerate(mtp_texts):
    print(f"\n[MTP prompt {i}]: {PROMPTS[i][:50]}...")
    print(f"[MTP output {i}]: {text[:200]}")

del llm_mtp
import gc
import torch
gc.collect()
torch.cuda.empty_cache()
print("\nMTP model unloaded, GPU memory released.\n")

# ------------------------------------------------------------------
# Step 2: Parallel-SD + MTP
# ------------------------------------------------------------------
print("=" * 60)
print("Step 2: Parallel-SD + MTP")
print("=" * 60)

t0 = time.time()
llm_parallel = LLM(
    model=MODEL_PATH,
    speculative_config={
        "method": "parallel",
        "model": MODEL_PATH,
        "num_speculative_tokens": NUM_SPEC_TOKENS,
        "draft_tensor_parallel_size": 1,
        "parallel_draft_method": "mtp",
        "parallel_top_k": 1,
    },
    dtype="bfloat16",
    tensor_parallel_size=TP_SIZE,
    max_num_seqs=1,
    max_model_len=MAX_MODEL_LEN,
    gpu_memory_utilization=GPU_MEM,
    trust_remote_code=True,
)
parallel_load_time = time.time() - t0
print(f"Parallel model loaded in {parallel_load_time:.1f}s")

t0 = time.time()
parallel_outputs = llm_parallel.generate(PROMPTS, sp)
parallel_gen_time = time.time() - t0
print(f"Parallel generation done in {parallel_gen_time:.1f}s")

parallel_texts = [o.outputs[0].text for o in parallel_outputs]
for i, text in enumerate(parallel_texts):
    print(f"\n[Parallel prompt {i}]: {PROMPTS[i][:50]}...")
    print(f"[Parallel output {i}]: {text[:200]}")

# ------------------------------------------------------------------
# Step 3: Compare
# ------------------------------------------------------------------
print("\n" + "=" * 60)
print("Comparison Results")
print("=" * 60)

all_match = True
for i in range(len(PROMPTS)):
    match = mtp_texts[i] == parallel_texts[i]
    status = "MATCH" if match else "MISMATCH"
    print(f"Prompt {i}: {status}")
    if not match:
        all_match = False
        print(f"  MTP:      {mtp_texts[i][:100]}...")
        print(f"  Parallel: {parallel_texts[i][:100]}...")

print(f"\nOverall: {'ALL MATCH' if all_match else 'SOME MISMATCH'}")
print(f"MTP gen time:      {mtp_gen_time:.2f}s")
print(f"Parallel gen time: {parallel_gen_time:.2f}s")

# Try to get cache stats
if hasattr(llm_parallel, 'llm_engine'):
    engine = llm_parallel.llm_engine
    # Attempt to retrieve cache statistics via workers
    print("\n(Cache statistics would be available in the worker process)")

del llm_parallel
print("\nDone.")
