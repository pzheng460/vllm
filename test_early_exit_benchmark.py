"""Benchmark: Parallel-SD with various early_exit_layer and top_k vs MTP baseline.

Tracks throughput, cache hit rate, and acceptance length for comparison.

Usage:
    # Default: 8 built-in prompts, spec_tokens=1
    CUDA_VISIBLE_DEVICES=0,1,2,3 python test_early_exit_benchmark.py

    # Speculative tokens = 3, gpqa_diamond dataset
    CUDA_VISIBLE_DEVICES=0,1,2,3 python test_early_exit_benchmark.py \
        --num-spec-tokens 3 --dataset gpqa_diamond --num-samples 50

    # Custom top-k and early-exit grid
    CUDA_VISIBLE_DEVICES=0,1,2,3 python test_early_exit_benchmark.py \
        --top-k 1 2 3 --early-exit -1 -3 -5

    # Include half-cache-hit and concurrent configs
    CUDA_VISIBLE_DEVICES=0,1,2,3 python test_early_exit_benchmark.py \
        --extra-configs

Configurations tested:
  1. MTP baseline (same num_speculative_tokens)
  2. Parallel-SD grid: early_exit × top_k combinations
  3. (optional) half_cache_hit and concurrent variants

Metrics compared:
  - Throughput (tok/s) and speedup vs MTP baseline
  - Mean acceptance length (includes bonus token) and ratio vs MTP
  - Cache hit rate (Parallel-SD only)
  - Output correctness vs MTP baseline
"""

import argparse
import gc
import json
import os
import tempfile
import time

import torch

# Stats file for cache hit collection from worker process
_CACHE_STATS_FILE = os.path.join(tempfile.gettempdir(),
                                  "vllm_cache_stats.json")
os.environ["VLLM_CACHE_STATS_FILE"] = _CACHE_STATS_FILE

# Spec decode acceptance stats file
_SPEC_STATS_FILE = os.path.join(tempfile.gettempdir(),
                                 "vllm_spec_stats.json")
os.environ["VLLM_SPEC_STATS_FILE"] = _SPEC_STATS_FILE

# Log stats frequently so acceptance metrics are flushed
os.environ["VLLM_LOG_STATS_INTERVAL"] = "1"

# Run EngineCore in-process so SpecDecodingStats flows to LoggingStatLogger
# without IPC serialization issues
os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"


# ---------------------------------------------------------------------------
# Dataset loaders
# ---------------------------------------------------------------------------

def get_builtin_prompts():
    """Diverse test prompts for benchmarking."""
    return [
        "Explain the theory of general relativity in detail.",
        "Write a Python function to implement a binary search tree with insert, delete, and search operations.",
        "What are the main differences between supervised, unsupervised, and reinforcement learning?",
        "Describe the process of photosynthesis step by step.",
        "Write a comprehensive guide on database normalization from 1NF to BCNF.",
        "Explain how a CPU pipeline works and what pipeline hazards are.",
        "What is the significance of the Turing machine in computer science?",
        "Describe the differences between TCP and UDP protocols with examples.",
    ]


def load_gpqa_diamond(num_samples=None):
    """Load GPQA Diamond dataset."""
    from datasets import load_dataset
    ds = load_dataset("fingertap/GPQA-Diamond", split="test")
    prompts = []
    for item in ds:
        q = item.get("Question", item.get("question", ""))
        choices = []
        for key in ["Correct Answer", "Incorrect Answer 1",
                    "Incorrect Answer 2", "Incorrect Answer 3"]:
            if key in item and item[key]:
                choices.append(item[key])
        prompt = f"{q}\n\nChoices:\n"
        for i, c in enumerate(choices):
            prompt += f"  {chr(65+i)}. {c}\n"
        prompt += "\nAnswer:"
        prompts.append(prompt)
    if num_samples:
        prompts = prompts[:num_samples]
    print(f"Loaded {len(prompts)} prompts from gpqa_diamond")
    return prompts


def load_gsm8k(num_samples=None):
    """Load GSM8K dataset."""
    from datasets import load_dataset
    ds = load_dataset("gsm8k", "main", split="test")
    prompts = [f"Solve: {item['question']}\nAnswer:" for item in ds]
    if num_samples:
        prompts = prompts[:num_samples]
    print(f"Loaded {len(prompts)} prompts from gsm8k")
    return prompts


def load_mt_bench(num_samples=None):
    """Load MT-Bench dataset."""
    from datasets import load_dataset
    ds = load_dataset("HuggingFaceH4/mt_bench_prompts", split="train")
    prompts = [item["prompt"][0] for item in ds]
    if num_samples:
        prompts = prompts[:num_samples]
    print(f"Loaded {len(prompts)} prompts from mt_bench")
    return prompts


DATASET_LOADERS = {
    "builtin": lambda n: get_builtin_prompts()[:n] if n else get_builtin_prompts(),
    "gpqa_diamond": load_gpqa_diamond,
    "gsm8k": load_gsm8k,
    "mt_bench": load_mt_bench,
}


def load_prompts(dataset_name, num_samples=None):
    """Load prompts from the specified dataset."""
    if dataset_name not in DATASET_LOADERS:
        raise ValueError(f"Unknown dataset: {dataset_name}. "
                         f"Available: {list(DATASET_LOADERS.keys())}")
    return DATASET_LOADERS[dataset_name](num_samples)


# ---------------------------------------------------------------------------
# Benchmark runner
# ---------------------------------------------------------------------------

def cleanup_gpu():
    """Force GPU memory cleanup between experiments."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def _reset_spec_cumulative(llm):
    """Reset cumulative spec decode stats in the logging pipeline."""
    try:
        mgr = llm.llm_engine.logger_manager
        if mgr is None:
            return
        for sl in mgr.stat_loggers:
            if hasattr(sl, 'per_engine_stat_loggers'):
                for pesl in sl.per_engine_stat_loggers.values():
                    if hasattr(pesl, 'spec_decoding_logging'):
                        pesl.spec_decoding_logging.reset_cumulative()
            elif hasattr(sl, 'spec_decoding_logging'):
                sl.spec_decoding_logging.reset_cumulative()
    except Exception:
        pass


def run_single_config(config_name, main_model, spec_config, prompts,
                      max_tokens=100, tp_size=4, max_model_len=4096,
                      gpu_mem=0.95, enforce_eager=True):
    """Run a single experiment configuration."""
    from vllm import LLM, SamplingParams

    # Clean stale stats files before run
    if os.path.exists(_CACHE_STATS_FILE):
        os.remove(_CACHE_STATS_FILE)
    if os.path.exists(_SPEC_STATS_FILE):
        os.remove(_SPEC_STATS_FILE)

    print(f"\n{'='*60}")
    print(f"Running: {config_name}")
    print(f"Spec config: {json.dumps(spec_config, indent=2)}")
    print(f"{'='*60}")

    extra_kwargs = {}
    if enforce_eager:
        extra_kwargs["enforce_eager"] = True

    llm = LLM(
        model=main_model,
        speculative_config=spec_config,
        gpu_memory_utilization=gpu_mem,
        max_model_len=max_model_len,
        tensor_parallel_size=tp_size,
        max_num_seqs=1,
        trust_remote_code=True,
        dtype="bfloat16",
        **extra_kwargs,
    )

    sampling_params = SamplingParams(
        max_tokens=max_tokens,
        temperature=0.0,
    )

    # Warmup run
    print("Warmup...")
    _ = llm.generate(prompts[:2], sampling_params)

    # Clean stale spec stats from warmup — force flush then remove
    try:
        llm.llm_engine.do_log_stats()
    except Exception:
        pass
    if os.path.exists(_SPEC_STATS_FILE):
        os.remove(_SPEC_STATS_FILE)
    # Reset cumulative acceptance counters so benchmark excludes warmup
    _reset_spec_cumulative(llm)

    # Benchmark run
    print("Benchmark run...")
    start = time.time()
    outputs = llm.generate(prompts, sampling_params)
    elapsed = time.time() - start

    total_tokens = sum(len(o.outputs[0].token_ids) for o in outputs)
    throughput = total_tokens / elapsed if elapsed > 0 else 0

    # Force final stats flush so acceptance metrics are written
    try:
        llm.llm_engine.do_log_stats()
    except Exception:
        pass

    # Get cache stats from file (written by worker process)
    spec_metrics = {}
    if os.path.exists(_CACHE_STATS_FILE):
        try:
            with open(_CACHE_STATS_FILE) as f:
                cache_stats = json.load(f)
            spec_metrics["cache_stats"] = {"global": cache_stats}
            os.remove(_CACHE_STATS_FILE)  # clean for next config
        except Exception:
            pass

    # Get spec decode acceptance stats from file
    if os.path.exists(_SPEC_STATS_FILE):
        try:
            with open(_SPEC_STATS_FILE) as f:
                acceptance_stats = json.load(f)
            spec_metrics["acceptance"] = acceptance_stats
            os.remove(_SPEC_STATS_FILE)  # clean for next config
        except Exception:
            pass

    output_texts = [o.outputs[0].text for o in outputs]

    # Make spec_config JSON-safe (filter out non-serializable objects)
    safe_spec_config = {}
    for k, v in spec_config.items():
        try:
            json.dumps(v)
            safe_spec_config[k] = v
        except (TypeError, ValueError):
            safe_spec_config[k] = str(v)

    result = {
        "config_name": config_name,
        "spec_config": safe_spec_config,
        "num_prompts": len(prompts),
        "total_output_tokens": total_tokens,
        "elapsed_seconds": round(elapsed, 3),
        "tokens_per_sec": round(throughput, 2),
        "spec_metrics": spec_metrics,
    }

    print(f"  Tokens: {total_tokens}")
    print(f"  Time: {elapsed:.3f}s")
    print(f"  Throughput: {throughput:.2f} tok/s")
    if spec_metrics.get("acceptance"):
        acc = spec_metrics["acceptance"]
        mal = acc.get("mean_acceptance_length", 0)
        dar = acc.get("draft_acceptance_rate", 0)
        print(f"  Mean acceptance length: {mal:.2f}")
        print(f"  Draft acceptance rate: {dar:.1f}%")
        per_pos = acc.get("per_position_acceptance_rate", [])
        if per_pos:
            rates_str = ", ".join(f"{r:.3f}" for r in per_pos)
            print(f"  Per-position acceptance: [{rates_str}]")

    if spec_metrics.get("cache_stats"):
        cs = spec_metrics["cache_stats"]
        if "global" in cs:
            g = cs["global"]
            total_lookups = g.get("hit", 0) + g.get("miss", 0) + g.get("half_hit", 0)
            if total_lookups > 0:
                hit_rate = g["hit"] / total_lookups * 100
                print(f"  Cache HIT rate: {hit_rate:.1f}% "
                      f"({g['hit']}/{total_lookups})")
                if g.get("half_hit", 0) > 0:
                    half_rate = g["half_hit"] / total_lookups * 100
                    print(f"  Half-HIT rate: {half_rate:.1f}%")

    del llm
    cleanup_gpu()

    return result, output_texts


def print_summary(all_results):
    """Print benchmark summary table."""
    print("\n" + "=" * 90)
    print("BENCHMARK SUMMARY")
    print("=" * 90)

    baseline_tps = all_results[0]["tokens_per_sec"]
    baseline_mal = all_results[0].get("spec_metrics", {}).get(
        "acceptance", {}).get("mean_acceptance_length", 0)

    print(f"\n{'Config':<35} {'tok/s':>8} {'vs MTP':>8}"
          f" {'Acc Len':>8} {'vs MTP':>8} {'Cache HIT':>10}")
    print("-" * 82)
    for r in all_results:
        name = r["config_name"]
        tps = r["tokens_per_sec"]
        speedup = tps / baseline_tps if baseline_tps > 0 else 0
        speedup_str = f"{speedup:.2f}x"

        # Acceptance length
        acc = r.get("spec_metrics", {}).get("acceptance", {})
        mal = acc.get("mean_acceptance_length", 0)
        mal_str = f"{mal:.2f}" if mal > 0 else "N/A"
        if mal > 0 and baseline_mal > 0:
            mal_ratio = mal / baseline_mal
            mal_cmp = f"{mal_ratio:.2f}x"
        else:
            mal_cmp = "N/A"

        # Cache hit
        cache_str = "N/A"
        cs = r.get("spec_metrics", {}).get("cache_stats", {})
        if "global" in cs:
            g = cs["global"]
            total = g.get("hit", 0) + g.get("miss", 0) + g.get("half_hit", 0)
            if total > 0:
                cache_str = f"{g['hit']/total*100:.1f}%"

        print(f"  {name:<33} {tps:>8.1f} {speedup_str:>8}"
              f" {mal_str:>8} {mal_cmp:>8} {cache_str:>10}")

    print()


def print_correctness(all_outputs):
    """Print correctness check vs MTP baseline."""
    baseline_texts = all_outputs.get("MTP baseline", [])
    if not baseline_texts:
        return
    print("CORRECTNESS CHECK (vs MTP baseline):")
    for cfg_name, texts in all_outputs.items():
        if cfg_name == "MTP baseline":
            continue
        matches = sum(1 for a, b in zip(baseline_texts, texts) if a == b)
        match_rate = matches / len(baseline_texts) * 100
        print(f"  {cfg_name}: {match_rate:.0f}% match "
              f"({matches}/{len(baseline_texts)})")
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Parallel-SD early-exit & top-k benchmark")
    parser.add_argument("--dataset", default="builtin",
                        choices=list(DATASET_LOADERS.keys()),
                        help="Dataset to use (default: builtin)")
    parser.add_argument("--num-samples", type=int, default=None,
                        help="Number of samples from dataset (default: all)")
    parser.add_argument("--max-tokens", type=int, default=100,
                        help="Max output tokens per prompt (default: 100)")
    parser.add_argument("--top-k", type=int, nargs="+", default=[1, 2, 3],
                        help="Top-k values to test (default: 1 2 3)")
    parser.add_argument("--early-exit", type=int, nargs="+", default=[-3],
                        help="Early-exit layer indices (default: -3)")
    parser.add_argument("--tp-size", type=int, default=4,
                        help="Tensor parallel size (default: 4)")
    parser.add_argument("--model", default="/mnt/data/weights/openPangu-R-72B-2512",
                        help="Main model path")
    parser.add_argument("--num-spec-tokens", type=int, default=1,
                        help="Number of speculative tokens (default: 1)")
    parser.add_argument("--gpu-mem", type=float, default=0.95,
                        help="GPU memory utilization (default: 0.95)")
    parser.add_argument("--max-model-len", type=int, default=4096,
                        help="Maximum model context length (default: 4096)")
    parser.add_argument("--extra-configs", action="store_true",
                        help="Also test half-cache-hit and concurrent variants")
    parser.add_argument("--save", default="early_exit_benchmark_results.json",
                        help="Output JSON file (default: early_exit_benchmark_results.json)")
    parser.add_argument("--no-targeted-branch", action="store_true",
                        help="Disable targeted branch to measure pure early-exit hit rate")
    args = parser.parse_args()

    if args.no_targeted_branch:
        os.environ["VLLM_DISABLE_TARGETED_BRANCH"] = "1"

    main_model = args.model
    prompts = load_prompts(args.dataset, args.num_samples)
    tp_size = args.tp_size
    num_spec_tokens = args.num_spec_tokens

    print("=" * 60)
    print("Parallel-SD Early-Exit & Top-K Benchmark")
    print(f"  Model:      {main_model}")
    print(f"  TP size:    {tp_size}")
    print(f"  Dataset:    {args.dataset} ({len(prompts)} prompts)")
    print(f"  Max tokens: {args.max_tokens}")
    print(f"  Spec tokens: {num_spec_tokens}")
    print(f"  Top-k grid: {args.top_k}")
    print(f"  Early-exit grid: {args.early_exit}")
    print(f"  Extra configs: {args.extra_configs}")
    print("=" * 60)

    all_results = []
    all_outputs = {}

    # ---- Config 0: MTP baseline ----
    cfg_mtp = {
        "method": "mtp",
        "num_speculative_tokens": num_spec_tokens,
    }
    r, texts = run_single_config(
        "MTP baseline",
        main_model, cfg_mtp, prompts,
        max_tokens=args.max_tokens, tp_size=tp_size,
        gpu_mem=args.gpu_mem, max_model_len=args.max_model_len,
    )
    all_results.append(r)
    all_outputs["MTP baseline"] = texts

    # ---- Grid: early_exit × top_k ----
    for ee in args.early_exit:
        for k in args.top_k:
            name = f"Parallel L={ee} k={k}"
            cfg = {
                "method": "parallel",
                "model": main_model,
                "num_speculative_tokens": num_spec_tokens,
                "draft_tensor_parallel_size": 1,
                "parallel_draft_method": "mtp",
                "parallel_top_k": k,
                "parallel_early_exit_layer": ee,
                "parallel_enable_half_cache_hit": False,
            }
            r, texts = run_single_config(
                name, main_model, cfg, prompts,
                max_tokens=args.max_tokens, tp_size=tp_size,
                gpu_mem=args.gpu_mem, max_model_len=args.max_model_len,
            )
            all_results.append(r)
            all_outputs[name] = texts

    # ---- Extra configs (optional) ----
    if args.extra_configs:
        ee0 = args.early_exit[0]  # use first early-exit value

        # half-cache-hit with k=1
        name = f"Parallel L={ee0} k=1 half-hit"
        cfg = {
            "method": "parallel",
            "model": main_model,
            "num_speculative_tokens": num_spec_tokens,
            "draft_tensor_parallel_size": 1,
            "parallel_draft_method": "mtp",
            "parallel_top_k": 1,
            "parallel_early_exit_layer": ee0,
            "parallel_enable_half_cache_hit": True,
        }
        r, texts = run_single_config(
            name, main_model, cfg, prompts,
            max_tokens=args.max_tokens, tp_size=tp_size,
            gpu_mem=args.gpu_mem, max_model_len=args.max_model_len,
        )
        all_results.append(r)
        all_outputs[name] = texts

        # concurrent with k=2
        name = f"Parallel L={ee0} k=2 concurrent"
        cfg = {
            "method": "parallel",
            "model": main_model,
            "num_speculative_tokens": num_spec_tokens,
            "draft_tensor_parallel_size": 1,
            "parallel_draft_method": "mtp",
            "parallel_top_k": 2,
            "parallel_early_exit_layer": ee0,
            "parallel_enable_half_cache_hit": False,
            "parallel_enable_concurrent": True,
        }
        r, texts = run_single_config(
            name, main_model, cfg, prompts,
            max_tokens=args.max_tokens, tp_size=tp_size,
            gpu_mem=args.gpu_mem, max_model_len=args.max_model_len,
        )
        all_results.append(r)
        all_outputs[name] = texts

    # ---- Summary ----
    print_summary(all_results)
    print_correctness(all_outputs)

    # ---- Save ----
    save_data = {
        "config": {
            "model": main_model,
            "dataset": args.dataset,
            "num_prompts": len(prompts),
            "max_tokens": args.max_tokens,
            "tp_size": tp_size,
            "num_spec_tokens": num_spec_tokens,
            "top_k_grid": args.top_k,
            "early_exit_grid": args.early_exit,
        },
        "results": all_results,
    }
    with open(args.save, "w") as f:
        json.dump(save_data, f, indent=2)
    print(f"Results saved to {args.save}")


if __name__ == "__main__":
    main()
