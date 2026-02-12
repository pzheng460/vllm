"""Parallel Speculative Decoding Experiment Script.

Benchmark and compare Parallel-SD against standard Eagle/MTP baselines
using various datasets.

Usage:
    # GPQA Diamond with different top-k values
    python examples/parallel_spec_decode_experiment.py \\
        --dataset gpqa_diamond --top-k 1 3 5

    # Compare with Eagle baseline
    python examples/parallel_spec_decode_experiment.py \\
        --dataset gsm8k --compare-eagle --num-samples 100

    # MTP mode
    python examples/parallel_spec_decode_experiment.py \\
        --main-model deepseek-ai/DeepSeek-V3 \\
        --draft-method mtp --dataset custom

    # Local dataset
    python examples/parallel_spec_decode_experiment.py \\
        --dataset local --local-path /path/to/data.jsonl \\
        --question-key prompt
"""

import argparse
import csv
import json
import os
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Dataset loaders
# ---------------------------------------------------------------------------

def load_gpqa_diamond(num_samples=None):
    """Load GPQA Diamond dataset."""
    try:
        from datasets import load_dataset
        ds = load_dataset("fingertap/GPQA-Diamond", split="train")
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
        return prompts
    except Exception as e:
        print(f"Failed to load gpqa_diamond: {e}")
        return _fallback_prompts(num_samples)


def load_gpqa_diamond_orig(num_samples=None):
    """Load GPQA Diamond original (requires auth)."""
    try:
        from datasets import load_dataset
        ds = load_dataset("Idavidrein/gpqa", "gpqa_diamond", split="train")
        prompts = [item.get("Question", "") for item in ds]
        if num_samples:
            prompts = prompts[:num_samples]
        return prompts
    except Exception as e:
        print(f"Failed to load gpqa_diamond_orig: {e}")
        return _fallback_prompts(num_samples)


def load_gsm8k(num_samples=None):
    """Load GSM8K dataset."""
    try:
        from datasets import load_dataset
        ds = load_dataset("gsm8k", "main", split="test")
        prompts = [
            f"Solve: {item['question']}\nAnswer:"
            for item in ds
        ]
        if num_samples:
            prompts = prompts[:num_samples]
        return prompts
    except Exception as e:
        print(f"Failed to load gsm8k: {e}")
        return _fallback_prompts(num_samples)


def load_humaneval(num_samples=None):
    """Load HumanEval dataset."""
    try:
        from datasets import load_dataset
        ds = load_dataset("openai/openai_humaneval", split="test")
        prompts = [item["prompt"] for item in ds]
        if num_samples:
            prompts = prompts[:num_samples]
        return prompts
    except Exception as e:
        print(f"Failed to load humaneval: {e}")
        return _fallback_prompts(num_samples)


def load_mmlu(num_samples=None):
    """Load MMLU dataset."""
    try:
        from datasets import load_dataset
        ds = load_dataset("cais/mmlu", "all", split="test")
        prompts = []
        for item in ds:
            q = item["question"]
            choices = item["choices"]
            prompt = f"{q}\n"
            for i, c in enumerate(choices):
                prompt += f"  {chr(65+i)}. {c}\n"
            prompt += "Answer:"
            prompts.append(prompt)
        if num_samples:
            prompts = prompts[:num_samples]
        return prompts
    except Exception as e:
        print(f"Failed to load mmlu: {e}")
        return _fallback_prompts(num_samples)


def load_mt_bench(num_samples=None):
    """Load MT-Bench dataset."""
    try:
        from datasets import load_dataset
        ds = load_dataset("HuggingFaceH4/mt_bench_prompts", split="train")
        prompts = [item["prompt"][0] for item in ds]
        if num_samples:
            prompts = prompts[:num_samples]
        return prompts
    except Exception as e:
        print(f"Failed to load mt_bench: {e}")
        return _fallback_prompts(num_samples)


def load_custom(num_samples=None):
    """Simple test prompts for quick testing."""
    prompts = [
        "Explain quantum computing in simple terms.",
        "Write a Python function to sort a list using merge sort.",
        "What are the main causes of climate change?",
        "Describe the difference between TCP and UDP protocols.",
        "Write a haiku about artificial intelligence.",
        "Explain the concept of recursion with an example.",
        "What is the theory of relativity?",
        "Write a SQL query to find duplicate records in a table.",
    ]
    if num_samples:
        prompts = prompts[:num_samples]
    return prompts


def load_local(path, question_key="question", num_samples=None):
    """Load dataset from a local file.

    Supports: JSON, JSONL, CSV, Parquet, directories with parquet files.
    """
    path = Path(path)

    if path.is_dir():
        # Scan for parquet files
        parquet_files = list(path.glob("*.parquet"))
        if not parquet_files:
            raise FileNotFoundError(
                f"No parquet files found in {path}"
            )
        try:
            import pandas as pd
            dfs = [pd.read_parquet(f) for f in parquet_files]
            df = pd.concat(dfs, ignore_index=True)
            prompts = df[question_key].tolist()
        except Exception as e:
            raise RuntimeError(f"Failed to load parquet dir: {e}") from e
    elif path.suffix == ".jsonl":
        prompts = []
        with open(path) as f:
            for line in f:
                item = json.loads(line.strip())
                prompts.append(item[question_key])
    elif path.suffix == ".json":
        with open(path) as f:
            data = json.load(f)
        if isinstance(data, list):
            prompts = [item[question_key] for item in data]
        elif isinstance(data, dict):
            # Try common keys
            for key in ["data", "samples", "items", "prompts"]:
                if key in data:
                    prompts = [item[question_key] for item in data[key]]
                    break
            else:
                raise ValueError(
                    f"JSON object has no recognized data key. "
                    f"Found: {list(data.keys())}"
                )
    elif path.suffix == ".csv":
        prompts = []
        with open(path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                prompts.append(row[question_key])
    elif path.suffix == ".parquet":
        try:
            import pandas as pd
            df = pd.read_parquet(path)
            prompts = df[question_key].tolist()
        except Exception as e:
            raise RuntimeError(f"Failed to load parquet: {e}") from e
    else:
        raise ValueError(f"Unsupported file format: {path.suffix}")

    if num_samples:
        prompts = prompts[:num_samples]
    return prompts


def _fallback_prompts(num_samples=None):
    """Fallback prompts when dataset loading fails."""
    prompts = load_custom()
    if num_samples:
        prompts = prompts[:num_samples]
    return prompts


DATASET_LOADERS = {
    "gpqa_diamond": load_gpqa_diamond,
    "gpqa_diamond_orig": load_gpqa_diamond_orig,
    "gsm8k": load_gsm8k,
    "humaneval": load_humaneval,
    "mmlu": load_mmlu,
    "mt_bench": load_mt_bench,
    "custom": load_custom,
}


def load_dataset_prompts(dataset_name, local_path=None,
                         question_key="question", num_samples=None):
    """Load prompts from the specified dataset."""
    if dataset_name == "local":
        if local_path is None:
            raise ValueError(
                "--local-path is required when --dataset=local"
            )
        return load_local(local_path, question_key, num_samples)

    # Auto-detect local file paths
    if os.path.exists(dataset_name):
        return load_local(dataset_name, question_key, num_samples)

    if dataset_name in DATASET_LOADERS:
        return DATASET_LOADERS[dataset_name](num_samples)

    raise ValueError(
        f"Unknown dataset: {dataset_name}. "
        f"Available: {list(DATASET_LOADERS.keys())} or 'local'"
    )


# ---------------------------------------------------------------------------
# Experiment runner
# ---------------------------------------------------------------------------

def run_experiment(
    main_model,
    draft_model,
    prompts,
    method,
    num_speculative_tokens,
    max_tokens,
    top_k=1,
    draft_method="eagle",
    enable_half_cache_hit=False,
    gpu_memory_utilization=0.9,
    max_model_len=4096,
    seed=42,
    system_prompt=None,
):
    """Run a single experiment configuration."""
    from vllm import LLM, SamplingParams

    spec_config = {
        "method": method,
        "num_speculative_tokens": num_speculative_tokens,
    }

    if method == "parallel":
        spec_config["parallel_top_k"] = top_k
        spec_config["parallel_draft_method"] = draft_method
        spec_config["parallel_enable_half_cache_hit"] = enable_half_cache_hit
        if draft_method != "mtp" and draft_model:
            spec_config["model"] = draft_model
    elif method in ("eagle", "eagle3"):
        if draft_model:
            spec_config["model"] = draft_model
    elif method == "mtp":
        pass  # MTP uses target model

    llm = LLM(
        model=main_model,
        speculative_config=spec_config,
        gpu_memory_utilization=gpu_memory_utilization,
        max_model_len=max_model_len,
        seed=seed,
    )

    sampling_params = SamplingParams(
        max_tokens=max_tokens,
        temperature=0.0,
    )

    # Build prompts with optional system prompt
    if system_prompt:
        formatted = [
            f"System: {system_prompt}\n\nUser: {p}\n\nAssistant:"
            for p in prompts
        ]
    else:
        formatted = prompts

    start_time = time.time()
    outputs = llm.generate(formatted, sampling_params)
    elapsed = time.time() - start_time

    # Collect results
    total_tokens = sum(len(o.outputs[0].token_ids) for o in outputs)
    throughput = total_tokens / elapsed if elapsed > 0 else 0

    # Try to get cache statistics from the speculator
    cache_stats = None
    try:
        if hasattr(llm, 'llm_engine'):
            engine = llm.llm_engine
            if hasattr(engine, 'model_executor'):
                executor = engine.model_executor
                # Navigate to the speculator
                if hasattr(executor, 'driver_worker'):
                    worker = executor.driver_worker
                    if hasattr(worker, 'model_runner'):
                        runner = worker.model_runner
                        if hasattr(runner, 'speculator'):
                            spec = runner.speculator
                            if hasattr(spec, 'get_cache_statistics'):
                                cache_stats = spec.get_cache_statistics()
    except Exception:
        pass

    result = {
        "method": method,
        "top_k": top_k,
        "draft_method": draft_method,
        "num_prompts": len(prompts),
        "total_tokens": total_tokens,
        "elapsed_seconds": elapsed,
        "throughput_tokens_per_sec": throughput,
        "avg_tokens_per_prompt": total_tokens / len(prompts),
        "cache_statistics": cache_stats,
    }

    # Cleanup
    del llm

    return result, outputs


def print_results(results):
    """Print experiment results in a formatted table."""
    print("\n" + "=" * 80)
    print("EXPERIMENT RESULTS")
    print("=" * 80)

    for r in results:
        print(f"\nMethod: {r['method']}", end="")
        if r["method"] == "parallel":
            print(f" (top_k={r['top_k']}, "
                  f"draft={r['draft_method']})", end="")
        print()
        print(f"  Prompts: {r['num_prompts']}")
        print(f"  Total tokens: {r['total_tokens']}")
        print(f"  Time: {r['elapsed_seconds']:.2f}s")
        print(f"  Throughput: {r['throughput_tokens_per_sec']:.1f} tok/s")
        print(f"  Avg tokens/prompt: {r['avg_tokens_per_prompt']:.1f}")
        if r.get("cache_statistics"):
            cs = r["cache_statistics"]
            if "global" in cs:
                g = cs["global"]
                total = g["hit"] + g["miss"] + g["half_hit"]
                hit_rate = g["hit"] / total * 100 if total > 0 else 0
                print(f"  Cache HIT rate: {hit_rate:.1f}% "
                      f"({g['hit']}/{total})")
                if g["half_hit"] > 0:
                    print(f"  Half-HIT: {g['half_hit']}")

    print("\n" + "=" * 80)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Parallel Speculative Decoding Experiment"
    )
    parser.add_argument(
        "--dataset", default="gpqa_diamond",
        help="Dataset: gpqa_diamond, gpqa_diamond_orig, gsm8k, "
             "humaneval, mmlu, mt_bench, custom, local, or file path"
    )
    parser.add_argument(
        "--local-path", default=None,
        help="Path to local dataset file (for --dataset=local)"
    )
    parser.add_argument(
        "--question-key", default="question",
        help="Key for question field in local dataset"
    )
    parser.add_argument(
        "--system-prompt", default=None,
        help="System prompt (for local dataset)"
    )
    parser.add_argument(
        "--num-samples", type=int, default=None,
        help="Number of samples to use (None=all)"
    )
    parser.add_argument(
        "--top-k", type=int, nargs="+", default=[1, 3, 5],
        help="Top-k values to test"
    )
    parser.add_argument(
        "--num-speculative-tokens", type=int, default=3,
        help="Number of speculative tokens"
    )
    parser.add_argument(
        "--max-tokens", type=int, default=512,
        help="Max tokens per prompt"
    )
    parser.add_argument(
        "--main-model", default="LLM-Research/Meta-Llama-3.1-8B-Instruct",
        help="Main/target model path"
    )
    parser.add_argument(
        "--draft-model", default=None,
        help="Draft model path (for Eagle methods)"
    )
    parser.add_argument(
        "--draft-method", default="eagle",
        choices=["eagle", "eagle3", "mtp"],
        help="Draft method for parallel-SD"
    )
    parser.add_argument(
        "--compare-eagle", action="store_true",
        help="Compare with Eagle baseline"
    )
    parser.add_argument(
        "--compare-mtp", action="store_true",
        help="Compare with MTP baseline"
    )
    parser.add_argument(
        "--gpu-memory-utilization", type=float, default=0.9,
        help="GPU memory utilization"
    )
    parser.add_argument(
        "--max-model-len", type=int, default=4096,
        help="Maximum model context length"
    )
    parser.add_argument(
        "--save-results", default=None,
        help="Save results to JSON file"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed"
    )

    args = parser.parse_args()

    # Load prompts
    print(f"Loading dataset: {args.dataset}")
    prompts = load_dataset_prompts(
        args.dataset,
        local_path=args.local_path,
        question_key=args.question_key,
        num_samples=args.num_samples,
    )
    print(f"Loaded {len(prompts)} prompts")

    all_results = []

    # Run Parallel-SD experiments for each top-k
    for top_k in args.top_k:
        print(f"\n--- Running Parallel-SD (top_k={top_k}) ---")
        result, _ = run_experiment(
            main_model=args.main_model,
            draft_model=args.draft_model,
            prompts=prompts,
            method="parallel",
            num_speculative_tokens=args.num_speculative_tokens,
            max_tokens=args.max_tokens,
            top_k=top_k,
            draft_method=args.draft_method,
            gpu_memory_utilization=args.gpu_memory_utilization,
            max_model_len=args.max_model_len,
            seed=args.seed,
            system_prompt=args.system_prompt,
        )
        all_results.append(result)

    # Eagle baseline comparison
    if args.compare_eagle and args.draft_model:
        print("\n--- Running Eagle baseline ---")
        result, _ = run_experiment(
            main_model=args.main_model,
            draft_model=args.draft_model,
            prompts=prompts,
            method="eagle",
            num_speculative_tokens=args.num_speculative_tokens,
            max_tokens=args.max_tokens,
            gpu_memory_utilization=args.gpu_memory_utilization,
            max_model_len=args.max_model_len,
            seed=args.seed,
            system_prompt=args.system_prompt,
        )
        all_results.append(result)

    # MTP baseline comparison
    if args.compare_mtp:
        print("\n--- Running MTP baseline ---")
        result, _ = run_experiment(
            main_model=args.main_model,
            draft_model=None,
            prompts=prompts,
            method="mtp",
            num_speculative_tokens=args.num_speculative_tokens,
            max_tokens=args.max_tokens,
            draft_method="mtp",
            gpu_memory_utilization=args.gpu_memory_utilization,
            max_model_len=args.max_model_len,
            seed=args.seed,
            system_prompt=args.system_prompt,
        )
        all_results.append(result)

    # Print results
    print_results(all_results)

    # Save results
    if args.save_results:
        with open(args.save_results, "w") as f:
            json.dump({
                "config": {
                    "dataset": args.dataset,
                    "num_samples": args.num_samples or len(prompts),
                    "num_speculative_tokens": args.num_speculative_tokens,
                    "max_tokens": args.max_tokens,
                    "main_model": args.main_model,
                    "draft_model": args.draft_model,
                    "draft_method": args.draft_method,
                    "seed": args.seed,
                },
                "results": all_results,
            }, f, indent=2)
        print(f"\nResults saved to {args.save_results}")


if __name__ == "__main__":
    main()
