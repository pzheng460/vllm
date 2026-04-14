"""Benchmark: MTP with early-exit (direct, no parallel proposer).

Tests different early-exit layers for MTP speculative decoding.
The key idea: instead of passing the target model's final hidden states
to MTP, we pass hidden states from an earlier layer (after applying norm).

Usage:
    # Default: 8 built-in prompts, test layers -1, -3, -5, -10, eager mode
    python test_mtp_early_exit.py

    # Specific layers and more speculative tokens
    python test_mtp_early_exit.py --early-exit -1 -2 -3 -5 -10 -20 \
        --num-spec-tokens 3

    # Use GPQA Diamond dataset (50 samples)
    python test_mtp_early_exit.py --dataset gpqa_diamond --num-samples 50

    # Full GPQA Diamond with long output and CUDA graph (piecewise)
    # Note: --no-eager enables CUDA graph; early-exit auto-downgrades
    # FULL_AND_PIECEWISE to PIECEWISE to preserve forward hooks.
    python test_mtp_early_exit.py --dataset gpqa_diamond \
        --max-tokens 4096 --no-eager --num-spec-tokens 3 \
        --early-exit -1 -3

    # Also test embed replacement mode (inputs_embeds from early-exit logits)
    python test_mtp_early_exit.py --early-exit -1 -3 \
        --num-spec-tokens 3 --early-exit-embed

    # Custom model with tensor parallelism
    python test_mtp_early_exit.py --model /path/to/model --tp-size 4

    # Force full output length (ignore EOS)
    python test_mtp_early_exit.py --max-tokens 4096 --ignore-eos

Arguments:
    --model             Target model path (default: openPangu-R-72B-2512)
    --dataset           Dataset: builtin, gpqa_diamond, gsm8k (default: builtin)
    --num-samples       Number of samples (default: all)
    --max-tokens        Max output tokens per prompt (default: 100)
    --num-spec-tokens   Number of speculative tokens (default: 1)
    --early-exit        Layer indices to test, e.g. -1 -3 -5 (default: -1 -3 -5 -10)
                        -1 = last layer (baseline, no early exit)
    --early-exit-embed  Also test embed replacement mode for non-baseline layers
    --tp-size           Tensor parallel size (default: 4)
    --gpu-mem           GPU memory utilization (default: 0.95)
    --max-model-len     Max model context length (default: 4096)
    --no-eager          Enable CUDA graph (piecewise mode for early-exit)
    --ignore-eos        Ignore EOS token to force full output length
    --save              Output JSON file (default: mtp_early_exit_results.json)

Metrics reported:
  - Throughput (tok/s)
  - Mean acceptance length (includes bonus token)
  - Draft acceptance rate
  - Per-position acceptance rate
  - Real-time cumulative acceptance rate during generation
  - Output correctness vs baseline (layer -1)
"""

import argparse
import gc
import json
import os
import tempfile
import threading
import time

import torch

# Stats file for collecting spec decode metrics from worker process.
# These env vars are consumed by vllm's internal metrics system and
# cannot be replaced with config parameters.
_SPEC_STATS_FILE = os.path.join(tempfile.gettempdir(),
                                "vllm_spec_stats.json")
os.environ["VLLM_SPEC_STATS_FILE"] = _SPEC_STATS_FILE
os.environ["VLLM_LOG_STATS_INTERVAL"] = "1"

# ---------------------------------------------------------------------------
# Dataset loaders
# ---------------------------------------------------------------------------

def get_builtin_prompts():
    return [
        "Explain the theory of general relativity in detail.",
        "Write a Python function to implement a binary search tree.",
        "What are the main differences between supervised and unsupervised learning?",
        "Describe the process of photosynthesis step by step.",
        "Write a guide on database normalization from 1NF to BCNF.",
        "Explain how a CPU pipeline works and what pipeline hazards are.",
        "What is the significance of the Turing machine in computer science?",
        "Describe the differences between TCP and UDP protocols.",
    ]


def load_gpqa_diamond(num_samples=None):
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
    from datasets import load_dataset
    ds = load_dataset("gsm8k", "main", split="test")
    prompts = [f"Solve: {item['question']}\nAnswer:" for item in ds]
    if num_samples:
        prompts = prompts[:num_samples]
    print(f"Loaded {len(prompts)} prompts from gsm8k")
    return prompts


def load_jsonl(path, num_samples=None):
    prompts = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            # Support common field names for the prompt text
            for key in ("prompt", "text", "input", "question", "content"):
                if key in item:
                    prompts.append(item[key])
                    break
            else:
                # If none of the known keys, use the first string value
                for v in item.values():
                    if isinstance(v, str) and v.strip():
                        prompts.append(v)
                        break
    if num_samples:
        prompts = prompts[:num_samples]
    print(f"Loaded {len(prompts)} prompts from {path}")
    return prompts


DATASET_LOADERS = {
    "builtin": lambda n: get_builtin_prompts()[:n] if n else get_builtin_prompts(),
    "gpqa_diamond": load_gpqa_diamond,
    "gsm8k": load_gsm8k,
}


def load_prompts(dataset_name, num_samples=None):
    # If it's a file path (json/jsonl), load directly
    if dataset_name.endswith(".jsonl") or dataset_name.endswith(".json"):
        if dataset_name.endswith(".jsonl"):
            return load_jsonl(dataset_name, num_samples)
        # .json: try as array of strings or array of objects
        with open(dataset_name, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list) and len(data) > 0:
            if isinstance(data[0], str):
                prompts = data
            else:
                prompts = []
                for item in data:
                    for key in ("prompt", "text", "input", "question", "content"):
                        if key in item:
                            prompts.append(item[key])
                            break
                    else:
                        for v in item.values():
                            if isinstance(v, str) and v.strip():
                                prompts.append(v)
                                break
        else:
            raise ValueError(f"Cannot parse prompts from {dataset_name}")
        if num_samples:
            prompts = prompts[:num_samples]
        print(f"Loaded {len(prompts)} prompts from {dataset_name}")
        return prompts

    if dataset_name not in DATASET_LOADERS:
        raise ValueError(f"Unknown dataset: {dataset_name}. "
                         f"Available: {list(DATASET_LOADERS.keys())} "
                         "or pass a .json/.jsonl file path")
    return DATASET_LOADERS[dataset_name](num_samples)


# ---------------------------------------------------------------------------
# Repetition analysis
# ---------------------------------------------------------------------------

def _ngram_repetition_rate(text, n=4):
    """Fraction of n-grams that are repeated at least once."""
    words = text.split()
    if len(words) < n:
        return 0.0
    ngrams = [tuple(words[i:i+n]) for i in range(len(words) - n + 1)]
    if not ngrams:
        return 0.0
    from collections import Counter
    counts = Counter(ngrams)
    repeated = sum(c - 1 for c in counts.values() if c > 1)
    return repeated / len(ngrams)


def _analyze_repetition(output_texts, token_counts, max_tokens):
    """Analyze outputs for repetition and max-token truncation."""
    num_hit_max = sum(1 for tc in token_counts if tc >= max_tokens)
    rep_rates = [_ngram_repetition_rate(t) for t in output_texts]
    # Consider >20% 4-gram repetition rate as "repetitive"
    num_repetitive = sum(1 for r in rep_rates if r > 0.2)
    avg_rep = sum(rep_rates) / len(rep_rates) if rep_rates else 0.0
    # Indices of worst offenders
    worst = sorted(range(len(rep_rates)), key=lambda i: -rep_rates[i])[:3]
    worst_info = [
        {"idx": i, "rep_rate": round(rep_rates[i], 3),
         "tokens": token_counts[i],
         "tail": output_texts[i][-200:] if output_texts[i] else ""}
        for i in worst if rep_rates[i] > 0.1
    ]
    return {
        "num_hit_max": num_hit_max,
        "num_repetitive": num_repetitive,
        "avg_rep_rate": round(avg_rep, 4),
        "worst_samples": worst_info,
    }


# ---------------------------------------------------------------------------
# Benchmark runner
# ---------------------------------------------------------------------------

def cleanup_gpu():
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


class _StatsMonitor:
    """Background thread that prints cumulative acceptance rate in real time."""

    def __init__(self, stats_file, config_name, interval=5.0):
        self.stats_file = stats_file
        self.config_name = config_name
        self.interval = interval
        self._stop = threading.Event()
        self._thread = None

    def start(self):
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2)

    def _run(self):
        last_drafts = 0
        while not self._stop.wait(self.interval):
            try:
                if not os.path.exists(self.stats_file):
                    continue
                with open(self.stats_file) as f:
                    stats = json.load(f)
                drafts = stats.get("num_drafts", 0)
                if drafts <= last_drafts:
                    continue
                last_drafts = drafts
                mal = stats.get("mean_acceptance_length", 0)
                dar = stats.get("draft_acceptance_rate", 0)
                per_pos = stats.get("per_position_acceptance_rate", [])
                rates_str = ", ".join(f"{r:.3f}" for r in per_pos)
                print(f"  [{self.config_name}] cumulative: "
                      f"drafts={drafts} MAL={mal:.2f} "
                      f"DAR={dar:.1f}% pos=[{rates_str}]",
                      flush=True)
            except Exception:
                pass


def run_single_config(config_name, main_model, spec_config, prompts,
                      max_tokens=100, tp_size=1, max_model_len=4096,
                      gpu_mem=0.9, enforce_eager=True, ignore_eos=False,
                      temperature=0.0, repetition_penalty=1.0,
                      max_num_seqs=256):
    """Run a single MTP early-exit experiment."""
    from vllm import LLM, SamplingParams

    if os.path.exists(_SPEC_STATS_FILE):
        os.remove(_SPEC_STATS_FILE)

    print(f"\n{'='*60}")
    print(f"Running: {config_name}")
    print(f"Spec config: {json.dumps(spec_config, indent=2)}")
    print(f"{'='*60}")

    extra_kwargs = {}
    if enforce_eager:
        extra_kwargs["enforce_eager"] = True

    llm_kwargs = dict(
        model=main_model,
        gpu_memory_utilization=gpu_mem,
        max_model_len=max_model_len,
        tensor_parallel_size=tp_size,
        max_num_seqs=max_num_seqs,
        trust_remote_code=True,
        dtype="bfloat16",
        disable_log_stats=False,
        **extra_kwargs,
    )
    if spec_config is not None:
        llm_kwargs["speculative_config"] = spec_config
    llm = LLM(**llm_kwargs)

    sampling_params = SamplingParams(
        max_tokens=max_tokens,
        temperature=temperature,
        ignore_eos=ignore_eos,
        repetition_penalty=repetition_penalty,
    )

    # Apply chat template manually to enable thinking mode.
    # The template already includes <s> (BOS), so we strip it to avoid
    # double-BOS when vLLM tokenizes with add_special_tokens=True.
    # Using llm.generate() instead of llm.chat() to stay compatible
    # with CUDA graphs (llm.chat can cause double-BOS + flash attn crash).
    try:
        tokenizer = llm.get_tokenizer()
        chat_prompts = []
        for p in prompts:
            msgs = [{"role": "user", "content": p}]
            text = tokenizer.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True)
            # Strip leading <s> to avoid double BOS
            if text.startswith("<s>"):
                text = text[3:]
            chat_prompts.append(text)
    except Exception:
        # Fallback: use raw prompts if chat template fails
        chat_prompts = prompts

    # Warmup
    print("Warmup...")
    _ = llm.generate(chat_prompts[:2], sampling_params)

    # Flush warmup stats
    for _ in range(2):
        try:
            llm.llm_engine.do_log_stats()
        except Exception:
            pass
        time.sleep(1)
    if os.path.exists(_SPEC_STATS_FILE):
        os.remove(_SPEC_STATS_FILE)
    _reset_spec_cumulative(llm)

    # Benchmark with real-time stats monitoring.
    # Run prompts in batches. If a CUDA error kills the engine,
    # rebuild LLM and continue from the next prompt.
    monitor = _StatsMonitor(_SPEC_STATS_FILE, config_name)
    print("Benchmark run...")
    monitor.start()
    start = time.time()

    from types import SimpleNamespace

    outputs = []
    skipped = []

    def _make_placeholder():
        return SimpleNamespace(
            outputs=[SimpleNamespace(
                text="[SKIPPED: engine crash]",
                token_ids=[])])

    def _rebuild_llm():
        nonlocal llm
        del llm
        cleanup_gpu()
        time.sleep(2)
        kw = dict(
            model=main_model,
            gpu_memory_utilization=gpu_mem,
            max_model_len=max_model_len,
            tensor_parallel_size=tp_size,
            max_num_seqs=max_num_seqs,
            trust_remote_code=True,
            dtype="bfloat16",
            disable_log_stats=False,
            **extra_kwargs,
        )
        if spec_config is not None:
            kw["speculative_config"] = spec_config
        llm = LLM(**kw)
        return llm

    def _is_engine_crash(e):
        cls = type(e).__name__
        msg = str(e)
        return ("EngineDeadError" in cls or "illegal memory" in msg
                or "EngineCore" in msg or "CUDA" in msg)

    def _run_batch(prompt_indices):
        """Run a batch of prompts. Returns (results, crashed_index).
        crashed_index is None if all succeeded."""
        nonlocal llm
        batch = [chat_prompts[j] for j in prompt_indices]
        try:
            batch_outputs = llm.generate(batch, sampling_params)
            return batch_outputs, None
        except Exception as e:
            if not _is_engine_crash(e):
                raise
            # Find which prompt crashed via binary search
            if len(prompt_indices) == 1:
                # Single prompt crashed
                print(f"  WARNING: Prompt {prompt_indices[0]} crashed. "
                      f"Skipping.")
                llm = _rebuild_llm()
                return [], prompt_indices[0]
            # Split and retry
            mid = len(prompt_indices) // 2
            left = prompt_indices[:mid]
            right = prompt_indices[mid:]
            print(f"  Batch [{prompt_indices[0]}..{prompt_indices[-1]}] "
                  f"crashed. Splitting into [{left[0]}..{left[-1]}] + "
                  f"[{right[0]}..{right[-1]}]...")
            llm = _rebuild_llm()
            # Try left half
            left_results, left_crash = _run_batch(left)
            if left_crash is not None:
                skipped.append(left_crash)
                # Insert placeholder at the crash position
                pos = left.index(left_crash)
                left_results.insert(pos, _make_placeholder())
            # Try right half
            right_results, right_crash = _run_batch(right)
            if right_crash is not None:
                skipped.append(right_crash)
                pos = right.index(right_crash)
                right_results.insert(pos, _make_placeholder())
            return left_results + right_results, None

    all_indices = list(range(len(chat_prompts)))
    batch_results, crash_idx = _run_batch(all_indices)
    if crash_idx is not None:
        skipped.append(crash_idx)
        batch_results.append(_make_placeholder())
    outputs.extend(batch_results)

    elapsed = time.time() - start
    monitor.stop()

    if skipped:
        print(f"  Skipped {len(skipped)} prompts due to crashes: {skipped}")

    total_tokens = sum(len(o.outputs[0].token_ids) for o in outputs)
    throughput = total_tokens / elapsed if elapsed > 0 else 0

    # Flush final stats
    try:
        llm.llm_engine.do_log_stats()
    except Exception:
        pass

    # Collect spec decode metrics
    spec_metrics = {}
    if os.path.exists(_SPEC_STATS_FILE):
        try:
            with open(_SPEC_STATS_FILE) as f:
                spec_metrics = json.load(f)
            os.remove(_SPEC_STATS_FILE)
        except Exception:
            pass

    output_texts = [o.outputs[0].text for o in outputs]
    token_counts = [len(o.outputs[0].token_ids) for o in outputs]

    # Repetition analysis
    rep_stats = _analyze_repetition(output_texts, token_counts, max_tokens)

    safe_spec_config = {}
    if spec_config is not None:
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
        "repetition": rep_stats,
    }

    print(f"  Tokens: {total_tokens}")
    print(f"  Time: {elapsed:.3f}s")
    print(f"  Throughput: {throughput:.2f} tok/s")
    if rep_stats["num_hit_max"]:
        print(f"  WARNING: {rep_stats['num_hit_max']}/{len(prompts)} "
              f"prompts hit max_tokens ({max_tokens})")
    if rep_stats["num_repetitive"]:
        print(f"  WARNING: {rep_stats['num_repetitive']}/{len(prompts)} "
              f"prompts have high repetition "
              f"(avg ngram_rep_rate={rep_stats['avg_rep_rate']:.1%})")
    if spec_metrics:
        mal = spec_metrics.get("mean_acceptance_length", 0)
        dar = spec_metrics.get("draft_acceptance_rate", 0)
        print(f"  Mean acceptance length: {mal:.2f}")
        print(f"  Draft acceptance rate: {dar:.1f}%")
        per_pos = spec_metrics.get("per_position_acceptance_rate", [])
        if per_pos:
            rates_str = ", ".join(f"{r:.3f}" for r in per_pos)
            print(f"  Per-position acceptance: [{rates_str}]")

    # Collect early-exit top-k stats
    ee_topk_file = os.path.join(tempfile.gettempdir(),
                                "vllm_ee_topk_stats.json")
    if os.path.exists(ee_topk_file):
        try:
            with open(ee_topk_file) as f:
                ee_topk = json.load(f)
            os.remove(ee_topk_file)
            result["ee_topk"] = ee_topk
            hit_rates = ee_topk.get("ee_topk_hit_rates", {})
            if hit_rates:
                rates_str = ", ".join(
                    f"top-{k}: {v:.1f}%"
                    for k, v in sorted(hit_rates.items(),
                                       key=lambda x: int(x[0])))
                print(f"  EE vs target: {rates_str}")
        except Exception:
            pass

    del llm
    cleanup_gpu()

    return result, output_texts


def print_summary(all_results):
    """Print benchmark summary table."""
    print("\n" + "=" * 90)
    print("MTP EARLY-EXIT BENCHMARK SUMMARY")
    print("=" * 90)

    baseline = all_results[0]
    baseline_tps = baseline["tokens_per_sec"]
    baseline_mal = baseline.get("spec_metrics", {}).get(
        "mean_acceptance_length", 0)

    # Header
    print(f"\n{'Config':<30} {'tok/s':>8} {'speedup':>8}"
          f" {'AccLen':>8} {'vs base':>8} {'DraftAcc':>9}")
    print("-" * 80)

    for r in all_results:
        name = r["config_name"]
        tps = r["tokens_per_sec"]
        speedup = tps / baseline_tps if baseline_tps > 0 else 0

        sm = r.get("spec_metrics", {})
        mal = sm.get("mean_acceptance_length", 0)
        dar = sm.get("draft_acceptance_rate", 0)

        mal_str = f"{mal:.2f}" if mal > 0 else "N/A"
        dar_str = f"{dar:.1f}%" if dar > 0 else "N/A"
        if mal > 0 and baseline_mal > 0:
            mal_ratio = f"{mal / baseline_mal:.2f}x"
        else:
            mal_ratio = "N/A"

        print(f"  {name:<28} {tps:>8.1f} {speedup:>7.2f}x"
              f" {mal_str:>8} {mal_ratio:>8} {dar_str:>9}")

    # Per-position detail
    print(f"\n{'Config':<30} Per-position acceptance rates")
    print("-" * 80)
    for r in all_results:
        name = r["config_name"]
        per_pos = r.get("spec_metrics", {}).get(
            "per_position_acceptance_rate", [])
        if per_pos:
            rates = " ".join(f"{p:.3f}" for p in per_pos)
            print(f"  {name:<28} [{rates}]")
        else:
            print(f"  {name:<28} N/A")

    # EE top-k hit rates vs target
    has_ee = any(r.get("ee_topk") for r in all_results)
    if has_ee:
        print(f"\n{'Config':<30} EE vs target top-k")
        print("-" * 80)
        for r in all_results:
            name = r["config_name"]
            ee = r.get("ee_topk", {})
            hit_rates = ee.get("ee_topk_hit_rates", {})
            if hit_rates:
                rates = "  ".join(
                    f"top-{k}: {v:.1f}%"
                    for k, v in sorted(hit_rates.items(),
                                       key=lambda x: int(x[0])))
                print(f"  {name:<28} {rates}")
            else:
                print(f"  {name:<28} N/A")
    print()


def print_correctness(all_outputs):
    """Check output correctness vs baseline."""
    baseline_key = list(all_outputs.keys())[0]
    baseline_texts = all_outputs[baseline_key]
    if not baseline_texts:
        return

    print(f"\nCORRECTNESS CHECK (vs {baseline_key}):")
    for cfg_name, texts in all_outputs.items():
        if cfg_name == baseline_key:
            continue
        matches = sum(1 for a, b in zip(baseline_texts, texts) if a == b)
        total = len(baseline_texts)
        pct = matches / total * 100 if total > 0 else 0
        print(f"  {cfg_name}: {pct:.0f}% match ({matches}/{total})")

        # Show up to 3 diffs
        diff_count = 0
        for i, (a, b) in enumerate(zip(baseline_texts, texts)):
            if a != b:
                diff_count += 1
                if diff_count <= 3:
                    # Find the first divergence point
                    common_len = 0
                    for ca, cb in zip(a, b):
                        if ca != cb:
                            break
                        common_len += 1
                    print(f"    Diff #{diff_count} at prompt {i} "
                          f"(diverges at char {common_len}):")
                    start = max(0, common_len - 20)
                    print(f"      baseline: ...{a[start:start+80]}...")
                    print(f"      current:  ...{b[start:start+80]}...")
        if diff_count > 3:
            print(f"    ... and {diff_count - 3} more diffs")
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="MTP early-exit benchmark (direct, no parallel proposer)")

    parser.add_argument("--model",
                        default="/mnt/data/weights/openPangu-R-72B-2512",
                        help="Target model path")
    parser.add_argument("--dataset", default="builtin",
                        help="Dataset name (builtin/gpqa_diamond/gsm8k) "
                             "or path to .json/.jsonl file")
    parser.add_argument("--num-samples", type=int, default=None,
                        help="Number of samples (default: all)")
    parser.add_argument("--max-tokens", type=int, default=100,
                        help="Max output tokens (default: 100)")
    parser.add_argument("--num-spec-tokens", type=int, default=1,
                        help="Number of speculative tokens (default: 1)")
    parser.add_argument("--early-exit", type=int, nargs="+",
                        default=[-1, -3, -5, -10],
                        help="Early-exit layer indices to test "
                             "(default: -1 -3 -5 -10). "
                             "-1 = last layer (baseline)")
    parser.add_argument("--tp-size", type=int, default=4,
                        help="Tensor parallel size (default: 4)")
    parser.add_argument("--gpu-mem", type=float, default=0.95,
                        help="GPU memory utilization (default: 0.95)")
    parser.add_argument("--max-num-seqs", type=int, default=256,
                        help="Max concurrent sequences (default: 256)")
    parser.add_argument("--max-model-len", type=int, default=4096,
                        help="Max model context length (default: 4096)")
    parser.add_argument("--early-exit-embed", action="store_true",
                        help="Also test early-exit embed mode "
                             "(inputs_embeds from early-exit logits)")
    parser.add_argument("--early-exit-embed-all", action="store_true",
                        help="Also test early-exit embed-all mode "
                             "(ALL positions' inputs_embeds from early-exit)")
    parser.add_argument("--mtp-skip-norm", action="store_true",
                        help="Also test passing un-normed hidden states "
                             "to MTP (skip target model final norm)")
    parser.add_argument("--no-eager", action="store_true",
                        help="Disable enforce_eager to enable CUDA graphs")
    parser.add_argument("--ignore-eos", action="store_true",
                        help="Ignore EOS token to force full output length")
    parser.add_argument("--temperature", type=float, default=0.0,
                        help="Sampling temperature (0=greedy, >0=rejection sampling)")
    parser.add_argument("--repetition-penalty", type=float, default=1.0,
                        help="Repetition penalty (1.0=off, 1.05~1.2=mild)")
    parser.add_argument("--ee-topk", type=int, nargs="+", default=None,
                        help="Enable early-exit top-k diagnostic with "
                             "specified k values, e.g. --ee-topk 1 3 5")
    parser.add_argument("--verify", action="store_true",
                        help="Run without spec decode first, then compare "
                             "MTP output against non-speculative baseline")
    parser.add_argument("--save-texts", action="store_true",
                        help="Save output texts to result JSON")
    parser.add_argument("--save",
                        default="mtp_early_exit_results.json",
                        help="Output JSON file")
    args = parser.parse_args()

    prompts = load_prompts(args.dataset, args.num_samples)

    # Sort early-exit layers so -1 (baseline) runs first if present
    early_exit_layers = sorted(args.early_exit, key=lambda x: -x)

    print("=" * 60)
    print("MTP Early-Exit Benchmark")
    print(f"  Model:        {args.model}")
    print(f"  TP size:      {args.tp_size}")
    print(f"  Dataset:      {args.dataset} ({len(prompts)} prompts)")
    print(f"  Max tokens:   {args.max_tokens}")
    print(f"  Spec tokens:  {args.num_spec_tokens}")
    print(f"  Early-exit:   {early_exit_layers}")
    print(f"  Embed mode:   {args.early_exit_embed}")
    print(f"  Embed-all:    {args.early_exit_embed_all}")
    print(f"  No-norm:      {args.mtp_skip_norm}")
    print(f"  EE top-k:     {args.ee_topk or 'disabled'}")
    print(f"  CUDA graph:   {args.no_eager}")
    print(f"  Ignore EOS:   {args.ignore_eos}")
    print("=" * 60)

    enforce_eager = not args.no_eager

    all_results = []
    all_outputs = {}

    # --verify: run non-speculative baseline first
    if args.verify:
        print("\n" + "=" * 60)
        print("VERIFY MODE: running non-speculative baseline")
        print("=" * 60)
        r_ns, texts_ns = run_single_config(
            "No-spec baseline", args.model, None, prompts,
            max_tokens=args.max_tokens,
            tp_size=args.tp_size,
            gpu_mem=args.gpu_mem,
            max_model_len=args.max_model_len,
            enforce_eager=enforce_eager,
            ignore_eos=args.ignore_eos,
            temperature=args.temperature,
            repetition_penalty=args.repetition_penalty,
            max_num_seqs=args.max_num_seqs,
        )
        all_results.append(r_ns)
        all_outputs["No-spec baseline"] = texts_ns

    for ee_layer in early_exit_layers:
        # Mode 1: hidden_states only (embed stays original)
        if ee_layer == -1:
            name = "MTP baseline (L=-1)"
        else:
            name = f"MTP ee L={ee_layer} hs-only"

        spec_config = {
            "method": "mtp",
            "num_speculative_tokens": args.num_spec_tokens,
            "early_exit_layer": ee_layer,
        }
        if args.ee_topk:
            spec_config["early_exit_topk"] = args.ee_topk

        r, texts = run_single_config(
            name, args.model, spec_config, prompts,
            max_tokens=args.max_tokens,
            tp_size=args.tp_size,
            gpu_mem=args.gpu_mem,
            max_model_len=args.max_model_len,
            enforce_eager=enforce_eager,
            ignore_eos=args.ignore_eos,
            temperature=args.temperature,
            repetition_penalty=args.repetition_penalty,
            max_num_seqs=args.max_num_seqs,
        )
        all_results.append(r)
        all_outputs[name] = texts

        # Mode 2: hidden_states + last-position embed from early-exit
        if args.early_exit_embed and ee_layer != -1:
            name = f"MTP ee L={ee_layer} hs+embed"
            if args.mtp_skip_norm:
                name += " no-norm"
            spec_config = {
                "method": "mtp",
                "num_speculative_tokens": args.num_spec_tokens,
                "early_exit_layer": ee_layer,
                "early_exit_replace_embed": True,
            }
            if args.mtp_skip_norm:
                spec_config["mtp_skip_norm"] = True
            if args.ee_topk:
                spec_config["early_exit_topk"] = args.ee_topk

            r, texts = run_single_config(
                name, args.model, spec_config, prompts,
                max_tokens=args.max_tokens,
                tp_size=args.tp_size,
                gpu_mem=args.gpu_mem,
                max_model_len=args.max_model_len,
                enforce_eager=enforce_eager,
                ignore_eos=args.ignore_eos,
                temperature=args.temperature,
                repetition_penalty=args.repetition_penalty,
                max_num_seqs=args.max_num_seqs,
            )
            all_results.append(r)
            all_outputs[name] = texts

        # Mode 3: hidden_states + ALL positions embed from early-exit
        if args.early_exit_embed_all and ee_layer != -1:
            name = f"MTP ee L={ee_layer} hs+embed-all"
            if args.mtp_skip_norm:
                name += " no-norm"
            spec_config = {
                "method": "mtp",
                "num_speculative_tokens": args.num_spec_tokens,
                "early_exit_layer": ee_layer,
                "early_exit_embed_all": True,
            }
            if args.mtp_skip_norm:
                spec_config["mtp_skip_norm"] = True
            if args.ee_topk:
                spec_config["early_exit_topk"] = args.ee_topk

            r, texts = run_single_config(
                name, args.model, spec_config, prompts,
                max_tokens=args.max_tokens,
                tp_size=args.tp_size,
                gpu_mem=args.gpu_mem,
                max_model_len=args.max_model_len,
                enforce_eager=enforce_eager,
                ignore_eos=args.ignore_eos,
                temperature=args.temperature,
                repetition_penalty=args.repetition_penalty,
                max_num_seqs=args.max_num_seqs,
            )
            all_results.append(r)
            all_outputs[name] = texts

        # Mode 4: un-normed hidden_states (skip target final norm)
        if args.mtp_skip_norm:
            if ee_layer == -1:
                name = "MTP baseline (L=-1) no-norm"
            else:
                name = f"MTP ee L={ee_layer} no-norm"
            spec_config = {
                "method": "mtp",
                "num_speculative_tokens": args.num_spec_tokens,
                "early_exit_layer": ee_layer,
                "mtp_skip_norm": True,
            }
            if args.ee_topk:
                spec_config["early_exit_topk"] = args.ee_topk

            r, texts = run_single_config(
                name, args.model, spec_config, prompts,
                max_tokens=args.max_tokens,
                tp_size=args.tp_size,
                gpu_mem=args.gpu_mem,
                max_model_len=args.max_model_len,
                enforce_eager=enforce_eager,
                ignore_eos=args.ignore_eos,
                temperature=args.temperature,
                repetition_penalty=args.repetition_penalty,
                max_num_seqs=args.max_num_seqs,
            )
            all_results.append(r)
            all_outputs[name] = texts

    # Summary
    print_summary(all_results)
    print_correctness(all_outputs)

    # Save results
    save_data = {
        "args": vars(args),
        "results": all_results,
    }
    if args.save_texts:
        save_data["outputs"] = {
            name: texts for name, texts in all_outputs.items()
        }
    with open(args.save, "w") as f:
        json.dump(save_data, f, indent=2, ensure_ascii=False)
    print(f"Results saved to {args.save}")


if __name__ == "__main__":
    main()
