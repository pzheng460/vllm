# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Experiment script for Parallel-SD early exit layer comparison.

Compares MTP baseline vs Parallel-SD with different early_exit_layer settings.
Uses vllm serve + /v1/chat/completions API for correct chat template handling.

Each configuration runs a vllm server in a subprocess to ensure clean GPU memory.

Usage:
    python run.py \
        --main-model /mnt/data/weights/openPangu-R-72B-2512 \
        --early-exit-layers "-1,-3,-5" \
        --top-k 1 \
        --max-tokens 100 \
        --save-results results.json
"""

import argparse
import json
import logging
import os
import re
import signal
import socket
import subprocess
import sys
import time

import requests

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# Default test prompts
DEFAULT_PROMPTS = [
    "Explain the theory of general relativity in simple terms.",
    "Write a Python function that computes the Fibonacci sequence.",
    "What are the main differences between TCP and UDP?",
    "Describe the process of photosynthesis step by step.",
    "What is the significance of the Turing test in AI?",
    "Explain how a transformer neural network works.",
    "What are the key principles of object-oriented programming?",
    "Describe the water cycle and its importance to Earth's climate.",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Parallel-SD early exit layer experiment"
    )
    parser.add_argument(
        "--main-model",
        type=str,
        default="/mnt/data/weights/openPangu-R-72B-2512",
        help="Path to main (target) model",
    )
    parser.add_argument(
        "--draft-model",
        type=str,
        default=None,
        help="Path to draft model (default: same as main model for MTP)",
    )
    parser.add_argument(
        "--early-exit-layers",
        type=str,
        default="-1,-3,-5",
        help="Comma-separated early exit layer indices "
             "(negative = relative to last, e.g. '-1,-3,-5')",
    )
    parser.add_argument(
        "--top-k",
        type=str,
        default="1",
        help="Comma-separated top-k values (e.g. '1,2,3')",
    )
    parser.add_argument(
        "--num-speculative-tokens",
        type=int,
        default=1,
        help="Number of speculative tokens",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=100,
        help="Maximum tokens to generate per prompt",
    )
    parser.add_argument(
        "--max-model-len",
        type=int,
        default=4096,
        help="Maximum model context length",
    )
    parser.add_argument(
        "--tensor-parallel-size",
        type=int,
        default=2,
        help="Tensor parallel size",
    )
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.95,
        help="GPU memory utilization",
    )
    parser.add_argument(
        "--max-num-seqs",
        type=int,
        default=1,
        help="Maximum number of sequences per batch",
    )
    parser.add_argument(
        "--prompts-file",
        type=str,
        default=None,
        help="Path to JSON/JSONL file with prompts (list of strings)",
    )
    parser.add_argument(
        "--num-prompts",
        type=int,
        default=None,
        help="Number of prompts to use (default: all)",
    )
    parser.add_argument(
        "--save-results",
        type=str,
        default=None,
        help="Save results to JSON file",
    )
    parser.add_argument(
        "--draft-method",
        type=str,
        default="mtp",
        choices=["eagle", "eagle3", "mtp"],
        help="Underlying draft method",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    parser.add_argument(
        "--enable-half-cache-hit",
        action="store_true",
        help="Enable half-cache-hit (prefix match fallback, top_k=1 only)",
    )
    parser.add_argument(
        "--skip-baseline",
        action="store_true",
        help="Skip MTP baseline run",
    )
    parser.add_argument(
        "--enforce-eager",
        action="store_true",
        help="Disable CUDA graphs",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=None,
        help="Server port (default: auto-select free port)",
    )
    return parser.parse_args()


# ------------------------------------------------------------------
# Server management
# ------------------------------------------------------------------

def find_free_port():
    """Find a free TCP port."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


def start_server(run_spec, port):
    """Start a vllm serve process and return the Popen object.

    Returns (proc, stderr_file_path) tuple. stderr is written to a temp file
    to avoid pipe buffer overflow with large model loading logs.
    """
    cmd = [
        sys.executable, "-m", "vllm.entrypoints.openai.api_server",
        "--model", run_spec["main_model"],
        "--trust-remote-code",
        "--tensor-parallel-size", str(run_spec["tensor_parallel_size"]),
        "--gpu-memory-utilization", str(run_spec["gpu_memory_utilization"]),
        "--max-model-len", str(run_spec["max_model_len"]),
        "--max-num-seqs", str(run_spec["max_num_seqs"]),
        "--seed", str(run_spec["seed"]),
        "--port", str(port),
        "--disable-log-requests",
    ]

    if run_spec.get("enforce_eager", False):
        cmd.append("--enforce-eager")

    spec_config = run_spec["spec_config"]
    if spec_config:
        cmd.extend(["--speculative-config", json.dumps(spec_config)])

    logger.info("Starting server: %s", " ".join(cmd[-6:]))

    import tempfile
    stderr_file = tempfile.NamedTemporaryFile(
        mode="w", prefix="vllm_stderr_", suffix=".log", delete=False
    )
    stdout_file = tempfile.NamedTemporaryFile(
        mode="w", prefix="vllm_stdout_", suffix=".log", delete=False
    )
    stderr_path = stderr_file.name
    stdout_path = stdout_file.name

    proc = subprocess.Popen(
        cmd, stdout=stdout_file, stderr=stderr_file,
        env=os.environ.copy(),
    )
    return proc, stderr_path, stdout_path


def wait_for_server(port, timeout=300):
    """Wait for the server to be ready."""
    url = f"http://localhost:{port}/health"
    start = time.time()
    while time.time() - start < timeout:
        try:
            r = requests.get(url, timeout=2)
            if r.status_code == 200:
                logger.info("Server ready on port %d (%.1fs)",
                            port, time.time() - start)
                return True
        except requests.ConnectionError:
            pass
        time.sleep(2)
    logger.error("Server did not start within %ds", timeout)
    return False


def stop_server(proc):
    """Gracefully stop the server process."""
    if proc.poll() is not None:
        return
    proc.send_signal(signal.SIGINT)
    try:
        proc.wait(timeout=30)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=10)
    logger.info("Server stopped")


# ------------------------------------------------------------------
# Metrics collection from Prometheus endpoint
# ------------------------------------------------------------------

def collect_prometheus_metrics(port, num_spec_tokens):
    """Collect spec decode metrics from the /metrics endpoint."""
    url = f"http://localhost:{port}/metrics"
    try:
        r = requests.get(url, timeout=5)
        text = r.text
    except requests.RequestException:
        return {"num_drafts": 0, "num_draft_tokens": 0,
                "num_accepted_tokens": 0,
                "acceptance_counts": [0] * num_spec_tokens}

    d = {"num_drafts": 0, "num_draft_tokens": 0,
         "num_accepted_tokens": 0,
         "acceptance_counts": [0] * num_spec_tokens}

    for line in text.splitlines():
        if line.startswith("#"):
            continue
        # Parse counter lines like:
        # vllm:spec_decode_num_drafts_total{...} 123.0
        if "spec_decode_num_drafts_total" in line:
            val = _parse_prom_value(line)
            if val is not None:
                d["num_drafts"] += int(val)
        elif "spec_decode_num_draft_tokens_total" in line:
            val = _parse_prom_value(line)
            if val is not None:
                d["num_draft_tokens"] += int(val)
        elif ("spec_decode_num_accepted_tokens_total" in line
              and "per_pos" not in line):
            val = _parse_prom_value(line)
            if val is not None:
                d["num_accepted_tokens"] += int(val)
        elif "spec_decode_num_accepted_tokens_per_pos_total" in line:
            # Extract position from labels: position="0"
            pos_match = re.search(r'position="(\d+)"', line)
            val = _parse_prom_value(line)
            if pos_match and val is not None:
                pos = int(pos_match.group(1))
                if pos < num_spec_tokens:
                    d["acceptance_counts"][pos] += int(val)

    return d


def _parse_prom_value(line):
    """Parse the numeric value from a Prometheus metric line."""
    parts = line.rsplit(" ", 1)
    if len(parts) == 2:
        try:
            return float(parts[1])
        except ValueError:
            pass
    return None


def compute_delta(before, after, num_spec_tokens):
    """Compute metric deltas between two snapshots."""
    nd = after["num_drafts"] - before["num_drafts"]
    ndt = after["num_draft_tokens"] - before["num_draft_tokens"]
    nat = after["num_accepted_tokens"] - before["num_accepted_tokens"]
    ac = [after["acceptance_counts"][i] - before["acceptance_counts"][i]
          for i in range(num_spec_tokens)]
    return {
        "num_drafts": nd,
        "num_draft_tokens": ndt,
        "num_accepted_tokens": nat,
        "mean_acceptance_length": 1 + (nat / nd) if nd > 0 else 1.0,
        "draft_acceptance_rate": (nat / ndt * 100) if ndt > 0 else 0.0,
        "per_position_acceptance_rate": [
            (ac[i] / nd * 100) if nd > 0 else 0.0
            for i in range(num_spec_tokens)
        ],
    }


# ------------------------------------------------------------------
# Chat completion requests
# ------------------------------------------------------------------

def send_chat_requests(port, prompts, max_tokens, seed, model_name):
    """Send chat completion requests sequentially. Returns outputs."""
    url = f"http://localhost:{port}/v1/chat/completions"
    outputs = []
    for prompt in prompts:
        body = {
            "model": model_name,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": 0,
            "seed": seed,
        }
        r = requests.post(url, json=body, timeout=120)
        r.raise_for_status()
        data = r.json()
        choice = data["choices"][0]
        text = choice["message"]["content"]
        usage = data.get("usage", {})
        outputs.append({
            "text": text,
            "completion_tokens": usage.get("completion_tokens", 0),
        })
    return outputs


# ------------------------------------------------------------------
# Single-config runner
# ------------------------------------------------------------------

def run_config(run_spec, port):
    """Run a single experiment config via vllm server.

    Returns result dict or None on failure.
    """
    config_name = run_spec["config_name"]
    prompts = run_spec["prompts"]
    num_spec = run_spec["num_speculative_tokens"]
    max_tokens = run_spec["max_tokens"]
    seed = run_spec["seed"]
    model_name = run_spec["main_model"]

    logger.info("=" * 60)
    logger.info("Running config: %s", config_name)
    logger.info("spec_config: %s",
                json.dumps(run_spec["spec_config"], indent=2))
    logger.info("=" * 60)

    # Clean up stale cache stats before starting server
    cache_stats_path = "/tmp/parallel_sd_cache_stats.json"
    if os.path.exists(cache_stats_path):
        os.unlink(cache_stats_path)

    # Start server
    proc, stderr_path, stdout_path = start_server(run_spec, port)
    try:
        if not wait_for_server(port):
            logger.error("Server failed to start for %s", config_name)
            proc.kill()
            # Read stderr from temp file
            try:
                with open(stderr_path, "r", errors="replace") as f:
                    stderr = f.read()
                for line in stderr.splitlines()[-20:]:
                    logger.error("  stderr: %s", line)
            except Exception:
                pass
            return None

        # Warmup request
        logger.info("Warmup request...")
        send_chat_requests(port, prompts[:1], max_tokens, seed, model_name)

        # Snapshot metrics before
        metrics_before = collect_prometheus_metrics(port, num_spec)

        # Timed run
        logger.info("Timed run with %d prompts...", len(prompts))
        start = time.perf_counter()
        outputs = send_chat_requests(port, prompts, max_tokens, seed, model_name)
        elapsed = time.perf_counter() - start

        # Snapshot metrics after
        metrics_after = collect_prometheus_metrics(port, num_spec)

    finally:
        stop_server(proc)
        # Read server logs from temp files
        server_stderr = ""
        server_stdout = ""
        try:
            with open(stderr_path, "r", errors="replace") as f:
                server_stderr = f.read()
            with open(stdout_path, "r", errors="replace") as f:
                server_stdout = f.read()
            logger.info("Server stderr: %d bytes, stdout: %d bytes",
                        len(server_stderr), len(server_stdout))
            # Check for cache miss debug info in stdout
            miss_lines = [l for l in server_stdout.splitlines()
                          if "Cache MISS debug" in l]
            if miss_lines:
                logger.info("  Cache miss debug (%d misses):", len(miss_lines))
                for ml in miss_lines[:10]:
                    logger.info("    %s", ml.strip())
            os.unlink(stderr_path)
            os.unlink(stdout_path)
        except Exception as e:
            logger.debug("Failed to read log files: %s", e)

    # Collect results
    total_output_tokens = sum(o["completion_tokens"] for o in outputs)
    tokens_per_sec = total_output_tokens / elapsed if elapsed > 0 else 0.0
    spec_metrics = compute_delta(metrics_before, metrics_after, num_spec)

    # Extract cache stats from temp file written by ParallelProposer
    cache_stats_path = "/tmp/parallel_sd_cache_stats.json"
    try:
        if os.path.exists(cache_stats_path):
            with open(cache_stats_path, "r") as f:
                cache_stats = json.load(f)
            spec_metrics["cache_hit_rate"] = cache_stats.get("hit_rate", 0)
            spec_metrics["cache_half_hit_rate"] = cache_stats.get(
                "half_hit_rate", 0)
            spec_metrics["cache_combined_hit_rate"] = cache_stats.get(
                "combined_hit_rate", 0)
            logger.info(
                "  Cache stats: hit=%d half=%d miss=%d "
                "(hit=%.1f%% half=%.1f%% combined=%.1f%%), layer=%d",
                cache_stats.get("total_hits", 0),
                cache_stats.get("total_half_hits", 0),
                cache_stats.get("total_misses", 0),
                cache_stats.get("hit_rate", 0),
                cache_stats.get("half_hit_rate", 0),
                cache_stats.get("combined_hit_rate", 0),
                cache_stats.get("early_exit_layer", 0),
            )
            os.unlink(cache_stats_path)
    except Exception as e:
        logger.debug("Failed to read cache stats: %s", e)

    # Show first output for sanity check
    if outputs:
        logger.info("Sample output: %s", outputs[0]["text"][:200])

    result = {
        "config_name": config_name,
        "spec_config": run_spec["spec_config"],
        "num_prompts": len(prompts),
        "total_output_tokens": total_output_tokens,
        "elapsed_seconds": round(elapsed, 3),
        "tokens_per_sec": round(tokens_per_sec, 2),
        "spec_metrics": spec_metrics,
        "output_texts": [o["text"] for o in outputs],
    }

    logger.info("Results for %s:", config_name)
    logger.info("  Total output tokens: %d", total_output_tokens)
    logger.info("  Elapsed: %.3fs", elapsed)
    logger.info("  Tokens/sec: %.2f", tokens_per_sec)
    logger.info("  Mean acceptance length: %.3f",
                spec_metrics["mean_acceptance_length"])
    logger.info("  Draft acceptance rate: %.2f%%",
                spec_metrics["draft_acceptance_rate"])
    logger.info("  Per-position acceptance: %s",
                [f"{r:.1f}%"
                 for r in spec_metrics["per_position_acceptance_rate"]])

    return result


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def load_prompts(args):
    """Load prompts from file or use defaults."""
    if args.prompts_file is not None:
        path = args.prompts_file
        if path.endswith(".jsonl"):
            prompts = []
            with open(path) as f:
                for line in f:
                    obj = json.loads(line.strip())
                    if isinstance(obj, str):
                        prompts.append(obj)
                    elif isinstance(obj, dict):
                        for key in ("prompt", "text", "question", "input"):
                            if key in obj:
                                prompts.append(obj[key])
                                break
        elif path.endswith(".json"):
            with open(path) as f:
                data = json.load(f)
            if isinstance(data, list):
                prompts = [
                    x if isinstance(x, str) else x.get("prompt", str(x))
                    for x in data
                ]
            else:
                prompts = data.get("prompts", data.get("data", []))
        else:
            raise ValueError(f"Unsupported file format: {path}")
    else:
        prompts = DEFAULT_PROMPTS

    if args.num_prompts is not None:
        prompts = prompts[: args.num_prompts]

    logger.info("Loaded %d prompts", len(prompts))
    return prompts


def compute_text_match_rate(baseline_texts, test_texts):
    """Compute character-level match rate between baseline and test outputs."""
    total = 0
    matched = 0
    for base, test in zip(baseline_texts, test_texts):
        min_len = min(len(base), len(test))
        total += min_len
        for i in range(min_len):
            if base[i] == test[i]:
                matched += 1
    return (matched / total * 100) if total > 0 else 0.0


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def main():
    args = parse_args()

    # Parse comma-separated early exit layers and top-k values
    early_exit_layers = [int(x) for x in args.early_exit_layers.split(",")]
    top_k_values = [int(x) for x in args.top_k.split(",")]

    prompts = load_prompts(args)
    results = []

    draft_model = args.draft_model or args.main_model
    port = args.port or find_free_port()

    # Common run_spec fields
    base_spec = {
        "main_model": args.main_model,
        "tensor_parallel_size": args.tensor_parallel_size,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "max_model_len": args.max_model_len,
        "max_num_seqs": args.max_num_seqs,
        "max_tokens": args.max_tokens,
        "num_speculative_tokens": args.num_speculative_tokens,
        "enforce_eager": args.enforce_eager,
        "seed": args.seed,
        "prompts": prompts,
    }

    # ----------------------------------------------------------------
    # Config 1: Baseline (no Parallel-SD)
    # ----------------------------------------------------------------
    baseline_result = None
    if not args.skip_baseline:
        run_spec = dict(base_spec)
        run_spec["config_name"] = f"{args.draft_method.upper()} baseline"
        spec_cfg = {
            "method": args.draft_method,
            "num_speculative_tokens": args.num_speculative_tokens,
        }
        # For MTP, don't specify model (vllm auto-fills from target).
        # For eagle/eagle3, need explicit draft model path.
        if args.draft_method not in ("mtp",):
            spec_cfg["model"] = draft_model
            spec_cfg["draft_tensor_parallel_size"] = 1
        run_spec["spec_config"] = spec_cfg
        baseline_result = run_config(run_spec, port)
        if baseline_result:
            results.append(baseline_result)

    # ----------------------------------------------------------------
    # Config 2+: Parallel-SD with different early_exit_layer and top_k
    # ----------------------------------------------------------------
    for layer in early_exit_layers:
        for top_k in top_k_values:
            run_spec = dict(base_spec)
            hh_tag = "+hh" if args.enable_half_cache_hit else ""
            run_spec["config_name"] = (
                f"Parallel{hh_tag} L={layer} k={top_k}"
            )
            spec_cfg = {
                "method": "parallel",
                "num_speculative_tokens": args.num_speculative_tokens,
                "parallel_draft_method": args.draft_method,
                "parallel_top_k": top_k,
                "parallel_early_exit_layer": layer,
                "parallel_enable_half_cache_hit": args.enable_half_cache_hit,
            }
            # For MTP, don't specify model (vllm auto-fills from target).
            # For eagle/eagle3, need explicit draft model path.
            if args.draft_method not in ("mtp",):
                spec_cfg["model"] = draft_model
                spec_cfg["draft_tensor_parallel_size"] = 1
            run_spec["spec_config"] = spec_cfg
            result = run_config(run_spec, port)
            if result is None:
                logger.error(
                    "Skipping failed config: Parallel layer=%d k=%d",
                    layer, top_k,
                )
                continue

            # Compute text match rate vs baseline
            if baseline_result is not None:
                match_rate = compute_text_match_rate(
                    baseline_result["output_texts"],
                    result["output_texts"],
                )
                result["text_match_rate_vs_baseline"] = round(match_rate, 2)
                logger.info("  Text match vs baseline: %.2f%%", match_rate)

            results.append(result)

    # ----------------------------------------------------------------
    # Summary
    # ----------------------------------------------------------------
    if not results:
        logger.error("No results collected!")
        return

    print("\n" + "=" * 100)
    print("EXPERIMENT SUMMARY")
    print("=" * 100)
    print(
        f"{'Config':<25} {'Tokens/s':>10} {'Acceptance':>12} "
        f"{'Accept Rate':>12} {'Hit%':>7} {'Half%':>7} {'Comb%':>7} "
        f"{'Match%':>8}"
    )
    print("-" * 100)
    for r in results:
        sm = r.get("spec_metrics", {})
        match_str = (
            f"{r.get('text_match_rate_vs_baseline', '-'):>7}"
            if isinstance(r.get("text_match_rate_vs_baseline"), float)
            else f"{'N/A':>7}"
        )
        hit_str = (f"{sm['cache_hit_rate']:>6.1f}"
                   if "cache_hit_rate" in sm else f"{'N/A':>6}")
        half_str = (f"{sm['cache_half_hit_rate']:>6.1f}"
                    if "cache_half_hit_rate" in sm else f"{'N/A':>6}")
        comb_str = (f"{sm['cache_combined_hit_rate']:>6.1f}"
                    if "cache_combined_hit_rate" in sm else f"{'N/A':>6}")
        print(
            f"{r['config_name']:<25} "
            f"{r['tokens_per_sec']:>10.2f} "
            f"{sm['mean_acceptance_length']:>12.3f} "
            f"{sm['draft_acceptance_rate']:>11.2f}% "
            f"{hit_str} {half_str} {comb_str} "
            f"{match_str}"
        )
    print("=" * 100)

    # Save results
    if args.save_results:
        save_data = []
        for r in results:
            r_copy = dict(r)
            r_copy.pop("output_texts", None)
            save_data.append(r_copy)

        with open(args.save_results, "w") as f:
            json.dump(save_data, f, indent=2)
        logger.info("Results saved to %s", args.save_results)


if __name__ == "__main__":
    main()
