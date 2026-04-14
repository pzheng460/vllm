# SPDX-License-Identifier: Apache-2.0
"""
Measure MTP speculative decoding acceptance rate on any HuggingFace
dataset.

Examples:

    # GPQA Diamond
    python scripts/test_mtp_acceptance_rate.py \
        --model-dir <model_path> \
        --dataset Idavidrein/gpqa --subset gpqa_diamond --split train \
        --text-column Question \
        --num-spec-tokens 3

    # MT-Bench
    python scripts/test_mtp_acceptance_rate.py \
        --model-dir <model_path> \
        --dataset philschmid/mt-bench --split train \
        --text-column turns \
        --num-spec-tokens 3

    # GSM8K
    python scripts/test_mtp_acceptance_rate.py \
        --model-dir <model_path> \
        --dataset openai/gsm8k --subset main --split test \
        --text-column question \
        --num-spec-tokens 3

    # MMLU
    python scripts/test_mtp_acceptance_rate.py \
        --model-dir <model_path> \
        --dataset cais/mmlu --subset all --split test \
        --text-column question \
        --num-spec-tokens 3

    # Use a local jsonl file (one JSON object per line with a "prompt" field)
    python scripts/test_mtp_acceptance_rate.py \
        --model-dir <model_path> \
        --dataset local --dataset-path /path/to/data.jsonl \
        --text-column prompt \
        --num-spec-tokens 3

    # Enable thinking (slow thinking / extended reasoning)
    python scripts/test_mtp_acceptance_rate.py \
        --model-dir <model_path> \
        --dataset openai/gsm8k --subset main --split test \
        --text-column question \
        --num-spec-tokens 3 \
        --enable-thinking --thinking-budget 4096

    # Completion mode (no chat template, text continuation)
    python scripts/test_mtp_acceptance_rate.py \
        --model-dir <model_path> \
        --dataset openai/gsm8k --subset main --split test \
        --text-column question \
        --num-spec-tokens 1 \
        --mode completion

    # Qwen3.5-35B-A3B chat + thinking, TP=8, save output
    python scripts/test_mtp_acceptance_rate.py \
        --model-dir <model_path> \
        --dataset Idavidrein/gpqa --subset gpqa_diamond --split train \
        --text-column Question \
        --num-spec-tokens 1 \
        --mode chat \
        --enable-thinking --thinking-budget 4096 \
        --max-tokens 1024 --max-model-len 4096 \
        --tp 8 --max-num-seqs 256 \
        --save-output output.json

    # MiMo-7B-RL chat + thinking
    python scripts/test_mtp_acceptance_rate.py \
        --model-dir <model_path> \
        --dataset Idavidrein/gpqa --subset gpqa_diamond --split train \
        --text-column Question \
        --num-spec-tokens 1 \
        --mode chat \
        --enable-thinking --thinking-budget 4096 \
        --max-tokens 4096 \
        --save-output output.json
"""

import argparse
import json
import time

from transformers import AutoTokenizer

from vllm import LLM, SamplingParams
from vllm.v1.metrics.reader import Counter, Vector

try:
    from vllm.config.reasoning import ReasoningConfig
except ImportError:
    ReasoningConfig = None


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------

def load_local_dataset(path: str, text_column: str) -> list[str]:
    """Load prompts from a local jsonl/json file."""
    texts = []
    with open(path) as f:
        if path.endswith(".json"):
            data = json.load(f)
            if isinstance(data, list):
                for row in data:
                    texts.append(str(row[text_column]))
            else:
                raise ValueError("JSON file must contain a list of objects")
        else:  # jsonl
            for line in f:
                row = json.loads(line)
                texts.append(str(row[text_column]))
    return texts


def load_hf_dataset(dataset_name: str, subset: str | None, split: str,
                    text_column: str) -> list[str]:
    """Load prompts from a HuggingFace dataset."""
    from datasets import load_dataset
    kwargs = {}
    if subset:
        kwargs["name"] = subset
    ds = load_dataset(dataset_name, split=split, **kwargs)
    print(f"Dataset columns: {ds.column_names}")
    print(f"Total samples: {len(ds)}")

    texts = []
    for row in ds:
        val = row[text_column]
        # Handle list-type columns (e.g. mt-bench "turns" is a list)
        if isinstance(val, list):
            val = "\n".join(str(v) for v in val)
        texts.append(str(val))
    return texts


def apply_chat_template(tokenizer, texts: list[str],
                        enable_thinking: bool = False) -> list[str]:
    """Wrap each text in a chat template."""
    prompts = []
    for text in texts:
        messages = [{"role": "user", "content": text}]
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
            enable_thinking=enable_thinking)
        prompts.append(prompt)
    return prompts


# ---------------------------------------------------------------------------
# Output parsing
# ---------------------------------------------------------------------------

def split_thinking(text: str) -> tuple[str | None, str]:
    """Split thinking and response from model output.

    Handles two cases:
    - Full tags: <think>...</think>response
    - Implicit start: ...thinking...</think>response
      (when chat template already emitted <think>)
    Returns (thinking, response) or (None, text) if no thinking found.
    """
    think_end = text.find("</think>")
    if think_end == -1:
        return None, text

    think_start = text.find("<think>")
    if think_start != -1:
        thinking = text[think_start + len("<think>"):think_end].strip()
    else:
        # Chat template already emitted <think>, output starts mid-thought
        thinking = text[:think_end].strip()

    response = text[think_end + len("</think>"):].strip()
    return thinking, response


# ---------------------------------------------------------------------------
# Metrics collection
# ---------------------------------------------------------------------------

def collect_spec_decode_metrics(llm, num_spec_tokens):
    """Collect cumulative speculative decoding metrics from vLLM."""
    metrics = llm.get_metrics()

    num_drafts = 0
    num_draft_tokens = 0
    num_accepted_tokens = 0
    acceptance_counts = [0] * num_spec_tokens

    for metric in metrics:
        if metric.name == "vllm:spec_decode_num_drafts":
            assert isinstance(metric, Counter)
            num_drafts += metric.value
        elif metric.name == "vllm:spec_decode_num_draft_tokens":
            assert isinstance(metric, Counter)
            num_draft_tokens += metric.value
        elif metric.name == "vllm:spec_decode_num_accepted_tokens":
            assert isinstance(metric, Counter)
            num_accepted_tokens += metric.value
        elif metric.name == "vllm:spec_decode_num_accepted_tokens_per_pos":
            assert isinstance(metric, Vector)
            for pos in range(len(metric.values)):
                if pos < num_spec_tokens:
                    acceptance_counts[pos] += metric.values[pos]

    return num_drafts, num_draft_tokens, num_accepted_tokens, acceptance_counts


def print_report(args, num_prompts, total_output_tokens, elapsed,
                 num_drafts, num_draft_tokens, num_accepted_tokens,
                 acceptance_counts,
                 total_thinking_tokens=0, total_response_tokens=0):
    """Print a summary report."""
    dataset_label = args.dataset
    if args.subset:
        dataset_label += f" / {args.subset}"

    print("\n" + "=" * 60)
    print("MTP Speculative Decoding - Acceptance Rate Report")
    print("=" * 60)
    print(f"Model:                {args.model_dir}")
    print(f"Dataset:              {dataset_label}")
    print(f"Mode:                 {args.mode}")
    print(f"Num prompts:          {num_prompts}")
    print(f"Num spec tokens:      {args.num_spec_tokens}")
    print(f"Thinking enabled:     {args.enable_thinking}")
    if args.thinking_budget is not None:
        print(f"Thinking budget:      {args.thinking_budget}")
    print(f"Total output tokens:  {total_output_tokens}")
    if args.enable_thinking and total_thinking_tokens > 0:
        print(f"  Thinking tokens:    {total_thinking_tokens}")
        print(f"  Response tokens:    {total_response_tokens}")
    print(f"Inference time:       {elapsed:.2f}s")
    print(f"Throughput:           {total_output_tokens / elapsed:.1f} tok/s")
    print("-" * 60)
    print(f"Num drafts:           {num_drafts}")
    print(f"Num draft tokens:     {num_draft_tokens}")
    print(f"Num accepted tokens:  {num_accepted_tokens}")

    if num_draft_tokens > 0:
        acceptance_rate = (num_accepted_tokens / num_draft_tokens) * 100
        print(f"Total acceptance rate:    {acceptance_rate:.2f}%")
    else:
        print("Total acceptance rate:    N/A (no draft tokens)")

    if num_drafts > 0:
        acceptance_length = 1 + (num_accepted_tokens / num_drafts)
        print(f"Mean acceptance length:   {acceptance_length:.4f}")
    else:
        print("Mean acceptance length:   N/A (no drafts)")

    print("-" * 60)
    print("Per-position acceptance rate:")
    for i, count in enumerate(acceptance_counts):
        rate = count / num_drafts if num_drafts > 0 else 0
        print(f"  Position {i}: {rate:.4f} ({count}/{num_drafts})")
    print("=" * 60)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="MTP acceptance rate on any dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter)

    # Dataset args
    ds = parser.add_argument_group("dataset")
    ds.add_argument("--dataset", type=str, default="Idavidrein/gpqa",
                    help="HuggingFace dataset ID, or 'local' for local file")
    ds.add_argument("--subset", type=str, default=None,
                    help="Dataset subset/config (e.g. gpqa_diamond, main)")
    ds.add_argument("--split", type=str, default="train",
                    help="Dataset split (default: train)")
    ds.add_argument("--text-column", type=str, default="Question",
                    help="Column name containing the prompt text")
    ds.add_argument("--dataset-path", type=str, default=None,
                    help="Path to local file (when --dataset=local)")
    ds.add_argument("--num-prompts", type=int, default=None,
                    help="Limit number of prompts (default: all)")

    # Model args
    md = parser.add_argument_group("model")
    md.add_argument("--model-dir", type=str, required=True,
                    help="Model path (must support MTP)")
    md.add_argument("--num-spec-tokens", type=int, default=3)
    md.add_argument("--tp", type=int, default=1)
    md.add_argument("--max-model-len", type=int, default=16384)
    md.add_argument("--enforce-eager", action="store_true")
    md.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    md.add_argument("--max-num-seqs", type=int, default=None)

    # Generation args
    gen = parser.add_argument_group("generation")
    gen.add_argument("--max-tokens", type=int, default=1024,
                       help="Max output tokens per request (default: 1024)")
    gen.add_argument("--temp", type=float, default=0.0)
    gen.add_argument("--top-p", type=float, default=1.0)
    gen.add_argument("--top-k", type=int, default=-1)

    # Thinking / reasoning
    think = parser.add_argument_group("thinking")
    think.add_argument("--enable-thinking", action="store_true",
                       help="Enable thinking/reasoning mode in chat template")
    think.add_argument("--thinking-budget", type=int, default=None,
                       help="Max tokens for thinking (default: no limit)")

    # Misc
    parser.add_argument("--mode", type=str, default="chat",
                        choices=["chat", "completion"],
                        help="Prompt mode: 'chat' applies chat template, "
                             "'completion' uses raw text for continuation "
                             "(default: chat)")
    parser.add_argument("--print-output", action="store_true",
                        help="Print generated text")
    parser.add_argument("--save-output", type=str, default=None,
                        help="Save outputs to a JSON file")

    return parser.parse_args()


def main():
    args = parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model_dir,
                                              trust_remote_code=True)

    # ---- Load dataset ----
    if args.dataset == "local":
        if not args.dataset_path:
            raise ValueError("--dataset-path is required when --dataset=local")
        print(f"Loading local dataset from {args.dataset_path} ...")
        texts = load_local_dataset(args.dataset_path, args.text_column)
    else:
        print(f"Loading dataset {args.dataset} "
              f"(subset={args.subset}, split={args.split}) ...")
        texts = load_hf_dataset(args.dataset, args.subset, args.split,
                                args.text_column)

    if args.num_prompts is not None:
        texts = texts[:args.num_prompts]

    # ---- Validate thinking args ----
    if args.enable_thinking and args.mode != "chat":
        raise ValueError("--enable-thinking requires --mode=chat")
    if args.thinking_budget is not None and not args.enable_thinking:
        raise ValueError("--thinking-budget requires --enable-thinking")
    if args.enable_thinking and args.thinking_budget is None:
        args.thinking_budget = 4096
        print(f"--thinking-budget not set, defaulting to {args.thinking_budget}")
    if args.enable_thinking and ReasoningConfig is None:
        print("WARNING: This vLLM version does not support reasoning_config "
              "or thinking_token_budget. --thinking-budget will be ignored. "
              "Use --max-tokens to control total output length.")

    # ---- Determine prompt mode ----
    use_chat = (args.mode == "chat")

    if use_chat:
        prompts = apply_chat_template(tokenizer, texts,
                                      enable_thinking=args.enable_thinking)
    else:
        prompts = texts

    print(f"Prepared {len(prompts)} prompts (mode={args.mode})")

    # ---- Build LLM with MTP spec decode ----
    speculative_config = {
        "method": "mtp",
        "num_speculative_tokens": args.num_spec_tokens,
    }

    extra_kwargs = {}
    if args.enable_thinking and ReasoningConfig is not None:
        extra_kwargs["reasoning_config"] = ReasoningConfig(
            think_start_str="<think>",
            think_end_str="</think>",
        )

    llm = LLM(
        model=args.model_dir,
        trust_remote_code=True,
        tensor_parallel_size=args.tp,
        enforce_eager=args.enforce_eager,
        gpu_memory_utilization=args.gpu_memory_utilization,
        speculative_config=speculative_config,
        disable_log_stats=False,
        max_model_len=args.max_model_len,
        max_num_seqs=args.max_num_seqs,
        **extra_kwargs,
    )

    sampling_kwargs = dict(
        temperature=args.temp,
        max_tokens=args.max_tokens,
        top_p=args.top_p,
        top_k=args.top_k,
    )
    if args.enable_thinking and ReasoningConfig is not None:
        sampling_kwargs["thinking_token_budget"] = args.thinking_budget

    sampling_params = SamplingParams(**sampling_kwargs)

    # ---- Run inference ----
    print(f"Running inference with MTP spec decoding "
          f"(num_spec_tokens={args.num_spec_tokens}) ...")
    start = time.perf_counter()
    outputs = llm.generate(prompts, sampling_params=sampling_params)
    elapsed = time.perf_counter() - start

    if args.print_output:
        for i, output in enumerate(outputs):
            text = output.outputs[0].text
            thinking, response = split_thinking(text)
            print("=" * 60)
            print(f"[Prompt {i}] {texts[i][:200]}...")
            if thinking is not None:
                print(f"[Thinking] {thinking}")
                print(f"[Response] {response}")
            else:
                print(f"[Output] {text}")

    if args.save_output:
        records = []
        for i, output in enumerate(outputs):
            text = output.outputs[0].text
            thinking, response = split_thinking(text)
            record = {
                "prompt": texts[i],
                "output": text,
                "num_tokens": len(output.outputs[0].token_ids),
            }
            if thinking is not None:
                record["thinking"] = thinking
                record["response"] = response
            records.append(record)
        with open(args.save_output, "w") as f:
            json.dump(records, f, ensure_ascii=False, indent=2)
        print(f"Saved {len(records)} outputs to {args.save_output}")

    # ---- Collect & report metrics ----
    total_output_tokens = sum(
        len(o.outputs[0].token_ids) for o in outputs)

    total_thinking_tokens = 0
    total_response_tokens = 0
    if args.enable_thinking:
        for output in outputs:
            text = output.outputs[0].text
            think_end = text.find("</think>")
            if think_end != -1:
                think_start = text.find("<think>")
                think_text = text[think_start + len("<think>"):think_end] \
                    if think_start != -1 else text[:think_end]
                resp_text = text[think_end + len("</think>"):]
                t_ids = tokenizer.encode(think_text, add_special_tokens=False)
                r_ids = tokenizer.encode(resp_text, add_special_tokens=False)
                total_thinking_tokens += len(t_ids)
                total_response_tokens += len(r_ids)
            else:
                # No </think> found, count all as thinking (budget exhausted)
                total_thinking_tokens += len(output.outputs[0].token_ids)

    num_drafts, num_draft_tokens, num_accepted_tokens, acceptance_counts = \
        collect_spec_decode_metrics(llm, args.num_spec_tokens)

    print_report(args, len(prompts), total_output_tokens, elapsed,
                 num_drafts, num_draft_tokens, num_accepted_tokens,
                 acceptance_counts,
                 total_thinking_tokens, total_response_tokens)


if __name__ == "__main__":
    main()
