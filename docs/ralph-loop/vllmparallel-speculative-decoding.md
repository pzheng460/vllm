# Parallel Speculative Decoding for vLLM v1

## Overview

Implement Parallel Speculative Decoding (Parallel-SD) in vLLM v0.14.0rc1 v1 engine. Based on the Mirror Speculative Decoding paper with adaptations, the core innovation is a **Reuse Cache** mechanism that pre-computes draft tokens for multiple speculative branches and caches them for reuse when the target model's actual outputs match a predicted branch.

**Priority model**: Pangu MTP (same interface as standard MTP, no special handling needed).
**Supported draft methods**: Eagle, Eagle3, MTP.

## Out of Scope

- Multi-device (multi-GPU/NPU) concurrent execution (future extension)
- Token Channel real implementation (stub interface only)
- Adaptive top-k auto-tuning (metrics tracked but no auto-adjustment)
- Draft model KV cache continuity between propose() rounds
- Cross-request ReuseCache sharing

---

## Architecture

### Wrapping Level

ParallelSpeculator wraps EagleSpeculator at the `vllm/v1/worker/gpu/spec_decode/` level (Speculator level, not Proposer level). This provides full control over GPU resources, CUDA graphs, and attention metadata.

```
┌─────────────────────────────────────────────────────────┐
│                  GPU Model Runner                       │
│  execute_model() → scoring → propose()                  │
└─────────────┬───────────────────────────────────────────┘
              │
              ▼
┌─────────────────────────────────────────────────────────┐
│              ParallelSpeculator                          │
│  - Manages ReuseCache (per-request)                      │
│  - Generates branches as virtual requests                │
│  - Cache HIT → return cached draft                       │
│  - Cache MISS → delegate to underlying speculator        │
│  - Metrics: HIT/MISS/HALF-HIT tracking                  │
│  - Pass-through mode for A/B testing                     │
│                                                          │
│  ┌──────────────────────────────────────────────┐       │
│  │         EagleSpeculator (underlying)          │       │
│  │  - Standard Eagle/MTP spec decode             │       │
│  │  - Unmodified propose() interface             │       │
│  │  - CUDA graph capture                         │       │
│  └──────────────────────────────────────────────┘       │
└─────────────────────────────────────────────────────────┘
```

### Core Flow

**Single propose() step:**

1. **Target model verifies** previous draft → produces `sampled_token_ids` (accepted_tokens + correction_token) + `hidden_states` at ALL positions
2. **Compute root_tokens**: Apply target model's own `lm_head` on hidden_states at each position → top-k tokens per position
3. **Generate branches**: Each (position, root_token) pair becomes a virtual request. All branches batched into a single draft model forward pass. Each branch autoregressively generates `num_speculative_tokens` continuation tokens.
4. **Store in ReuseCache**: All branches stored with their lookup keys
5. **Cache lookup**: Use `sampled_token_ids` (filtered of -1) as lookup key
6. **HIT**: Return cached branch's continuation tokens as draft
7. **MISS**: Fall back to standard `propose()` via underlying EagleSpeculator (double forward pass accepted)

### PREFILL vs DECODE Detection

- **PREFILL**: `sampled_token_ids[0]` has 1 element (only correction_token). Only branch at last position. Branch count = `top_k`.
- **DECODE**: `sampled_token_ids[0]` has multiple elements (num_spec_tokens + 1). Branch at ALL positions. Branch count = `(num_spec_tokens + 1) * top_k`.

### Branch Generation (Virtual Requests)

Each branch is treated as an independent virtual request in the draft model's batched forward pass:
- Independent KV cache allocation per branch
- Pre-allocated per request at request start time
- Fresh KV cache each round (no cross-round dependency)
- CUDA graphs with padding to max batch size for static shapes

### Key Design: DECODE Mode

```
input_ids = [last_accepted, d1, d2, d3]  (num_spec_tokens + 1)
hidden_states = target_model_hidden_states at all 4 positions
root_tokens = top-k(lm_head(hidden_states)) at each position

Branch 0: pos=0, context=[], root=a → key=(a), draft=[x1,x2,x3]
Branch 1: pos=1, context=[d1], root=b → key=(d1,b), draft=[y1,y2,y3]
Branch 2: pos=2, context=[d1,d2], root=c → key=(d1,d2,c), draft=[z1,z2,z3]
Branch 3: pos=3, context=[d1,d2,d3], root=d → key=(d1,d2,d3,d), draft=[w1,w2,w3]

Lookup: sampled_token_ids=[d1,d2,z0] (filter -1)
HIT if z0 == c → return [z1,z2,z3]
```

### Key Design: PREFILL Mode

```
input_ids = [prompt tokens...]
hidden_states = target_model_hidden_states (only last position used)
root_tokens = top-k(lm_head(hidden_states[-1]))  (k=2 example)

Branch 0: root=a → key=(a), draft=[x1,x2,x3]
Branch 1: root=b → key=(b), draft=[y1,y2,y3]

Lookup: sampled_token_ids=[z0] (correction_token only)
HIT if z0 == a or z0 == b
```

### Half-Cache Hit

When `parallel_enable_half_cache_hit=True` and `top_k==1` and full match fails:
- Try prefix match: all tokens except last match a cache key
- If prefix matches, return that branch's draft tokens
- Half-Cache Hit and Cache Miss are mutually exclusive

---

## Components

### ReuseCache (`vllm/v1/spec_decode/reuse_cache.py`)

```python
@dataclass
class ReuseCacheEntry:
    continuation_tokens: list[int]  # Draft tokens after root
    base_position: int = 0          # Branch starting position

class ReuseCache:
    """Per-request cache manager with Input-Prefix-Aware Keys."""

    def __init__(self, num_speculative_tokens: int, device: torch.device,
                 enable_half_cache_hit: bool = False):
        self.cache: dict[tuple[int, ...], ReuseCacheEntry] = {}
        self._hit_count: int = 0
        self._miss_count: int = 0
        self._half_hit_count: int = 0

    def store_branch(self, lookup_key: tuple[int, ...],
                     branch_tokens: list[int],
                     base_position: int = 0) -> None: ...

    def lookup(self, lookup_key: list[int]) -> Optional[list[int]]:
        """Lookup with optional half-cache-hit fallback."""
        ...

    def clear(self) -> None: ...

    def get_statistics(self) -> dict[str, int]:
        """Return {hit, miss, half_hit} counts."""
        ...

    def reset_statistics(self) -> None: ...
```

**Properties:**
- Per-request scope (one ReuseCache per active request)
- Cleared and re-populated each propose() round
- Keys are tuples of token IDs (prefix + root_token)
- Statistics tracked per-request and aggregated globally

### ParallelSpeculator (`vllm/v1/worker/gpu/spec_decode/parallel_speculator.py`)

```python
class ParallelSpeculator:
    """Wraps EagleSpeculator with Reuse Cache mechanism."""

    def __init__(self, vllm_config: VllmConfig, device: torch.device):
        self._underlying: EagleSpeculator  # Standard Eagle/MTP speculator
        self._reuse_caches: dict[str, ReuseCache]  # Per-request caches
        self.top_k: int  # From config: parallel_top_k
        self.draft_method: str  # From config: parallel_draft_method
        self.enable_half_cache_hit: bool
        self.enable_concurrent: bool  # Phase 2
        self.early_exit_layer: int  # Phase 2
        self._passthrough: bool  # Pass-through mode for A/B testing

    def load_model(self, target_model: nn.Module) -> None:
        """Delegate to underlying + setup early exit hooks."""
        ...

    def propose(self, input_batch, sampling_metadata,
                all_hidden_states: Optional[torch.Tensor] = None,
                **kwargs) -> torch.Tensor:
        """Main entry: branch generation → cache lookup → return draft."""
        ...

    def _compute_root_tokens(self, hidden_states: torch.Tensor,
                             target_lm_head: nn.Module,
                             positions: list[int],
                             top_k: int) -> torch.Tensor:
        """Apply target lm_head to hidden_states, return top-k tokens."""
        ...

    def _generate_branches(self, root_tokens, hidden_states,
                           input_ids, ...) -> list[Branch]:
        """Generate all branches as virtual requests via batched propose()."""
        ...

    def _cache_lookup(self, request_id: str,
                      sampled_token_ids: list[int]) -> Optional[list[int]]:
        """Look up ReuseCache for matching branch."""
        ...

    def get_cache_statistics(self) -> dict[str, dict[str, int]]:
        """Per-request and global cache statistics."""
        ...

    def reset_cache_statistics(self) -> None: ...
```

### Config Changes (`vllm/config/speculative.py`)

Add `"parallel"` to `SpeculativeMethod` type:
```python
SpeculativeMethod = Literal[
    "ngram", "medusa", "mlp_speculator", "draft_model", "suffix",
    "parallel",  # NEW
    EagleModelTypes,
]
```

New config parameters:
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `parallel_draft_method` | str | "eagle" | Underlying draft method: "eagle", "eagle3", "mtp" |
| `parallel_top_k` | int | 1 | Top-κ candidates per position |
| `parallel_enable_half_cache_hit` | bool | False | Enable prefix matching (only when top_k=1) |
| `parallel_early_exit_layer` | int | -1 | Early exit layer index (-1 = last layer) |
| `parallel_enable_concurrent` | bool | False | Enable concurrent draft+target execution |

**Smart defaults**: If `method="parallel"` and no `parallel_draft_method` specified:
- If `model` path provided → default to "eagle"
- If no `model` path → default to "mtp" (uses target model's built-in MTP layers)

### Routing (`vllm/v1/worker/gpu/spec_decode/__init__.py`)

```python
def init_speculator(vllm_config, device):
    if speculative_config.method == "parallel":
        return ParallelSpeculator(vllm_config, device)
    elif speculative_config.use_eagle():
        return EagleSpeculator(vllm_config, device)
    raise NotImplementedError(...)
```

### Model Runner Changes (`vllm/v1/worker/gpu/model_runner.py`)

- During target model forward pass (scoring/verification), save the full `hidden_states` tensor
- Pass `all_hidden_states` to `speculator.propose()` as an optional parameter
- For non-parallel methods, this parameter is ignored (backward-compatible)

### Early Exit (Phase 2)

- Register PyTorch `register_forward_hook()` on target model layer at index `early_exit_layer`
- Hook captures intermediate hidden_states during forward pass
- Model-agnostic: works with any transformer that follows `model.model.layers[i]` convention
- Fallback layer name mappings for non-standard models

### Concurrent Execution (Phase 2)

- When `parallel_enable_concurrent=True` and `early_exit_layer != -1`:
  - Target model forward on default CUDA stream
  - At early exit layer, extract hidden_states, signal draft model
  - Draft model branch generation on separate CUDA stream
  - Synchronization barrier before cache lookup
- Configurable via config flag (default: off)

### TokenChannel Stub (`vllm/v1/spec_decode/token_channel.py`)

```python
class TokenChannel:
    """Stub interface for future NPU cross-device communication."""

    def send_top_k_candidates(self, token_ids, probabilities, batch_size, ...):
        raise NotImplementedError("TokenChannel is a reserved interface")

    def recv_top_k_candidates(self, batch_size, ...):
        raise NotImplementedError("TokenChannel is a reserved interface")

    def send_speculative_tokens(self, spec_token_ids, batch_size, ...):
        raise NotImplementedError("TokenChannel is a reserved interface")

    def recv_speculative_tokens(self, batch_size, ...):
        raise NotImplementedError("TokenChannel is a reserved interface")
```

### Observability

- **DEBUG logging**: All cache operations (store, lookup, hit, miss, half-hit)
- **Metrics**: HIT/MISS/HALF-HIT counts per request and global aggregation
- **Cache operation latency**: Tracked and logged
- **Memory tracking**: Branch KV cache allocation monitoring
- **Pass-through mode**: Config flag to disable Parallel-SD and behave as pure EagleSpeculator (for A/B comparison)

---

## User Stories

### US-1: ReuseCache Data Structure

**Description**: Implement the ReuseCache and ReuseCacheEntry classes with store, lookup, half-hit, clear, and statistics functionality.

**Acceptance Criteria**:
- `store_branch()` stores entries keyed by tuple of token IDs
- `lookup()` returns continuation tokens on exact key match, None on miss
- `lookup()` with half-cache-hit enabled tries prefix match when exact fails (only when top_k=1)
- `clear()` removes all entries
- `get_statistics()` returns accurate hit/miss/half_hit counts
- `reset_statistics()` zeroes all counters
- Unit tests pass: 10+ test cases covering store/lookup/miss/half-hit/clear/stats

### US-2: SpeculativeConfig Extensions

**Description**: Add "parallel" method type and parallel-specific configuration parameters to SpeculativeConfig.

**Acceptance Criteria**:
- `method="parallel"` accepted in config without error
- All 5 new parameters parseable from speculative_config dict
- Smart defaults: eagle when model path provided, mtp when not
- Validation: `parallel_enable_half_cache_hit` only valid when `parallel_top_k == 1`
- `use_eagle()` returns appropriate value based on `parallel_draft_method`
- Config unit tests pass

### US-3: ParallelSpeculator Core

**Description**: Implement ParallelSpeculator class that wraps EagleSpeculator, computes root tokens, generates branches as virtual requests, manages ReuseCache, and handles HIT/MISS flow.

**Acceptance Criteria**:
- Wraps EagleSpeculator (delegates `load_model()`, `set_attn()`, etc.)
- Computes root_tokens using target model's lm_head on hidden_states
- Generates branches as virtual requests in batched propose() call
- Stores all branches in per-request ReuseCache
- Cache HIT returns cached draft tokens without extra draft model call
- Cache MISS falls back to standard propose()
- PREFILL mode: only branch at last position (branch count = top_k)
- DECODE mode: branch at all positions (branch count = (num_spec + 1) * top_k)
- Pass-through mode works (pure EagleSpeculator behavior)
- Unit tests with mock EagleSpeculator pass

### US-4: Routing and Model Runner Integration

**Description**: Add routing for "parallel" method in init_speculator() and modify model runner to pass all_hidden_states to propose().

**Acceptance Criteria**:
- `init_speculator()` returns ParallelSpeculator when `method="parallel"`
- Model runner extracts full hidden_states tensor during target model forward
- `all_hidden_states` passed to `propose()` as optional parameter
- Non-parallel methods unaffected (backward-compatible)
- Integration test: ParallelSpeculator loads and runs with real Eagle model

### US-5: KV Cache and CUDA Graph Support

**Description**: Implement per-request KV cache pre-allocation for virtual branch requests and CUDA graph support with padding.

**Acceptance Criteria**:
- Each request pre-allocates KV cache slots for max_branches at request start
- KV slots released when request completes
- CUDA graphs captured with padded max batch size
- Branch virtual requests use pre-allocated KV slots
- Fresh KV cache each propose() round (no cross-round dependency)

### US-6: Metrics and Observability

**Description**: Implement cache statistics tracking, DEBUG logging, and pass-through mode.

**Acceptance Criteria**:
- HIT/MISS/HALF-HIT counts tracked per request
- Global aggregation across all requests
- Cache operation latency logged at DEBUG level
- `get_cache_statistics()` returns structured metrics dict
- Pass-through mode disables all Parallel-SD logic
- Metrics visible in experiment script output

### US-7: Early Exit Implementation (Phase 2)

**Description**: Implement early exit hidden_states extraction using PyTorch forward hooks on target model layers.

**Acceptance Criteria**:
- Forward hook registered on `model.model.layers[early_exit_layer]`
- Hook captures intermediate hidden_states during target model forward
- `parallel_early_exit_layer=-1` uses last layer (default, equivalent to no early exit)
- Negative indices supported (relative to last layer)
- Works with LLaMA, DeepSeek, and Pangu model architectures
- Fallback layer name mappings for non-standard models

### US-8: Concurrent Execution (Phase 2)

**Description**: Implement optional concurrent execution of draft model on separate CUDA stream while target model continues processing.

**Acceptance Criteria**:
- `parallel_enable_concurrent=True` enables multi-stream execution
- Draft model branch generation runs on separate CUDA stream
- Target model continues on default stream after early exit point
- Synchronization barrier before cache lookup
- Disabled by default (sequential execution)
- No correctness difference between concurrent and sequential modes

### US-9: Experiment Script

**Description**: Implement comprehensive experiment script supporting all datasets and comparison modes.

**Acceptance Criteria**:
- Script at `examples/parallel_spec_decode_experiment.py`
- Supports datasets: gpqa_diamond, gpqa_diamond_orig, gsm8k, humaneval, mmlu, mt_bench, custom, local
- Local dataset support: JSON, JSONL, CSV, Parquet, directories
- `--compare-eagle` flag for Eagle baseline comparison
- `--compare-mtp` flag for MTP baseline comparison
- `--top-k` accepts list of values to test
- `--save-results` outputs structured JSON
- Reports: cache HIT/MISS rates, throughput, acceptance rates
- All CLI arguments as specified in CLAUDE.md

### US-10: TokenChannel Stub

**Description**: Create TokenChannel stub interface for future NPU cross-device communication.

**Acceptance Criteria**:
- File at `vllm/v1/spec_decode/token_channel.py`
- All 4 methods defined with NotImplementedError
- Docstrings document intended usage and communication complexity
- No actual communication implementation

### US-11: Integration Tests

**Description**: Comprehensive integration test suite for Parallel-SD.

**Acceptance Criteria**:
- Token-exact match: Parallel-SD with top_k=1 produces identical output to standard Eagle
- Statistical: Parallel-SD with top_k>1 achieves HIT rate > 0 on test prompts
- PREFILL/DECODE mode transitions work correctly
- Half-cache-hit produces valid (non-error) output
- Multi-request batch works correctly
- Cache statistics are accurate (counts match actual hits/misses)
- Test with both Eagle and MTP underlying methods

---

## Implementation Phases

### Phase 1: Minimal Working Parallel-SD

**User Stories**: US-1, US-2, US-3, US-4, US-5, US-6

**Files Changed**:
| File | Action | Description |
|------|--------|-------------|
| `vllm/v1/spec_decode/reuse_cache.py` | NEW | ReuseCache + ReuseCacheEntry |
| `vllm/v1/worker/gpu/spec_decode/parallel_speculator.py` | NEW | ParallelSpeculator class |
| `vllm/config/speculative.py` | MODIFY | Add "parallel" method + params |
| `vllm/v1/worker/gpu/spec_decode/__init__.py` | MODIFY | Add routing for "parallel" |
| `vllm/v1/worker/gpu/model_runner.py` | MODIFY | Pass all_hidden_states to propose() |
| `tests/v1/spec_decode/test_reuse_cache.py` | NEW | ReuseCache unit tests |
| `tests/v1/spec_decode/test_parallel_speculator.py` | NEW | ParallelSpeculator unit tests |

**Verification**:
- `pytest tests/v1/spec_decode/test_reuse_cache.py` passes
- `pytest tests/v1/spec_decode/test_parallel_speculator.py` passes
- Existing Eagle/MTP tests still pass (no regression)
- Manual test with LLaMA + Eagle: `method="parallel"` produces correct output
- Token-exact match verified for top_k=1

### Phase 2: Early Exit + Concurrent Execution

**User Stories**: US-7, US-8

**Files Changed**:
| File | Action | Description |
|------|--------|-------------|
| `vllm/v1/worker/gpu/spec_decode/parallel_speculator.py` | MODIFY | Add early exit hooks + concurrent stream |
| `vllm/v1/worker/gpu/model_runner.py` | MODIFY | Support early exit hidden_states extraction |
| `tests/v1/spec_decode/test_parallel_speculator.py` | MODIFY | Add early exit + concurrent tests |

**Verification**:
- Early exit with sequential execution produces correct output
- Concurrent execution produces identical output to sequential
- `parallel_enable_concurrent=False` (default) behavior unchanged from Phase 1
- Performance metrics show overlap when concurrent enabled

### Phase 3: Experiment Script + Polish

**User Stories**: US-9, US-10, US-11

**Files Changed**:
| File | Action | Description |
|------|--------|-------------|
| `examples/parallel_spec_decode_experiment.py` | NEW | Full experiment script |
| `vllm/v1/spec_decode/token_channel.py` | NEW | TokenChannel stub |
| `tests/v1/spec_decode/test_parallel_integration.py` | NEW | Integration tests |

**Verification**:
- Experiment script runs on custom dataset without errors
- All dataset loaders work (gpqa_diamond, gsm8k, humaneval, etc.)
- Comparison modes (--compare-eagle, --compare-mtp) produce valid reports
- Integration tests pass for both Eagle and MTP backends
- TokenChannel stub imports without error

---

## Configuration Reference

### Basic Usage (Eagle)
```python
llm = LLM(
    model="LLM-Research/Meta-Llama-3.1-8B-Instruct",
    speculative_config={
        "method": "parallel",
        "model": "vllm-ascend/EAGLE-LLaMA3.1-Instruct-8B",
        "num_speculative_tokens": 3,
    },
)
```

### Basic Usage (MTP / Pangu)
```python
llm = LLM(
    model="deepseek-ai/DeepSeek-V3",  # or Pangu model
    speculative_config={
        "method": "parallel",
        "num_speculative_tokens": 3,
        "parallel_draft_method": "mtp",
    },
)
```

### Full Configuration
```python
llm = LLM(
    model="LLM-Research/Meta-Llama-3.1-8B-Instruct",
    speculative_config={
        "method": "parallel",
        "model": "vllm-ascend/EAGLE-LLaMA3.1-Instruct-8B",
        "num_speculative_tokens": 3,
        "parallel_top_k": 1,
        "parallel_draft_method": "eagle",
        "parallel_enable_half_cache_hit": False,
        "parallel_early_exit_layer": -1,
        "parallel_enable_concurrent": False,
    },
)
```

---

## Error Handling

- All errors propagate up and interrupt execution (no silent fallback)
- Branch generation errors (OOM, numerical instability) are fatal
- Invalid configuration (e.g., half_cache_hit with top_k>1) raises ValueError at init
- Edge cases (EOS in sampled_token_ids, all drafts rejected) always processed through full Parallel-SD flow

## Memory Model

- **ReuseCache**: Python dict of token tuples → small memory footprint
- **Branch KV cache**: Pre-allocated per request. Max = `(num_spec_tokens + 1) * top_k * num_spec_tokens` KV blocks per request
- **Hidden states**: Full tensor from target model, O(seq_len * hidden_dim) per request per round
- **CUDA graphs**: Padded to max batch size including virtual branch requests
