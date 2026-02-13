# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Parallel Speculative Decoding Speculator.

Wraps EagleSpeculator with a Reuse Cache mechanism that pre-computes
draft token branches and caches them for reuse when the target model's
actual outputs match a predicted branch.
"""

import logging
import time
from typing import Any

import torch
import torch.nn as nn

from vllm.config import VllmConfig
from vllm.v1.attention.backends.utils import AttentionMetadataBuilder
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.spec_decode.reuse_cache import ReuseCache
from vllm.v1.worker.gpu.block_table import BlockTables
from vllm.v1.worker.gpu.input_batch import InputBatch
from vllm.v1.worker.gpu.sample.metadata import SamplingMetadata
from vllm.v1.worker.gpu.spec_decode.eagle import EagleSpeculator

logger = logging.getLogger(__name__)


class ParallelSpeculator:
    """Parallel Speculative Decoding speculator.

    Wraps an EagleSpeculator with a Reuse Cache mechanism. On each
    propose() call:
    1. (Phase 4) Generates branches from target model hidden_states
       and stores all branches in the per-request ReuseCache
    2. (Phase 1) Looks up cache with sampled_token_ids
    3. (Phase 2) Runs standard eagle propose as fallback
    4. (Phase 3) Cache HIT: merges cached draft tokens into result
       Cache MISS: returns fallback result as-is
    """

    def __init__(self, vllm_config: VllmConfig, device: torch.device):
        self.vllm_config = vllm_config
        self.device = device

        spec_config = vllm_config.speculative_config
        assert spec_config is not None
        assert spec_config.method == "parallel"

        self.num_speculative_tokens = spec_config.num_speculative_tokens
        self.top_k = spec_config.parallel_top_k
        self.draft_method = spec_config.parallel_draft_method
        self.enable_half_cache_hit = spec_config.parallel_enable_half_cache_hit
        self.early_exit_layer = spec_config.parallel_early_exit_layer
        self.enable_concurrent = spec_config.parallel_enable_concurrent
        self._passthrough = False

        # Create the underlying EagleSpeculator
        # Temporarily set method to the underlying draft method for
        # EagleSpeculator initialization
        original_method = spec_config.method
        spec_config.method = self.draft_method
        self._underlying = EagleSpeculator(vllm_config, device)
        spec_config.method = original_method

        # Per-request reuse caches: request_id -> ReuseCache
        self._reuse_caches: dict[str, ReuseCache] = {}

        # Global statistics
        self._global_hit: int = 0
        self._global_miss: int = 0
        self._global_half_hit: int = 0
        self._total_propose_time_ns: int = 0
        self._propose_count: int = 0

        # Store reference to target model's lm_head (set during load_model)
        self._target_lm_head: nn.Module | None = None

        # Early exit support
        self._early_exit_hook_handle = None
        self._early_exit_hidden_states: torch.Tensor | None = None

        # Concurrent execution support
        self._draft_stream: torch.cuda.Stream | None = None
        if self.enable_concurrent and device.type == "cuda":
            self._draft_stream = torch.cuda.Stream(device=device)

        logger.info(
            "ParallelSpeculator initialized: draft_method=%s, top_k=%d, "
            "half_cache_hit=%s, early_exit_layer=%d, concurrent=%s",
            self.draft_method, self.top_k, self.enable_half_cache_hit,
            self.early_exit_layer, self.enable_concurrent,
        )

    # ------------------------------------------------------------------
    # Delegation to underlying EagleSpeculator
    # ------------------------------------------------------------------

    def load_model(self, target_model: nn.Module) -> None:
        self._underlying.load_model(target_model)
        # After load_model, the draft model shares lm_head with target,
        # so we can use self._underlying.model.compute_logits() for
        # root token computation.
        if hasattr(self._underlying.model, "compute_logits"):
            self._target_lm_head = True  # Flag: compute_logits available
            logger.info(
                "ParallelSpeculator: compute_logits available for root tokens"
            )

        # Set up early exit forward hook if needed
        if self.early_exit_layer != -1:
            self._setup_early_exit_hook(target_model)

    def set_attn(
        self,
        kv_cache_config: KVCacheConfig,
        attn_metadata_builders: list[AttentionMetadataBuilder],
        block_tables: BlockTables,
    ) -> None:
        self._underlying.set_attn(
            kv_cache_config, attn_metadata_builders, block_tables
        )

    def capture_model(self) -> None:
        self._underlying.capture_model()

    # ------------------------------------------------------------------
    # Early Exit Support
    # ------------------------------------------------------------------

    # Standard layer accessor patterns for common model architectures
    _LAYER_ACCESSOR_PATTERNS = [
        "model.layers",        # LLaMA, DeepSeek, Pangu, most HF models
        "transformer.layers",  # Some GPT-style models
        "encoder.layers",      # Encoder models
    ]

    def _find_model_layers(
        self, target_model: nn.Module,
    ) -> nn.ModuleList | None:
        """Find transformer layers in the target model.

        Tries standard naming conventions for layer access.

        Returns:
            ModuleList of transformer layers, or None if not found.
        """
        for pattern in self._LAYER_ACCESSOR_PATTERNS:
            obj = target_model
            try:
                for attr in pattern.split("."):
                    obj = getattr(obj, attr)
                if isinstance(obj, (nn.ModuleList, list)):
                    return obj
            except AttributeError:
                continue
        return None

    def _setup_early_exit_hook(self, target_model: nn.Module) -> None:
        """Register a forward hook on the early exit layer.

        The hook captures intermediate hidden_states during the target
        model's forward pass. Supports negative indexing relative to
        the last layer.

        Args:
            target_model: The target model to hook into.
        """
        layers = self._find_model_layers(target_model)
        if layers is None:
            logger.warning(
                "Could not find transformer layers in target model. "
                "Early exit disabled."
            )
            self.early_exit_layer = -1
            return

        num_layers = len(layers)
        # Resolve negative index
        if self.early_exit_layer < 0:
            layer_idx = num_layers + self.early_exit_layer
        else:
            layer_idx = self.early_exit_layer

        if layer_idx < 0 or layer_idx >= num_layers:
            raise ValueError(
                f"early_exit_layer={self.early_exit_layer} is out of range "
                f"for model with {num_layers} layers. "
                f"Resolved index: {layer_idx}"
            )

        target_layer = layers[layer_idx]

        def early_exit_hook(module, input, output):
            # Output from transformer layer is typically
            # (hidden_states,) or hidden_states
            if isinstance(output, tuple):
                hidden_states = output[0]
            else:
                hidden_states = output
            self._early_exit_hidden_states = hidden_states.detach()

        # Remove previous hook if exists
        if self._early_exit_hook_handle is not None:
            self._early_exit_hook_handle.remove()

        self._early_exit_hook_handle = target_layer.register_forward_hook(
            early_exit_hook
        )
        logger.info(
            "Early exit hook registered on layer %d/%d",
            layer_idx, num_layers,
        )

    def get_early_exit_hidden_states(self) -> torch.Tensor | None:
        """Get the hidden_states captured by the early exit hook.

        Returns:
            Hidden states tensor if captured, None otherwise.
        """
        hs = self._early_exit_hidden_states
        self._early_exit_hidden_states = None  # Consume
        return hs

    # ------------------------------------------------------------------
    # Concurrent Execution Support
    # ------------------------------------------------------------------

    def _run_on_draft_stream(self, fn, *args, **kwargs):
        """Run a function on the draft CUDA stream.

        If concurrent execution is disabled or no draft stream exists,
        runs on the default stream.

        Args:
            fn: Function to execute.
            *args, **kwargs: Arguments to pass to fn.

        Returns:
            Result of fn.
        """
        if self._draft_stream is not None and self.enable_concurrent:
            with torch.cuda.stream(self._draft_stream):
                return fn(*args, **kwargs)
        else:
            return fn(*args, **kwargs)

    def sync_draft_stream(self) -> None:
        """Synchronize the draft CUDA stream with the default stream.

        Call this before accessing results from concurrent execution.
        """
        if self._draft_stream is not None:
            self._draft_stream.synchronize()

    # ------------------------------------------------------------------
    # Pass-through properties from underlying speculator
    # ------------------------------------------------------------------

    @property
    def model(self):
        return self._underlying.model

    @property
    def num_speculative_steps(self):
        return self._underlying.num_speculative_steps

    @property
    def hidden_states(self):
        return self._underlying.hidden_states

    @hidden_states.setter
    def hidden_states(self, value):
        self._underlying.hidden_states = value

    @property
    def draft_tokens(self):
        return self._underlying.draft_tokens

    @property
    def input_buffers(self):
        return self._underlying.input_buffers

    @property
    def kv_cache_config(self):
        return self._underlying.kv_cache_config

    @property
    def attn_metadata_builders(self):
        return self._underlying.attn_metadata_builders

    @property
    def block_tables(self):
        return self._underlying.block_tables

    @property
    def cudagraph_manager(self):
        return self._underlying.cudagraph_manager

    @property
    def max_num_reqs(self):
        return self._underlying.max_num_reqs

    @property
    def idx_mapping(self):
        return self._underlying.idx_mapping

    @property
    def temperature(self):
        return self._underlying.temperature

    @property
    def seeds(self):
        return self._underlying.seeds

    @property
    def max_model_len(self):
        return self._underlying.max_model_len

    @property
    def method(self):
        return self._underlying.method

    @property
    def speculative_config(self):
        return self._underlying.speculative_config

    @property
    def draft_model_config(self):
        return self._underlying.draft_model_config

    @property
    def scheduler_config(self):
        return self._underlying.scheduler_config

    @property
    def hidden_size(self):
        return self._underlying.hidden_size

    @property
    def vocab_size(self):
        return self._underlying.vocab_size

    @property
    def dtype(self):
        return self._underlying.dtype

    # ------------------------------------------------------------------
    # ReuseCache management
    # ------------------------------------------------------------------

    def _get_or_create_cache(self, request_id: str) -> ReuseCache:
        """Get or create a ReuseCache for a request."""
        if request_id not in self._reuse_caches:
            self._reuse_caches[request_id] = ReuseCache(
                num_speculative_tokens=self.num_speculative_tokens,
                enable_half_cache_hit=self.enable_half_cache_hit,
            )
        return self._reuse_caches[request_id]

    def remove_request_cache(self, request_id: str) -> None:
        """Remove a request's cache when the request completes."""
        if request_id in self._reuse_caches:
            # Aggregate stats before removing
            stats = self._reuse_caches[request_id].get_statistics()
            self._global_hit += stats["hit"]
            self._global_miss += stats["miss"]
            self._global_half_hit += stats["half_hit"]
            del self._reuse_caches[request_id]

    # ------------------------------------------------------------------
    # Core: Root token computation
    # ------------------------------------------------------------------

    def _compute_root_tokens(
        self,
        hidden_states: torch.Tensor,
        positions: list[int] | torch.Tensor,
        top_k: int,
    ) -> torch.Tensor:
        """Compute root tokens (dummy_sampled_token_ids) using
        the draft model's compute_logits on hidden_states at specified
        positions.

        The draft model shares lm_head weights with the target model,
        so compute_logits produces target-equivalent logits.

        Args:
            hidden_states: Full hidden_states tensor from target model.
                Shape: [num_tokens, hidden_size]
            positions: Token position indices to extract hidden_states from.
            top_k: Number of top candidates per position.

        Returns:
            Tensor of shape [num_positions, top_k] with token IDs.
        """
        if self._target_lm_head is None:
            raise RuntimeError(
                "compute_logits not available on draft model. "
                "Ensure load_model() was called."
            )

        if isinstance(positions, list):
            pos_tensor = torch.tensor(
                positions, dtype=torch.long, device=self.device
            )
        else:
            pos_tensor = positions

        # Extract hidden_states at specified positions
        selected_hs = hidden_states[pos_tensor]  # [num_pos, hidden_size]

        # Compute logits using draft model's compute_logits
        # (shares lm_head weights with target model)
        with torch.no_grad():
            logits = self._underlying.model.compute_logits(
                selected_hs
            )  # [num_pos, vocab]

        # Get top-k token IDs
        _, top_k_ids = torch.topk(logits, k=top_k, dim=-1)  # [num_pos, k]
        return top_k_ids

    # ------------------------------------------------------------------
    # Core: Branch generation and cache population
    # ------------------------------------------------------------------

    def _is_prefill(self, sampled_token_ids: torch.Tensor | list) -> bool:
        """Detect PREFILL vs DECODE mode from sampled_token_ids shape."""
        if isinstance(sampled_token_ids, torch.Tensor):
            num_tokens = sampled_token_ids.shape[-1] if sampled_token_ids.dim() > 1 else 1
        else:
            if isinstance(sampled_token_ids[0], (list, tuple)):
                num_tokens = len(sampled_token_ids[0])
            else:
                num_tokens = 1
        return num_tokens <= 1

    def _generate_branches_for_next_round(
        self,
        input_batch: InputBatch,
        sampling_metadata: SamplingMetadata,
        last_hidden_states: torch.Tensor,
        aux_hidden_states: list[torch.Tensor] | None,
        all_hidden_states: torch.Tensor,
        num_sampled: torch.Tensor,
        num_rejected: torch.Tensor,
        last_sampled: torch.Tensor,
        next_prefill_tokens: torch.Tensor,
        sampled_token_ids: torch.Tensor | None,
    ) -> None:
        """Generate all branch continuations and store in per-request caches.

        For each request, determines PREFILL vs DECODE mode and generates
        branches at appropriate positions by calling the underlying eagle
        speculator with modified num_rejected / last_sampled tensors.

        Branch results are stored in per-request ReuseCache instances
        for lookup in the next round's Phase 1.
        """
        if self._target_lm_head is None:
            return  # Can't compute root tokens without lm_head

        num_reqs = input_batch.num_reqs
        query_start_loc_np = input_batch.query_start_loc_np

        # Clear all per-request caches for a fresh round
        for i in range(num_reqs):
            cache = self._get_or_create_cache(input_batch.req_ids[i])
            cache.clear()

        # Analyze each request: determine mode and branch positions
        request_infos: list[dict | None] = []
        for i in range(num_reqs):
            ns = num_sampled[i].item()
            if ns == 0:
                # Chunked prefill, not yet complete — skip
                request_infos.append(None)
                continue

            qs = int(query_start_loc_np[i])
            qe = int(query_start_loc_np[i + 1])
            ql = qe - qs

            # Per-request PREFILL vs DECODE detection
            if (sampled_token_ids is None
                    or sampled_token_ids.dim() == 1
                    or sampled_token_ids.shape[1] <= 1):
                is_prefill = True
            else:
                valid = int((sampled_token_ids[i] != -1).sum().item())
                is_prefill = (valid <= 1)

            # Get input_ids for cache key construction
            req_input_ids = input_batch.input_ids[qs:qe].tolist()

            if is_prefill:
                # PREFILL: only branch at the last position
                branch_positions = [ql - 1]
            else:
                # DECODE: branch at all positions (up to num_spec + 1)
                max_pos = min(ql, self.num_speculative_tokens + 1)
                branch_positions = list(range(max_pos))

            request_infos.append({
                'is_prefill': is_prefill,
                'qs': qs,
                'ql': ql,
                'input_ids': req_input_ids,
                'branch_positions': branch_positions,
            })

        # Find max number of branch positions across all requests
        max_branch_pos = 0
        for info in request_infos:
            if info is not None:
                max_branch_pos = max(
                    max_branch_pos, len(info['branch_positions'])
                )

        if max_branch_pos == 0:
            return

        # Pre-compute root tokens (batched) for each position index
        # root_tokens_per_pos[pos_idx] = (active_req_indices, top_k_ids)
        root_tokens_per_pos: list[tuple[list[int], torch.Tensor] | None] = []
        for pos_idx in range(max_branch_pos):
            hs_indices: list[int] = []
            active_reqs: list[int] = []
            for i in range(num_reqs):
                info = request_infos[i]
                if info is None or pos_idx >= len(info['branch_positions']):
                    continue
                pos = info['branch_positions'][pos_idx]
                hs_indices.append(info['qs'] + pos)
                active_reqs.append(i)

            if not active_reqs:
                root_tokens_per_pos.append(None)
                continue

            top_k_ids = self._compute_root_tokens(
                all_hidden_states, hs_indices, self.top_k
            )
            root_tokens_per_pos.append((active_reqs, top_k_ids))

        # For each (pos_idx, k_idx): build modified tensors, call propose,
        # and store the resulting draft tokens in per-request caches.
        for pos_idx in range(max_branch_pos):
            rt_info = root_tokens_per_pos[pos_idx]
            if rt_info is None:
                continue
            active_reqs, top_k_ids = rt_info

            for k_idx in range(self.top_k):
                # Clone tensors so we can modify per-request values
                branch_num_rejected = num_rejected.clone()
                branch_last_sampled = last_sampled.clone()
                branch_num_sampled = num_sampled.clone()

                branch_keys: list[tuple[int, tuple[int, ...]]] = []

                for j, req_idx in enumerate(active_reqs):
                    info = request_infos[req_idx]
                    pos = info['branch_positions'][pos_idx]
                    root_token = top_k_ids[j, k_idx].item()

                    if info['is_prefill']:
                        # Force use of last_sampled, process full query
                        branch_num_sampled[req_idx] = 1
                        branch_num_rejected[req_idx] = 0
                    else:
                        # Truncate query to position p+1
                        branch_num_rejected[req_idx] = (
                            info['ql'] - (pos + 1)
                        )

                    # Set root token as the last_sampled for this request
                    branch_last_sampled[req_idx] = root_token

                    # Build cache key
                    if info['is_prefill']:
                        key = (root_token,)
                    else:
                        draft_prefix = info['input_ids'][1:pos + 1]
                        key = tuple(draft_prefix) + (root_token,)

                    branch_keys.append((req_idx, key))

                if not branch_keys:
                    continue

                # Call underlying propose with modified parameters
                branch_result = self._underlying.propose(
                    input_batch=input_batch,
                    sampling_metadata=sampling_metadata,
                    last_hidden_states=last_hidden_states,
                    aux_hidden_states=aux_hidden_states,
                    num_sampled=branch_num_sampled,
                    num_rejected=branch_num_rejected,
                    last_sampled=branch_last_sampled,
                    next_prefill_tokens=next_prefill_tokens,
                ).clone()  # Clone: propose returns view of internal buffer

                # Store draft tokens in per-request caches
                for req_idx, key in branch_keys:
                    req_id = input_batch.req_ids[req_idx]
                    cache = self._get_or_create_cache(req_id)
                    draft_tokens = branch_result[req_idx].tolist()
                    cache.store_branch(key, draft_tokens)

        logger.debug(
            "Branch generation complete: num_reqs=%d, max_pos=%d, "
            "top_k=%d",
            num_reqs, max_branch_pos, self.top_k,
        )

    # ------------------------------------------------------------------
    # Main propose() entry point
    # ------------------------------------------------------------------

    @torch.inference_mode()
    def propose(
        self,
        input_batch: InputBatch,
        sampling_metadata: SamplingMetadata,
        last_hidden_states: torch.Tensor,
        aux_hidden_states: list[torch.Tensor] | None,
        num_sampled: torch.Tensor,
        num_rejected: torch.Tensor,
        last_sampled: torch.Tensor,
        next_prefill_tokens: torch.Tensor,
        # Parallel-SD specific
        all_hidden_states: torch.Tensor | None = None,
        sampled_token_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Generate draft tokens with Parallel-SD cache mechanism.

        Implements a 4-phase algorithm:
          Phase 4: Generate branches for current round (populate cache)
          Phase 1: Look up sampled_token_ids in current round's cache
          Phase 2: Standard eagle propose as fallback
          Phase 3: Merge cache hits into fallback result

        Args:
            input_batch: Current input batch.
            sampling_metadata: Sampling parameters.
            last_hidden_states: Hidden states from target model last layer.
            aux_hidden_states: Auxiliary hidden states (for eagle3).
            num_sampled: Number of sampled tokens per request.
            num_rejected: Number of rejected tokens per request.
            last_sampled: Last sampled token per request.
            next_prefill_tokens: Next prefill tokens.
            all_hidden_states: Full hidden_states from target model
                at all positions (for branch generation).
            sampled_token_ids: Sampled token IDs from target model
                verification (accepted_tokens + correction_token).

        Returns:
            Draft token tensor of shape [num_reqs, num_speculative_steps].
        """
        start_time = time.monotonic_ns()

        # Pass-through mode: delegate to underlying speculator
        if self._passthrough or all_hidden_states is None:
            result = self._underlying.propose(
                input_batch=input_batch,
                sampling_metadata=sampling_metadata,
                last_hidden_states=last_hidden_states,
                aux_hidden_states=aux_hidden_states,
                num_sampled=num_sampled,
                num_rejected=num_rejected,
                last_sampled=last_sampled,
                next_prefill_tokens=next_prefill_tokens,
            )
            elapsed = time.monotonic_ns() - start_time
            self._total_propose_time_ns += elapsed
            self._propose_count += 1
            return result

        num_reqs = input_batch.num_reqs

        # --------------------------------------------------------------
        # Phase 4: Generate branches for current round (populate cache)
        # Must run BEFORE cache lookup because both branch keys and
        # lookup keys reference the SAME draft tokens (current round's
        # input_ids being verified by the target model).
        # --------------------------------------------------------------
        early_hs = self.get_early_exit_hidden_states()
        branch_hs = early_hs if early_hs is not None else all_hidden_states

        self._generate_branches_for_next_round(
            input_batch=input_batch,
            sampling_metadata=sampling_metadata,
            last_hidden_states=last_hidden_states,
            aux_hidden_states=aux_hidden_states,
            all_hidden_states=branch_hs,
            num_sampled=num_sampled,
            num_rejected=num_rejected,
            last_sampled=last_sampled,
            next_prefill_tokens=next_prefill_tokens,
            sampled_token_ids=sampled_token_ids,
        )

        # --------------------------------------------------------------
        # Phase 1: Cache Lookup (in current round's cache)
        # --------------------------------------------------------------
        cache_hits: dict[int, list[int]] = {}
        if sampled_token_ids is not None:
            for i in range(num_reqs):
                req_id = input_batch.req_ids[i]
                cache = self._reuse_caches.get(req_id)
                if cache is None:
                    continue

                # Extract lookup key from sampled_token_ids.
                # Use num_sampled to determine valid token count because
                # rejection_sample() uses torch.empty — positions beyond
                # num_sampled contain uninitialized garbage, not -1.
                n = int(num_sampled[i].item())
                if n <= 0:
                    continue

                if sampled_token_ids.dim() == 1:
                    tokens = [sampled_token_ids[i].item()]
                else:
                    tokens = sampled_token_ids[i, :n].tolist()

                hit = cache.lookup(tokens)
                if hit is not None and len(hit) > 0:
                    cache_hits[i] = hit

        # --------------------------------------------------------------
        # Phase 2: Fallback — standard eagle propose
        # (Runs to maintain correct eagle KV cache state)
        # --------------------------------------------------------------
        fallback_result = self._underlying.propose(
            input_batch=input_batch,
            sampling_metadata=sampling_metadata,
            last_hidden_states=last_hidden_states,
            aux_hidden_states=aux_hidden_states,
            num_sampled=num_sampled,
            num_rejected=num_rejected,
            last_sampled=last_sampled,
            next_prefill_tokens=next_prefill_tokens,
        ).clone()  # MUST clone — returns VIEW of internal buffer

        # --------------------------------------------------------------
        # Phase 3: Merge cache hits into fallback result
        # --------------------------------------------------------------
        for req_idx, cached_tokens in cache_hits.items():
            n = min(len(cached_tokens), fallback_result.shape[1])
            for t in range(n):
                fallback_result[req_idx, t] = cached_tokens[t]

        if cache_hits:
            logger.debug(
                "Cache hits merged: %d/%d requests",
                len(cache_hits), num_reqs,
            )

        elapsed = time.monotonic_ns() - start_time
        self._total_propose_time_ns += elapsed
        self._propose_count += 1

        return fallback_result

    # ------------------------------------------------------------------
    # Pass-through mode
    # ------------------------------------------------------------------

    def set_passthrough(self, enabled: bool) -> None:
        """Enable/disable pass-through mode for A/B testing."""
        self._passthrough = enabled
        logger.info("ParallelSpeculator pass-through mode: %s", enabled)

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------

    def get_cache_statistics(self) -> dict[str, Any]:
        """Get cache statistics (per-request and global).

        Returns:
            Dict with:
            - 'global': {hit, miss, half_hit} aggregated counts
            - 'per_request': {request_id: {hit, miss, half_hit}} for
              active requests
            - 'avg_propose_time_us': average propose latency in microseconds
        """
        per_request = {}
        for req_id, cache in self._reuse_caches.items():
            per_request[req_id] = cache.get_statistics()

        # Add current active request stats to global counts
        active_hit = sum(s["hit"] for s in per_request.values())
        active_miss = sum(s["miss"] for s in per_request.values())
        active_half = sum(s["half_hit"] for s in per_request.values())

        avg_propose = (
            (self._total_propose_time_ns / self._propose_count / 1000)
            if self._propose_count > 0 else 0.0
        )

        return {
            "global": {
                "hit": self._global_hit + active_hit,
                "miss": self._global_miss + active_miss,
                "half_hit": self._global_half_hit + active_half,
            },
            "per_request": per_request,
            "avg_propose_time_us": avg_propose,
            "passthrough": self._passthrough,
        }

    def reset_cache_statistics(self) -> None:
        """Reset all statistics counters."""
        self._global_hit = 0
        self._global_miss = 0
        self._global_half_hit = 0
        self._total_propose_time_ns = 0
        self._propose_count = 0
        for cache in self._reuse_caches.values():
            cache.reset_statistics()

    # ------------------------------------------------------------------
    # Pass-through for other EagleSpeculator methods/attributes
    # ------------------------------------------------------------------

    def run_model(self, *args, **kwargs):
        return self._underlying.run_model(*args, **kwargs)

    def generate_draft(self, *args, **kwargs):
        return self._underlying.generate_draft(*args, **kwargs)
