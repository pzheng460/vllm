# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Parallel Speculative Decoding Proposer.

Wraps EagleProposer with a Reuse Cache mechanism that pre-computes
draft token branches and caches them for reuse when the target model's
actual outputs match a predicted branch.

This module operates at the Proposer level (vllm/v1/spec_decode/)
and wraps the runtime EagleProposer, unlike ParallelSpeculator which
wraps EagleSpeculator at the Speculator level (vllm/v1/worker/gpu/spec_decode/).
"""

import logging
from typing import Any

import torch
import torch.nn as nn

from vllm.config import VllmConfig
from vllm.v1.attention.backend import CommonAttentionMetadata
from vllm.v1.spec_decode.eagle import EagleProposer
from vllm.v1.spec_decode.reuse_cache import ReuseCache

logger = logging.getLogger(__name__)


class ParallelProposer:
    """Parallel Speculative Decoding proposer.

    Wraps an EagleProposer with a Reuse Cache mechanism. On each
    propose() call:
    1. (Phase 4) Generates branches and stores in current round's cache
    2. (Phase 1) Looks up sampled_token_ids in current round's cache
    3. (Phase 2) Runs standard fallback propose
    4. (Phase 3) Merges cache hits into fallback result
    """

    # Standard layer accessor patterns for common model architectures
    _LAYER_ACCESSOR_PATTERNS = [
        "model.layers",        # LLaMA, DeepSeek, Pangu, most HF models
        "transformer.layers",  # Some GPT-style models
        "encoder.layers",      # Encoder models
    ]

    def __init__(
        self,
        vllm_config: VllmConfig,
        device: torch.device,
        runner=None,
    ):
        self._underlying = EagleProposer(vllm_config, device, runner)

        spec_config = vllm_config.speculative_config
        assert spec_config is not None
        assert spec_config.method == "parallel"

        self._num_speculative_tokens = spec_config.num_speculative_tokens
        self.top_k = spec_config.parallel_top_k
        self.enable_half_cache_hit = spec_config.parallel_enable_half_cache_hit
        self.early_exit_layer = spec_config.parallel_early_exit_layer

        # Per-request reuse caches: request_id -> ReuseCache
        self._reuse_caches: dict[str, ReuseCache] = {}

        # Target model's lm_head for root token computation
        self._target_lm_head: nn.Module | None = None

        # Early exit support
        self._early_exit_hook_handle = None
        self._total_propose_calls = 0
        self._total_cache_hits = 0
        self._early_exit_hidden_states: torch.Tensor | None = None

        # State for current propose call (set before propose, cleared after)
        self._sampled_token_ids: torch.Tensor | list | None = None
        self._request_ids: list[str] | None = None

        # Global statistics
        self._global_hit: int = 0
        self._global_miss: int = 0
        self._global_half_hit: int = 0

        logger.info(
            "ParallelProposer initialized: top_k=%d, "
            "half_cache_hit=%s, num_spec_tokens=%d, "
            "early_exit_layer=%d",
            self.top_k, self.enable_half_cache_hit,
            self._num_speculative_tokens, self.early_exit_layer,
        )

    def __getattr__(self, name):
        """Delegate attribute access to the underlying EagleProposer."""
        return getattr(self._underlying, name)

    # ------------------------------------------------------------------
    # Early Exit Support
    # ------------------------------------------------------------------

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
            Consumes the captured states (one-time read).
        """
        hs = self._early_exit_hidden_states
        self._early_exit_hidden_states = None  # Consume
        return hs

    # ------------------------------------------------------------------
    # Overridden methods
    # ------------------------------------------------------------------

    def load_model(self, target_model: nn.Module) -> None:
        self._underlying.load_model(target_model)

        # Save reference to target model for root token computation.
        # IMPORTANT: Must use TARGET model's compute_logits, NOT the
        # draft model's. The draft model's compute_logits (MTP) applies
        # SharedHead.norm(hidden_states) internally, but target_hidden_states
        # already has final RMSNorm applied from the target model's forward
        # pass. Using the draft model's compute_logits would double-normalize.
        self._target_model = target_model
        if hasattr(target_model, "compute_logits"):
            self._target_lm_head = True
            logger.info(
                "ParallelProposer: using TARGET model's compute_logits "
                "for root tokens (avoids double normalization)"
            )

        # Set up early exit forward hook if needed
        if self.early_exit_layer != -1:
            self._setup_early_exit_hook(target_model)

    def propose(
        self,
        target_token_ids: torch.Tensor,
        target_positions: torch.Tensor,
        target_hidden_states: torch.Tensor,
        next_token_ids: torch.Tensor,
        last_token_indices: torch.Tensor | None,
        common_attn_metadata: CommonAttentionMetadata,
        sampling_metadata: "SamplingMetadata",
        mm_embed_inputs: tuple[list[torch.Tensor], torch.Tensor] | None = None,
        num_rejected_tokens_gpu: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Generate draft tokens with Parallel-SD cache mechanism.

        Implements a 4-phase algorithm:
          Phase 4: Generate branches for current round (populate cache)
          Phase 1: Look up sampled_token_ids in current round's cache
          Phase 2: Standard propose as fallback
          Phase 3: Merge cache hits into fallback result
        """
        batch_size = next_token_ids.shape[0]

        # Resolve last_token_indices
        if last_token_indices is None:
            effective_lti = common_attn_metadata.query_start_loc[1:] - 1
        else:
            effective_lti = last_token_indices

        # Get early exit hidden states (if available) for branch generation
        early_exit_hs = self.get_early_exit_hidden_states()
        using_early_exit = early_exit_hs is not None
        branch_hs = early_exit_hs if using_early_exit else target_hidden_states

        # Phase 4: Branch Generation for CURRENT round
        # Must run first so cache is populated before lookup.
        # Uses early exit hidden states (if configured) for root token
        # computation, but fallback propose still uses target_hidden_states
        # (draft model needs last-layer HS).
        try:
            self._generate_branches(
                target_token_ids, target_positions,
                branch_hs, target_hidden_states,
                next_token_ids, effective_lti, common_attn_metadata,
                sampling_metadata, mm_embed_inputs,
                num_rejected_tokens_gpu, batch_size,
                is_early_exit=using_early_exit,
            )
        except RuntimeError as e:
            logger.debug("Branch generation skipped: %s", e)

        # Phase 1: Cache Lookup (in CURRENT round's cache)
        # Now that branches are generated and stored, lookup by
        # sampled_token_ids finds matching branches from this round.
        cache_hits = self._cache_lookup(batch_size)

        # Phase 2: Fallback (standard propose)
        fallback = self._underlying.propose(
            target_token_ids, target_positions, target_hidden_states,
            next_token_ids, last_token_indices, common_attn_metadata,
            sampling_metadata, mm_embed_inputs, num_rejected_tokens_gpu,
        ).clone()  # MUST clone — propose returns view of internal buffer

        # Phase 3: Merge cache hits
        for req_idx, cached_tokens in cache_hits.items():
            n = min(len(cached_tokens), fallback.shape[1])
            for t in range(n):
                fallback[req_idx, t] = cached_tokens[t]

        # Track cache hit statistics (using ReuseCache's own counters)
        self._total_propose_calls += batch_size
        self._total_cache_hits += len(cache_hits)
        if self._total_propose_calls % 20 == 0 and self._total_propose_calls > 0:
            # Collect detailed stats from all active caches
            global_stats = self.get_cache_statistics()["global"]
            total_hit = global_stats["hit"]
            total_half = global_stats["half_hit"]
            total_miss = global_stats["miss"]
            total_lookups = total_hit + total_half + total_miss
            hit_rate = (total_hit / total_lookups * 100
                        if total_lookups > 0 else 0.0)
            half_rate = (total_half / total_lookups * 100
                         if total_lookups > 0 else 0.0)
            logger.warning(
                "Parallel-SD cache stats: hit=%d half=%d miss=%d "
                "total=%d (hit=%.1f%% half=%.1f%%), "
                "early_exit_layer=%d",
                total_hit, total_half, total_miss, total_lookups,
                hit_rate, half_rate, self.early_exit_layer,
            )
            # Write to a temp file for external collection
            try:
                import json as _json
                stats = {
                    "total_calls": self._total_propose_calls,
                    "total_lookups": total_lookups,
                    "total_hits": total_hit,
                    "total_half_hits": total_half,
                    "total_misses": total_miss,
                    "hit_rate": round(hit_rate, 2),
                    "half_hit_rate": round(half_rate, 2),
                    "combined_hit_rate": round(
                        (total_hit + total_half) / total_lookups * 100
                        if total_lookups > 0 else 0.0, 2),
                    "early_exit_layer": self.early_exit_layer,
                }
                with open("/tmp/parallel_sd_cache_stats.json", "w") as _f:
                    _json.dump(stats, _f)
            except Exception:
                pass

        # Clear per-call state
        self._sampled_token_ids = None
        self._request_ids = None

        return fallback

    # ------------------------------------------------------------------
    # State management
    # ------------------------------------------------------------------

    def set_sampled_token_ids(
        self,
        sampled_token_ids: torch.Tensor | list,
        request_ids: list[str],
    ) -> None:
        """Store sampled_token_ids for cache lookup in the next propose().

        Must be called before propose() in each round.
        """
        self._sampled_token_ids = sampled_token_ids
        self._request_ids = request_ids

    def remove_request_cache(self, req_id: str) -> None:
        """Remove a request's cache when the request finishes."""
        if req_id in self._reuse_caches:
            stats = self._reuse_caches[req_id].get_statistics()
            self._global_hit += stats["hit"]
            self._global_miss += stats["miss"]
            self._global_half_hit += stats["half_hit"]
            del self._reuse_caches[req_id]

    # ------------------------------------------------------------------
    # Phase 1: Cache Lookup
    # ------------------------------------------------------------------

    def _cache_lookup(self, batch_size: int) -> dict[int, list[int]]:
        """Look up sampled_token_ids in current round's caches."""
        cache_hits: dict[int, list[int]] = {}
        if self._sampled_token_ids is None or self._request_ids is None:
            return cache_hits

        for i in range(min(batch_size, len(self._request_ids))):
            req_id = self._request_ids[i]
            cache = self._reuse_caches.get(req_id)
            if cache is None:
                continue

            # Extract tokens for this request
            if isinstance(self._sampled_token_ids, torch.Tensor):
                if self._sampled_token_ids.dim() == 1:
                    tokens = [self._sampled_token_ids[i].item()]
                else:
                    tokens = self._sampled_token_ids[i].tolist()
            else:
                tokens = list(self._sampled_token_ids[i])

            # Filter -1 (rejected/padding)
            filtered = [t for t in tokens if t != -1]
            if not filtered:
                continue

            hit = cache.lookup(filtered)
            if hit is not None and len(hit) > 0:
                cache_hits[i] = hit

        return cache_hits

    # ------------------------------------------------------------------
    # Phase 4: Branch Generation
    # ------------------------------------------------------------------

    def _get_or_create_cache(self, req_id: str) -> ReuseCache:
        """Get or create a ReuseCache for a request."""
        if req_id not in self._reuse_caches:
            self._reuse_caches[req_id] = ReuseCache(
                num_speculative_tokens=self._num_speculative_tokens,
                enable_half_cache_hit=self.enable_half_cache_hit,
            )
        return self._reuse_caches[req_id]

    def _compute_root_tokens(
        self,
        hidden_states: torch.Tensor,
        positions: list[int],
        top_k: int,
        is_early_exit: bool = False,
    ) -> torch.Tensor:
        """Compute root tokens from hidden states.

        Uses different compute_logits paths depending on the source:
        - target_hidden_states (post-norm): Use TARGET model's compute_logits
          which expects already-normalized hidden states (no extra norm).
        - early_exit hidden states (pre-norm): Use DRAFT model's compute_logits
          which applies SharedHead.norm() internally (needed for un-normed hs).

        Args:
            hidden_states: [num_tokens, hidden_size]
            positions: Absolute token indices to extract.
            top_k: Number of top candidates per position.
            is_early_exit: True if hidden_states come from an early exit
                hook (pre-norm), False if from target model output (post-norm).

        Returns:
            Tensor of shape [num_positions, top_k] with token IDs.
        """
        pos_tensor = torch.tensor(
            positions, dtype=torch.long, device=hidden_states.device,
        )
        selected_hs = hidden_states[pos_tensor]

        with torch.no_grad():
            if is_early_exit:
                # Early exit HS are pre-norm → use draft model's
                # compute_logits which applies SharedHead.norm() internally
                logits = self._underlying.model.compute_logits(selected_hs)
            else:
                # Target HS are post-norm → use target model's
                # compute_logits which does NOT apply extra norm
                logits = self._target_model.compute_logits(selected_hs)

        if logits is None:
            logger.warning(
                "_compute_root_tokens: logits is None! "
                "is_early_exit=%s, positions=%s, hs_shape=%s",
                is_early_exit, positions, hidden_states.shape,
            )
            # Fallback: return zeros
            return torch.zeros(
                len(positions), top_k,
                dtype=torch.long, device=hidden_states.device,
            )

        _, top_k_ids = torch.topk(logits, k=top_k, dim=-1)
        return top_k_ids

    def _generate_branches(
        self,
        target_token_ids: torch.Tensor,
        target_positions: torch.Tensor,
        branch_hidden_states: torch.Tensor,
        target_hidden_states: torch.Tensor,
        next_token_ids: torch.Tensor,
        last_token_indices: torch.Tensor,
        common_attn_metadata: CommonAttentionMetadata,
        sampling_metadata: "SamplingMetadata",
        mm_embed_inputs: tuple | None,
        num_rejected_tokens_gpu: torch.Tensor | None,
        batch_size: int,
        is_early_exit: bool = False,
    ) -> None:
        """Generate branch continuations and store in per-request caches.

        For each request, determines PREFILL vs DECODE mode and generates
        branches at appropriate positions.

        Args:
            branch_hidden_states: Hidden states for root token computation
                (may be from early exit layer).
            target_hidden_states: Last-layer hidden states for draft model
                propose calls.
            is_early_exit: True if branch_hidden_states come from early
                exit hook (pre-norm), affecting compute_logits path.
        """
        if self._target_lm_head is None or self._request_ids is None:
            return

        qsl = common_attn_metadata.query_start_loc
        if qsl.device.type != "cpu":
            qsl_cpu = qsl.cpu()
        else:
            qsl_cpu = qsl

        # Analyze requests and determine branch positions
        request_infos: list[dict | None] = []
        for i in range(batch_size):
            if i >= len(self._request_ids):
                request_infos.append(None)
                continue

            qs = int(qsl_cpu[i].item())
            qe = int(qsl_cpu[i + 1].item())
            ql = qe - qs

            # PREFILL vs DECODE detection (use query length, not valid token count)
            is_prefill = self._is_prefill_request(i, ql)

            # Get input_ids for cache key construction
            req_input_ids = target_token_ids[qs:qe].tolist()

            if is_prefill:
                branch_positions = [ql - 1]
            else:
                max_pos = min(ql, self._num_speculative_tokens + 1)
                branch_positions = list(range(max_pos))

            request_infos.append({
                "is_prefill": is_prefill,
                "qs": qs,
                "ql": ql,
                "input_ids": req_input_ids,
                "branch_positions": branch_positions,
            })

        # Clear caches for fresh round
        for i in range(batch_size):
            if i < len(self._request_ids):
                cache = self._get_or_create_cache(self._request_ids[i])
                cache.clear()

        # Find max branch positions
        max_branch_pos = 0
        for info in request_infos:
            if info is not None:
                max_branch_pos = max(
                    max_branch_pos, len(info["branch_positions"])
                )

        if max_branch_pos == 0:
            return

        # Pre-compute root tokens per position
        root_tokens_per_pos: list[
            tuple[list[int], torch.Tensor] | None
        ] = []
        for pos_idx in range(max_branch_pos):
            hs_indices: list[int] = []
            active_reqs: list[int] = []
            for i in range(batch_size):
                info = request_infos[i]
                if info is None or pos_idx >= len(info["branch_positions"]):
                    continue
                pos = info["branch_positions"][pos_idx]
                hs_indices.append(info["qs"] + pos)
                active_reqs.append(i)

            if not active_reqs:
                root_tokens_per_pos.append(None)
                continue

            top_k_ids = self._compute_root_tokens(
                branch_hidden_states, hs_indices, self.top_k,
                is_early_exit=is_early_exit,
            )
            root_tokens_per_pos.append((active_reqs, top_k_ids))

        # Save common_attn_metadata state before branch generation.
        # EagleProposer.propose() modifies common_attn_metadata in place
        # (seq_lens, num_actual_tokens, query_start_loc, etc.), so we
        # must save and restore after each branch call to prevent
        # cumulative corruption.
        saved_seq_lens = common_attn_metadata.seq_lens.clone()
        saved_num_actual_tokens = common_attn_metadata.num_actual_tokens
        saved_max_query_len = common_attn_metadata.max_query_len
        saved_query_start_loc = common_attn_metadata.query_start_loc
        saved_query_start_loc_cpu = common_attn_metadata.query_start_loc_cpu
        saved_seq_lens_cpu = common_attn_metadata._seq_lens_cpu
        saved_num_computed_tokens_cpu = (
            common_attn_metadata._num_computed_tokens_cpu
        )

        # Generate branches: for each (pos, k), call underlying propose
        for pos_idx in range(max_branch_pos):
            rt_info = root_tokens_per_pos[pos_idx]
            if rt_info is None:
                continue
            active_reqs, top_k_ids = rt_info

            for k_idx in range(self.top_k):
                branch_next_tokens = next_token_ids.clone()
                branch_lti = last_token_indices.clone()
                branch_keys: list[tuple[int, tuple[int, ...]]] = []

                for j, req_idx in enumerate(active_reqs):
                    info = request_infos[req_idx]
                    pos = info["branch_positions"][pos_idx]
                    root_token = top_k_ids[j, k_idx].item()

                    # Set root token as the next_token for this request
                    branch_next_tokens[req_idx] = root_token

                    # Adjust last_token_indices to point to position p
                    branch_lti[req_idx] = info["qs"] + pos

                    # Build cache key
                    if info["is_prefill"]:
                        key = (root_token,)
                    else:
                        draft_prefix = info["input_ids"][1:pos + 1]
                        key = tuple(draft_prefix) + (root_token,)

                    branch_keys.append((req_idx, key))

                if not branch_keys:
                    continue

                # Call underlying propose with modified inputs
                branch_result = self._underlying.propose(
                    target_token_ids, target_positions,
                    target_hidden_states,
                    branch_next_tokens, branch_lti,
                    common_attn_metadata, sampling_metadata,
                    mm_embed_inputs, num_rejected_tokens_gpu,
                ).clone()

                # Restore common_attn_metadata to pre-branch state
                common_attn_metadata.seq_lens.copy_(saved_seq_lens)
                common_attn_metadata.num_actual_tokens = (
                    saved_num_actual_tokens
                )
                common_attn_metadata.max_query_len = saved_max_query_len
                common_attn_metadata.query_start_loc = (
                    saved_query_start_loc
                )
                common_attn_metadata.query_start_loc_cpu = (
                    saved_query_start_loc_cpu
                )
                common_attn_metadata._seq_lens_cpu = saved_seq_lens_cpu
                common_attn_metadata._num_computed_tokens_cpu = (
                    saved_num_computed_tokens_cpu
                )

                # Store draft tokens in per-request caches
                for req_idx, key in branch_keys:
                    req_id = self._request_ids[req_idx]
                    cache = self._get_or_create_cache(req_id)
                    draft_tokens = branch_result[req_idx].tolist()
                    cache.store_branch(key, draft_tokens)

        logger.debug(
            "Branch generation: batch_size=%d, max_pos=%d, top_k=%d",
            batch_size, max_branch_pos, self.top_k,
        )

    def _is_prefill_request(self, req_idx: int, query_length: int) -> bool:
        """Detect if request is in PREFILL mode.

        Uses query length to distinguish PREFILL (long prompt) from DECODE
        (short verification input of num_spec + 1 tokens). This is more
        reliable than counting valid tokens in sampled_token_ids, because
        after a rejected draft, sampled_token_ids has only 1 valid token
        which would be incorrectly classified as PREFILL.
        """
        return query_length > self._num_speculative_tokens + 1

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------

    def get_cache_statistics(self) -> dict[str, Any]:
        """Get cache statistics."""
        per_request = {}
        for req_id, cache in self._reuse_caches.items():
            per_request[req_id] = cache.get_statistics()

        active_hit = sum(s["hit"] for s in per_request.values())
        active_miss = sum(s["miss"] for s in per_request.values())
        active_half = sum(s["half_hit"] for s in per_request.values())

        return {
            "global": {
                "hit": self._global_hit + active_hit,
                "miss": self._global_miss + active_miss,
                "half_hit": self._global_half_hit + active_half,
            },
            "per_request": per_request,
        }

    def reset_cache_statistics(self) -> None:
        """Reset all statistics counters."""
        self._global_hit = 0
        self._global_miss = 0
        self._global_half_hit = 0
        for cache in self._reuse_caches.values():
            cache.reset_statistics()
