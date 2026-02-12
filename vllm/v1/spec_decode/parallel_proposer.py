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
    1. Looks up sampled_token_ids in previous round's cache (Phase 1)
    2. Generates branches for NEXT round's cache (Phase 4)
    3. Runs standard fallback propose (Phase 2)
    4. Merges cache hits into fallback result (Phase 3)
    """

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

        # Per-request reuse caches: request_id -> ReuseCache
        self._reuse_caches: dict[str, ReuseCache] = {}

        # Target model's lm_head for root token computation
        self._target_lm_head: nn.Module | None = None

        # State for current propose call (set before propose, cleared after)
        self._sampled_token_ids: torch.Tensor | list | None = None
        self._request_ids: list[str] | None = None

        # Global statistics
        self._global_hit: int = 0
        self._global_miss: int = 0
        self._global_half_hit: int = 0

        logger.info(
            "ParallelProposer initialized: top_k=%d, "
            "half_cache_hit=%s, num_spec_tokens=%d",
            self.top_k, self.enable_half_cache_hit,
            self._num_speculative_tokens,
        )

    def __getattr__(self, name):
        """Delegate attribute access to the underlying EagleProposer."""
        return getattr(self._underlying, name)

    # ------------------------------------------------------------------
    # Overridden methods
    # ------------------------------------------------------------------

    def load_model(self, target_model: nn.Module) -> None:
        self._underlying.load_model(target_model)
        # After load_model, the draft model shares lm_head with target,
        # so we can use self._underlying.model.compute_logits() for
        # root token computation.
        if hasattr(self._underlying.model, "compute_logits"):
            self._target_lm_head = True  # Flag: compute_logits available
            logger.info(
                "ParallelProposer: compute_logits available for root tokens"
            )

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
          Phase 1: Look up sampled_token_ids in previous round's cache
          Phase 4: Generate branches for next round (runs before fallback
                   so fallback leaves correct KV cache state)
          Phase 2: Standard propose as fallback
          Phase 3: Merge cache hits into fallback result
        """
        batch_size = next_token_ids.shape[0]

        # Resolve last_token_indices
        if last_token_indices is None:
            effective_lti = common_attn_metadata.query_start_loc[1:] - 1
        else:
            effective_lti = last_token_indices

        # Phase 1: Cache Lookup
        cache_hits = self._cache_lookup(batch_size)

        # Phase 4: Branch Generation for NEXT round
        # NOTE: Branch generation requires calling propose() multiple times,
        # which corrupts the attention metadata builder's internal state.
        # For num_speculative_tokens == 1, branch generation doesn't help
        # anyway (each branch produces only 1 deterministic token at temp=0).
        # Skip branch generation until multi-call propose() is supported.
        if self._num_speculative_tokens > 1:
            try:
                self._generate_branches(
                    target_token_ids, target_positions,
                    target_hidden_states,
                    next_token_ids, effective_lti, common_attn_metadata,
                    sampling_metadata, mm_embed_inputs,
                    num_rejected_tokens_gpu, batch_size,
                )
            except RuntimeError as e:
                logger.debug("Branch generation skipped: %s", e)

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

        if cache_hits:
            logger.debug(
                "Cache hits merged: %d/%d requests",
                len(cache_hits), batch_size,
            )

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
        """Look up sampled_token_ids in previous round's caches."""
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
    ) -> torch.Tensor:
        """Compute root tokens using the draft model's compute_logits.

        The draft model shares lm_head weights with the target model,
        so compute_logits produces target-equivalent logits.

        Args:
            hidden_states: [num_tokens, hidden_size]
            positions: Absolute token indices to extract.
            top_k: Number of top candidates per position.

        Returns:
            Tensor of shape [num_positions, top_k] with token IDs.
        """
        pos_tensor = torch.tensor(
            positions, dtype=torch.long, device=hidden_states.device,
        )
        selected_hs = hidden_states[pos_tensor]

        with torch.no_grad():
            logits = self._underlying.model.compute_logits(selected_hs)

        _, top_k_ids = torch.topk(logits, k=top_k, dim=-1)
        return top_k_ids

    def _generate_branches(
        self,
        target_token_ids: torch.Tensor,
        target_positions: torch.Tensor,
        target_hidden_states: torch.Tensor,
        next_token_ids: torch.Tensor,
        last_token_indices: torch.Tensor,
        common_attn_metadata: CommonAttentionMetadata,
        sampling_metadata: "SamplingMetadata",
        mm_embed_inputs: tuple | None,
        num_rejected_tokens_gpu: torch.Tensor | None,
        batch_size: int,
    ) -> None:
        """Generate branch continuations and store in per-request caches.

        For each request, determines PREFILL vs DECODE mode and generates
        branches at appropriate positions.
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

            # PREFILL vs DECODE detection
            is_prefill = self._is_prefill_request(i)

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
                target_hidden_states, hs_indices, self.top_k,
            )
            root_tokens_per_pos.append((active_reqs, top_k_ids))

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

    def _is_prefill_request(self, req_idx: int) -> bool:
        """Detect if request is in PREFILL mode."""
        if self._sampled_token_ids is None:
            return True
        if isinstance(self._sampled_token_ids, torch.Tensor):
            if self._sampled_token_ids.dim() == 1:
                return True
            row = self._sampled_token_ids[req_idx]
            valid = int((row != -1).sum().item())
            return valid <= 1
        else:
            tokens = self._sampled_token_ids[req_idx]
            if isinstance(tokens, (list, tuple)):
                valid = sum(1 for t in tokens if t != -1)
                return valid <= 1
            return True

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
