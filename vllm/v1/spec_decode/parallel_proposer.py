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

import numpy as np
import torch
import torch.nn as nn

from vllm.config import CUDAGraphMode, VllmConfig
from vllm.forward_context import set_forward_context
from vllm.v1.attention.backend import CommonAttentionMetadata
from vllm.v1.sample.ops.penalties import apply_all_penalties
from vllm.v1.spec_decode.eagle import PADDING_SLOT_ID, EagleProposer
from vllm.v1.spec_decode.reuse_cache import ReuseCache

logger = logging.getLogger(__name__)


class ParallelProposer:
    """Parallel Speculative Decoding proposer.

    Wraps an EagleProposer with a Reuse Cache mechanism. On each
    propose() call:
    1. (Phase 4) Generates branches from current hidden_states → cache
    2. (Phase 1) Looks up sampled_token_ids in current round's cache
    3. (Phase 2) Runs standard fallback propose
    4. (Phase 3) Merges cache hits into fallback result

    Branch generation MUST run before cache lookup because both the
    branch keys and lookup keys reference the same draft tokens (the
    ones currently being verified by the target model).
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
        self._draft_method = spec_config.parallel_draft_method or "eagle"
        self.top_k = spec_config.parallel_top_k
        self.enable_half_cache_hit = spec_config.parallel_enable_half_cache_hit
        self.early_exit_layer = spec_config.parallel_early_exit_layer
        self.enable_concurrent = getattr(
            spec_config, "parallel_enable_concurrent", False,
        )

        # Per-request reuse caches: request_id -> ReuseCache
        self._reuse_caches: dict[str, ReuseCache] = {}

        # Target model's lm_head for root token computation
        self._target_lm_head: nn.Module | None = None
        # Target model's final norm (for Eagle early-exit)
        self._target_norm: nn.Module | None = None

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

        # CUDA stream for concurrent branch generation
        self._draft_stream: torch.cuda.Stream | None = None
        if self.enable_concurrent and device.type == "cuda":
            self._draft_stream = torch.cuda.Stream(device=device)

        logger.info(
            "ParallelProposer initialized: top_k=%d, "
            "half_cache_hit=%s, num_spec_tokens=%d, "
            "early_exit_layer=%d, concurrent=%s",
            self.top_k, self.enable_half_cache_hit,
            self._num_speculative_tokens, self.early_exit_layer,
            self.enable_concurrent,
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

    # Standard norm accessor patterns for common model architectures
    _NORM_ACCESSOR_PATTERNS = [
        "model.norm",         # LLaMA, Qwen2, DeepSeek, most HF models
        "transformer.norm",   # Some GPT-style models
    ]

    def _find_target_norm(
        self, target_model: nn.Module,
    ) -> nn.Module | None:
        """Find the final RMSNorm/LayerNorm in the target model.

        Tries standard naming conventions for the final normalization layer.

        Returns:
            The norm module, or None if not found.
        """
        for pattern in self._NORM_ACCESSOR_PATTERNS:
            obj = target_model
            try:
                for attr in pattern.split("."):
                    obj = getattr(obj, attr)
                return obj
            except AttributeError:
                continue
        return None

    def _apply_target_norm(
        self, hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        """Apply the target model's final norm to hidden states.

        Used for Eagle early-exit: intermediate layer hidden states are
        pre-norm and need normalization before being fed to lm_head.

        Args:
            hidden_states: Pre-norm hidden states from early exit layer.

        Returns:
            Normalized hidden states.
        """
        if self._target_norm is not None:
            return self._target_norm(hidden_states)
        # Fallback: return as-is (may produce incorrect logits)
        logger.warning(
            "_apply_target_norm: no target norm found, "
            "returning un-normalized hidden states"
        )
        return hidden_states

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

        # For Eagle early-exit, we need the target model's final norm
        # because Eagle's compute_logits does NOT apply norm internally
        # (unlike MTP's SharedHead which has norm built in).
        if self._draft_method in ("eagle", "eagle3"):
            self._target_norm = self._find_target_norm(target_model)
            if self._target_norm is not None:
                logger.info(
                    "ParallelProposer: found target model's final norm "
                    "for Eagle early-exit normalization"
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

        Algorithm (same-round cache):
          Phase 4: Generate branches from current hidden_states → cache
          Phase 1: Look up sampled_token_ids in current round's cache
          Phase 2: Standard propose as fallback
          Phase 3: Merge cache hits into fallback result

        Branch generation MUST run before cache lookup because both
        the branch keys and the lookup keys reference the SAME draft
        tokens (the ones being verified in the current round).
        """
        batch_size = next_token_ids.shape[0]

        # Resolve last_token_indices
        if last_token_indices is None:
            effective_lti = common_attn_metadata.query_start_loc[1:] - 1
        else:
            effective_lti = last_token_indices

        # Phase 4: Generate branches for current round (populate cache)
        # Uses early exit hidden states (if configured) for root token
        # computation.
        early_exit_hs = self.get_early_exit_hidden_states()
        using_early_exit = early_exit_hs is not None
        branch_hs = early_exit_hs if using_early_exit else target_hidden_states

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

        # Phase 1: Cache Lookup (in current round's cache)
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

        # Track cache hit statistics
        self._total_propose_calls += batch_size
        self._total_cache_hits += len(cache_hits)
        if self._total_propose_calls % 20 == 0 and self._total_propose_calls > 0:
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
        sampling_metadata: "SamplingMetadata | None" = None,
        active_reqs: list[int] | None = None,
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
            sampling_metadata: If provided, penalties (repetition, frequency,
                presence) are applied to logits to match rejection sampler.
            active_reqs: Request indices corresponding to each position,
                needed for per-request penalty lookup.

        Returns:
            Tensor of shape [num_positions, top_k] with token IDs.
        """
        pos_tensor = torch.tensor(
            positions, dtype=torch.long, device=hidden_states.device,
        )
        selected_hs = hidden_states[pos_tensor]

        with torch.no_grad():
            if is_early_exit:
                if self._draft_method == "mtp":
                    logits = self._underlying.model.compute_logits(
                        selected_hs)
                else:
                    normed_hs = self._apply_target_norm(selected_hs)
                    logits = self._target_model.compute_logits(normed_hs)
            else:
                logits = self._target_model.compute_logits(selected_hs)

        if logits is None:
            logger.warning(
                "_compute_root_tokens: logits is None! "
                "is_early_exit=%s, positions=%s, hs_shape=%s",
                is_early_exit, positions, hidden_states.shape,
            )
            return torch.zeros(
                len(positions), top_k,
                dtype=torch.long, device=hidden_states.device,
            )

        # Apply sampling penalties (repetition, frequency, presence) to
        # match the rejection sampler. Without this, root tokens computed
        # from raw logits may not match the penalized argmax that the
        # rejection sampler produces, causing cache misses.
        if (sampling_metadata is not None
                and not sampling_metadata.no_penalties
                and active_reqs is not None):
            logits = self._apply_sampling_penalties(
                logits, sampling_metadata, active_reqs)

        _, top_k_ids = torch.topk(logits, k=top_k, dim=-1)
        return top_k_ids

    def _apply_sampling_penalties(
        self,
        logits: torch.Tensor,
        sampling_metadata: "SamplingMetadata",
        active_reqs: list[int],
    ) -> torch.Tensor:
        """Apply the same penalties that the rejection sampler applies.

        This ensures root tokens match what the rejection sampler actually
        produces, preventing cache misses due to penalty-modified argmax.
        """
        num_positions = logits.shape[0]
        if num_positions == 0:
            return logits

        # Build per-position → per-request mapping
        repeat_indices = torch.tensor(
            active_reqs, dtype=torch.long,
            device=logits.device)

        # Get penalty parameters expanded to per-position
        prompt_token_ids = sampling_metadata.prompt_token_ids
        if prompt_token_ids is None:
            return logits
        prompt_token_ids = prompt_token_ids[repeat_indices]
        presence_penalties = sampling_metadata.presence_penalties[
            repeat_indices]
        frequency_penalties = sampling_metadata.frequency_penalties[
            repeat_indices]
        repetition_penalties = sampling_metadata.repetition_penalties[
            repeat_indices]

        # Use same output_token_ids for all positions of same request
        # (ignoring per-position spec token differences, which is a
        # negligible approximation)
        output_token_ids = [
            sampling_metadata.output_token_ids[r] for r in active_reqs]

        # Convert logits to float32 for penalty application
        logits_f32 = logits.to(torch.float32)
        logits_f32 = apply_all_penalties(
            logits_f32,
            prompt_token_ids,
            presence_penalties,
            frequency_penalties,
            repetition_penalties,
            output_token_ids,
        )
        return logits_f32

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
                sampling_metadata=sampling_metadata,
                active_reqs=active_reqs,
            )
            root_tokens_per_pos.append((active_reqs, top_k_ids))

        # Collect all branches into a flat list for batched generation.
        branches: list[dict] = []
        for pos_idx in range(max_branch_pos):
            rt_info = root_tokens_per_pos[pos_idx]
            if rt_info is None:
                continue
            active_reqs, top_k_ids = rt_info

            for k_idx in range(self.top_k):
                for j, req_idx in enumerate(active_reqs):
                    info = request_infos[req_idx]
                    pos = info["branch_positions"][pos_idx]
                    root_token = top_k_ids[j, k_idx].item()

                    if info["is_prefill"]:
                        key = (root_token,)
                    else:
                        draft_prefix = info["input_ids"][1:pos + 1]
                        key = tuple(draft_prefix) + (root_token,)

                    branches.append({
                        "req_idx": req_idx,
                        "pos": pos,
                        "root_token": root_token,
                        "cache_key": key,
                        "qs": info["qs"],
                        "ql": info["ql"],
                    })

        if not branches:
            return

        # Decide: batched (fast) vs serial (fallback) generation.
        eagle = self._underlying
        total_expanded_tokens = sum(b["ql"] for b in branches)
        use_batched = (
            not eagle.uses_mrope
            and total_expanded_tokens <= eagle.max_num_tokens
            and len(branches) + 1 <= eagle.arange.shape[0]
        )

        if use_batched:
            self._batched_branch_generate(
                branches, target_token_ids, target_positions,
                target_hidden_states, common_attn_metadata,
                num_rejected_tokens_gpu,
            )
        else:
            self._serial_branch_generate(
                branches, target_token_ids, target_positions,
                target_hidden_states, next_token_ids,
                last_token_indices, common_attn_metadata,
                sampling_metadata, mm_embed_inputs,
                num_rejected_tokens_gpu, batch_size,
            )

        logger.debug(
            "Branch generation: batch_size=%d, branches=%d, "
            "batched=%s, max_pos=%d, top_k=%d",
            batch_size, len(branches), use_batched,
            max_branch_pos, self.top_k,
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
    # Serial branch generation (fallback)
    # ------------------------------------------------------------------

    def _serial_branch_generate(
        self,
        branches: list[dict],
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
        """Fallback: generate branches via underlying propose calls.

        Groups non-conflicting branches (different req_idx) into
        single propose calls to reduce overhead.
        """
        # Save common_attn_metadata state.
        saved_seq_lens = common_attn_metadata.seq_lens.clone()
        saved_num_actual_tokens = common_attn_metadata.num_actual_tokens
        saved_max_query_len = common_attn_metadata.max_query_len
        saved_query_start_loc = common_attn_metadata.query_start_loc
        saved_query_start_loc_cpu = common_attn_metadata.query_start_loc_cpu
        saved_seq_lens_cpu = common_attn_metadata._seq_lens_cpu
        saved_num_computed_tokens_cpu = (
            common_attn_metadata._num_computed_tokens_cpu
        )

        # Group non-conflicting branches into batches. Branches
        # with different req_idx can share a single propose call.
        call_groups: list[list[dict]] = []
        for b in branches:
            placed = False
            for group in call_groups:
                if not any(g["req_idx"] == b["req_idx"] for g in group):
                    group.append(b)
                    placed = True
                    break
            if not placed:
                call_groups.append([b])

        for group in call_groups:
            branch_next_tokens = next_token_ids.clone()
            branch_lti = last_token_indices.clone()

            for b in group:
                branch_next_tokens[b["req_idx"]] = b["root_token"]
                branch_lti[b["req_idx"]] = b["qs"] + b["pos"]

            branch_result = self._underlying.propose(
                target_token_ids, target_positions,
                target_hidden_states,
                branch_next_tokens, branch_lti,
                common_attn_metadata, sampling_metadata,
                mm_embed_inputs, num_rejected_tokens_gpu,
            ).clone()

            # Restore common_attn_metadata
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

            for b in group:
                req_id = self._request_ids[b["req_idx"]]
                cache = self._get_or_create_cache(req_id)
                draft_tokens = branch_result[b["req_idx"]].tolist()
                cache.store_branch(b["cache_key"], draft_tokens)

    # ------------------------------------------------------------------
    # Batched branch generation (fast path)
    # ------------------------------------------------------------------

    def _batched_branch_generate(
        self,
        branches: list[dict],
        target_token_ids: torch.Tensor,
        target_positions: torch.Tensor,
        target_hidden_states: torch.Tensor,
        common_attn_metadata: CommonAttentionMetadata,
        num_rejected_tokens_gpu: torch.Tensor | None,
    ) -> None:
        """Generate draft tokens for all branches in a single batched
        set of draft model forwards.

        Instead of calling self._underlying.propose() N times (one per
        branch), this method:
        1. Expands the batch to treat each branch as a virtual request
        2. Runs one first-forward (prefill) for all branches
        3. Runs num_spec-1 decode forwards for all branches
        4. Stores results in per-request caches

        All branches of the same request share the same KV cache block
        table (race condition on writes is acceptable since branch
        results are stored in the reuse cache, not used directly).
        """
        eagle = self._underlying
        device = eagle.device
        num_branches = len(branches)
        num_spec = eagle.num_speculative_tokens

        # Handle eagle3 hidden state combination
        if eagle.method == "eagle3":
            from vllm.model_executor.models.llama_eagle3 import (
                Eagle3LlamaForCausalLM,
            )
            assert isinstance(eagle.model, Eagle3LlamaForCausalLM)
            target_hidden_states = eagle.model.combine_hidden_states(
                target_hidden_states
            )

        # ----------------------------------------------------------
        # Step 1: Build expanded tensors for the first forward
        # ----------------------------------------------------------
        total_tokens = sum(b["ql"] for b in branches)

        # Pre-allocate expanded tensors
        exp_input_ids = torch.empty(
            total_tokens, dtype=torch.int32, device=device,
        )
        exp_positions = torch.empty(
            total_tokens, dtype=torch.int64, device=device,
        )
        exp_hidden_states = torch.empty(
            (total_tokens, eagle.hidden_size),
            dtype=eagle.dtype, device=device,
        )
        exp_qsl = torch.zeros(
            num_branches + 1, dtype=torch.int32, device=device,
        )
        exp_lti = torch.empty(
            num_branches, dtype=torch.int64, device=device,
        )
        exp_seq_lens = torch.empty(
            num_branches,
            dtype=common_attn_metadata.seq_lens.dtype,
            device=device,
        )
        exp_slot_mapping = torch.empty(
            total_tokens,
            dtype=common_attn_metadata.slot_mapping.dtype,
            device=device,
        )

        # Map branch -> original request for block table replication
        branch_req_indices = torch.empty(
            num_branches, dtype=torch.long, device=device,
        )

        offset = 0
        for b_idx, branch in enumerate(branches):
            qs = branch["qs"]
            ql = branch["ql"]
            pos = branch["pos"]

            # Shift input_ids by 1 (same as EagleProposer):
            # input_ids[:-1] = target_token_ids[1:]
            # input_ids[last_token_index] = root_token
            if ql > 1:
                exp_input_ids[offset:offset + ql - 1] = (
                    target_token_ids[qs + 1:qs + ql]
                )
            # Fill the last position (not covered by shift).
            # Must be a valid token to avoid out-of-range embedding.
            exp_input_ids[offset + ql - 1] = (
                target_token_ids[qs + ql - 1]
            )
            # Set the root token at the branch position
            exp_input_ids[offset + pos] = branch["root_token"]

            # Copy positions and hidden_states from original request
            exp_positions[offset:offset + ql] = (
                target_positions[qs:qs + ql]
            )
            exp_hidden_states[offset:offset + ql] = (
                target_hidden_states[qs:qs + ql]
            )

            # Copy slot_mapping from original request
            exp_slot_mapping[offset:offset + ql] = (
                common_attn_metadata.slot_mapping[qs:qs + ql]
            )

            # Set per-branch metadata
            exp_qsl[b_idx + 1] = offset + ql
            exp_lti[b_idx] = offset + pos
            exp_seq_lens[b_idx] = (
                common_attn_metadata.seq_lens[branch["req_idx"]]
            )
            branch_req_indices[b_idx] = branch["req_idx"]

            offset += ql

        # Replicate block table rows for branches
        exp_block_table = (
            common_attn_metadata.block_table_tensor[branch_req_indices]
        )

        # Build expanded CommonAttentionMetadata
        exp_qsl_cpu = exp_qsl.cpu()
        exp_cam = CommonAttentionMetadata(
            query_start_loc=exp_qsl,
            seq_lens=exp_seq_lens,
            query_start_loc_cpu=exp_qsl_cpu,
            _seq_lens_cpu=None,
            _num_computed_tokens_cpu=None,
            num_reqs=num_branches,
            num_actual_tokens=total_tokens,
            max_query_len=max(b["ql"] for b in branches),
            max_seq_len=int(exp_seq_lens.max().item()),
            block_table_tensor=exp_block_table,
            slot_mapping=exp_slot_mapping,
            causal=True,
        )

        # ----------------------------------------------------------
        # Step 2: Run the first forward (prefill-like)
        # ----------------------------------------------------------
        if eagle.attn_metadata_builder is None:
            attn_metadata_builder = eagle._get_attention_metadata_builder()
        else:
            attn_metadata_builder = eagle.attn_metadata_builder

        attn_metadata = attn_metadata_builder.build_for_drafting(
            common_attn_metadata=exp_cam, draft_index=0,
        )
        per_layer_attn_metadata = {
            name: attn_metadata for name in eagle.attn_layer_names
        }
        # Handle indexer layers if present
        if eagle.draft_indexer_metadata_builder:
            draft_indexer_md = (
                eagle.draft_indexer_metadata_builder.build_for_drafting(
                    common_attn_metadata=exp_cam, draft_index=0,
                )
            )
            for name in eagle.indexer_layer_names:
                per_layer_attn_metadata[name] = draft_indexer_md

        # Copy to eagle's persistent buffers
        eagle.input_ids[:total_tokens] = exp_input_ids
        eagle._set_positions(total_tokens, exp_positions)
        eagle.hidden_states[:total_tokens] = exp_hidden_states

        input_ids_arg = eagle.input_ids[:total_tokens]
        inputs_embeds_arg = None
        if eagle.supports_mm_inputs:
            eagle.inputs_embeds[:total_tokens] = (
                eagle.model.embed_input_ids(input_ids_arg)
            )
            input_ids_arg = None
            inputs_embeds_arg = eagle.inputs_embeds[:total_tokens]

        with set_forward_context(
            per_layer_attn_metadata,
            eagle.vllm_config,
            num_tokens=total_tokens,
            cudagraph_runtime_mode=CUDAGraphMode.NONE,
        ):
            ret_hs = eagle.model(
                input_ids=input_ids_arg,
                positions=eagle._get_positions(total_tokens),
                hidden_states=eagle.hidden_states[:total_tokens],
                inputs_embeds=inputs_embeds_arg,
            )
            if eagle.method == "mtp":
                last_hs = ret_hs
                hs = last_hs
            else:
                last_hs, hs = ret_hs

        # Extract hidden states at branch positions
        sample_hs = last_hs[exp_lti]
        logits = eagle.model.compute_logits(sample_hs)

        # Early exit for single draft token
        if num_spec == 1:
            draft_ids = logits.argmax(dim=-1).view(-1, 1)
            self._store_batched_results(branches, draft_ids)
            return

        # Prepare for subsequent decode-style forwards
        positions = exp_positions[exp_lti]
        if eagle.method in (
            "deepseek_mtp", "ernie_mtp",
            "longcat_flash_mtp", "pangu_ultra_moe_mtp",
        ):
            hs = eagle.hidden_states[exp_lti]
        else:
            hs = hs[exp_lti]

        draft_ids = logits.argmax(dim=-1)
        draft_ids_list = [draft_ids]

        # ----------------------------------------------------------
        # Step 3: Update metadata for decode-style forwards
        # ----------------------------------------------------------
        exp_cam.num_actual_tokens = num_branches
        exp_cam.max_query_len = 1
        exp_cam.query_start_loc = eagle.arange[:num_branches + 1]
        exp_cam.query_start_loc_cpu = torch.from_numpy(
            np.arange(num_branches + 1, dtype=np.int32)
        )

        # Apply num_rejected_tokens_gpu adjustment (same as
        # EagleProposer does after first forward).
        if num_spec > 1 and num_rejected_tokens_gpu is not None:
            exp_rejected = num_rejected_tokens_gpu[branch_req_indices]
            exp_cam.seq_lens -= exp_rejected
        exp_cam._seq_lens_cpu = None
        exp_cam._num_computed_tokens_cpu = None

        # ----------------------------------------------------------
        # Step 4: Generate remaining draft tokens
        # ----------------------------------------------------------
        block_size = attn_metadata_builder.kv_cache_spec.block_size

        for token_index in range(num_spec - 1):
            input_ids = draft_ids_list[-1].int()
            positions += 1
            exceeds = positions >= eagle.max_model_len
            clamped_pos = torch.where(exceeds, 0, positions)

            exp_cam.seq_lens += 1
            exp_cam.seq_lens.masked_fill_(exceeds, 1)
            if exp_cam._seq_lens_cpu is not None:
                exp_cam._seq_lens_cpu += 1
            if exp_cam._num_computed_tokens_cpu is not None:
                exp_cam._num_computed_tokens_cpu += 1

            # Compute slot mapping for new positions
            block_numbers = clamped_pos // block_size
            block_ids = exp_cam.block_table_tensor.gather(
                dim=1, index=block_numbers.view(-1, 1),
            ).view(-1)
            exp_cam.slot_mapping = (
                block_ids * block_size + clamped_pos % block_size
            )
            exp_cam.slot_mapping.masked_fill_(exceeds, PADDING_SLOT_ID)

            # Build attention metadata
            attn_metadata = attn_metadata_builder.build_for_drafting(
                common_attn_metadata=exp_cam,
                draft_index=token_index + 1,
            )
            per_layer_attn_metadata = {
                name: attn_metadata
                for name in eagle.attn_layer_names
            }

            # Copy to eagle buffers
            eagle.input_ids[:num_branches] = input_ids
            eagle._set_positions(num_branches, clamped_pos)
            eagle.hidden_states[:num_branches] = hs

            input_ids_arg = eagle.input_ids[:num_branches]
            inputs_embeds_arg = None
            if eagle.supports_mm_inputs:
                eagle.inputs_embeds[:num_branches] = (
                    eagle.model.embed_input_ids(input_ids_arg)
                )
                input_ids_arg = None
                inputs_embeds_arg = (
                    eagle.inputs_embeds[:num_branches]
                )

            with set_forward_context(
                per_layer_attn_metadata,
                eagle.vllm_config,
                num_tokens=num_branches,
                cudagraph_runtime_mode=CUDAGraphMode.NONE,
            ):
                ret_hs = eagle.model(
                    input_ids=input_ids_arg,
                    positions=eagle._get_positions(num_branches),
                    hidden_states=eagle.hidden_states[:num_branches],
                    inputs_embeds=inputs_embeds_arg,
                )
                if eagle.method == "mtp":
                    last_hs = ret_hs
                    hs = ret_hs
                else:
                    last_hs, hs = ret_hs

            hs = hs[:num_branches]
            logits = eagle.model.compute_logits(last_hs[:num_branches])
            draft_ids = logits.argmax(dim=-1)
            draft_ids_list.append(draft_ids)

        # [num_branches, num_speculative_tokens]
        all_draft_ids = torch.stack(draft_ids_list, dim=1)
        self._store_batched_results(branches, all_draft_ids)

    def _store_batched_results(
        self,
        branches: list[dict],
        all_draft_ids: torch.Tensor,
    ) -> None:
        """Store batched branch generation results into per-request caches."""
        for b_idx, branch in enumerate(branches):
            req_id = self._request_ids[branch["req_idx"]]
            cache = self._get_or_create_cache(req_id)
            draft_tokens = all_draft_ids[b_idx].tolist()
            cache.store_branch(branch["cache_key"], draft_tokens)

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
