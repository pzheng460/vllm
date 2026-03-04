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

import json as _json
import logging
import os
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
        self.disable_targeted_branch = bool(
            os.environ.get("VLLM_DISABLE_TARGETED_BRANCH", "")
        )
        # When targeted branch is disabled, force half-cache-hit on so
        # cache misses still use early-exit branch tokens (wrong root but
        # same prefix), rather than falling back to standard MTP propose
        # which would contaminate acceptance length statistics.
        if self.disable_targeted_branch:
            self.enable_half_cache_hit = True
        self._max_num_seqs = vllm_config.scheduler_config.max_num_seqs

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
        self._early_exit_residual: torch.Tensor | None = None

        # State for current propose call (set before propose, cleared after)
        self._sampled_token_ids: torch.Tensor | list | None = None
        self._request_ids: list[str] | None = None

        # Global statistics
        self._global_hit: int = 0
        self._global_miss: int = 0
        self._global_half_hit: int = 0

        # CUDA stream for async branch generation (always created on CUDA)
        self._draft_stream: torch.cuda.Stream | None = None
        self._device = device
        if device.type == "cuda":
            self._draft_stream = torch.cuda.Stream(device=device)

        # Pre-computed async branch state
        self._stored_batch_state: dict | None = None
        self._standard_branches_launched: bool = False
        self._standard_branches_done: bool = False
        self._request_infos_proposer: list[dict | None] = []

        # GPU-native async branch gen results (no CPU transfer until sync)
        self._async_draft_ids: torch.Tensor | None = None
        self._async_root_tokens: torch.Tensor | None = None

        # Cross-device (heterogeneous) deployment
        self._draft_device: torch.device | None = None
        self._remote_eagle: EagleProposer | None = None
        self._remote_forward_context: dict | None = None
        self._remote_target_model: nn.Module | None = None
        self._remote_target_norm: nn.Module | None = None
        self._kv_sync_event: torch.cuda.Event | None = None
        self._vllm_config = vllm_config

        draft_device_str = getattr(
            spec_config, "parallel_draft_device", None,
        )
        if draft_device_str is not None:
            self._draft_device = torch.device(draft_device_str)
            # Create the draft stream on the remote device
            self._draft_stream = torch.cuda.Stream(
                device=self._draft_device,
            )
            self._kv_sync_event = torch.cuda.Event()

        logger.info(
            "ParallelProposer initialized: top_k=%d, "
            "half_cache_hit=%s, num_spec_tokens=%d, "
            "early_exit_layer=%d, concurrent=%s, draft_device=%s",
            self.top_k, self.enable_half_cache_hit,
            self._num_speculative_tokens, self.early_exit_layer,
            self.enable_concurrent, self._draft_device,
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
        self,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Apply the target model's final norm to hidden states.

        For models using fused_add_rms_norm (e.g. Pangu), the decoder layer
        returns (hidden_states, residual) where hidden_states is the MLP
        output.  The final norm must be: norm(hidden_states + residual).
        RMSNorm with two arguments does fused_add_rms_norm internally.

        Args:
            hidden_states: Pre-norm hidden states from early exit layer.
            residual: Residual tensor for fused_add_rms_norm, or None.

        Returns:
            Normalized hidden states ready for compute_logits.
        """
        if self._target_norm is not None:
            if residual is not None:
                # fused_add_rms_norm: norm(hs + residual)
                normed, _ = self._target_norm(hidden_states, residual)
                return normed
            return self._target_norm(hidden_states)
        # Fallback: return as-is (may produce incorrect logits)
        logger.warning_once(
            "_apply_target_norm: no target norm found, "
            "returning un-normalized hidden states"
        )
        if residual is not None:
            return hidden_states + residual
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
            # (hidden_states, residual) or hidden_states.
            # Models using fused_add_rms_norm (e.g. Pangu) return
            # (hidden_states, residual) where hidden_states is the MLP
            # output and residual carries the accumulated representation.
            # Both must be captured for correct norm application:
            # final_hs = norm(hidden_states, residual) = rms_norm(hs + res)
            if isinstance(output, tuple):
                hidden_states = output[0]
                residual = output[1] if len(output) > 1 else None
            else:
                hidden_states = output
                residual = None
            # MUST clone! fused_add_rms_norm in the next layer modifies
            # hidden_states IN-PLACE. Without clone, the draft stream's
            # D2D copy races with the in-place modification.
            hs = hidden_states.detach().clone()
            res = residual.detach().clone() if residual is not None else None
            self._early_exit_hidden_states = hs
            self._early_exit_residual = res

            # Strategy B: Launch standard branch gen on draft stream
            # Apply norm here to produce post-norm HS for the async path
            if (self._draft_stream is not None
                    and self._stored_batch_state is not None
                    and not self._standard_branches_launched):
                normed_hs = self._apply_target_norm(hs, res)
                self._launch_async_standard_branch_gen(
                    normed_hs, is_early_exit=False,
                )

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

    def get_early_exit_hidden_states(
        self,
    ) -> tuple[torch.Tensor, torch.Tensor | None] | None:
        """Get the hidden_states (and residual) captured by early exit hook.

        Returns:
            (hidden_states, residual) tuple if captured, None otherwise.
            residual may be None for models that don't use fused_add_rms_norm.
            Consumes the captured states (one-time read).
        """
        hs = self._early_exit_hidden_states
        if hs is None:
            return None
        res = self._early_exit_residual
        self._early_exit_hidden_states = None  # Consume
        self._early_exit_residual = None
        return hs, res

    # ------------------------------------------------------------------
    # Async branch generation (CUDA stream pipeline)
    # ------------------------------------------------------------------

    def store_batch_state(
        self,
        target_token_ids: torch.Tensor,
        target_positions: torch.Tensor,
        common_attn_metadata: "CommonAttentionMetadata",
        sampling_metadata: "SamplingMetadata",
        request_ids: list[str] | None = None,
        mm_embed_inputs: tuple | None = None,
        num_rejected_tokens_gpu: torch.Tensor | None = None,
    ) -> None:
        """Store pre-forward state for async branch generation.

        Must be called before the target model forward pass so that
        the early exit hook or post-forward launch can use this state.
        Only active when enable_concurrent is True.
        """
        if not self.enable_concurrent:
            # Still set request_ids for targeted branch tracking
            if request_ids is not None:
                self._request_ids = request_ids
            return

        # Set request_ids early so pre-compute and async gen can use them
        if request_ids is not None:
            self._request_ids = request_ids

        # Sync Eagle KV cache to remote device (overlaps with target fwd)
        if self._draft_device is not None:
            self._sync_eagle_kv_to_remote(common_attn_metadata)

        # Pre-compute ALL CPU-heavy metadata here (before target forward)
        metadata = self._precompute_branch_metadata(
            target_token_ids, target_positions,
            common_attn_metadata, num_rejected_tokens_gpu,
        )

        self._stored_batch_state = {
            'metadata': metadata,
            # Keep refs for sync fallback and targeted branches
            'target_token_ids': target_token_ids,
            'target_positions': target_positions,
            'common_attn_metadata': common_attn_metadata,
            'sampling_metadata': sampling_metadata,
            'mm_embed_inputs': mm_embed_inputs,
            'num_rejected_tokens_gpu': num_rejected_tokens_gpu,
        }
        self._standard_branches_launched = False
        self._standard_branches_done = False
        self._async_draft_ids = None
        self._async_root_tokens = None

    def _sync_eagle_kv_to_remote(
        self,
        common_attn_metadata: "CommonAttentionMetadata",
    ) -> None:
        """Copy only USED blocks of GPU 0 Eagle KV cache to remote device.

        Instead of copying the entire KV cache (~9 GB), we extract the
        block indices referenced by the current batch's block table and
        only transfer those blocks (~4 MB for typical workloads).
        """
        # Lazy allocation: KV caches are bound after load_model
        if not getattr(self, '_remote_kv_allocated', False):
            self._allocate_remote_kv_cache()
            self._remote_kv_allocated = True

        if not hasattr(self, '_kv_cache_pairs') or not self._kv_cache_pairs:
            return

        # Extract unique used block indices from the block table
        num_reqs = common_attn_metadata.num_reqs
        block_table = common_attn_metadata.block_table_tensor[:num_reqs]
        used_blocks = block_table.reshape(-1).unique().long()
        # Filter out invalid block indices (negative values)
        used_blocks = used_blocks[used_blocks >= 0]

        if used_blocks.numel() == 0:
            return

        # Selective block copy: gather from GPU 0 → D2D → scatter to GPU 1
        used_blocks_remote = used_blocks.to(
            self._draft_device, non_blocking=True,
        )
        for layer_name, gpu0_kv, remote_kv in self._kv_cache_pairs:
            if isinstance(gpu0_kv, list):
                for src, dst in zip(gpu0_kv, remote_kv):
                    # src/dst shape: (2, num_blocks, block_size, kv_heads, head)
                    src_blocks = torch.index_select(
                        src, dim=1, index=used_blocks,
                    )
                    dst.index_copy_(
                        1, used_blocks_remote,
                        src_blocks.to(self._draft_device, non_blocking=True),
                    )
            else:
                src_blocks = torch.index_select(
                    gpu0_kv, dim=1, index=used_blocks,
                )
                remote_kv.index_copy_(
                    1, used_blocks_remote,
                    src_blocks.to(self._draft_device, non_blocking=True),
                )

        # Record event on default stream so draft_stream can wait
        if self._kv_sync_event is not None:
            self._kv_sync_event.record(torch.cuda.current_stream())

    def _precompute_branch_metadata(
        self,
        target_token_ids: torch.Tensor,
        target_positions: torch.Tensor,
        common_attn_metadata: "CommonAttentionMetadata",
        num_rejected_tokens_gpu: torch.Tensor | None,
    ) -> dict | None:
        """Pre-compute all CPU-heavy metadata for GPU-native branch gen.

        ALL CPU-GPU sync points (.item(), .tolist(), .cpu()) happen here,
        BEFORE the target forward pass. The GPU-native branch gen method
        can then run with zero CPU-GPU sync.

        Returns None if GPU-native path is not applicable (top_k>1,
        mrope, too many tokens, etc.), causing fallback to sync path.
        """
        eagle = self._underlying

        # GPU path only supports top_k=1 (default)
        if self.top_k != 1:
            return None

        # GPU path doesn't support mrope
        if eagle.uses_mrope:
            return None

        if self._target_lm_head is None or self._request_ids is None:
            return None

        device = eagle.device
        batch_size = len(self._request_ids)
        if batch_size == 0:
            return None

        # --- CPU sync: analyze requests ---
        request_infos = self._analyze_requests_proposer(
            target_token_ids, common_attn_metadata, batch_size,
        )

        # Build branch descriptors (top_k=1: one branch per position)
        branches: list[dict] = []
        for i in range(batch_size):
            info = request_infos[i]
            if info is None:
                continue
            for pos in info["branch_positions"]:
                if info["is_prefill"]:
                    key_prefix: tuple[int, ...] = ()
                else:
                    key_prefix = tuple(info["input_ids"][1:pos + 1])

                branches.append({
                    "req_idx": i,
                    "pos": pos,
                    "qs": info["qs"],
                    "ql": info["ql"],
                    "key_prefix": key_prefix,
                })

        if not branches:
            return None

        num_branches = len(branches)
        total_tokens = sum(b["ql"] for b in branches)

        # Check limits
        if total_tokens > eagle.max_num_tokens:
            return None
        if num_branches + 1 > eagle.arange.shape[0]:
            return None

        # --- Build expanded GPU tensors ---
        exp_input_ids = torch.empty(
            total_tokens, dtype=torch.int32, device=device,
        )
        exp_positions = torch.empty(
            total_tokens, dtype=torch.int64, device=device,
        )
        exp_slot_mapping = torch.empty(
            total_tokens,
            dtype=common_attn_metadata.slot_mapping.dtype,
            device=device,
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

        # Gather/scatter indices for GPU-native branch gen
        hs_gather_indices = torch.empty(
            num_branches, dtype=torch.long, device=device,
        )
        root_scatter_indices = torch.empty(
            num_branches, dtype=torch.long, device=device,
        )
        hs_fill_indices = torch.empty(
            total_tokens, dtype=torch.long, device=device,
        )

        # Branch-to-request mapping (GPU)
        branch_req_indices = torch.empty(
            num_branches, dtype=torch.long, device=device,
        )

        # Cache key prefixes (CPU, for post-sync cache storage)
        cache_key_prefixes: list[tuple[int, ...]] = []
        branch_to_req: list[int] = []

        offset = 0
        for b_idx, branch in enumerate(branches):
            qs = branch["qs"]
            ql = branch["ql"]
            pos = branch["pos"]

            # Shift input_ids by 1 (same as EagleProposer)
            if ql > 1:
                exp_input_ids[offset:offset + ql - 1] = (
                    target_token_ids[qs + 1:qs + ql]
                )
            # Fill the last position with a valid token
            exp_input_ids[offset + ql - 1] = (
                target_token_ids[qs + ql - 1]
            )
            # Root token position will be overwritten by GPU scatter

            # Copy positions and slot_mapping
            exp_positions[offset:offset + ql] = (
                target_positions[qs:qs + ql]
            )
            exp_slot_mapping[offset:offset + ql] = (
                common_attn_metadata.slot_mapping[qs:qs + ql]
            )

            # Per-branch metadata
            exp_qsl[b_idx + 1] = offset + ql
            exp_lti[b_idx] = offset + pos
            exp_seq_lens[b_idx] = (
                common_attn_metadata.seq_lens[branch["req_idx"]]
            )
            branch_req_indices[b_idx] = branch["req_idx"]

            # Gather/scatter indices
            hs_gather_indices[b_idx] = qs + pos
            root_scatter_indices[b_idx] = offset + pos
            hs_fill_indices[offset:offset + ql] = torch.arange(
                qs, qs + ql, device=device,
            )

            # CPU data for post-sync cache storage
            cache_key_prefixes.append(branch["key_prefix"])
            branch_to_req.append(branch["req_idx"])

            offset += ql

        # Replicate block table rows
        exp_block_table = (
            common_attn_metadata.block_table_tensor[branch_req_indices]
        )

        # CPU metadata (sync OK here, before target forward)
        exp_qsl_cpu = exp_qsl.cpu()
        max_ql = max(b["ql"] for b in branches)
        max_seq_len = int(exp_seq_lens.max().item())

        # Decode-phase query_start_loc_cpu
        decode_qsl_cpu = torch.from_numpy(
            np.arange(num_branches + 1, dtype=np.int32)
        )

        return {
            'request_infos': request_infos,
            'num_branches': num_branches,
            'total_tokens': total_tokens,
            'max_ql': max_ql,
            'max_seq_len': max_seq_len,
            'exp_input_ids': exp_input_ids,
            'exp_positions': exp_positions,
            'exp_slot_mapping': exp_slot_mapping,
            'exp_qsl': exp_qsl,
            'exp_lti': exp_lti,
            'exp_seq_lens': exp_seq_lens,
            'exp_block_table': exp_block_table,
            'exp_qsl_cpu': exp_qsl_cpu,
            'decode_qsl_cpu': decode_qsl_cpu,
            'hs_gather_indices': hs_gather_indices,
            'root_scatter_indices': root_scatter_indices,
            'hs_fill_indices': hs_fill_indices,
            'cache_key_prefixes': cache_key_prefixes,
            'branch_to_req': branch_to_req,
            'branch_req_indices': branch_req_indices,
            'num_rejected_tokens_gpu': num_rejected_tokens_gpu,
            'cross_device': self._draft_device is not None,
        }

    def _launch_async_standard_branch_gen(
        self,
        hidden_states: torch.Tensor,
        is_early_exit: bool = False,
    ) -> None:
        """Launch standard branch generation on the draft CUDA stream.

        Uses GPU-native path (zero CPU-GPU sync) when pre-computed
        metadata is available. Falls back to sync-heavy path otherwise.

        Args:
            hidden_states: Hidden states for root token computation and
                eagle model input (early exit or full forward).
            is_early_exit: True if hidden_states come from early exit hook.
        """
        self._standard_branches_launched = True
        state = self._stored_batch_state
        if state is None:
            return

        metadata = state.get('metadata')
        if metadata is None:
            # No GPU-native path (top_k>1, mrope, etc.)
            # Fall back to sync-heavy path
            batch_size = (
                len(self._request_ids) if self._request_ids else 0
            )
            if batch_size == 0:
                return

            cam = state['common_attn_metadata']
            effective_lti = cam.query_start_loc[1:] - 1

            default_stream = torch.cuda.current_stream()
            try:
                with torch.cuda.stream(self._draft_stream):
                    self._draft_stream.wait_stream(default_stream)
                    self._generate_standard_branches(
                        target_token_ids=state['target_token_ids'],
                        target_positions=state['target_positions'],
                        branch_hidden_states=hidden_states,
                        target_hidden_states=hidden_states,
                        last_token_indices=effective_lti,
                        common_attn_metadata=cam,
                        sampling_metadata=state['sampling_metadata'],
                        mm_embed_inputs=state['mm_embed_inputs'],
                        num_rejected_tokens_gpu=(
                            state['num_rejected_tokens_gpu']
                        ),
                        batch_size=batch_size,
                        is_early_exit=is_early_exit,
                    )
            except Exception as e:
                logger.warning("Async branch gen (sync path) failed: %s", e)
                return

            self._standard_branches_done = True
            return

        # GPU-native path: zero CPU-GPU sync in the hook
        default_stream = torch.cuda.current_stream()
        use_remote = (
            metadata.get('cross_device', False)
            and self._remote_eagle is not None
        )
        try:
            with torch.cuda.stream(self._draft_stream):
                self._draft_stream.wait_stream(default_stream)
                # Wait for KV sync to complete on the draft device
                if use_remote and self._kv_sync_event is not None:
                    self._draft_stream.wait_event(self._kv_sync_event)
                if use_remote:
                    self._generate_standard_branches_gpu_remote(
                        hidden_states, metadata,
                        is_early_exit=is_early_exit,
                    )
                else:
                    self._generate_standard_branches_gpu(
                        hidden_states, metadata,
                        is_early_exit=is_early_exit,
                    )
        except Exception as e:
            logger.warning("Async GPU branch gen failed: %s", e)
            self._async_draft_ids = None
            self._async_root_tokens = None
            return

        self._standard_branches_done = True

    def launch_standard_branches_if_not_started(
        self,
        hidden_states: torch.Tensor,
    ) -> None:
        """Strategy A: launch async branch gen after target forward.

        Called from model_runner after the target forward completes.
        No-op if already launched by the early exit hook (Strategy B).

        Args:
            hidden_states: Full target model hidden states.
        """
        if self._standard_branches_launched:
            return  # Already launched by early exit hook
        if self._draft_stream is None or self._stored_batch_state is None:
            return
        self._launch_async_standard_branch_gen(
            hidden_states, is_early_exit=False,
        )

    def _generate_standard_branches_gpu(
        self,
        hidden_states: torch.Tensor,
        metadata: dict,
        is_early_exit: bool = False,
    ) -> None:
        """GPU-native branch generation with ZERO CPU-GPU sync points.

        All CPU-heavy work was done in _precompute_branch_metadata().
        This method only enqueues GPU kernels and returns immediately,
        allowing the target forward to continue on the default stream.

        Results are stored as GPU tensors (_async_draft_ids,
        _async_root_tokens) and transferred to CPU later in
        _store_async_results_in_cache().

        Args:
            hidden_states: Hidden states from early exit or full forward.
            metadata: Pre-computed metadata from _precompute_branch_metadata.
            is_early_exit: Whether hidden_states are from early exit hook.
        """
        eagle = self._underlying
        num_branches = metadata['num_branches']
        total_tokens = metadata['total_tokens']
        num_spec = eagle.num_speculative_tokens

        # Unpack pre-computed tensors
        exp_input_ids = metadata['exp_input_ids']
        exp_positions = metadata['exp_positions']
        exp_slot_mapping = metadata['exp_slot_mapping']
        exp_qsl = metadata['exp_qsl']
        exp_lti = metadata['exp_lti']
        exp_seq_lens = metadata['exp_seq_lens']
        exp_block_table = metadata['exp_block_table']
        hs_gather_indices = metadata['hs_gather_indices']
        root_scatter_indices = metadata['root_scatter_indices']
        hs_fill_indices = metadata['hs_fill_indices']

        # Handle eagle3 hidden state combination
        if eagle.method == "eagle3":
            from vllm.model_executor.models.llama_eagle3 import (
                Eagle3LlamaForCausalLM,
            )
            assert isinstance(eagle.model, Eagle3LlamaForCausalLM)
            hidden_states = eagle.model.combine_hidden_states(
                hidden_states
            )

        # Step 1: Compute root tokens (GPU only, no .item()!)
        selected_hs = hidden_states[hs_gather_indices]

        with torch.no_grad():
            if is_early_exit:
                # Early exit HS are pre-norm: apply TARGET model's final
                # norm before computing logits.  Using the target norm
                # (not MTP SharedHead norm) is critical because the HS
                # come from the target model's intermediate layers.
                normed_hs = self._apply_target_norm(selected_hs)
                root_logits = self._target_model.compute_logits(
                    normed_hs)
            else:
                root_logits = self._target_model.compute_logits(
                    selected_hs)

        if root_logits is None:
            return

        # Argmax stays on GPU — no CPU transfer!
        root_tokens = root_logits.argmax(dim=-1)
        self._async_root_tokens = root_tokens

        # Step 2: Scatter root tokens into pre-built exp_input_ids
        exp_input_ids[root_scatter_indices] = root_tokens.int()

        # Step 3: Fill exp_hidden_states from hidden_states
        exp_hidden_states = hidden_states[hs_fill_indices]

        # Step 4: Build CommonAttentionMetadata from pre-computed tensors
        exp_cam = CommonAttentionMetadata(
            query_start_loc=exp_qsl,
            seq_lens=exp_seq_lens.clone(),
            query_start_loc_cpu=metadata['exp_qsl_cpu'],
            _seq_lens_cpu=None,
            _num_computed_tokens_cpu=None,
            num_reqs=num_branches,
            num_actual_tokens=total_tokens,
            max_query_len=metadata['max_ql'],
            max_seq_len=metadata['max_seq_len'],
            block_table_tensor=exp_block_table,
            slot_mapping=exp_slot_mapping,
            causal=True,
        )

        # Step 5: Run eagle first forward (prefill-like)
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
            self._async_draft_ids = draft_ids
            return

        # Prepare for decode-style forwards
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

        # Step 6: Update metadata for decode-style forwards
        exp_cam.num_actual_tokens = num_branches
        exp_cam.max_query_len = 1
        exp_cam.query_start_loc = eagle.arange[:num_branches + 1]
        exp_cam.query_start_loc_cpu = metadata['decode_qsl_cpu']

        # Apply num_rejected_tokens_gpu adjustment
        num_rejected_tokens_gpu = metadata.get('num_rejected_tokens_gpu')
        if num_spec > 1 and num_rejected_tokens_gpu is not None:
            exp_rejected = num_rejected_tokens_gpu[
                metadata['branch_req_indices']
            ]
            exp_cam.seq_lens -= exp_rejected
        exp_cam._seq_lens_cpu = None
        exp_cam._num_computed_tokens_cpu = None

        # Step 7: Generate remaining draft tokens (decode loop)
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
            logits = eagle.model.compute_logits(
                last_hs[:num_branches])
            draft_ids = logits.argmax(dim=-1)
            draft_ids_list.append(draft_ids)

        # Store results as GPU tensors (NO CPU transfer!)
        self._async_draft_ids = torch.stack(draft_ids_list, dim=1)

    def _generate_standard_branches_gpu_remote(
        self,
        hidden_states: torch.Tensor,
        metadata: dict,
        is_early_exit: bool = False,
    ) -> None:
        """Cross-device GPU-native branch generation on the remote Eagle.

        Same logic as _generate_standard_branches_gpu but runs on
        self._draft_device using self._remote_eagle. Temporarily swaps
        the compilation_config.static_forward_context so that
        set_forward_context picks up the remote Eagle's attention layers.

        All source tensors are D2D copied from GPU 0 to the draft device.
        Results are stored as GPU tensors on the draft device and
        transferred to CPU in _store_async_results_in_cache().
        """
        eagle = self._remote_eagle
        draft_device = self._draft_device
        num_branches = metadata['num_branches']
        total_tokens = metadata['total_tokens']
        num_spec = eagle.num_speculative_tokens

        # --- D2D transfer: move expanded tensors to draft device ---
        exp_input_ids = metadata['exp_input_ids'].to(
            draft_device, non_blocking=True)
        exp_positions = metadata['exp_positions'].to(
            draft_device, non_blocking=True)
        exp_slot_mapping = metadata['exp_slot_mapping'].to(
            draft_device, non_blocking=True)
        exp_qsl = metadata['exp_qsl'].to(
            draft_device, non_blocking=True)
        exp_lti = metadata['exp_lti'].to(
            draft_device, non_blocking=True)
        exp_seq_lens = metadata['exp_seq_lens'].to(
            draft_device, non_blocking=True)
        exp_block_table = metadata['exp_block_table'].to(
            draft_device, non_blocking=True)
        hs_gather_indices = metadata['hs_gather_indices'].to(
            draft_device, non_blocking=True)
        root_scatter_indices = metadata['root_scatter_indices'].to(
            draft_device, non_blocking=True)
        hs_fill_indices = metadata['hs_fill_indices'].to(
            draft_device, non_blocking=True)

        # D2D transfer hidden states
        if eagle.method == "eagle3":
            from vllm.model_executor.models.llama_eagle3 import (
                Eagle3LlamaForCausalLM,
            )
            assert isinstance(eagle.model, Eagle3LlamaForCausalLM)
            hidden_states = eagle.model.combine_hidden_states(
                hidden_states)
        remote_hs = hidden_states.to(draft_device, non_blocking=True)

        # Step 1: Compute root tokens on draft device
        selected_hs = remote_hs[hs_gather_indices]

        with torch.no_grad():
            if is_early_exit:
                if self._draft_method == "mtp":
                    root_logits = eagle.model.compute_logits(selected_hs)
                else:
                    if self._remote_target_norm is not None:
                        normed_hs = self._remote_target_norm(selected_hs)
                    else:
                        normed_hs = selected_hs
                    if self._remote_target_model is not None:
                        root_logits = (
                            self._remote_target_model.compute_logits(
                                normed_hs))
                    else:
                        # Fallback: transfer back to GPU 0
                        normed_hs_gpu0 = normed_hs.to(
                            self._device, non_blocking=True)
                        root_logits = self._target_model.compute_logits(
                            normed_hs_gpu0).to(
                            draft_device, non_blocking=True)
            else:
                if self._remote_target_model is not None:
                    root_logits = (
                        self._remote_target_model.compute_logits(
                            selected_hs))
                else:
                    selected_gpu0 = selected_hs.to(
                        self._device, non_blocking=True)
                    root_logits = self._target_model.compute_logits(
                        selected_gpu0).to(
                        draft_device, non_blocking=True)

        if root_logits is None:
            return

        root_tokens = root_logits.argmax(dim=-1)
        self._async_root_tokens = root_tokens

        # Step 2: Scatter root tokens into exp_input_ids
        exp_input_ids[root_scatter_indices] = root_tokens.int()

        # Step 3: Fill exp_hidden_states from remote hidden states
        exp_hidden_states = remote_hs[hs_fill_indices]

        # Step 4: Build CommonAttentionMetadata on draft device
        exp_cam = CommonAttentionMetadata(
            query_start_loc=exp_qsl,
            seq_lens=exp_seq_lens.clone(),
            query_start_loc_cpu=metadata['exp_qsl_cpu'],
            _seq_lens_cpu=None,
            _num_computed_tokens_cpu=None,
            num_reqs=num_branches,
            num_actual_tokens=total_tokens,
            max_query_len=metadata['max_ql'],
            max_seq_len=metadata['max_seq_len'],
            block_table_tensor=exp_block_table,
            slot_mapping=exp_slot_mapping,
            causal=True,
        )

        # Step 5: Run eagle first forward (prefill-like)
        # Temporarily swap static_forward_context for remote Eagle
        compilation_config = self._vllm_config.compilation_config
        original_sfc = compilation_config.static_forward_context
        compilation_config.static_forward_context = (
            self._remote_forward_context
        )

        try:
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
            if eagle.draft_indexer_metadata_builder:
                draft_indexer_md = (
                    eagle.draft_indexer_metadata_builder.build_for_drafting(
                        common_attn_metadata=exp_cam, draft_index=0,
                    )
                )
                for name in eagle.indexer_layer_names:
                    per_layer_attn_metadata[name] = draft_indexer_md

            # Copy to eagle's persistent buffers (on draft device)
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
                self._async_draft_ids = draft_ids
                return

            # Prepare for decode-style forwards
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

            # Update metadata for decode-style forwards
            exp_cam.num_actual_tokens = num_branches
            exp_cam.max_query_len = 1
            exp_cam.query_start_loc = eagle.arange[:num_branches + 1]
            exp_cam.query_start_loc_cpu = metadata['decode_qsl_cpu']

            num_rejected_tokens_gpu = metadata.get(
                'num_rejected_tokens_gpu')
            if num_spec > 1 and num_rejected_tokens_gpu is not None:
                exp_rejected = num_rejected_tokens_gpu[
                    metadata['branch_req_indices']
                ].to(draft_device, non_blocking=True)
                exp_cam.seq_lens -= exp_rejected
            exp_cam._seq_lens_cpu = None
            exp_cam._num_computed_tokens_cpu = None

            # Generate remaining draft tokens (decode loop)
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

                block_numbers = clamped_pos // block_size
                block_ids = exp_cam.block_table_tensor.gather(
                    dim=1, index=block_numbers.view(-1, 1),
                ).view(-1)
                exp_cam.slot_mapping = (
                    block_ids * block_size + clamped_pos % block_size
                )
                exp_cam.slot_mapping.masked_fill_(
                    exceeds, PADDING_SLOT_ID)

                attn_metadata = attn_metadata_builder.build_for_drafting(
                    common_attn_metadata=exp_cam,
                    draft_index=token_index + 1,
                )
                per_layer_attn_metadata = {
                    name: attn_metadata
                    for name in eagle.attn_layer_names
                }

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
                logits = eagle.model.compute_logits(
                    last_hs[:num_branches])
                draft_ids = logits.argmax(dim=-1)
                draft_ids_list.append(draft_ids)

            # Store results as GPU tensors on draft device
            self._async_draft_ids = torch.stack(draft_ids_list, dim=1)

        finally:
            # Always restore the original static_forward_context
            compilation_config.static_forward_context = original_sfc

    def _store_async_results_in_cache(self) -> None:
        """Store GPU branch gen results in per-request caches.

        Called from propose() after draft_stream.synchronize().
        Does bulk CPU transfer of GPU tensors and stores in cache.
        """
        state = self._stored_batch_state
        if state is None:
            return
        metadata = state.get('metadata')
        if metadata is None:
            return

        # Set request_infos for targeted branches
        self._request_infos_proposer = metadata['request_infos']

        # Clear per-request caches for fresh round
        batch_size = len(self._request_ids) if self._request_ids else 0
        for i in range(batch_size):
            if i < len(self._request_ids):
                cache = self._get_or_create_cache(self._request_ids[i])
                cache.clear()

        # Bulk CPU transfer (one transfer each, not per-branch)
        draft_ids_cpu = self._async_draft_ids.cpu()
        root_tokens_cpu = self._async_root_tokens.cpu()

        cache_key_prefixes = metadata['cache_key_prefixes']
        branch_to_req = metadata['branch_to_req']

        for b_idx in range(len(branch_to_req)):
            req_idx = branch_to_req[b_idx]
            req_id = self._request_ids[req_idx]
            cache = self._get_or_create_cache(req_id)

            # Construct cache key: prefix + root_token
            root_token = int(root_tokens_cpu[b_idx].item())
            cache_key = cache_key_prefixes[b_idx] + (root_token,)

            # Store draft tokens
            draft_tokens = draft_ids_cpu[b_idx].tolist()
            cache.store_branch(cache_key, draft_tokens)

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

        # For early-exit, we need the target model's final norm because
        # early-exit hidden states are pre-norm (haven't gone through the
        # model's final RMSNorm).  This is needed for both Eagle and MTP:
        # the target norm must be used instead of MTP's SharedHead norm
        # since the hidden states come from the target model's layers.
        self._target_norm = self._find_target_norm(target_model)
        if self._target_norm is not None:
            logger.info(
                "ParallelProposer: found target model's final norm "
                "for early-exit normalization"
            )

        # Load remote Eagle on separate GPU if configured
        if self._draft_device is not None:
            self._load_remote_eagle(target_model)

        # Set up early exit forward hook if needed
        if self.early_exit_layer != -1:
            self._setup_early_exit_hook(target_model)

    def _load_remote_eagle(self, target_model: nn.Module) -> None:
        """Load a second Eagle model on the remote draft device.

        Creates an isolated static_forward_context so the remote Eagle's
        attention layers register independently from the GPU 0 layers.
        Also copies lm_head and norm to the remote device for root token
        computation without cross-device round-trips.
        """
        from copy import deepcopy

        from vllm.config import get_layers_from_vllm_config
        from vllm.model_executor.layers.attention_layer_base import (
            AttentionLayerBase,
        )
        from vllm.model_executor.model_loader import get_model

        draft_device = self._draft_device
        compilation_config = self._vllm_config.compilation_config

        # Save original static_forward_context
        original_sfc = compilation_config.static_forward_context

        # Create isolated forward context for the remote Eagle
        isolated_sfc: dict = {}
        compilation_config.static_forward_context = isolated_sfc

        try:
            # Create remote EagleProposer on the draft device
            remote_eagle = EagleProposer(
                self._vllm_config, draft_device, self._underlying.runner,
            )
            # Temporarily redirect device_config so get_model loads on
            # the draft device instead of the default GPU 0
            original_device = self._vllm_config.device_config.device
            self._vllm_config.device_config.device = draft_device
            try:
                with torch.device(draft_device):
                    remote_eagle.load_model(target_model)
            finally:
                self._vllm_config.device_config.device = original_device

            # After load_model, shared weights (embed_tokens, lm_head) may
            # point to GPU 0 tensors. Move them to the draft device.
            self._ensure_remote_eagle_on_device(remote_eagle, draft_device)
            logger.info(
                "Remote Eagle model loaded on %s, layers: %s",
                draft_device, remote_eagle.attn_layer_names,
            )
        finally:
            # Restore original static_forward_context
            compilation_config.static_forward_context = original_sfc

        self._remote_eagle = remote_eagle
        self._remote_forward_context = isolated_sfc
        self._remote_kv_allocated = False

        # Create attention metadata builder for the remote Eagle on
        # the draft device. We can't use _get_attention_metadata_builder
        # because it searches runner.attn_groups which only has GPU 0
        # layers. Instead, create a builder directly from the remote
        # forward context's layer objects.
        self._create_remote_attn_metadata_builder(
            remote_eagle, isolated_sfc, draft_device,
        )

        # Copy lm_head and norm to remote device for root token computation
        # These are small (~30MB for lm_head, <1MB for norm)
        self._remote_target_model = self._create_remote_logits_module(
            target_model, draft_device,
        )
        if self._target_norm is not None:
            self._remote_target_norm = deepcopy(
                self._target_norm,
            ).to(draft_device)
            logger.info(
                "Target norm copied to %s for remote root token computation",
                draft_device,
            )

        logger.info(
            "Cross-device Parallel-SD initialized: "
            "target=%s, draft=%s",
            self._device, draft_device,
        )

    @staticmethod
    def _ensure_remote_eagle_on_device(
        remote_eagle: EagleProposer,
        device: torch.device,
    ) -> None:
        """Move any cross-device shared weights to the target device.

        EagleProposer.load_model() shares embed_tokens and lm_head with
        the target model (on GPU 0). For the remote eagle, we need copies
        on the draft device.
        """
        model = remote_eagle.model
        if model is None:
            return

        # Check embed_tokens
        if hasattr(model, "model") and hasattr(model.model, "embed_tokens"):
            embed = model.model.embed_tokens
            if hasattr(embed, "weight") and embed.weight.device != device:
                from copy import deepcopy
                model.model.embed_tokens = deepcopy(embed).to(device)
                logger.info(
                    "Copied embed_tokens to %s for remote Eagle", device,
                )

        # Check lm_head
        if hasattr(model, "lm_head"):
            lm_head = model.lm_head
            if hasattr(lm_head, "weight") and lm_head.weight.device != device:
                from copy import deepcopy
                model.lm_head = deepcopy(lm_head).to(device)
                logger.info(
                    "Copied lm_head to %s for remote Eagle", device,
                )

    def _create_remote_attn_metadata_builder(
        self,
        remote_eagle: EagleProposer,
        remote_sfc: dict,
        draft_device: torch.device,
    ) -> None:
        """Create an attention metadata builder for the remote Eagle.

        The remote Eagle can't use _get_attention_metadata_builder()
        because that searches runner.attn_groups which only has GPU 0
        layers. Instead, we create a builder directly from the remote
        forward context's attention layer.
        """
        from vllm.model_executor.layers.attention_layer_base import (
            AttentionLayerBase,
        )

        if not remote_eagle.attn_layer_names:
            logger.warning(
                "Remote Eagle has no attention layer names; "
                "metadata builder not created."
            )
            return

        first_layer_name = remote_eagle.attn_layer_names[0]
        remote_layer = remote_sfc.get(first_layer_name)
        if remote_layer is None:
            logger.warning(
                "Remote layer %s not found in forward context; "
                "metadata builder not created.",
                first_layer_name,
            )
            return

        if not isinstance(remote_layer, AttentionLayerBase):
            logger.warning(
                "Remote layer %s is not an AttentionLayerBase; "
                "metadata builder not created.",
                first_layer_name,
            )
            return

        backend = remote_layer.get_attn_backend()
        kv_cache_spec = remote_layer.get_kv_cache_spec(self._vllm_config)
        builder = backend.get_builder_cls()(
            kv_cache_spec,
            remote_eagle.attn_layer_names,
            self._vllm_config,
            draft_device,
        )
        remote_eagle.attn_metadata_builder = builder
        logger.info(
            "Created attention metadata builder for remote Eagle "
            "on %s (layer: %s)",
            draft_device, first_layer_name,
        )

    def _create_remote_logits_module(
        self,
        target_model: nn.Module,
        device: torch.device,
    ) -> nn.Module:
        """Create a lightweight module on the remote device for logits.

        vllm's ParallelLMHead raises RuntimeError on forward() — its
        weights are meant to be used directly by the sampler. We extract
        the weight tensor and create a simple linear projection.
        """
        class RemoteLogitsModule(nn.Module):
            """Computes logits via matmul with lm_head weight."""

            def __init__(self, weight: torch.Tensor):
                super().__init__()
                # Store as buffer (not parameter) to avoid optimizer
                self.register_buffer("weight", weight)

            def compute_logits(self, hidden_states: torch.Tensor):
                return torch.nn.functional.linear(
                    hidden_states, self.weight,
                )

        # Find the lm_head weight in the target model
        lm_head_weight = None
        if hasattr(target_model, "lm_head"):
            lm_head = target_model.lm_head
            if hasattr(lm_head, "weight"):
                lm_head_weight = lm_head.weight
        if lm_head_weight is None and hasattr(
            target_model, "language_model"
        ):
            lm = target_model.language_model
            if hasattr(lm, "lm_head") and hasattr(lm.lm_head, "weight"):
                lm_head_weight = lm.lm_head.weight

        if lm_head_weight is None:
            logger.warning(
                "Could not find lm_head weight in target model for "
                "remote device. Root token computation will fall "
                "back to GPU 0."
            )
            return None

        # Clone weight to the remote device
        remote_weight = lm_head_weight.detach().clone().to(device)
        module = RemoteLogitsModule(remote_weight)
        module.eval()
        weight_mb = (
            remote_weight.numel() * remote_weight.element_size()
            / 1024 / 1024
        )
        logger.info(
            "lm_head weight copied to %s for remote root token "
            "computation (%.1f MB)",
            device, weight_mb,
        )
        return module

    def _allocate_remote_kv_cache(self) -> None:
        """Allocate mirrored KV cache on the remote draft device.

        The remote Eagle KV cache has the same shape as the GPU 0 Eagle
        KV cache. We copy the entire cache before each branch gen round
        so that the remote Eagle can attend to the same KV history.
        """
        if self._remote_eagle is None or self._draft_device is None:
            return

        original_sfc = self._vllm_config.compilation_config \
            .static_forward_context
        remote_sfc = self._remote_forward_context
        if remote_sfc is None:
            return

        # Map GPU 0 Eagle layer names to their KV caches
        eagle_layer_names = self._underlying.attn_layer_names
        self._kv_cache_pairs: list[tuple[str, Any, Any]] = []

        for layer_name in eagle_layer_names:
            gpu0_layer = original_sfc.get(layer_name)
            remote_layer = remote_sfc.get(layer_name)
            if gpu0_layer is None or remote_layer is None:
                continue
            # Each attention layer has kv_cache attribute set by the
            # model runner during KV cache allocation
            if (hasattr(gpu0_layer, "kv_cache")
                    and gpu0_layer.kv_cache is not None):
                # The KV cache is a list of tensors (one per virtual
                # engine). Allocate matching tensors on remote device.
                gpu0_kv = gpu0_layer.kv_cache
                if isinstance(gpu0_kv, list):
                    remote_kv = [
                        torch.empty_like(t, device=self._draft_device)
                        for t in gpu0_kv
                    ]
                else:
                    remote_kv = torch.empty_like(
                        gpu0_kv, device=self._draft_device,
                    )
                remote_layer.kv_cache = remote_kv
                self._kv_cache_pairs.append(
                    (layer_name, gpu0_kv, remote_kv),
                )
                kv_size = (
                    sum(t.numel() * t.element_size()
                        for t in (gpu0_kv if isinstance(gpu0_kv, list)
                                  else [gpu0_kv]))
                )
                logger.info(
                    "Allocated mirrored KV cache for %s on %s "
                    "(%.1f MB)",
                    layer_name, self._draft_device,
                    kv_size / 1024 / 1024,
                )

        if not self._kv_cache_pairs:
            logger.warning(
                "No KV cache pairs found for remote Eagle. "
                "KV sync will be skipped."
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
        # Standard branches may have been pre-computed on the draft stream.
        # Only targeted branches (which need correction_token from scoring)
        # are run synchronously here.
        async_ok = False
        if self._standard_branches_launched:
            if self._standard_branches_done:
                # Synchronize the draft CUDA stream
                if self._draft_stream is not None:
                    self._draft_stream.synchronize()
                # Store GPU results in cache (bulk CPU transfer)
                if (self._async_draft_ids is not None
                        and self._stored_batch_state is not None):
                    self._store_async_results_in_cache()
                    async_ok = True
            self._standard_branches_done = False
            self._standard_branches_launched = False

        if async_ok:
            self._stored_batch_state = None
        else:
            # No async branches — run full synchronous branch gen
            self._stored_batch_state = None
            early_exit_result = self.get_early_exit_hidden_states()
            using_early_exit = early_exit_result is not None
            if using_early_exit:
                early_exit_hs, early_exit_res = early_exit_result
                # Apply the target model's final norm to early-exit HS
                # immediately, producing post-norm HS equivalent to the
                # target model's output.  This avoids threading the
                # residual through branch generation methods.
                branch_hs = self._apply_target_norm(
                    early_exit_hs, early_exit_res)
            else:
                branch_hs = target_hidden_states

            try:
                self._generate_branches(
                    target_token_ids, target_positions,
                    branch_hs, target_hidden_states,
                    next_token_ids, effective_lti, common_attn_metadata,
                    sampling_metadata, mm_embed_inputs,
                    num_rejected_tokens_gpu, batch_size,
                    # Post-norm HS can use the same path as non-early-exit
                    is_early_exit=False,
                )
            except RuntimeError as e:
                logger.debug("Branch generation skipped: %s", e)

        # Phase 1: Cache Lookup (only standard branches)
        cache_hits = self._cache_lookup(batch_size)

        # Phase 2: For cache misses, generate targeted branches as fallback.
        # Targeted branches use the actual correction_token as root, so their
        # draft tokens are correct. This is cheaper than a full propose call.
        # Can be disabled via VLLM_DISABLE_TARGETED_BRANCH to measure
        # pure early-exit prediction quality.
        cache_misses = set(range(batch_size)) - set(cache_hits.keys())
        if cache_misses and not self.disable_targeted_branch:
            try:
                targeted_hits = self._generate_targeted_branches(
                    target_token_ids, target_positions,
                    target_hidden_states, next_token_ids,
                    effective_lti, common_attn_metadata,
                    sampling_metadata, mm_embed_inputs,
                    num_rejected_tokens_gpu, batch_size,
                )
                cache_hits.update(targeted_hits)
            except RuntimeError as e:
                logger.debug("Targeted branch gen skipped: %s", e)

        # Phase 3: Construct result — cache hits + fallback for remaining
        if len(cache_hits) == batch_size and batch_size > 0:
            # All resolved (standard hits + targeted hits)
            result = torch.zeros(
                batch_size, self._num_speculative_tokens,
                dtype=torch.int64, device=next_token_ids.device,
            )
            for req_idx, cached_tokens in cache_hits.items():
                n = min(len(cached_tokens), result.shape[1])
                for t in range(n):
                    result[req_idx, t] = cached_tokens[t]
        else:
            # Still some misses — run full fallback propose
            result = self._underlying.propose(
                target_token_ids, target_positions, target_hidden_states,
                next_token_ids, last_token_indices, common_attn_metadata,
                sampling_metadata, mm_embed_inputs, num_rejected_tokens_gpu,
            ).clone()  # MUST clone — propose returns view of internal buffer
            for req_idx, cached_tokens in cache_hits.items():
                n = min(len(cached_tokens), result.shape[1])
                for t in range(n):
                    result[req_idx, t] = cached_tokens[t]

        # Track cache hit statistics
        self._total_propose_calls += batch_size
        self._total_cache_hits += len(cache_hits)

        # Clear per-call state
        self._sampled_token_ids = None
        self._request_ids = None

        return result

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

            # Log and dump cumulative cache stats
            total = (self._global_hit + self._global_miss
                     + self._global_half_hit)
            hit_rate = self._global_hit / total if total > 0 else 0.0
            logger.info(
                "ParallelProposer cache stats: hit=%d miss=%d "
                "half_hit=%d total=%d hit_rate=%.3f",
                self._global_hit, self._global_miss,
                self._global_half_hit, total, hit_rate,
            )
            stats_file = os.environ.get("VLLM_CACHE_STATS_FILE")
            if stats_file:
                try:
                    with open(stats_file, "w") as f:
                        _json.dump({
                            "hit": self._global_hit,
                            "miss": self._global_miss,
                            "half_hit": self._global_half_hit,
                            "total": total,
                            "hit_rate": hit_rate,
                        }, f)
                except Exception as e:
                    logger.warning("Failed to write cache stats: %s", e)

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
                # Early exit HS are pre-norm: apply TARGET model's final
                # norm before computing logits.  Using the target norm
                # (not MTP SharedHead norm) is critical because the HS
                # come from the target model's intermediate layers.
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

    def _analyze_requests_proposer(
        self,
        target_token_ids: torch.Tensor,
        common_attn_metadata: CommonAttentionMetadata,
        batch_size: int,
    ) -> list[dict | None]:
        """Analyze requests to determine PREFILL/DECODE mode and positions.

        Returns per-request info dicts (or None for skipped requests).
        """
        qsl = common_attn_metadata.query_start_loc
        if qsl.device.type != "cpu":
            qsl_cpu = qsl.cpu()
        else:
            qsl_cpu = qsl

        request_infos: list[dict | None] = []
        for i in range(batch_size):
            if self._request_ids and i >= len(self._request_ids):
                request_infos.append(None)
                continue

            qs = int(qsl_cpu[i].item())
            qe = int(qsl_cpu[i + 1].item())
            ql = qe - qs

            if ql == 0:
                request_infos.append(None)
                continue

            is_prefill = self._is_prefill_request(i, ql)
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

        return request_infos

    def _generate_standard_branches(
        self,
        target_token_ids: torch.Tensor,
        target_positions: torch.Tensor,
        branch_hidden_states: torch.Tensor,
        target_hidden_states: torch.Tensor,
        last_token_indices: torch.Tensor,
        common_attn_metadata: CommonAttentionMetadata,
        sampling_metadata: "SamplingMetadata",
        mm_embed_inputs: tuple | None,
        num_rejected_tokens_gpu: torch.Tensor | None,
        batch_size: int,
        is_early_exit: bool = False,
    ) -> None:
        """Generate standard branch continuations (no scoring needed).

        Analyzes requests, computes root tokens, generates standard
        branches and stores them in per-request caches. Stores
        request_infos for use by _generate_targeted_branches().

        Does NOT need next_token_ids from scoring — overrides per-branch
        with root_token. Does NOT inject targeted branches.
        """
        if self._target_lm_head is None or self._request_ids is None:
            return

        # Analyze requests
        request_infos = self._analyze_requests_proposer(
            target_token_ids, common_attn_metadata, batch_size,
        )
        self._request_infos_proposer = request_infos

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

        # Collect standard branches
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

        # Generate draft tokens for standard branches
        eagle = self._underlying
        num_branches = len(branches)
        total_expanded_tokens = sum(b["ql"] for b in branches)
        use_batched = (
            not eagle.uses_mrope
            and total_expanded_tokens <= eagle.max_num_tokens
            and num_branches + 1 <= eagle.arange.shape[0]
            # Ensure expanded batch doesn't exceed attention builder's
            # pre-allocated buffers (e.g., static_sink_attention).
            and num_branches <= self._max_num_seqs
        )

        # Construct default next_token_ids (overridden per-branch anyway)
        device = target_token_ids.device
        default_next_tokens = torch.zeros(
            batch_size, dtype=torch.int64, device=device,
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
                target_hidden_states, default_next_tokens,
                last_token_indices, common_attn_metadata,
                sampling_metadata, mm_embed_inputs,
                num_rejected_tokens_gpu, batch_size,
            )

        logger.debug(
            "Standard branch gen: batch_size=%d, branches=%d, "
            "batched=%s, max_pos=%d, top_k=%d",
            batch_size, len(branches), use_batched,
            max_branch_pos, self.top_k,
        )

    def _generate_targeted_branches(
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
    ) -> dict[int, list[int]]:
        """Generate targeted branches (guaranteed cache hit).

        For each request, generates a branch whose root_token =
        correction_token at the matching position. Returns draft tokens
        directly without storing in cache, avoiding the redundant
        store-then-lookup round trip.

        Uses request_infos stored by _generate_standard_branches().

        Returns:
            dict mapping req_idx -> draft token list.
        """
        if self._sampled_token_ids is None or self._request_ids is None:
            return {}

        request_infos = self._request_infos_proposer
        if not request_infos:
            return {}

        branches: list[dict] = []
        for i in range(batch_size):
            if i >= len(request_infos):
                continue
            info = request_infos[i]
            if info is None or i >= len(self._request_ids):
                continue

            correction_token = next_token_ids[i].item()

            if info["is_prefill"]:
                matching_pos = info["ql"] - 1
                key = (correction_token,)
            else:
                if isinstance(self._sampled_token_ids, torch.Tensor):
                    if self._sampled_token_ids.dim() == 1:
                        tokens = [self._sampled_token_ids[i].item()]
                    else:
                        tokens = self._sampled_token_ids[i].tolist()
                else:
                    tokens = list(self._sampled_token_ids[i])
                filtered = [t for t in tokens if t != -1]
                if not filtered:
                    continue
                matching_pos = len(filtered) - 1
                key = tuple(filtered)

            # Skip if already covered by standard branches
            req_id = self._request_ids[i]
            cache = self._reuse_caches.get(req_id)
            if cache is not None and key in cache._cache:
                continue

            branches.append({
                "req_idx": i,
                "pos": matching_pos,
                "root_token": correction_token,
                "cache_key": key,
                "qs": info["qs"],
                "ql": info["ql"],
            })

        if not branches:
            return {}

        logger.debug(
            "Targeted branch gen: %d branches (batch=%d)",
            len(branches), batch_size,
        )

        # Targeted branches are few (at most batch_size), use serial gen
        # Return results directly instead of storing in cache
        return self._serial_branch_generate(
            branches, target_token_ids, target_positions,
            target_hidden_states, next_token_ids,
            last_token_indices, common_attn_metadata,
            sampling_metadata, mm_embed_inputs,
            num_rejected_tokens_gpu, batch_size,
            store_in_cache=False,
        )

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
                (may be from early exit layer, post-norm applied in caller).
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

        # Store request_infos for _generate_targeted_branches()
        self._request_infos_proposer = request_infos

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

        # NOTE: Targeted branches are NOT injected here. They are generated
        # after cache lookup (Phase 1) only for cache misses, as a fallback
        # to avoid the expensive full propose call. See propose() Phase 2.

        if not branches:
            return

        # Decide: batched (fast) vs serial (fallback) generation.
        eagle = self._underlying
        num_branches = len(branches)
        total_expanded_tokens = sum(b["ql"] for b in branches)
        use_batched = (
            not eagle.uses_mrope
            and total_expanded_tokens <= eagle.max_num_tokens
            and num_branches + 1 <= eagle.arange.shape[0]
            # Ensure expanded batch doesn't exceed attention builder's
            # pre-allocated buffers (e.g., static_sink_attention).
            and num_branches <= self._max_num_seqs
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
        store_in_cache: bool = True,
    ) -> dict[int, list[int]]:
        """Fallback: generate branches via underlying propose calls.

        Groups non-conflicting branches (different req_idx) into
        single propose calls to reduce overhead.

        Args:
            store_in_cache: If True, store results in per-request caches.
                If False, return results directly as {req_idx: draft_tokens}.

        Returns:
            dict mapping req_idx -> draft token list (always returned,
            but primarily useful when store_in_cache=False).
        """
        results: dict[int, list[int]] = {}
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
                draft_tokens = branch_result[b["req_idx"]].tolist()
                if store_in_cache:
                    req_id = self._request_ids[b["req_idx"]]
                    cache = self._get_or_create_cache(req_id)
                    cache.store_branch(b["cache_key"], draft_tokens)
                results[b["req_idx"]] = draft_tokens

        return results

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
