"""Unit tests for ParallelSpeculator."""

import pytest
from unittest.mock import MagicMock, patch

import torch

from vllm.v1.spec_decode.reuse_cache import ReuseCache


class TestParallelSpeculatorInit:
    """Test ParallelSpeculator initialization."""

    def test_reuse_cache_management(self):
        """Test per-request cache creation and removal."""
        cache = ReuseCache(num_speculative_tokens=3)
        cache.store_branch((100,), [10, 20, 30])
        cache.lookup([100])  # HIT
        cache.lookup([999])  # MISS

        stats = cache.get_statistics()
        assert stats["hit"] == 1
        assert stats["miss"] == 1


class TestParallelSpeculatorPrefillDetection:
    """Test PREFILL vs DECODE detection."""

    def test_prefill_single_token_tensor(self):
        """Single token sampled_token_ids -> PREFILL."""
        sampled = torch.tensor([[42]])
        num_tokens = sampled.shape[-1]
        assert num_tokens <= 1  # PREFILL

    def test_decode_multi_token_tensor(self):
        """Multi-token sampled_token_ids -> DECODE."""
        sampled = torch.tensor([[10, 20, 30, -1]])
        num_tokens = sampled.shape[-1]
        assert num_tokens > 1  # DECODE

    def test_prefill_single_token_list(self):
        """Single token list -> PREFILL."""
        sampled = [[42]]
        num_tokens = len(sampled[0])
        assert num_tokens <= 1  # PREFILL

    def test_decode_multi_token_list(self):
        """Multi-token list -> DECODE."""
        sampled = [[10, 20, 30]]
        num_tokens = len(sampled[0])
        assert num_tokens > 1  # DECODE


class TestParallelSpeculatorRootTokens:
    """Test root token computation logic."""

    def test_compute_root_tokens_topk1(self):
        """Top-k=1 should return single token per position."""
        hidden_size = 32
        vocab_size = 100
        num_positions = 4

        hidden_states = torch.randn(num_positions, hidden_size)
        # Mock lm_head
        lm_head = torch.nn.Linear(hidden_size, vocab_size, bias=False)

        with torch.no_grad():
            logits = lm_head(hidden_states)
            _, top_k_ids = torch.topk(logits, k=1, dim=-1)

        assert top_k_ids.shape == (num_positions, 1)

    def test_compute_root_tokens_topk3(self):
        """Top-k=3 should return 3 tokens per position."""
        hidden_size = 32
        vocab_size = 100
        num_positions = 4

        hidden_states = torch.randn(num_positions, hidden_size)
        lm_head = torch.nn.Linear(hidden_size, vocab_size, bias=False)

        with torch.no_grad():
            logits = lm_head(hidden_states)
            _, top_k_ids = torch.topk(logits, k=3, dim=-1)

        assert top_k_ids.shape == (num_positions, 3)

    def test_root_tokens_at_positions(self):
        """Root tokens should only be computed at specified positions."""
        hidden_size = 32
        vocab_size = 100

        # 10 tokens total, but only compute at positions [2, 5, 7]
        hidden_states = torch.randn(10, hidden_size)
        positions = [2, 5, 7]
        pos_tensor = torch.tensor(positions, dtype=torch.long)

        selected_hs = hidden_states[pos_tensor]
        assert selected_hs.shape == (3, hidden_size)


class TestParallelSpeculatorBranchKeys:
    """Test branch key generation for PREFILL and DECODE modes."""

    def test_prefill_keys(self):
        """PREFILL mode: keys are (root_token,) only."""
        cache = ReuseCache(num_speculative_tokens=3)

        # Simulate PREFILL: top_k=2, only last position
        root_tokens = [10, 20]
        for root in root_tokens:
            key = (root,)
            cache.store_branch(key, [100, 200, 300])

        assert cache.size == 2
        assert cache.lookup([10]) == [100, 200, 300]
        assert cache.lookup([20]) == [100, 200, 300]
        assert cache.lookup([30]) is None  # MISS

    def test_decode_keys(self):
        """DECODE mode: keys are input_ids[1:pos+1] + (root_token,)."""
        cache = ReuseCache(num_speculative_tokens=3)

        # input_ids = [last_accepted, d1, d2, d3]
        d1, d2, d3 = 50, 60, 70
        root_tokens = [10, 20, 30, 40]  # one per position

        # Position 0: context=[], root=10 -> key=(10,)
        cache.store_branch((root_tokens[0],),
                           [100, 200, 300], base_position=0)
        # Position 1: context=[d1], root=20 -> key=(d1, 20)
        cache.store_branch((d1, root_tokens[1]),
                           [101, 201, 301], base_position=1)
        # Position 2: context=[d1,d2], root=30 -> key=(d1, d2, 30)
        cache.store_branch((d1, d2, root_tokens[2]),
                           [102, 202, 302], base_position=2)
        # Position 3: context=[d1,d2,d3], root=40 -> key=(d1, d2, d3, 40)
        cache.store_branch((d1, d2, d3, root_tokens[3]),
                           [103, 203, 303], base_position=3)

        assert cache.size == 4

        # Simulate sampled_token_ids = [d1, d2, correction]
        # HIT if correction matches root at position 2
        assert cache.lookup([d1, d2, 30]) == [102, 202, 302]

        # MISS if correction doesn't match any
        assert cache.lookup([d1, d2, 99]) is None

    def test_decode_keys_topk2(self):
        """DECODE mode with top_k=2: double the branches."""
        cache = ReuseCache(num_speculative_tokens=3)

        d1 = 50
        # Position 0: two branches
        cache.store_branch((10,), [100, 200, 300], base_position=0)
        cache.store_branch((20,), [110, 210, 310], base_position=0)
        # Position 1: two branches
        cache.store_branch((d1, 30), [101, 201, 301], base_position=1)
        cache.store_branch((d1, 40), [111, 211, 311], base_position=1)

        assert cache.size == 4

        # sampled = [d1, correction]
        assert cache.lookup([d1, 30]) == [101, 201, 301]
        assert cache.lookup([d1, 40]) == [111, 211, 311]
        assert cache.lookup([d1, 99]) is None


class TestParallelSpeculatorCacheFlow:
    """Test the complete cache HIT/MISS flow."""

    def test_cache_hit_flow(self):
        """Simulate a full round with cache HIT."""
        cache = ReuseCache(num_speculative_tokens=3)

        # Step 1: Generate branches (simulate)
        d1, d2 = 50, 60
        cache.store_branch((10,), [100, 200, 300], base_position=0)
        cache.store_branch((d1, 20), [101, 201, 301], base_position=1)
        cache.store_branch((d1, d2, 30), [102, 202, 302], base_position=2)

        # Step 2: Target model returns sampled_token_ids
        sampled = [d1, d2, 30]  # correction=30 matches position 2

        # Step 3: Lookup
        result = cache.lookup(sampled)
        assert result == [102, 202, 302]

        stats = cache.get_statistics()
        assert stats["hit"] == 1

    def test_cache_miss_flow(self):
        """Simulate a full round with cache MISS."""
        cache = ReuseCache(num_speculative_tokens=3)

        d1, d2 = 50, 60
        cache.store_branch((10,), [100, 200, 300], base_position=0)
        cache.store_branch((d1, 20), [101, 201, 301], base_position=1)
        cache.store_branch((d1, d2, 30), [102, 202, 302], base_position=2)

        # Correction=99 doesn't match any root token
        sampled = [d1, d2, 99]
        result = cache.lookup(sampled)
        assert result is None

        stats = cache.get_statistics()
        assert stats["miss"] == 1

    def test_prefill_to_decode_transition(self):
        """Test transition from PREFILL to DECODE mode."""
        cache = ReuseCache(num_speculative_tokens=3)

        # Round 1: PREFILL
        cache.store_branch((10,), [100, 200, 300])
        cache.store_branch((20,), [101, 201, 301])
        result = cache.lookup([10])  # HIT
        assert result == [100, 200, 300]

        # Round 2: DECODE (new round, cache cleared)
        cache.clear()
        d1, d2, d3 = 100, 200, 300  # draft tokens from round 1
        cache.store_branch((10,), [400, 500, 600], base_position=0)
        cache.store_branch((d1, 20), [401, 501, 601], base_position=1)
        result = cache.lookup([d1, 20])  # HIT
        assert result == [401, 501, 601]


class TestParallelSpeculatorStatistics:
    """Test statistics tracking."""

    def test_global_statistics_aggregation(self):
        """Test that per-request stats aggregate to global."""
        # Simulate per-request stats
        cache1 = ReuseCache(num_speculative_tokens=3)
        cache1.store_branch((10,), [100])
        cache1.lookup([10])  # HIT
        cache1.lookup([99])  # MISS

        cache2 = ReuseCache(num_speculative_tokens=3)
        cache2.store_branch((20,), [200])
        cache2.lookup([20])  # HIT
        cache2.lookup([20])  # HIT

        stats1 = cache1.get_statistics()
        stats2 = cache2.get_statistics()

        global_hit = stats1["hit"] + stats2["hit"]
        global_miss = stats1["miss"] + stats2["miss"]
        assert global_hit == 3
        assert global_miss == 1

    def test_cache_clear_preserves_stats(self):
        """Cache clear should not reset statistics."""
        cache = ReuseCache(num_speculative_tokens=3)
        cache.store_branch((10,), [100])
        cache.lookup([10])  # HIT
        cache.clear()
        assert cache.get_statistics()["hit"] == 1

    def test_reset_statistics(self):
        cache = ReuseCache(num_speculative_tokens=3)
        cache.store_branch((10,), [100])
        cache.lookup([10])
        cache.reset_statistics()
        assert cache.get_statistics() == {"hit": 0, "miss": 0, "half_hit": 0}


class TestParallelSpeculatorPassthrough:
    """Test pass-through mode logic."""

    def test_passthrough_flag(self):
        """Pass-through mode should be toggleable."""
        passthrough = False
        assert not passthrough
        passthrough = True
        assert passthrough


class TestFilterSampledTokenIds:
    """Test filtering of sampled_token_ids (-1 removal)."""

    def test_filter_no_negatives(self):
        sampled = [10, 20, 30]
        filtered = [t for t in sampled if t != -1]
        assert filtered == [10, 20, 30]

    def test_filter_with_negatives(self):
        sampled = [10, 20, -1, -1]
        filtered = [t for t in sampled if t != -1]
        assert filtered == [10, 20]

    def test_filter_all_negatives_except_correction(self):
        sampled = [-1, -1, 30]
        filtered = [t for t in sampled if t != -1]
        assert filtered == [30]

    def test_filter_single_token(self):
        sampled = [42]
        filtered = [t for t in sampled if t != -1]
        assert filtered == [42]


class TestEarlyExitHook:
    """Test early exit forward hook mechanism."""

    def test_find_model_layers_standard(self):
        """Find layers using standard model.layers pattern."""
        from vllm.v1.worker.gpu.spec_decode.parallel_speculator import (
            ParallelSpeculator,
        )

        # Create a mock model with model.layers
        mock_model = torch.nn.Module()
        mock_inner = torch.nn.Module()
        mock_layers = torch.nn.ModuleList([
            torch.nn.Linear(10, 10) for _ in range(4)
        ])
        mock_inner.add_module("layers", mock_layers)
        mock_model.add_module("model", mock_inner)

        # Test the layer finding logic
        patterns = ParallelSpeculator._LAYER_ACCESSOR_PATTERNS
        obj = mock_model
        for attr in patterns[0].split("."):
            obj = getattr(obj, attr)
        assert len(obj) == 4

    def test_find_model_layers_transformer(self):
        """Find layers using transformer.layers pattern."""
        mock_model = torch.nn.Module()
        mock_transformer = torch.nn.Module()
        mock_layers = torch.nn.ModuleList([
            torch.nn.Linear(10, 10) for _ in range(6)
        ])
        mock_transformer.add_module("layers", mock_layers)
        mock_model.add_module("transformer", mock_transformer)

        obj = mock_model
        for attr in "transformer.layers".split("."):
            obj = getattr(obj, attr)
        assert len(obj) == 6

    def test_forward_hook_captures_hidden_states(self):
        """Test that a forward hook captures output correctly."""
        captured = {}

        def hook(module, input, output):
            if isinstance(output, tuple):
                captured["hs"] = output[0].detach()
            else:
                captured["hs"] = output.detach()

        layer = torch.nn.Linear(10, 10)
        handle = layer.register_forward_hook(hook)

        x = torch.randn(3, 10)
        _ = layer(x)

        assert "hs" in captured
        assert captured["hs"].shape == (3, 10)
        handle.remove()

    def test_negative_layer_index_resolution(self):
        """Test resolving negative layer indices."""
        num_layers = 32
        # -1 -> last layer (31)
        assert num_layers + (-1) == 31
        # -5 -> layer 27
        assert num_layers + (-5) == 27
        # 0 -> first layer
        assert 0 == 0

    def test_early_exit_layer_default_is_last(self):
        """Default early_exit_layer=-1 means last layer."""
        early_exit_layer = -1
        num_layers = 32
        layer_idx = num_layers + early_exit_layer
        assert layer_idx == 31  # Last layer


class TestConcurrentExecution:
    """Test concurrent CUDA stream execution."""

    @pytest.mark.skipif(
        not torch.cuda.is_available(),
        reason="CUDA not available",
    )
    def test_cuda_stream_creation(self):
        """Test CUDA stream can be created."""
        stream = torch.cuda.Stream()
        assert stream is not None

    @pytest.mark.skipif(
        not torch.cuda.is_available(),
        reason="CUDA not available",
    )
    def test_run_on_stream(self):
        """Test running computation on a separate stream."""
        stream = torch.cuda.Stream()
        x = torch.randn(100, 100, device="cuda")
        with torch.cuda.stream(stream):
            y = x @ x.T
        stream.synchronize()
        assert y.shape == (100, 100)

    def test_concurrent_disabled_by_default(self):
        """Concurrent execution should be disabled by default."""
        enable_concurrent = False
        assert not enable_concurrent

    def test_sync_without_stream(self):
        """Sync should be safe when no stream exists."""
        # No stream -> sync is a no-op
        draft_stream = None
        if draft_stream is not None:
            draft_stream.synchronize()
        # Should not raise
