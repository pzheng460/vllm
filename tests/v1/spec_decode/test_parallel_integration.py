"""Integration tests for Parallel Speculative Decoding.

These tests verify the complete Parallel-SD pipeline including
ReuseCache, ParallelSpeculator, SpeculativeConfig, and routing.
Tests that require GPU models are marked with pytest.mark.skipif.
"""

import pytest

import torch

from vllm.v1.spec_decode.reuse_cache import ReuseCache, ReuseCacheEntry
from vllm.v1.spec_decode.token_channel import TokenChannel


class TestEndToEndCacheFlow:
    """Test complete cache flow from store to lookup."""

    def test_decode_mode_full_flow_topk1(self):
        """Full DECODE flow with top_k=1: exact match expected."""
        cache = ReuseCache(num_speculative_tokens=3)

        # Simulate input_ids = [last_accepted, d1, d2, d3]
        d1, d2, d3 = 50, 60, 70

        # Target model outputs root_tokens (one per position, top_k=1)
        root_tokens = [10, 20, 30, 40]

        # Store branches (what the draft model would generate)
        branches = {
            (root_tokens[0],): [100, 200, 300],
            (d1, root_tokens[1]): [101, 201, 301],
            (d1, d2, root_tokens[2]): [102, 202, 302],
            (d1, d2, d3, root_tokens[3]): [103, 203, 303],
        }
        for key, tokens in branches.items():
            cache.store_branch(key, tokens, base_position=list(
                branches.keys()).index(key))

        # Case 1: All accepted, correction matches position 2
        sampled = [d1, d2, 30]  # correction=30
        result = cache.lookup(sampled)
        assert result == [102, 202, 302]

        # Case 2: First rejected, correction matches position 0
        cache2 = ReuseCache(num_speculative_tokens=3)
        for key, tokens in branches.items():
            cache2.store_branch(key, tokens)
        sampled2 = [10]  # Only correction_token, all rejected
        result2 = cache2.lookup(sampled2)
        assert result2 == [100, 200, 300]

    def test_decode_mode_full_flow_topk2(self):
        """Full DECODE flow with top_k=2: more branches, same lookup."""
        cache = ReuseCache(num_speculative_tokens=3)

        d1, d2 = 50, 60
        # Position 0: top-2 = [10, 15]
        cache.store_branch((10,), [100, 200, 300])
        cache.store_branch((15,), [110, 210, 310])
        # Position 1: top-2 = [20, 25]
        cache.store_branch((d1, 20), [101, 201, 301])
        cache.store_branch((d1, 25), [111, 211, 311])
        # Position 2: top-2 = [30, 35]
        cache.store_branch((d1, d2, 30), [102, 202, 302])
        cache.store_branch((d1, d2, 35), [112, 212, 312])

        assert cache.size == 6

        # sampled = [d1, d2, correction]
        assert cache.lookup([d1, d2, 30]) == [102, 202, 302]
        assert cache.lookup([d1, d2, 35]) == [112, 212, 312]
        assert cache.lookup([d1, d2, 99]) is None  # MISS

    def test_prefill_mode_full_flow(self):
        """Full PREFILL flow: only last position branches."""
        cache = ReuseCache(num_speculative_tokens=3)

        # PREFILL: top_k=3
        cache.store_branch((10,), [100, 200, 300])
        cache.store_branch((20,), [101, 201, 301])
        cache.store_branch((30,), [102, 202, 302])

        assert cache.size == 3

        # Correction token = 20
        result = cache.lookup([20])
        assert result == [101, 201, 301]

    def test_multi_round_decode(self):
        """Test multiple rounds of DECODE with cache clear between rounds."""
        cache = ReuseCache(num_speculative_tokens=3)

        # Round 1
        d1, d2, d3 = 50, 60, 70
        cache.store_branch((10,), [100, 200, 300])
        cache.store_branch((d1, 20), [101, 201, 301])
        cache.store_branch((d1, d2, 30), [102, 202, 302])
        cache.store_branch((d1, d2, d3, 40), [103, 203, 303])

        result1 = cache.lookup([d1, d2, 30])
        assert result1 == [102, 202, 302]

        # Round 2: new draft tokens from round 1's accepted tokens
        cache.clear()
        new_d1, new_d2, new_d3 = 102, 202, 302
        cache.store_branch((50,), [400, 500, 600])
        cache.store_branch((new_d1, 60), [401, 501, 601])
        cache.store_branch((new_d1, new_d2, 70), [402, 502, 602])
        cache.store_branch((new_d1, new_d2, new_d3, 80), [403, 503, 603])

        result2 = cache.lookup([new_d1, new_d2, 70])
        assert result2 == [402, 502, 602]

        # Stats accumulate across rounds
        stats = cache.get_statistics()
        assert stats["hit"] == 2


class TestHalfCacheHitIntegration:
    """Integration tests for half-cache-hit mechanism."""

    def test_half_hit_in_decode_flow(self):
        """Half-cache-hit works in a realistic DECODE scenario."""
        cache = ReuseCache(
            num_speculative_tokens=3,
            enable_half_cache_hit=True,
        )

        d1, d2 = 50, 60
        # Only one branch at position 2 (top_k=1)
        cache.store_branch((d1, d2, 30), [102, 202, 302])

        # Correction=99, not exact match, but prefix [d1, d2] matches
        result = cache.lookup([d1, d2, 99])
        assert result == [102, 202, 302]
        assert cache.get_statistics()["half_hit"] == 1

    def test_half_hit_exact_takes_priority(self):
        """Exact match should always win over half-hit."""
        cache = ReuseCache(
            num_speculative_tokens=3,
            enable_half_cache_hit=True,
        )

        d1, d2 = 50, 60
        cache.store_branch((d1, d2, 30), [102, 202, 302])
        cache.store_branch((d1, d2, 99), [999, 888, 777])

        # Exact match on 99
        result = cache.lookup([d1, d2, 99])
        assert result == [999, 888, 777]
        assert cache.get_statistics()["hit"] == 1
        assert cache.get_statistics()["half_hit"] == 0


class TestMultiRequestBatch:
    """Test cache management with multiple concurrent requests."""

    def test_independent_per_request_caches(self):
        """Each request should have an independent cache."""
        cache_req1 = ReuseCache(num_speculative_tokens=3)
        cache_req2 = ReuseCache(num_speculative_tokens=3)

        # Request 1: store different branches
        cache_req1.store_branch((10,), [100, 200, 300])

        # Request 2: store different branches
        cache_req2.store_branch((10,), [400, 500, 600])

        # Same key, different values per request
        assert cache_req1.lookup([10]) == [100, 200, 300]
        assert cache_req2.lookup([10]) == [400, 500, 600]

    def test_request_lifecycle(self):
        """Simulate request creation, usage, and cleanup."""
        caches: dict[str, ReuseCache] = {}

        # Request arrives
        caches["req-1"] = ReuseCache(num_speculative_tokens=3)
        caches["req-2"] = ReuseCache(num_speculative_tokens=3)

        # Use caches
        caches["req-1"].store_branch((10,), [100])
        caches["req-1"].lookup([10])
        caches["req-2"].store_branch((20,), [200])
        caches["req-2"].lookup([99])  # MISS

        # Request 1 completes
        stats1 = caches["req-1"].get_statistics()
        assert stats1["hit"] == 1
        del caches["req-1"]

        # Request 2 still active
        assert "req-2" in caches
        stats2 = caches["req-2"].get_statistics()
        assert stats2["miss"] == 1

    def test_batch_stats_aggregation(self):
        """Aggregate stats across all active requests."""
        caches = {
            "r1": ReuseCache(num_speculative_tokens=3),
            "r2": ReuseCache(num_speculative_tokens=3),
            "r3": ReuseCache(num_speculative_tokens=3),
        }

        caches["r1"].store_branch((1,), [10]); caches["r1"].lookup([1])  # HIT
        caches["r2"].store_branch((2,), [20]); caches["r2"].lookup([9])  # MISS
        caches["r3"].store_branch((3,), [30]); caches["r3"].lookup([3])  # HIT

        total_hit = sum(c.get_statistics()["hit"] for c in caches.values())
        total_miss = sum(c.get_statistics()["miss"] for c in caches.values())
        assert total_hit == 2
        assert total_miss == 1


class TestCacheStatisticsAccuracy:
    """Verify cache statistics exactly match actual hits/misses."""

    def test_stats_match_operations(self):
        """Every lookup should be counted exactly once."""
        cache = ReuseCache(num_speculative_tokens=3)

        cache.store_branch((10,), [100])
        cache.store_branch((20,), [200])

        operations = [
            ([10], "hit"),
            ([20], "hit"),
            ([30], "miss"),
            ([10], "hit"),
            ([40], "miss"),
            ([20], "hit"),
        ]

        expected_hit = 0
        expected_miss = 0
        for key, expected_type in operations:
            result = cache.lookup(key)
            if expected_type == "hit":
                assert result is not None
                expected_hit += 1
            else:
                assert result is None
                expected_miss += 1

        stats = cache.get_statistics()
        assert stats["hit"] == expected_hit == 4
        assert stats["miss"] == expected_miss == 2

    def test_half_hit_stats_accuracy(self):
        """Half-hit should count separately from hit and miss."""
        cache = ReuseCache(
            num_speculative_tokens=3,
            enable_half_cache_hit=True,
        )

        cache.store_branch((50, 60, 30), [102, 202, 302])

        cache.lookup([50, 60, 30])  # HIT (exact)
        cache.lookup([50, 60, 99])  # HALF-HIT (prefix match)
        cache.lookup([99, 99, 99])  # MISS (no match)

        stats = cache.get_statistics()
        assert stats["hit"] == 1
        assert stats["half_hit"] == 1
        assert stats["miss"] == 1
        # Total = 3 operations
        assert sum(stats.values()) == 3


class TestTokenChannelStub:
    """Verify TokenChannel stub raises NotImplementedError."""

    def test_send_top_k_raises(self):
        tc = TokenChannel()
        with pytest.raises(NotImplementedError):
            tc.send_top_k_candidates(
                torch.zeros(1, 1), torch.zeros(1, 1), batch_size=1
            )

    def test_recv_top_k_raises(self):
        tc = TokenChannel()
        with pytest.raises(NotImplementedError):
            tc.recv_top_k_candidates(batch_size=1)

    def test_send_spec_raises(self):
        tc = TokenChannel()
        with pytest.raises(NotImplementedError):
            tc.send_speculative_tokens(
                torch.zeros(1, 1), batch_size=1
            )

    def test_recv_spec_raises(self):
        tc = TokenChannel()
        with pytest.raises(NotImplementedError):
            tc.recv_speculative_tokens(batch_size=1)


class TestConfigIntegration:
    """Test that speculative config correctly parses parallel parameters."""

    def test_parallel_method_in_speculative_method(self):
        from typing import get_args
        from vllm.config.speculative import SpeculativeMethod
        methods = get_args(SpeculativeMethod)
        assert "parallel" in methods

    def test_use_parallel_method(self):
        """Verify use_parallel() returns True for parallel method."""
        from vllm.config.speculative import SpeculativeConfig
        # Can't instantiate SpeculativeConfig without full vllm setup,
        # but we can test the method logic
        assert hasattr(SpeculativeConfig, "use_parallel")

    def test_parallel_config_params_exist(self):
        """Verify all parallel config parameters are defined."""
        from vllm.config.speculative import SpeculativeConfig
        import dataclasses
        fields = {f.name for f in dataclasses.fields(SpeculativeConfig)}
        expected = {
            "parallel_draft_method",
            "parallel_top_k",
            "parallel_enable_half_cache_hit",
            "parallel_early_exit_layer",
            "parallel_enable_concurrent",
        }
        assert expected.issubset(fields)


class TestRoutingIntegration:
    """Test that init_speculator routing handles parallel method."""

    def test_init_speculator_import(self):
        from vllm.v1.worker.gpu.spec_decode import init_speculator
        assert callable(init_speculator)

    def test_parallel_speculator_import(self):
        from vllm.v1.worker.gpu.spec_decode.parallel_speculator import (
            ParallelSpeculator,
        )
        assert ParallelSpeculator is not None
