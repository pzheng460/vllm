"""Unit tests for Parallel-SD ReuseCache."""

import pytest

from vllm.v1.spec_decode.reuse_cache import ReuseCache, ReuseCacheEntry


class TestReuseCacheEntry:
    """Tests for ReuseCacheEntry dataclass."""

    def test_basic_creation(self):
        entry = ReuseCacheEntry(
            continuation_tokens=[10, 20, 30],
            base_position=2,
        )
        assert entry.continuation_tokens == [10, 20, 30]
        assert entry.base_position == 2

    def test_default_base_position(self):
        entry = ReuseCacheEntry(continuation_tokens=[1, 2, 3])
        assert entry.base_position == 0


class TestReuseCacheStore:
    """Tests for ReuseCache.store_branch()."""

    def test_store_single_branch(self):
        cache = ReuseCache(num_speculative_tokens=3)
        cache.store_branch(
            lookup_key=(100,),
            branch_tokens=[10, 20, 30],
            base_position=0,
        )
        assert cache.size == 1

    def test_store_multiple_branches(self):
        cache = ReuseCache(num_speculative_tokens=3)
        cache.store_branch((100,), [10, 20, 30], base_position=0)
        cache.store_branch((200, 100), [40, 50, 60], base_position=1)
        cache.store_branch((200, 300, 100), [70, 80, 90], base_position=2)
        assert cache.size == 3

    def test_store_overwrites_same_key(self):
        cache = ReuseCache(num_speculative_tokens=3)
        cache.store_branch((100,), [10, 20, 30])
        cache.store_branch((100,), [40, 50, 60])
        assert cache.size == 1
        result = cache.lookup([100])
        assert result == [40, 50, 60]

    def test_store_decode_mode_keys(self):
        """Test DECODE mode key patterns: input_ids[1:pos+1] + root_token."""
        cache = ReuseCache(num_speculative_tokens=3)
        # Position 0: context=[], root=a -> key=(a,)
        cache.store_branch((10,), [100, 200, 300], base_position=0)
        # Position 1: context=[d1], root=b -> key=(d1, b)
        cache.store_branch((50, 20), [101, 201, 301], base_position=1)
        # Position 2: context=[d1,d2], root=c -> key=(d1, d2, c)
        cache.store_branch((50, 60, 30), [102, 202, 302], base_position=2)
        assert cache.size == 3

    def test_store_prefill_mode_keys(self):
        """Test PREFILL mode key patterns: (root_token,)."""
        cache = ReuseCache(num_speculative_tokens=3)
        # Top-k=2, only last position
        cache.store_branch((10,), [100, 200, 300], base_position=0)
        cache.store_branch((20,), [101, 201, 301], base_position=0)
        assert cache.size == 2


class TestReuseCacheLookup:
    """Tests for ReuseCache.lookup()."""

    def test_lookup_exact_hit(self):
        cache = ReuseCache(num_speculative_tokens=3)
        cache.store_branch((100,), [10, 20, 30])
        result = cache.lookup([100])
        assert result == [10, 20, 30]

    def test_lookup_miss(self):
        cache = ReuseCache(num_speculative_tokens=3)
        cache.store_branch((100,), [10, 20, 30])
        result = cache.lookup([999])
        assert result is None

    def test_lookup_empty_cache(self):
        cache = ReuseCache(num_speculative_tokens=3)
        result = cache.lookup([100])
        assert result is None

    def test_lookup_multi_token_key_hit(self):
        cache = ReuseCache(num_speculative_tokens=3)
        cache.store_branch((50, 60, 30), [102, 202, 302])
        result = cache.lookup([50, 60, 30])
        assert result == [102, 202, 302]

    def test_lookup_multi_token_key_miss(self):
        cache = ReuseCache(num_speculative_tokens=3)
        cache.store_branch((50, 60, 30), [102, 202, 302])
        # Same prefix but different last token
        result = cache.lookup([50, 60, 99])
        assert result is None

    def test_lookup_returns_copy(self):
        """Ensure lookup returns a copy, not a reference."""
        cache = ReuseCache(num_speculative_tokens=3)
        cache.store_branch((100,), [10, 20, 30])
        result = cache.lookup([100])
        result[0] = 999
        # Original should be unchanged
        result2 = cache.lookup([100])
        assert result2 == [10, 20, 30]

    def test_lookup_decode_flow(self):
        """Simulate full DECODE mode: store branches, lookup with
        sampled_token_ids."""
        cache = ReuseCache(num_speculative_tokens=3)
        # input_ids = [last_accepted, d1, d2, d3]
        # root_tokens = [a, b, c, d] (top-1 at each position)
        d1, d2, d3 = 50, 60, 70
        a, b, c, d = 10, 20, 30, 40

        cache.store_branch((a,), [100, 200, 300], base_position=0)
        cache.store_branch((d1, b), [101, 201, 301], base_position=1)
        cache.store_branch((d1, d2, c), [102, 202, 302], base_position=2)
        cache.store_branch((d1, d2, d3, d), [103, 203, 303],
                           base_position=3)

        # sampled_token_ids = [d1, d2, correction] (filter -1)
        # HIT if correction == c
        result = cache.lookup([d1, d2, c])
        assert result == [102, 202, 302]

        # MISS if correction != c
        result_miss = cache.lookup([d1, d2, 99])
        assert result_miss is None

    def test_lookup_prefill_flow(self):
        """Simulate PREFILL mode: store top-k branches, lookup correction."""
        cache = ReuseCache(num_speculative_tokens=3)
        cache.store_branch((10,), [100, 200, 300])
        cache.store_branch((20,), [101, 201, 301])

        # HIT
        assert cache.lookup([10]) == [100, 200, 300]
        assert cache.lookup([20]) == [101, 201, 301]
        # MISS
        assert cache.lookup([30]) is None


class TestReuseCacheHalfHit:
    """Tests for half-cache-hit (prefix matching)."""

    def test_half_hit_disabled_by_default(self):
        cache = ReuseCache(num_speculative_tokens=3,
                           enable_half_cache_hit=False)
        cache.store_branch((50, 60, 30), [102, 202, 302])
        # Different last token, half-hit disabled -> MISS
        result = cache.lookup([50, 60, 99])
        assert result is None
        stats = cache.get_statistics()
        assert stats["half_hit"] == 0
        assert stats["miss"] == 1

    def test_half_hit_enabled(self):
        cache = ReuseCache(num_speculative_tokens=3,
                           enable_half_cache_hit=True)
        cache.store_branch((50, 60, 30), [102, 202, 302])
        # Same prefix [50, 60] but different last token -> HALF-HIT
        result = cache.lookup([50, 60, 99])
        assert result == [102, 202, 302]
        stats = cache.get_statistics()
        assert stats["half_hit"] == 1
        assert stats["miss"] == 0

    def test_half_hit_prefers_exact(self):
        """Exact match should take priority over half-hit."""
        cache = ReuseCache(num_speculative_tokens=3,
                           enable_half_cache_hit=True)
        cache.store_branch((50, 60, 30), [102, 202, 302])
        cache.store_branch((50, 60, 99), [999, 888, 777])
        result = cache.lookup([50, 60, 99])
        assert result == [999, 888, 777]
        stats = cache.get_statistics()
        assert stats["hit"] == 1
        assert stats["half_hit"] == 0

    def test_half_hit_single_token_key(self):
        """Half-hit requires key length > 1."""
        cache = ReuseCache(num_speculative_tokens=3,
                           enable_half_cache_hit=True)
        cache.store_branch((10,), [100, 200, 300])
        # Single token key, different -> MISS (no prefix to match)
        result = cache.lookup([20])
        assert result is None
        assert cache.get_statistics()["miss"] == 1

    def test_half_hit_length_mismatch(self):
        """Half-hit only matches same-length keys."""
        cache = ReuseCache(num_speculative_tokens=3,
                           enable_half_cache_hit=True)
        cache.store_branch((50, 60, 30), [102, 202, 302])
        # Different length -> no half-hit
        result = cache.lookup([50, 60])
        assert result is None
        assert cache.get_statistics()["miss"] == 1


class TestReuseCacheClear:
    """Tests for ReuseCache.clear()."""

    def test_clear_empties_cache(self):
        cache = ReuseCache(num_speculative_tokens=3)
        cache.store_branch((100,), [10, 20, 30])
        cache.store_branch((200,), [40, 50, 60])
        assert cache.size == 2
        cache.clear()
        assert cache.size == 0

    def test_clear_does_not_reset_statistics(self):
        cache = ReuseCache(num_speculative_tokens=3)
        cache.store_branch((100,), [10, 20, 30])
        cache.lookup([100])  # HIT
        cache.clear()
        stats = cache.get_statistics()
        assert stats["hit"] == 1  # Stats preserved


class TestReuseCacheStatistics:
    """Tests for statistics tracking."""

    def test_initial_statistics(self):
        cache = ReuseCache(num_speculative_tokens=3)
        stats = cache.get_statistics()
        assert stats == {"hit": 0, "miss": 0, "half_hit": 0}

    def test_hit_miss_counting(self):
        cache = ReuseCache(num_speculative_tokens=3)
        cache.store_branch((100,), [10, 20, 30])

        cache.lookup([100])  # HIT
        cache.lookup([100])  # HIT
        cache.lookup([999])  # MISS

        stats = cache.get_statistics()
        assert stats["hit"] == 2
        assert stats["miss"] == 1
        assert stats["half_hit"] == 0

    def test_half_hit_counting(self):
        cache = ReuseCache(num_speculative_tokens=3,
                           enable_half_cache_hit=True)
        cache.store_branch((50, 60, 30), [102, 202, 302])

        cache.lookup([50, 60, 30])  # HIT
        cache.lookup([50, 60, 99])  # HALF-HIT
        cache.lookup([99, 99, 99])  # MISS

        stats = cache.get_statistics()
        assert stats["hit"] == 1
        assert stats["half_hit"] == 1
        assert stats["miss"] == 1

    def test_reset_statistics(self):
        cache = ReuseCache(num_speculative_tokens=3)
        cache.store_branch((100,), [10, 20, 30])
        cache.lookup([100])  # HIT
        cache.lookup([999])  # MISS

        cache.reset_statistics()
        stats = cache.get_statistics()
        assert stats == {"hit": 0, "miss": 0, "half_hit": 0}

    def test_latency_statistics(self):
        cache = ReuseCache(num_speculative_tokens=3)
        cache.store_branch((100,), [10, 20, 30])
        cache.lookup([100])

        latency = cache.get_latency_statistics()
        assert "avg_lookup_us" in latency
        assert "avg_store_us" in latency
        assert latency["avg_lookup_us"] >= 0
        assert latency["avg_store_us"] >= 0

    def test_latency_statistics_empty(self):
        cache = ReuseCache(num_speculative_tokens=3)
        latency = cache.get_latency_statistics()
        assert latency["avg_lookup_us"] == 0.0
        assert latency["avg_store_us"] == 0.0

    def test_reset_clears_latency(self):
        cache = ReuseCache(num_speculative_tokens=3)
        cache.store_branch((100,), [10, 20, 30])
        cache.lookup([100])
        cache.reset_statistics()
        latency = cache.get_latency_statistics()
        assert latency["avg_lookup_us"] == 0.0
        assert latency["avg_store_us"] == 0.0


class TestReuseCacheSize:
    """Tests for cache size property."""

    def test_empty_cache_size(self):
        cache = ReuseCache(num_speculative_tokens=3)
        assert cache.size == 0

    def test_size_after_stores(self):
        cache = ReuseCache(num_speculative_tokens=3)
        cache.store_branch((1,), [10])
        assert cache.size == 1
        cache.store_branch((2,), [20])
        assert cache.size == 2

    def test_size_after_clear(self):
        cache = ReuseCache(num_speculative_tokens=3)
        cache.store_branch((1,), [10])
        cache.clear()
        assert cache.size == 0
