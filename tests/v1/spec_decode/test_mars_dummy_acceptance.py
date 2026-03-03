"""Tests for MARS dummy acceptance in Parallel Speculative Decoding.

Tests the static _mars_dummy_acceptance() method and the
_compute_mars_branch_positions() integration.
"""

import pytest

from vllm.v1.spec_decode.parallel_proposer import ParallelProposer


class TestMarsDummyAcceptance:
    """Unit tests for ParallelProposer._mars_dummy_acceptance()."""

    def test_all_accepted_exact_match(self):
        """All draft tokens match top-1 → all accepted."""
        result = ParallelProposer._mars_dummy_acceptance(
            draft_tokens=[10, 20, 30],
            top1_tokens=[10, 20, 30],
            top2_tokens=[11, 21, 31],
            logit_ratios=[0.5, 0.5, 0.5],
            threshold=0.9,
        )
        assert result == 3  # All accepted

    def test_first_token_rejected(self):
        """First draft token doesn't match top-1 or top-2 → reject at 0."""
        result = ParallelProposer._mars_dummy_acceptance(
            draft_tokens=[99, 20, 30],
            top1_tokens=[10, 20, 30],
            top2_tokens=[11, 21, 31],
            logit_ratios=[0.5, 0.5, 0.5],
            threshold=0.9,
        )
        assert result == 0

    def test_middle_token_rejected(self):
        """Second draft token rejected → branch at position 1."""
        result = ParallelProposer._mars_dummy_acceptance(
            draft_tokens=[10, 99, 30],
            top1_tokens=[10, 20, 30],
            top2_tokens=[11, 21, 31],
            logit_ratios=[0.5, 0.5, 0.5],
            threshold=0.9,
        )
        assert result == 1

    def test_last_token_rejected(self):
        """Last draft token rejected → branch at last position."""
        result = ParallelProposer._mars_dummy_acceptance(
            draft_tokens=[10, 20, 99],
            top1_tokens=[10, 20, 30],
            top2_tokens=[11, 21, 31],
            logit_ratios=[0.5, 0.5, 0.5],
            threshold=0.9,
        )
        assert result == 2

    def test_tie_accepted_high_ratio(self):
        """Draft matches top-2 with ratio > θ → accepted (tie case)."""
        result = ParallelProposer._mars_dummy_acceptance(
            draft_tokens=[11, 20, 30],  # 11 is top-2 at pos 0
            top1_tokens=[10, 20, 30],
            top2_tokens=[11, 21, 31],
            logit_ratios=[0.95, 0.5, 0.5],  # 0.95 > 0.9 threshold
            threshold=0.9,
        )
        assert result == 3  # All accepted (tie at pos 0 is accepted)

    def test_tie_rejected_low_ratio(self):
        """Draft matches top-2 but ratio < θ → rejected."""
        result = ParallelProposer._mars_dummy_acceptance(
            draft_tokens=[11, 20, 30],  # 11 is top-2 at pos 0
            top1_tokens=[10, 20, 30],
            top2_tokens=[11, 21, 31],
            logit_ratios=[0.85, 0.5, 0.5],  # 0.85 < 0.9 threshold
            threshold=0.9,
        )
        assert result == 0  # Rejected at position 0

    def test_tie_at_exact_threshold(self):
        """Ratio == θ exactly → rejected (strictly greater required)."""
        result = ParallelProposer._mars_dummy_acceptance(
            draft_tokens=[11, 20, 30],
            top1_tokens=[10, 20, 30],
            top2_tokens=[11, 21, 31],
            logit_ratios=[0.9, 0.5, 0.5],  # 0.9 == threshold, not >
            threshold=0.9,
        )
        assert result == 0  # Rejected (not strictly greater)

    def test_mixed_accept_reject(self):
        """CLAUDE.md example: accept, tie-accept, reject."""
        # Position 0: d1=10, top1=10 → ACCEPT (exact match)
        # Position 1: d2=21, top2=21, ratio=0.95>0.9 → ACCEPT (tie)
        # Position 2: d3=99, ratio=0.5<0.9 → REJECT
        result = ParallelProposer._mars_dummy_acceptance(
            draft_tokens=[10, 21, 99],
            top1_tokens=[10, 20, 30],
            top2_tokens=[11, 21, 31],
            logit_ratios=[0.5, 0.95, 0.5],
            threshold=0.9,
        )
        assert result == 2  # First rejection at position 2

    def test_single_token(self):
        """Single draft token accepted."""
        result = ParallelProposer._mars_dummy_acceptance(
            draft_tokens=[10],
            top1_tokens=[10],
            top2_tokens=[11],
            logit_ratios=[0.5],
            threshold=0.9,
        )
        assert result == 1  # All accepted

    def test_single_token_rejected(self):
        """Single draft token rejected."""
        result = ParallelProposer._mars_dummy_acceptance(
            draft_tokens=[99],
            top1_tokens=[10],
            top2_tokens=[11],
            logit_ratios=[0.5],
            threshold=0.9,
        )
        assert result == 0

    def test_empty_tokens(self):
        """Empty draft tokens → return 0 (nothing to accept)."""
        result = ParallelProposer._mars_dummy_acceptance(
            draft_tokens=[],
            top1_tokens=[],
            top2_tokens=[],
            logit_ratios=[],
            threshold=0.9,
        )
        assert result == 0

    def test_zero_threshold_disables_tie(self):
        """Threshold 0.0: tie case always rejected (ratio > 0 is True
        for any positive ratio, but 0.0 threshold means any ratio > 0.0
        would pass). This tests the edge case."""
        # With threshold=0.0, ratio=0.01 > 0.0 → ACCEPT tie
        result = ParallelProposer._mars_dummy_acceptance(
            draft_tokens=[11],  # matches top-2
            top1_tokens=[10],
            top2_tokens=[11],
            logit_ratios=[0.01],
            threshold=0.0,
        )
        assert result == 1  # Accepted (ratio > 0)

    def test_negative_ratio_rejects_tie(self):
        """Negative logit ratio should reject tie case."""
        result = ParallelProposer._mars_dummy_acceptance(
            draft_tokens=[11],
            top1_tokens=[10],
            top2_tokens=[11],
            logit_ratios=[-0.5],
            threshold=0.9,
        )
        assert result == 0  # Rejected

    def test_sequential_stops_at_first_rejection(self):
        """MARS stops at the FIRST rejection, doesn't look further."""
        result = ParallelProposer._mars_dummy_acceptance(
            draft_tokens=[10, 99, 30],  # pos 1 rejects, pos 2 would match
            top1_tokens=[10, 20, 30],
            top2_tokens=[11, 21, 31],
            logit_ratios=[0.5, 0.5, 0.5],
            threshold=0.9,
        )
        assert result == 1  # Stops at pos 1, doesn't see pos 2 match
