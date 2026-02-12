"""Token Channel - Reserved interface for NPU cross-device communication.

This module provides a stub interface for future cross-device token
communication in Parallel Speculative Decoding. When the draft model
and target model run on different devices (e.g., NPU + GPU), this
channel handles lightweight O(B*kappa) token-level communication
instead of O(B*K*D) hidden state transfers.

Communication complexity:
    - Token Channel: O(B * kappa) where B = batch_size, kappa = top_k
    - Traditional: O(B * K * D) where K = num_layers, D = hidden_dim

Currently NOT implemented. All methods raise NotImplementedError.
"""

import torch


class TokenChannel:
    """Lightweight token communication channel for cross-device scenarios.

    Reserved interface for future NPU/multi-device deployment where
    the draft model runs on a separate device from the target model.
    """

    def send_top_k_candidates(
        self,
        token_ids: torch.Tensor,
        probabilities: torch.Tensor,
        batch_size: int,
        top_k: int = 1,
    ) -> None:
        """Send top-k candidate tokens to the draft model device.

        Args:
            token_ids: Token IDs of shape [batch_size, top_k].
            probabilities: Token probabilities of shape [batch_size, top_k].
            batch_size: Number of requests in the batch.
            top_k: Number of candidates per position.

        Raises:
            NotImplementedError: Always (stub interface).
        """
        raise NotImplementedError(
            "TokenChannel is a reserved interface for future NPU "
            "cross-device communication. Not yet implemented."
        )

    def recv_top_k_candidates(
        self,
        batch_size: int,
        top_k: int = 1,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Receive top-k candidate tokens from the target model device.

        Args:
            batch_size: Number of requests in the batch.
            top_k: Number of candidates per position.

        Returns:
            Tuple of (token_ids, probabilities).

        Raises:
            NotImplementedError: Always (stub interface).
        """
        raise NotImplementedError(
            "TokenChannel is a reserved interface for future NPU "
            "cross-device communication. Not yet implemented."
        )

    def send_speculative_tokens(
        self,
        spec_token_ids: torch.Tensor,
        batch_size: int,
        num_speculative_tokens: int = 1,
    ) -> None:
        """Send speculative (draft) tokens to the target model device.

        Args:
            spec_token_ids: Draft token IDs of shape
                [batch_size, num_speculative_tokens].
            batch_size: Number of requests in the batch.
            num_speculative_tokens: Number of draft tokens per request.

        Raises:
            NotImplementedError: Always (stub interface).
        """
        raise NotImplementedError(
            "TokenChannel is a reserved interface for future NPU "
            "cross-device communication. Not yet implemented."
        )

    def recv_speculative_tokens(
        self,
        batch_size: int,
        num_speculative_tokens: int = 1,
    ) -> torch.Tensor:
        """Receive speculative (draft) tokens from the draft model device.

        Args:
            batch_size: Number of requests in the batch.
            num_speculative_tokens: Number of draft tokens per request.

        Returns:
            Draft token IDs tensor.

        Raises:
            NotImplementedError: Always (stub interface).
        """
        raise NotImplementedError(
            "TokenChannel is a reserved interface for future NPU "
            "cross-device communication. Not yet implemented."
        )
