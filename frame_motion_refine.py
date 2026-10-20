# coding: utf-8
"""Lightweight temporal motion refinement residual for General-stage tokens."""
import torch
import torch.nn as nn


class FrameMotionRefine(nn.Module):
    """
    Local temporal Conv1d residual branch on [N, T, D] tokens.

    Used after General MoE-FFN to smooth frame-level jitter without a new loss.
    """

    def __init__(
        self,
        dim: int = 512,
        kernel_size: int = 5,
        dropout: float = 0.1,
    ):
        super().__init__()
        if kernel_size < 1 or kernel_size % 2 == 0:
            raise ValueError(f"kernel_size must be odd positive, got {kernel_size}")
        padding = kernel_size // 2
        self.layer_norm = nn.LayerNorm(dim, eps=1e-6)
        self.temporal = nn.Sequential(
            nn.Conv1d(dim, dim, kernel_size=kernel_size, padding=padding, groups=1),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(dim, dim, kernel_size=1),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor, token_mask: torch.Tensor = None) -> torch.Tensor:
        """
        x: [N, T, D]
        token_mask: [N, T] True=valid frame (optional)
        """
        assert x.ndim == 3, f"expected [N,T,D], got {x.shape}"
        h = self.layer_norm(x)
        h = h.transpose(1, 2)  # [N,D,T]
        h = self.temporal(h)
        h = h.transpose(1, 2)  # [N,T,D]
        if token_mask is not None:
            h = h * token_mask.bool().unsqueeze(-1).to(dtype=h.dtype)
        return h
