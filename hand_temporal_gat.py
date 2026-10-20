# coding: utf-8
"""Local temporal multi-head self-attention residual on hand joints only."""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class HandTemporalGAT(nn.Module):
    """
    Local temporal MHSA over hand joints (default indices [8, 50)).

    Input/output: [N, C, T, J]
    Hand slice [N, C, T, 42] -> [N*42, T, C] attention with window_size,
    then scatter delta back; non-hand joints stay 0.
    """

    def __init__(
        self,
        in_channels: int = 16,
        num_joints: int = 50,
        hand_joint_start: int = 8,
        hand_joint_end: int = 50,
        heads: int = 4,
        window_size: int = 5,
    ):
        super().__init__()
        if window_size < 1 or window_size % 2 == 0:
            raise ValueError(f"window_size must be odd >=1, got {window_size}")
        if in_channels % heads != 0:
            raise ValueError(
                f"in_channels ({in_channels}) must be divisible by heads ({heads})"
            )
        if not (0 <= hand_joint_start < hand_joint_end <= num_joints):
            raise ValueError(
                f"invalid hand range [{hand_joint_start}, {hand_joint_end}) "
                f"for num_joints={num_joints}"
            )

        self.in_channels = in_channels
        self.num_joints = num_joints
        self.hand_joint_start = hand_joint_start
        self.hand_joint_end = hand_joint_end
        self.num_hand_joints = hand_joint_end - hand_joint_start
        self.heads = heads
        self.head_dim = in_channels // heads
        self.window_size = window_size
        self.radius = window_size // 2

        self.norm = nn.LayerNorm(in_channels)
        self.q_proj = nn.Linear(in_channels, in_channels, bias=False)
        self.k_proj = nn.Linear(in_channels, in_channels, bias=False)
        self.v_proj = nn.Linear(in_channels, in_channels, bias=False)
        self.out_proj = nn.Linear(in_channels, in_channels, bias=False)
        # R^h_w zero-init; w indexes window slots
        self.rel_bias = nn.Parameter(torch.zeros(heads, window_size))

    def _boundary_mask(self, t: int, device: torch.device) -> torch.Tensor:
        """True where window slot falls outside [0, T). Shape [T, W]."""
        frames = torch.arange(t, device=device)[:, None]
        offsets = torch.arange(self.window_size, device=device)[None, :] - self.radius
        abs_pos = frames + offsets
        return (abs_pos < 0) | (abs_pos >= t)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.ndim == 4, f"expected [N,C,T,J], got {x.shape}"
        n, c, t, j = x.shape
        assert c == self.in_channels, f"expected C={self.in_channels}, got {c}"
        assert j == self.num_joints, f"expected J={self.num_joints}, got {j}"

        hand = x[..., self.hand_joint_start : self.hand_joint_end]
        # [N,C,T,H] -> [N*H, T, C]
        h = rearrange(hand, "n c t hj -> (n hj) t c")
        h = self.norm(h)

        nj = n * self.num_hand_joints
        q = self.q_proj(h).view(nj, t, self.heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(h).view(nj, t, self.heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(h).view(nj, t, self.heads, self.head_dim).transpose(1, 2)

        k_pad = F.pad(k, (0, 0, self.radius, self.radius))
        v_pad = F.pad(v, (0, 0, self.radius, self.radius))
        k_win = k_pad.unfold(dimension=2, size=self.window_size, step=1)
        v_win = v_pad.unfold(dimension=2, size=self.window_size, step=1)

        logits = torch.einsum("nhtd,nhtdw->nhtw", q, k_win) / math.sqrt(self.head_dim)
        logits = logits + self.rel_bias[None, :, None, :]
        invalid = self._boundary_mask(t, x.device)
        logits = logits.masked_fill(invalid[None, None, :, :], float("-inf"))
        attn = F.softmax(logits, dim=-1)
        attn = torch.nan_to_num(attn, nan=0.0)

        out = torch.einsum("nhtw,nhtdw->nhtd", attn, v_win)
        out = out.transpose(1, 2).reshape(nj, t, self.in_channels)
        out = self.out_proj(out)
        out = rearrange(
            out, "(n hj) t c -> n c t hj", n=n, hj=self.num_hand_joints
        )

        delta = x.new_zeros(n, c, t, j)
        delta[..., self.hand_joint_start : self.hand_joint_end] = out
        assert delta.shape == x.shape
        return delta
