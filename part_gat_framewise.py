# coding: utf-8
"""Frame-wise Part-level GAT (dense on P body parts per frame)."""
import math
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from body_part_partition import build_joint_to_part, build_joints_of_part


class PartPool(nn.Module):
    """Mean-pool joints within each body part: [N,C,T,J] -> [N,C,T,P]."""

    def __init__(self, num_joints: int = 50, num_parts: int = 6):
        super().__init__()
        self.num_joints = num_joints
        self.num_parts = num_parts
        joints_of_part = build_joints_of_part(num_joints, num_parts)
        self.joints_of_part: List[List[int]] = joints_of_part
        # register indices as buffers for device moves
        for p, idx in enumerate(joints_of_part):
            self.register_buffer(f"part_idx_{p}", torch.tensor(idx, dtype=torch.long))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.ndim == 4, f"expected [N,C,T,J], got {x.shape}"
        n, c, t, j = x.shape
        assert j == self.num_joints, f"expected J={self.num_joints}, got {j}"

        parts = []
        for p in range(self.num_parts):
            idx = getattr(self, f"part_idx_{p}")
            parts.append(x.index_select(dim=-1, index=idx).mean(dim=-1))
        return torch.stack(parts, dim=-1)  # [N,C,T,P]


class PartBroadcast(nn.Module):
    """Broadcast part features back to joints: [N,C,T,P] -> [N,C,T,J]."""

    def __init__(self, num_joints: int = 50, num_parts: int = 6):
        super().__init__()
        self.num_joints = num_joints
        self.num_parts = num_parts
        self.register_buffer(
            "joint_to_part", build_joint_to_part(num_joints, num_parts)
        )

    def forward(self, p: torch.Tensor) -> torch.Tensor:
        assert p.ndim == 4, f"expected [N,C,T,P], got {p.shape}"
        assert p.shape[-1] == self.num_parts, (
            f"expected P={self.num_parts}, got {p.shape[-1]}"
        )
        # advanced index on last dim: [N,C,T,J]
        return p.index_select(dim=-1, index=self.joint_to_part)


class FramewisePartGAT(nn.Module):
    """
    Dense GAT over P body parts inside each frame.

        E^h_ab = Q_a K_b^T / sqrt(d) + B^h_ab

    No skeleton mask / condition gate / gloss / MoE / temporal attention.
    """

    def __init__(
        self,
        in_channels: int = 16,
        num_parts: int = 6,
        gat_channels: int = 64,
        gat_heads: int = 2,
        bias_init: str = "zero",
    ):
        super().__init__()
        if gat_channels % gat_heads != 0:
            raise ValueError(
                f"gat_channels ({gat_channels}) must be divisible by "
                f"gat_heads ({gat_heads})"
            )
        head_dim = gat_channels // gat_heads

        self.in_channels = in_channels
        self.num_parts = num_parts
        self.gat_channels = gat_channels
        self.gat_heads = gat_heads
        self.head_dim = head_dim

        self.in_proj = nn.Conv2d(in_channels, gat_channels, kernel_size=1)
        self.out_proj = nn.Conv2d(gat_channels, in_channels, kernel_size=1)
        self.norm = nn.LayerNorm(gat_channels)

        self.q_proj = nn.Linear(gat_channels, gat_channels, bias=False)
        self.k_proj = nn.Linear(gat_channels, gat_channels, bias=False)
        self.v_proj = nn.Linear(gat_channels, gat_channels, bias=False)

        if bias_init in ("zero", "zeros"):
            init_bias = torch.zeros(gat_heads, num_parts, num_parts)
        else:
            raise ValueError(
                f"part_gat_bias_init must be 'zero'/'zeros', got {bias_init!r}"
            )
        self.graph_bias = nn.Parameter(init_bias)

    def forward(self, p: torch.Tensor) -> torch.Tensor:
        assert p.ndim == 4, f"expected [N,C,T,P], got {p.shape}"
        n, c, t, num_p = p.shape
        assert c == self.in_channels, f"expected C={self.in_channels}, got {c}"
        assert num_p == self.num_parts, f"expected P={self.num_parts}, got {num_p}"

        h = self.in_proj(p)  # [N,Cg,T,P]
        h = h.permute(0, 2, 3, 1).contiguous()  # [N,T,P,Cg]
        h = h.view(n * t, num_p, self.gat_channels)
        h = self.norm(h)

        nt = n * t
        q = self.q_proj(h).view(nt, num_p, self.gat_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(h).view(nt, num_p, self.gat_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(h).view(nt, num_p, self.gat_heads, self.head_dim).transpose(1, 2)

        logits = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        logits = logits + self.graph_bias.unsqueeze(0)
        attn = F.softmax(logits, dim=-1)
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).reshape(nt, num_p, self.gat_channels)
        out = out.view(n, t, num_p, self.gat_channels).permute(0, 3, 1, 2).contiguous()
        out = self.out_proj(out)

        assert out.shape == p.shape, f"PartGAT out {out.shape} != in {p.shape}"
        return out
