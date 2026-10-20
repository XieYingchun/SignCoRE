# coding: utf-8
"""IntraFrameGAT-noCond + skelinit + timestep low-rank bias (ported from FGDM)."""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from skeleton_graph_bias_gat import build_skeleton_graph_bias
from body_part_partition import (
    build_joint_to_part,
    build_structural_part_pair_bias,
)


class IntraFrameGATNoCondTimestepLowRankBias(nn.Module):
    """
    Intra-frame GAT (no gloss gate) with skeleton-init B^h plus
    timestep-conditioned low-rank bias, and optional part-guided bias:

        E^h_ij(τ) = QK/sqrt(d) + B^h_ij + u_i^h(τ)+v_j^h(τ)
                    + g * P^h_{part(i), part(j)}

    Call reset_special_parameters() after initialize_model(), which otherwise
    zeros any parameter whose name contains "bias".
    """

    def __init__(
        self,
        in_channels: int = 16,
        num_joints: int = 50,
        condition_dim: int = 512,
        gat_channels: int = 64,
        gat_heads: int = 4,
        head_dim: int = 16,
        graph_bias_init: str = "skeleton",
        skeleton_layout: str = "whole",
        use_timestep_lowrank_bias: bool = True,
        use_part_guided_bias: bool = False,
        part_guided_num_parts: int = 5,
        part_guided_bias_init: str = "zero",
        part_guided_gate_init: float = 0.0,
    ):
        super().__init__()
        if gat_heads * head_dim != gat_channels:
            raise ValueError(
                f"gat_heads ({gat_heads}) * head_dim ({head_dim}) "
                f"must equal gat_channels ({gat_channels})"
            )

        self.in_channels = in_channels
        self.num_joints = num_joints
        self.gat_channels = gat_channels
        self.gat_heads = gat_heads
        self.head_dim = head_dim
        self.condition_dim = condition_dim
        self.graph_bias_init = graph_bias_init
        self.skeleton_layout = skeleton_layout
        self.use_timestep_lowrank_bias = bool(use_timestep_lowrank_bias)
        self.use_part_guided_bias = bool(use_part_guided_bias)
        self.part_guided_num_parts = int(part_guided_num_parts)
        self.part_guided_bias_init = str(part_guided_bias_init)
        self.part_guided_gate_init = float(part_guided_gate_init)

        self.in_proj = nn.Conv2d(in_channels, gat_channels, kernel_size=1)
        self.out_proj = nn.Conv2d(gat_channels, in_channels, kernel_size=1)
        self.norm = nn.LayerNorm(gat_channels)

        self.q_proj = nn.Linear(gat_channels, gat_channels, bias=False)
        self.k_proj = nn.Linear(gat_channels, gat_channels, bias=False)
        self.v_proj = nn.Linear(gat_channels, gat_channels, bias=False)

        self.graph_bias = nn.Parameter(
            torch.zeros(gat_heads, num_joints, num_joints)
        )

        if self.use_timestep_lowrank_bias:
            out_dim = 2 * gat_heads * num_joints
            self.time_bias_hidden = nn.Linear(condition_dim, condition_dim)
            self.time_bias_act = nn.SiLU()
            self.time_bias_out = nn.Linear(condition_dim, out_dim)

        if self.use_part_guided_bias:
            j2p = build_joint_to_part(
                num_joints=num_joints, num_parts=self.part_guided_num_parts
            )
            self.register_buffer("joint_to_part", j2p, persistent=False)
            self.part_pair_bias = nn.Parameter(
                torch.zeros(
                    gat_heads, self.part_guided_num_parts, self.part_guided_num_parts
                )
            )
            self.part_guided_gate = nn.Parameter(
                torch.full((1,), self.part_guided_gate_init)
            )

        self.reset_special_parameters()

    def reset_special_parameters(self):
        """Restore initialization that initialize_model() must not overwrite."""
        if self.graph_bias_init in ("zeros", "zero"):
            init_bias = torch.zeros(
                self.gat_heads,
                self.num_joints,
                self.num_joints,
                device=self.graph_bias.device,
                dtype=self.graph_bias.dtype,
            )
        elif self.graph_bias_init == "skeleton":
            init_bias = build_skeleton_graph_bias(
                self.num_joints,
                self.gat_heads,
                layout=self.skeleton_layout,
            ).to(
                device=self.graph_bias.device,
                dtype=self.graph_bias.dtype,
            )
        else:
            raise ValueError(
                f"unsupported graph_bias_init={self.graph_bias_init!r}"
            )

        with torch.no_grad():
            self.graph_bias.copy_(init_bias)

        if self.use_timestep_lowrank_bias:
            # Must be non-zero so the timestep branch has usable features.
            nn.init.xavier_uniform_(self.time_bias_hidden.weight)
            nn.init.zeros_(self.time_bias_hidden.bias)
            # Keep zero-init so training starts equivalent to no timestep bias.
            nn.init.zeros_(self.time_bias_out.weight)
            nn.init.zeros_(self.time_bias_out.bias)

        if self.use_part_guided_bias:
            init_name = self.part_guided_bias_init.lower()
            if init_name in ("zeros", "zero"):
                part_init = torch.zeros_like(self.part_pair_bias)
            elif init_name == "structural":
                part_init = build_structural_part_pair_bias(
                    self.gat_heads, self.part_guided_num_parts
                ).to(
                    device=self.part_pair_bias.device,
                    dtype=self.part_pair_bias.dtype,
                )
            else:
                raise ValueError(
                    f"unsupported part_guided_bias_init={self.part_guided_bias_init!r}"
                )
            with torch.no_grad():
                self.part_pair_bias.copy_(part_init)
                self.part_guided_gate.fill_(self.part_guided_gate_init)

    def _timestep_lowrank_bias(self, time_embed: torch.Tensor, t: int) -> torch.Tensor:
        assert time_embed.ndim == 2, f"expected time_embed [N,D], got {time_embed.shape}"
        n = time_embed.shape[0]
        h, j = self.gat_heads, self.num_joints

        uv = self.time_bias_out(self.time_bias_act(self.time_bias_hidden(time_embed)))
        u, v = uv.chunk(2, dim=-1)
        u = u.view(n, h, j)
        v = v.view(n, h, j)

        u = u.unsqueeze(1).expand(-1, t, -1, -1).reshape(n * t, h, j)
        v = v.unsqueeze(1).expand(-1, t, -1, -1).reshape(n * t, h, j)
        return u.unsqueeze(-1) + v.unsqueeze(-2)

    def _part_guided_joint_bias(self) -> torch.Tensor:
        """Expand [H,P,P] part-pair bias to [H,J,J] via joint_to_part."""
        pi = self.joint_to_part  # [J]
        # [H,P,P] -> [H,J,P] -> [H,J,J]
        return self.part_pair_bias[:, pi][:, :, pi]

    def forward(self, x, condition=None, mask=None, time_embed=None):
        del condition, mask
        assert x.ndim == 4, f"expected x [N,C,T,J], got {x.shape}"
        n, c, t, j = x.shape
        assert c == self.in_channels, f"expected C={self.in_channels}, got {c}"
        assert j == self.num_joints, f"expected J={self.num_joints}, got {j}"

        h = self.in_proj(x)
        h = rearrange(h, "n c t j -> (n t) j c")
        h = self.norm(h)

        nt = n * t
        q = self.q_proj(h).view(nt, j, self.gat_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(h).view(nt, j, self.gat_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(h).view(nt, j, self.gat_heads, self.head_dim).transpose(1, 2)

        logits = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        logits = logits + self.graph_bias.unsqueeze(0)
        if self.use_timestep_lowrank_bias:
            assert time_embed is not None, "time_embed required for timestep low-rank bias"
            assert time_embed.shape[0] == n, "time_embed batch must match x"
            logits = logits + self._timestep_lowrank_bias(time_embed, t)
        if self.use_part_guided_bias:
            logits = logits + self.part_guided_gate * self._part_guided_joint_bias().unsqueeze(0)

        attn = F.softmax(logits, dim=-1)
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).reshape(nt, j, self.gat_channels)
        out = rearrange(out, "(n t) j c -> n c t j", n=n, t=t)

        out = self.out_proj(out)
        assert out.shape == x.shape, f"output {out.shape} != input {x.shape}"
        return out
