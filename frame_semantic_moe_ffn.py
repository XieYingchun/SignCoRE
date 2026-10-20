# coding: utf-8
"""Frame-wise semantic MoE-FFN for General-stage tokens [N,T,C]."""
import torch
import torch.nn as nn
import torch.nn.functional as F


class FrameSemanticMoEFFN(nn.Module):
    """
    Replace Positionwise FFN with top-k MoE over frame tokens.

    Router input is a subset of frame token and timestep embedding (same dim).
    Experts always see the frame token.
    Returns: y [N,T,C], aux_loss (load-balance + z-loss, already weighted).
    """

    def __init__(
        self,
        dim: int = 512,
        num_experts: int = 4,
        top_k: int = 1,
        hidden_mult: int = 2,
        dropout: float = 0.1,
        load_balance_weight: float = 0.001,
        z_loss_weight: float = 0.0001,
        router_use_time: bool = True,
        router_use_frame: bool = True,
        router_pool: str = "frame",
    ):
        super().__init__()
        if top_k < 1 or top_k > num_experts:
            raise ValueError(f"top_k={top_k} invalid for num_experts={num_experts}")

        self.dim = dim
        self.num_experts = num_experts
        self.top_k = top_k
        self.load_balance_weight = float(load_balance_weight)
        self.z_loss_weight = float(z_loss_weight)
        self.router_use_time = bool(router_use_time)
        self.router_use_frame = bool(router_use_frame)
        self.router_pool = str(router_pool)
        if self.router_pool not in ("frame", "sequence"):
            raise ValueError(
                f"router_pool must be 'frame' or 'sequence', got {self.router_pool}"
            )
        if not self.router_use_time and not self.router_use_frame:
            raise ValueError("router needs frame features, timestep embedding, or both")

        self.layer_norm = nn.LayerNorm(dim, eps=1e-6)
        self.router = nn.Linear(dim, num_experts)
        hidden = dim * hidden_mult
        self.experts = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(dim, hidden),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden, dim),
                    nn.Dropout(dropout),
                )
                for _ in range(num_experts)
            ]
        )

    def forward(
        self,
        x: torch.Tensor,
        time_embed: torch.Tensor,
        token_mask: torch.Tensor = None,
    ):
        """
        x: [N, T, C]  (residual stream into FFN, same as PositionwiseFFN input)
        time_embed: [N, C] diffusion timestep embedding; required when router_use_time is True
        token_mask: [N, T] True=valid frame, False=padding (optional)
        """
        assert x.ndim == 3, f"expected [N,T,C], got {x.shape}"
        n, t, c = x.shape

        residual = x
        x_norm = self.layer_norm(x)
        if self.router_use_time:
            assert time_embed is not None and time_embed.ndim == 2, (
                f"expected time_embed [N,C], got {None if time_embed is None else time_embed.shape}"
            )
            assert time_embed.shape[0] == n and time_embed.shape[1] == c
            t_exp = time_embed[:, None, :].expand(-1, t, -1)
        else:
            t_exp = None
        if self.router_use_frame:
            frame_in = x_norm
            if self.router_pool == "sequence":
                frame_in = self._sequence_mean(x_norm, token_mask)
            router_in = frame_in if t_exp is None else frame_in + t_exp
        else:
            router_in = t_exp

        logits = self.router(router_in)  # [N,T,E]
        probs = F.softmax(logits, dim=-1)
        topk_probs, topk_idx = torch.topk(probs, self.top_k, dim=-1)
        # top-1: keep raw softmax so task loss can train the router.
        # top-k>1: renormalize selected experts to sum to 1.
        if self.top_k > 1:
            topk_probs = topk_probs / (topk_probs.sum(dim=-1, keepdim=True) + 1e-9)

        gates = torch.zeros_like(probs)
        gates.scatter_(-1, topk_idx, topk_probs)

        # expert_outs: [N,T,E,C]
        expert_outs = torch.stack([expert(x_norm) for expert in self.experts], dim=2)
        y = (gates.unsqueeze(-1) * expert_outs).sum(dim=2)
        y = y + residual

        aux = self.load_balance_weight * self._load_balance_loss(
            probs, gates, token_mask
        )
        aux = aux + self.z_loss_weight * self._z_loss(logits, token_mask)
        return y, aux

    def _sequence_mean(
        self, x_norm: torch.Tensor, token_mask: torch.Tensor = None
    ) -> torch.Tensor:
        """Masked mean over time, broadcast back to every frame. [N,T,C]."""
        if token_mask is None:
            pooled = x_norm.mean(dim=1, keepdim=True)
        else:
            valid = token_mask.bool().to(dtype=x_norm.dtype).unsqueeze(-1)
            denom = valid.sum(dim=1, keepdim=True).clamp_min(1.0)
            pooled = (x_norm * valid).sum(dim=1, keepdim=True) / denom
        return pooled.expand_as(x_norm)

    def _load_balance_loss(
        self,
        probs: torch.Tensor,
        gates: torch.Tensor,
        token_mask: torch.Tensor = None,
    ) -> torch.Tensor:
        # Switch-Transformer style; optionally only over valid frames.
        if token_mask is None:
            importance = probs.mean(dim=(0, 1))  # [E]
            load = (gates > 0).float().mean(dim=(0, 1))  # [E]
        else:
            valid = token_mask.bool().unsqueeze(-1).float()  # [N,T,1]
            denom = valid.sum().clamp_min(1.0)
            importance = (probs * valid).sum(dim=(0, 1)) / denom
            load = ((gates > 0).float() * valid).sum(dim=(0, 1)) / denom
        return self.num_experts * torch.sum(importance * load)

    def _z_loss(
        self, logits: torch.Tensor, token_mask: torch.Tensor = None
    ) -> torch.Tensor:
        # Encourage finite router logits: mean(square(logsumexp))
        z = torch.square(torch.logsumexp(logits, dim=-1))  # [N,T]
        if token_mask is None:
            return z.mean()
        valid = token_mask.bool().float()
        return (z * valid).sum() / valid.sum().clamp_min(1.0)
