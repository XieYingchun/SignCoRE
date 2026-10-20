# coding: utf-8
"""Transformer decoder layer with optional FrameSemanticMoE-FFN (copy of Sign-IDD layer)."""
import torch
import torch.nn as nn
from torch import Tensor

from transformer_layers import MultiHeadedAttention, PositionwiseFeedForward
from frame_semantic_moe_ffn import FrameSemanticMoEFFN
from frame_motion_refine import FrameMotionRefine


class TransformerDecoderLayerMoE(nn.Module):
    """
    Same as TransformerDecoderLayer, but FFN can be FrameSemanticMoE-FFN.
    When use_moe_ffn=True, forward also needs time_embed and returns moe_aux.
    Optional MotionRefine residual after MoE (zero-init gate).
    """

    def __init__(
        self,
        size: int = 0,
        ff_size: int = 0,
        num_heads: int = 0,
        dropout: float = 0.1,
        decoder_trg_trg: bool = True,
        use_moe_ffn: bool = False,
        moe_num_experts: int = 4,
        moe_top_k: int = 1,
        moe_hidden_mult: int = 2,
        moe_load_balance_weight: float = 0.001,
        moe_z_loss_weight: float = 0.0001,
        use_motion_refine: bool = False,
        motion_refine_kernel: int = 5,
        motion_refine_dropout: float = 0.1,
        motion_refine_gate_init: float = 0.0,
        moe_router_use_time: bool = True,
        moe_router_use_frame: bool = True,
        moe_router_pool: str = "frame",
    ):
        super().__init__()
        self.size = size
        self.use_moe_ffn = bool(use_moe_ffn)
        self.moe_router_use_time = bool(moe_router_use_time)
        self.moe_router_use_frame = bool(moe_router_use_frame)
        self.moe_router_pool = str(moe_router_pool)
        self.use_motion_refine = bool(use_motion_refine) and self.use_moe_ffn

        self.trg_trg_att = MultiHeadedAttention(num_heads, size, dropout=dropout)
        self.src_trg_att = MultiHeadedAttention(num_heads, size, dropout=dropout)

        if self.use_moe_ffn:
            self.feed_forward = FrameSemanticMoEFFN(
                dim=size,
                num_experts=moe_num_experts,
                top_k=moe_top_k,
                hidden_mult=moe_hidden_mult,
                dropout=dropout,
                load_balance_weight=moe_load_balance_weight,
                z_loss_weight=moe_z_loss_weight,
                router_use_time=self.moe_router_use_time,
                router_use_frame=self.moe_router_use_frame,
                router_pool=self.moe_router_pool,
            )
        else:
            self.feed_forward = PositionwiseFeedForward(size, ff_size=ff_size)

        if self.use_motion_refine:
            self.motion_refine = FrameMotionRefine(
                dim=size,
                kernel_size=motion_refine_kernel,
                dropout=motion_refine_dropout,
            )
            self.motion_refine_gate = nn.Parameter(
                torch.full((1,), float(motion_refine_gate_init))
            )
        else:
            self.motion_refine = None
            self.motion_refine_gate = None

        self.x_layer_norm = nn.LayerNorm(size, eps=1e-6)
        self.dec_layer_norm = nn.LayerNorm(size, eps=1e-6)
        self.dropout = nn.Dropout(dropout)
        self.decoder_trg_trg = decoder_trg_trg

    def forward(
        self,
        x: Tensor = None,
        memory: Tensor = None,
        src_mask: Tensor = None,
        trg_mask: Tensor = None,
        padding_mask: Tensor = None,
        time_embed: Tensor = None,
    ):
        h1 = self.x_layer_norm(x)
        if self.decoder_trg_trg:
            h1 = self.trg_trg_att(
                h1, h1, h1, mask=trg_mask, padding_mask=padding_mask
            )
        h1 = self.dropout(h1) + x

        h1_norm = self.dec_layer_norm(h1)
        h2 = self.src_trg_att(memory, memory, h1_norm, mask=src_mask)

        ffn_in = self.dropout(h2) + h1
        moe_aux = ffn_in.new_zeros(())
        if self.use_moe_ffn:
            assert time_embed is not None, "time_embed required for MoE-FFN"
            # In this codebase padding_mask True = valid (see batch.py /
            # MultiHeadedAttention masked_fill(~padding_mask, 0)).
            token_mask = None
            if padding_mask is not None:
                pm = padding_mask.bool()
                if pm.dim() == 4:
                    # [N,1,T,T] or [N,1,T,F] -> valid if any key/feature is True
                    token_mask = pm[:, 0].any(dim=-1)
                elif pm.dim() == 3:
                    token_mask = pm.squeeze(1)
                elif pm.dim() == 2:
                    token_mask = pm
                else:
                    raise ValueError(
                        f"unsupported padding_mask shape for MoE: {tuple(pm.shape)}"
                    )
            o, moe_aux = self.feed_forward(
                ffn_in, time_embed=time_embed, token_mask=token_mask
            )
            if self.use_motion_refine:
                motion = self.motion_refine(o, token_mask=token_mask)
                o = o + self.motion_refine_gate * motion
        else:
            o = self.feed_forward(ffn_in)

        return o, h2, moe_aux
