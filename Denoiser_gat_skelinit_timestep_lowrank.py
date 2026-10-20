# coding: utf-8
"""Denoiser = Focal(IntraFrameGAT+TCN) x N + General Transformer (no SCG)."""
import math

import torch
import torch.nn as nn
from einops import rearrange
from torch import Tensor

from helpers import freeze_params, subsequent_mask
from transformer_layers import PositionalEncoding, TransformerDecoderLayer
from f_stage_gat_skelinit_timestep_lowrank import F_Stage
from ID import ID


class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


class DenoiserGatSkelinitTimestepLowrank(nn.Module):
    """FGDM-style Focal GAT + General Transformer denoiser (ID inside)."""

    def __init__(
        self,
        num_focal_layers: int = 3,
        num_general_layers: int = 2,
        num_heads: int = 4,
        hidden_size: int = 512,
        ff_size: int = 2048,
        dropout: float = 0.1,
        emb_dropout: float = 0.1,
        freeze: bool = False,
        trg_size: int = 150,
        decoder_trg_trg_: bool = True,
        gcn_hidden_size: int = 16,
        gat_channels: int = 64,
        gat_heads: int = 4,
        head_dim: int = 16,
        gat_use_condition: bool = False,
        gat_graph_bias_init: str = "skeleton",
        gat_dense: bool = True,
        use_skeleton_hard_mask: bool = False,
        use_timestep_lowrank_bias: bool = True,
        use_part_gat: bool = False,
        part_gat_num_parts: int = 6,
        part_gat_channels: int = 64,
        part_gat_heads: int = 2,
        part_gat_bias_init: str = "zero",
        part_gate_init: float = 0.0,
        **kwargs,
    ):
        super().__init__()
        del kwargs

        self.in_feature_size = trg_size
        self.out_feature_size = trg_size
        self.joints = int(trg_size // 3)
        self.num_focal_layers = num_focal_layers
        self.num_general_layers = num_general_layers

        focal_kwargs = dict(
            num_joints=self.joints,
            condition_dim=hidden_size,
            gat_channels=gat_channels,
            gat_heads=gat_heads,
            head_dim=head_dim,
            gat_use_condition=gat_use_condition,
            gat_graph_bias_init=gat_graph_bias_init,
            gat_dense=gat_dense,
            use_skeleton_hard_mask=use_skeleton_hard_mask,
            dropout=dropout,
            use_timestep_lowrank_bias=use_timestep_lowrank_bias,
            use_part_gat=use_part_gat,
            part_gat_num_parts=part_gat_num_parts,
            part_gat_channels=part_gat_channels,
            part_gat_heads=part_gat_heads,
            part_gat_bias_init=part_gat_bias_init,
            part_gate_init=part_gate_init,
        )

        self.focal_embed = nn.Conv2d(7, gcn_hidden_size, 1)
        self.Focal = nn.ModuleList([
            F_Stage(gcn_hidden_size, **focal_kwargs)
            for _ in range(self.num_focal_layers)
        ])

        self.f2g_embed = nn.Linear(gcn_hidden_size * self.joints, hidden_size)
        self.pe = PositionalEncoding(hidden_size, mask_count=True)
        self.pos_drop = nn.Dropout(p=emb_dropout)
        self.emb_dropout = nn.Dropout(p=emb_dropout)
        self.General = nn.ModuleList([
            TransformerDecoderLayer(
                size=hidden_size, ff_size=ff_size, num_heads=num_heads,
                dropout=dropout, decoder_trg_trg=decoder_trg_trg_
            ) for _ in range(self.num_general_layers)
        ])

        self.layer_norm_mid = nn.LayerNorm(hidden_size, eps=1e-6)
        self.output_layer_mid = nn.Linear(hidden_size, hidden_size, bias=False)
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(hidden_size),
            nn.Linear(hidden_size, hidden_size * 2),
            nn.GELU(),
            nn.Linear(hidden_size * 2, hidden_size),
        )
        self.output_layer = nn.Sequential(
            nn.LayerNorm(hidden_size, eps=1e-6),
            nn.Linear(hidden_size, trg_size, bias=False)
        )

        if freeze:
            freeze_params(self)

    def forward(
        self,
        t,
        trg_embed: Tensor = None,
        encoder_output: Tensor = None,
        src_mask: Tensor = None,
        trg_mask: Tensor = None,
        **kwargs,
    ):
        del kwargs
        assert trg_mask is not None, "trg_mask required for Transformer"

        t_emb = self.time_mlp(t)
        time_embed = t_emb[:, None, :].repeat(1, encoder_output.shape[1], 1)
        condition = self.pos_drop(encoder_output + time_embed)

        sub_mask = subsequent_mask(trg_embed.size(1)).type_as(trg_mask)
        padding_mask = trg_mask

        # ID inside denoiser (FGDM style); do NOT also ID in ACD shell
        trg_embed = ID(trg_embed)
        trg_f = self.focal_embed(rearrange(trg_embed, "n t (v c) -> n c t v", v=self.joints))

        moe_aux_loss = trg_f.new_zeros(())
        for stage in self.Focal:
            trg_f, stage_aux = stage(trg_f, condition, src_mask, time_embed=t_emb)
            moe_aux_loss = moe_aux_loss + stage_aux

        trg_g = self.emb_dropout(self.pe(self.f2g_embed(
            rearrange(trg_f, "n c t v -> n t (v c)"))))

        for idx, stage in enumerate(self.General):
            trg_g, _ = stage(
                x=trg_g, memory=condition,
                src_mask=src_mask, trg_mask=sub_mask,
                padding_mask=padding_mask,
            )
            if idx < 1:
                trg_g = self.output_layer_mid(self.layer_norm_mid(trg_g))

        output = self.output_layer(trg_g)
        return output, trg_g, moe_aux_loss
