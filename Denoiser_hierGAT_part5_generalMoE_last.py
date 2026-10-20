# coding: utf-8
"""
Denoiser = Focal(hierGAT+PartGAT-P5) + General[0] FFN + General[1] MoE-FFN.

Copied/adapted from Denoiser_gat_skelinit_timestep_lowrank.py; original untouched.
"""
import math

import torch
import torch.nn as nn
from einops import rearrange
from torch import Tensor

from helpers import freeze_params, subsequent_mask
from transformer_layers import PositionalEncoding
from transformer_decoder_layer_moe import TransformerDecoderLayerMoE
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


class DenoiserHierGATPart5GeneralMoELast(nn.Module):
    """Focal unchanged; only last General layer uses FrameSemanticMoE-FFN."""

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
        use_part_gat: bool = True,
        part_gat_num_parts: int = 5,
        part_gat_channels: int = 64,
        part_gat_heads: int = 2,
        part_gat_bias_init: str = "zero",
        part_gate_init: float = 0.0,
        general_moe_num_experts: int = 4,
        general_moe_top_k: int = 1,
        general_moe_hidden_mult: int = 2,
        general_moe_load_balance_weight: float = 0.001,
        general_moe_z_loss_weight: float = 0.0001,
        moe_router_use_time: bool = True,
        moe_router_use_frame: bool = True,
        moe_router_pool: str = "frame",
        use_general_moe: bool = True,
        general_moe_all_layers: bool = False,
        use_general_motion_refine: bool = False,
        motion_refine_kernel: int = 5,
        motion_refine_dropout: float = 0.1,
        motion_refine_gate_init: float = 0.0,
        use_part_guided_bias: bool = False,
        part_guided_num_parts: int = 5,
        part_guided_bias_init: str = "zero",
        part_guided_gate_init: float = 0.0,
        use_hand_temporal_gat: bool = False,
        hand_temporal_only_last_block: bool = False,
        hand_temporal_last_n_blocks: int = None,
        hand_temporal_window: int = 5,
        hand_temporal_heads: int = 4,
        hand_temporal_gate_init: float = 0.0,
        hand_joint_start: int = 8,
        hand_joint_end: int = 50,
        use_stage_separated_focal: bool = False,
        joint_stage_layers: int = 2,
        part_stage_layers: int = 1,
        **kwargs,
    ):
        super().__init__()
        del kwargs

        if num_general_layers < 1:
            raise ValueError("need >=1 General layers")

        self.in_feature_size = trg_size
        self.out_feature_size = trg_size
        self.joints = int(trg_size // 3)
        self.num_focal_layers = num_focal_layers
        self.num_general_layers = num_general_layers
        self.use_general_moe = bool(use_general_moe)
        self.moe_router_use_time = bool(moe_router_use_time)
        self.moe_router_use_frame = bool(moe_router_use_frame)
        self.moe_router_pool = str(moe_router_pool)
        self.general_moe_all_layers = bool(general_moe_all_layers)
        self.use_general_motion_refine = bool(use_general_motion_refine)
        self.use_part_guided_bias = bool(use_part_guided_bias)
        self.use_hand_temporal_gat = bool(use_hand_temporal_gat)
        self.hand_temporal_only_last_block = bool(hand_temporal_only_last_block)
        # Optional override: enable HandTemporal on the last N Focal blocks.
        # None keeps legacy behavior (only_last_block / all blocks).
        self.hand_temporal_last_n_blocks = (
            None
            if hand_temporal_last_n_blocks is None
            else int(hand_temporal_last_n_blocks)
        )
        self.hand_temporal_window = int(hand_temporal_window)
        self.hand_temporal_gate_init = float(hand_temporal_gate_init)
        self.use_stage_separated_focal = bool(use_stage_separated_focal)
        self.joint_stage_layers = int(joint_stage_layers)
        self.part_stage_layers = int(part_stage_layers)

        if self.use_stage_separated_focal and self.use_hand_temporal_gat:
            raise ValueError(
                "use_stage_separated_focal and use_hand_temporal_gat are "
                "mutually exclusive controlled variants"
            )

        focal_base = dict(
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
            part_gat_num_parts=part_gat_num_parts,
            part_gat_channels=part_gat_channels,
            part_gat_heads=part_gat_heads,
            part_gat_bias_init=part_gat_bias_init,
            part_gate_init=part_gate_init,
            use_part_guided_bias=self.use_part_guided_bias,
            part_guided_num_parts=part_guided_num_parts,
            part_guided_bias_init=part_guided_bias_init,
            part_guided_gate_init=part_guided_gate_init,
            hand_temporal_window=hand_temporal_window,
            hand_temporal_heads=hand_temporal_heads,
            hand_temporal_gate_init=hand_temporal_gate_init,
            hand_joint_start=hand_joint_start,
            hand_joint_end=hand_joint_end,
        )

        self.focal_embed = nn.Conv2d(7, gcn_hidden_size, 1)
        if self.use_stage_separated_focal:
            if self.joint_stage_layers + self.part_stage_layers != self.num_focal_layers:
                raise ValueError(
                    f"joint_stage_layers({self.joint_stage_layers}) + "
                    f"part_stage_layers({self.part_stage_layers}) must equal "
                    f"num_focal_layers({self.num_focal_layers})"
                )
            stages = []
            for _ in range(self.joint_stage_layers):
                stages.append(
                    F_Stage(
                        gcn_hidden_size,
                        use_joint_gat=True,
                        use_part_gat=False,
                        use_hand_temporal_gat=False,
                        **focal_base,
                    )
                )
            for _ in range(self.part_stage_layers):
                stages.append(
                    F_Stage(
                        gcn_hidden_size,
                        use_joint_gat=False,
                        use_part_gat=True,
                        use_hand_temporal_gat=False,
                        **focal_base,
                    )
                )
            self.Focal = nn.ModuleList(stages)
        else:
            stages = []
            for i in range(self.num_focal_layers):
                use_ht = self._hand_temporal_enabled_at(i)
                stages.append(
                    F_Stage(
                        gcn_hidden_size,
                        use_joint_gat=True,
                        use_part_gat=use_part_gat,
                        use_hand_temporal_gat=use_ht,
                        **focal_base,
                    )
                )
            self.Focal = nn.ModuleList(stages)

        self.f2g_embed = nn.Linear(gcn_hidden_size * self.joints, hidden_size)
        self.pe = PositionalEncoding(hidden_size, mask_count=True)
        self.pos_drop = nn.Dropout(p=emb_dropout)
        self.emb_dropout = nn.Dropout(p=emb_dropout)

        general_layers = []
        for i in range(self.num_general_layers):
            # Default: MoE only on last General; optional: all / none.
            if not self.use_general_moe:
                use_moe = False
            else:
                use_moe = self.general_moe_all_layers or (
                    i == self.num_general_layers - 1
                )
            general_layers.append(
                TransformerDecoderLayerMoE(
                    size=hidden_size,
                    ff_size=ff_size,
                    num_heads=num_heads,
                    dropout=dropout,
                    decoder_trg_trg=decoder_trg_trg_,
                    use_moe_ffn=use_moe,
                    moe_num_experts=general_moe_num_experts,
                    moe_top_k=general_moe_top_k,
                    moe_hidden_mult=general_moe_hidden_mult,
                    moe_load_balance_weight=general_moe_load_balance_weight,
                    moe_z_loss_weight=general_moe_z_loss_weight,
                    moe_router_use_time=self.moe_router_use_time,
                    moe_router_use_frame=self.moe_router_use_frame,
                    moe_router_pool=self.moe_router_pool,
                    # MotionRefine only on MoE General layers
                    use_motion_refine=use_moe and self.use_general_motion_refine,
                    motion_refine_kernel=motion_refine_kernel,
                    motion_refine_dropout=motion_refine_dropout,
                    motion_refine_gate_init=motion_refine_gate_init,
                )
            )
        self.General = nn.ModuleList(general_layers)

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
            nn.Linear(hidden_size, trg_size, bias=False),
        )

        if freeze:
            freeze_params(self)

    def _hand_temporal_enabled_at(self, block_idx: int) -> bool:
        """Whether Focal[block_idx] should include HandTemporalGAT."""
        if not self.use_hand_temporal_gat:
            return False
        n = self.num_focal_layers
        if self.hand_temporal_last_n_blocks is not None:
            last_n = self.hand_temporal_last_n_blocks
            if last_n < 1 or last_n > n:
                raise ValueError(
                    f"hand_temporal_last_n_blocks={last_n} out of range "
                    f"for num_focal_layers={n}"
                )
            return block_idx >= n - last_n
        if self.hand_temporal_only_last_block:
            return block_idx == n - 1
        return True

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

        trg_embed = ID(trg_embed)
        trg_f = self.focal_embed(
            rearrange(trg_embed, "n t (v c) -> n c t v", v=self.joints)
        )

        moe_aux_loss = trg_f.new_zeros(())
        for stage in self.Focal:
            trg_f, stage_aux = stage(trg_f, condition, src_mask, time_embed=t_emb)
            moe_aux_loss = moe_aux_loss + stage_aux

        trg_g = self.emb_dropout(
            self.pe(self.f2g_embed(rearrange(trg_f, "n c t v -> n t (v c)")))
        )

        for idx, stage in enumerate(self.General):
            trg_g, _, stage_moe = stage(
                x=trg_g,
                memory=condition,
                src_mask=src_mask,
                trg_mask=sub_mask,
                padding_mask=padding_mask,
                time_embed=t_emb if stage.use_moe_ffn else None,
            )
            moe_aux_loss = moe_aux_loss + stage_moe
            # Mid projection only between General layers (skip when single General)
            if idx < self.num_general_layers - 1:
                trg_g = self.output_layer_mid(self.layer_norm_mid(trg_g))

        output = self.output_layer(trg_g)
        return output, trg_g, moe_aux_loss
