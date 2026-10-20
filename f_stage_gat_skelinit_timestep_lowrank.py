# coding: utf-8
"""F_Stage = IntraFrameGAT(skelinit+timestep lowrank) + optional PartGAT + optional HandTemporalGAT + TCN."""
import torch
import torch.nn as nn

from intra_frame_gat_skelinit_timestep_lowrank import IntraFrameGATNoCondTimestepLowRankBias
from part_gat_framewise import FramewisePartGAT, PartBroadcast, PartPool
from hand_temporal_gat import HandTemporalGAT


class F_Stage(nn.Module):
    """Focal stage: optional Joint GAT / Part GAT / HandTemporal residual + TCN."""

    def __init__(
        self,
        in_channels,
        Temp_kernel_size=9,
        stride=1,
        dropout=0,
        layout="whole",
        strategy="spatial",
        use_A="True",
        residual=True,
        num_joints=50,
        condition_dim=512,
        gat_channels=64,
        gat_heads=4,
        head_dim=16,
        gat_use_condition=True,
        gat_graph_bias_init="zeros",
        gat_dense=True,
        use_skeleton_hard_mask=False,
        gat_use_fixed_skel_prior=False,
        gat_skel_prior_learnable_scale=True,
        gat_skel_prior_per_head_scale=True,
        use_temporal_gat=False,
        max_temporal_len=300,
        use_joint_moe=False,
        joint_moe_experts=4,
        joint_moe_topk=1,
        joint_moe_hidden_mult=2,
        use_joint_gat=True,
        use_part_gat=False,
        part_gat_num_parts=6,
        part_gat_channels=64,
        part_gat_heads=2,
        part_gat_bias_init="zero",
        part_gate_init=0.0,
        use_timestep_lowrank_bias=True,
        use_part_guided_bias=False,
        part_guided_num_parts=5,
        part_guided_bias_init="zero",
        part_guided_gate_init=0.0,
        use_hand_temporal_gat=False,
        hand_temporal_window=5,
        hand_temporal_heads=4,
        hand_temporal_gate_init=0.0,
        hand_joint_start=8,
        hand_joint_end=50,
    ):
        super().__init__()
        del layout, strategy, use_A, residual
        del gat_use_condition, gat_use_fixed_skel_prior
        del gat_skel_prior_learnable_scale, gat_skel_prior_per_head_scale
        del use_temporal_gat, max_temporal_len, use_joint_moe
        del joint_moe_experts, joint_moe_topk, joint_moe_hidden_mult

        if gat_dense is not True:
            raise ValueError("only dense J×J GAT is supported (gat_dense=True)")
        if use_skeleton_hard_mask:
            raise ValueError(
                "use_skeleton_hard_mask=True is not supported in this ablation"
            )

        self.use_joint_gat = bool(use_joint_gat)
        self.use_part_gat = bool(use_part_gat)
        self.use_hand_temporal_gat = bool(use_hand_temporal_gat)
        if not (
            self.use_joint_gat or self.use_part_gat or self.use_hand_temporal_gat
        ):
            raise ValueError(
                "F_Stage needs at least one of JointGAT / PartGAT / HandTemporalGAT"
            )

        bias_init = gat_graph_bias_init
        if bias_init == "zero":
            bias_init = "zeros"

        if self.use_joint_gat:
            self.gat = IntraFrameGATNoCondTimestepLowRankBias(
                in_channels=in_channels,
                num_joints=num_joints,
                condition_dim=condition_dim,
                gat_channels=gat_channels,
                gat_heads=gat_heads,
                head_dim=head_dim,
                graph_bias_init=bias_init,
                use_timestep_lowrank_bias=use_timestep_lowrank_bias,
                use_part_guided_bias=use_part_guided_bias,
                part_guided_num_parts=part_guided_num_parts,
                part_guided_bias_init=part_guided_bias_init,
                part_guided_gate_init=part_guided_gate_init,
            )
        else:
            self.gat = None

        if self.use_part_gat:
            self.part_pool = PartPool(
                num_joints=num_joints, num_parts=part_gat_num_parts
            )
            self.part_gat = FramewisePartGAT(
                in_channels=in_channels,
                num_parts=part_gat_num_parts,
                gat_channels=part_gat_channels,
                gat_heads=part_gat_heads,
                bias_init=part_gat_bias_init,
            )
            self.part_broadcast = PartBroadcast(
                num_joints=num_joints, num_parts=part_gat_num_parts
            )
            self.part_gate = nn.Parameter(
                torch.tensor([float(part_gate_init)], dtype=torch.float32)
            )

        if self.use_hand_temporal_gat:
            self.hand_temporal = HandTemporalGAT(
                in_channels=in_channels,
                num_joints=num_joints,
                hand_joint_start=hand_joint_start,
                hand_joint_end=hand_joint_end,
                heads=hand_temporal_heads,
                window_size=hand_temporal_window,
            )
            self.hand_temporal_gate = nn.Parameter(
                torch.tensor([float(hand_temporal_gate_init)], dtype=torch.float32)
            )
        else:
            self.hand_temporal = None
            self.hand_temporal_gate = None

        self.tcn = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                in_channels,
                in_channels,
                (Temp_kernel_size, 1),
                (stride, 1),
                ((Temp_kernel_size - 1) // 2, 0),
            ),
            nn.BatchNorm2d(in_channels),
            nn.Dropout(dropout, inplace=True),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, condition, mask, time_embed=None):
        x_input = x
        res = x
        moe_aux = x.new_zeros(())

        if self.use_joint_gat:
            x = self.gat(x, condition, mask, time_embed=time_embed)

        if self.use_part_gat:
            p = self.part_pool(x)
            p = self.part_gat(p)
            x_part = self.part_broadcast(p)
            assert x_part.shape == x.shape, (
                f"part broadcast {x_part.shape} != joint feat {x.shape}"
            )
            x = x + self.part_gate * x_part

        if self.use_hand_temporal_gat:
            x = x + self.hand_temporal_gate * self.hand_temporal(x)

        x = self.tcn(x)
        x = self.relu(x + res)
        assert x.shape == x_input.shape, (
            f"F_Stage out {x.shape} != in {x_input.shape}"
        )
        return x, moe_aux
