# coding: utf-8
"""Skeleton graph bias init for IntraFrameGAT skelinit (ported from FGDM, no SCG)."""
import torch

# PHOENIX whole-body 50 joints (same as FGDM module/utils/graph.py layout=whole)
_WHOLE_NEIGHBOR_EDGES = [
    (1, 0), (1, 2), (2, 3), (3, 4), (1, 5), (5, 6),
    (6, 7), (7, 8), (8, 9), (9, 10), (10, 11), (11, 12),
    (8, 13), (13, 14), (14, 15), (15, 16), (8, 17), (17, 18),
    (18, 19), (19, 20), (8, 21), (21, 22), (22, 23), (23, 24),
    (8, 25), (25, 26), (26, 27), (27, 28), (4, 29), (29, 30),
    (30, 31), (31, 32), (32, 33), (29, 34), (34, 35), (35, 36),
    (36, 37), (29, 38), (38, 39), (39, 40), (40, 41), (29, 42),
    (42, 43), (43, 44), (44, 45), (29, 46), (46, 47), (47, 48), (48, 49),
]


def build_skeleton_prior_jj(
    num_joints: int,
    edge_value: float = 0.1,
    self_value: float = 0.2,
    layout: str = "whole",
) -> torch.Tensor:
    """Fixed skeleton prior [J, J]: edge_value on undirected edges, self_value on diagonal."""
    if layout != "whole":
        raise ValueError(f"only layout='whole' is supported here, got {layout!r}")
    if num_joints != 50:
        raise ValueError(f"layout whole has 50 joints, expected {num_joints}")

    prior = torch.zeros(num_joints, num_joints)
    for i, j in _WHOLE_NEIGHBOR_EDGES:
        if i != j:
            prior[i, j] = edge_value
            prior[j, i] = edge_value
    for i in range(num_joints):
        prior[i, i] = self_value
    return prior


def build_skeleton_graph_bias(
    num_joints: int,
    num_heads: int,
    edge_value: float = 0.1,
    self_value: float = 0.2,
    layout: str = "whole",
) -> torch.Tensor:
    """Dense [heads, J, J] bias init for GAT graph_bias."""
    return (
        build_skeleton_prior_jj(num_joints, edge_value, self_value, layout)
        .unsqueeze(0)
        .expand(num_heads, -1, -1)
        .clone()
    )
