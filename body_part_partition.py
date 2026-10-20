# coding: utf-8
"""Body-part partition for PHOENIX whole-body 50 joints.

Layout matches FGDM/Sign-IDD graph.py layout='whole':
  8 body joints + 21 left-hand + 21 right-hand.

Body (center=1):
  0 nose/head, 1 neck/torso,
  2/3/4 right shoulder/elbow/wrist,
  5/6/7 left shoulder/elbow/wrist.
Hands:
  8..28  left hand (21),
  29..49 right hand (21).
"""
from typing import Dict, List, Tuple

import torch

# 6-part: face_head / torso separate
PART_NAMES_6: Tuple[str, ...] = (
    "left_hand",
    "right_hand",
    "left_arm",
    "right_arm",
    "face_head",
    "torso",
)
PART_TO_JOINTS_6: Dict[str, List[int]] = {
    "left_hand": list(range(8, 29)),
    "right_hand": list(range(29, 50)),
    "left_arm": [5, 6, 7],
    "right_arm": [2, 3, 4],
    "face_head": [0],
    "torso": [1],
}

# 5-part: merge joints 0+1 into upper_body
PART_NAMES_5: Tuple[str, ...] = (
    "left_hand",
    "right_hand",
    "left_arm",
    "right_arm",
    "upper_body",
)
PART_TO_JOINTS_5: Dict[str, List[int]] = {
    "left_hand": list(range(8, 29)),
    "right_hand": list(range(29, 50)),
    "left_arm": [5, 6, 7],
    "right_arm": [2, 3, 4],
    "upper_body": [0, 1],
}

# 3-part: keep both hands; merge arms + head + torso into body
PART_NAMES_3: Tuple[str, ...] = (
    "left_hand",
    "right_hand",
    "body",
)
PART_TO_JOINTS_3: Dict[str, List[int]] = {
    "left_hand": list(range(8, 29)),
    "right_hand": list(range(29, 50)),
    "body": [0, 1, 2, 3, 4, 5, 6, 7],
}

# 2-part: merge both hands; merge arms + head + torso into body
PART_NAMES_2: Tuple[str, ...] = (
    "hands",
    "body",
)
PART_TO_JOINTS_2: Dict[str, List[int]] = {
    "hands": list(range(8, 50)),
    "body": [0, 1, 2, 3, 4, 5, 6, 7],
}

# backward-compatible aliases
PART_NAMES = PART_NAMES_6
PART_TO_JOINTS = PART_TO_JOINTS_6


def _layout_for(num_parts: int) -> Tuple[Tuple[str, ...], Dict[str, List[int]]]:
    if num_parts == 6:
        return PART_NAMES_6, PART_TO_JOINTS_6
    if num_parts == 5:
        return PART_NAMES_5, PART_TO_JOINTS_5
    if num_parts == 3:
        return PART_NAMES_3, PART_TO_JOINTS_3
    if num_parts == 2:
        return PART_NAMES_2, PART_TO_JOINTS_2
    raise ValueError(f"supported part counts are 2, 3, 5 or 6, got {num_parts}")

def build_joint_to_part(num_joints: int = 50, num_parts: int = 6) -> torch.Tensor:
    """Return LongTensor [J] mapping each joint index to part id."""
    if num_joints != 50:
        raise ValueError(f"only 50-joint whole layout supported, got {num_joints}")

    part_names, part_to_joints = _layout_for(num_parts)
    joint_to_part = torch.full((num_joints,), -1, dtype=torch.long)
    for part_id, name in enumerate(part_names):
        for j in part_to_joints[name]:
            if joint_to_part[j] >= 0:
                raise ValueError(f"joint {j} assigned to multiple parts")
            joint_to_part[j] = part_id

    if (joint_to_part < 0).any():
        missing = (joint_to_part < 0).nonzero(as_tuple=False).view(-1).tolist()
        raise ValueError(f"joints not assigned to any part: {missing}")
    return joint_to_part


def build_joints_of_part(num_joints: int = 50, num_parts: int = 6) -> List[List[int]]:
    """Return list of length P; each entry is joint indices of that part."""
    joint_to_part = build_joint_to_part(num_joints, num_parts)
    joints_of_part = [[] for _ in range(num_parts)]
    for j, p in enumerate(joint_to_part.tolist()):
        joints_of_part[p].append(j)
    return joints_of_part


def build_structural_part_pair_bias(
    num_heads: int, num_parts: int = 5
) -> torch.Tensor:
    """
    Structural prior for PartGAT-P5 part-pair bias [H, P, P].

    Part ids: 0 left_hand, 1 right_hand, 2 left_arm, 3 right_arm, 4 upper_body.
    All listed edges are bidirectional; unspecified pairs stay 0.
    """
    if num_parts != 5:
        raise ValueError(
            f"structural part-pair prior currently defined for P=5, got {num_parts}"
        )
    bias = torch.zeros(num_heads, num_parts, num_parts)
    # self-loops
    for p in range(num_parts):
        bias[:, p, p] = 0.20
    # bidirectional pairs
    pairs = (
        (0, 2, 0.10),  # left_hand-left_arm
        (1, 3, 0.10),  # right_hand-right_arm
        (2, 4, 0.10),  # left_arm-upper_body
        (3, 4, 0.10),  # right_arm-upper_body
        (0, 1, 0.10),  # left_hand-right_hand
        (0, 4, 0.05),  # left_hand-upper_body
        (1, 4, 0.05),  # right_hand-upper_body
    )
    for i, j, v in pairs:
        bias[:, i, j] = v
        bias[:, j, i] = v
    return bias
