# coding: utf-8
import torch
import torch.nn as nn

from helpers import getSkeletalModelStructure


class Loss(nn.Module):

    def __init__(self, cfg, target_pad=0.0):
        super(Loss, self).__init__()

        self.loss = cfg["training"]["loss"].lower()
        self.bone_loss = cfg["training"]["bone_loss"].lower()

        if self.loss == "l1":
            self.criterion = nn.L1Loss()
        elif self.loss == "mse":
            self.criterion = nn.MSELoss()
        else:
            print("Loss not found - revert to default L1 loss")
            self.criterion = nn.L1Loss()

        if self.bone_loss == "l1":
            self.criterion_bone = nn.L1Loss()
        elif self.bone_loss == "mse":
            self.criterion_bone = nn.MSELoss()
        else:
            print("Loss not found - revert to default MSE loss")
            self.criterion_bone = nn.MSELoss()

        model_cfg = cfg["model"]
        train_cfg = cfg["training"]

        self.target_pad = target_pad
        self.loss_scale = model_cfg.get("loss_scale", 1.0)

        # Motion saliency weighting (train-only when apply_motion_saliency=True)
        self.use_motion_saliency_weight = bool(
            train_cfg.get(
                "use_motion_saliency_weight",
                model_cfg.get("use_motion_saliency_weight", False),
            )
        )
        self.motion_saliency_beta = float(
            train_cfg.get(
                "motion_saliency_beta",
                model_cfg.get("motion_saliency_beta", 0.5),
            )
        )
        joints = train_cfg.get(
            "motion_saliency_joints",
            model_cfg.get("motion_saliency_joints", [8, 50]),
        )
        self.motion_saliency_joint_start = int(joints[0])
        self.motion_saliency_joint_end = int(joints[1])
        self.motion_saliency_norm = str(
            train_cfg.get(
                "motion_saliency_norm",
                model_cfg.get("motion_saliency_norm", "per_sequence"),
            )
        ).lower()
        self.motion_saliency_clamp = float(
            train_cfg.get(
                "motion_saliency_clamp",
                model_cfg.get("motion_saliency_clamp", 3.0),
            )
        )
        self.motion_saliency_debug = bool(
            train_cfg.get(
                "motion_saliency_debug",
                model_cfg.get("motion_saliency_debug", False),
            )
        )
        self._motion_saliency_debug_printed = False

    def _valid_frame_mask(self, targets: torch.Tensor) -> torch.Tensor:
        """Whole-frame validity [N, T]: True if any coord in the frame is non-pad."""
        assert targets.ndim == 3, f"expected [N,T,D], got {targets.shape}"
        return (targets != self.target_pad).any(dim=-1)

    def _hand_motion_weights(
        self, targets: torch.Tensor, valid_frame: torch.Tensor
    ) -> torch.Tensor:
        """
        Per-sequence hand-motion saliency weights [N, T].
        v_t from GT hand joints; normalized on valid frames only.
        """
        n, t, d = targets.shape
        assert d % 3 == 0, f"expected 3D joints flat, got D={d}"
        j = d // 3
        x0 = targets.view(n, t, j, 3)
        hand = x0[
            :, :, self.motion_saliency_joint_start : self.motion_saliency_joint_end, :
        ]
        # ||x_t - x_{t-1}||_2 per hand joint -> mean over joints -> [N,T]
        delta = hand[:, 1:] - hand[:, :-1]
        speed = torch.norm(delta, p=2, dim=-1).mean(dim=-1)  # [N, T-1]
        v = torch.zeros(n, t, device=targets.device, dtype=targets.dtype)
        v[:, 1:] = speed
        v[:, 0] = 0.0

        if self.motion_saliency_norm != "per_sequence":
            raise ValueError(
                f"unsupported motion_saliency_norm={self.motion_saliency_norm}"
            )

        valid = valid_frame.to(dtype=v.dtype)
        mean_v = (v * valid).sum(dim=1, keepdim=True) / valid.sum(
            dim=1, keepdim=True
        ).clamp(min=1.0)
        v_norm = v / (mean_v + 1e-6)
        v_norm = torch.clamp(v_norm, 0.0, self.motion_saliency_clamp)
        w = 1.0 + self.motion_saliency_beta * v_norm
        # invalid frames: keep weight unused (multiplied by valid later)
        return w

    def forward(self, preds, targets, apply_motion_saliency: bool = False):
        """
        apply_motion_saliency: only True for train batch loss when enabled in cfg.
        Val/test/inference must leave this False (default).
        """
        loss_mask = targets != self.target_pad

        preds_masked = preds * loss_mask
        targets_masked = targets * loss_mask

        preds_masked_length, preds_masked_direct = get_length_direct(preds_masked)
        targets_masked_length, targets_masked_direct = get_length_direct(targets_masked)

        preds_masked_length = preds_masked_length * loss_mask[:, :, :50]
        targets_masked_length = targets_masked_length * loss_mask[:, :, :50]
        preds_masked_direct = preds_masked_direct * loss_mask[:, :, :150]
        targets_masked_direct = targets_masked_direct * loss_mask[:, :, :150]

        # Bone loss: original unweighted mean (unchanged)
        l_bone = self.criterion_bone(preds_masked_direct, targets_masked_direct)

        use_saliency = bool(apply_motion_saliency) and self.use_motion_saliency_weight
        if use_saliency:
            valid_frame = self._valid_frame_mask(targets)  # [N,T]
            w_t = self._hand_motion_weights(targets, valid_frame)  # [N,T]
            # frame L1 over pose coords
            err = (preds - targets).abs().mean(dim=-1)  # [N,T]
            weight = w_t * valid_frame.to(dtype=err.dtype)
            denom = weight.sum().clamp(min=1e-8)
            l_pose = (err * weight).sum() / denom

            if self.motion_saliency_debug and not self._motion_saliency_debug_printed:
                w_valid = w_t[valid_frame]
                print(
                    "[motion_saliency] "
                    f"valid_frame.sum()={int(valid_frame.sum().item())} "
                    f"w_t min/mean/max="
                    f"{float(w_valid.min()):.4f}/"
                    f"{float(w_valid.mean()):.4f}/"
                    f"{float(w_valid.max()):.4f} "
                    f"L_pose={float(l_pose):.6f} "
                    f"L_bone={float(l_bone):.6f} "
                    f"total={float(l_pose + 0.1 * l_bone):.6f} "
                    f"nan_pose={bool(torch.isnan(l_pose))} "
                    f"nan_bone={bool(torch.isnan(l_bone))}"
                )
                self._motion_saliency_debug_printed = True
        else:
            l_pose = self.criterion(preds_masked, targets_masked)

        loss = l_pose + 0.1 * l_bone

        if self.loss_scale != 1.0:
            loss = loss * self.loss_scale

        return loss


def get_length_direct(trg):
    trg_reshaped = trg.view(trg.shape[0], trg.shape[1], 50, 3)
    trg_list = trg_reshaped.split(1, dim=2)
    trg_list_squeeze = [t.squeeze(dim=2) for t in trg_list]
    skeletons = getSkeletalModelStructure()

    length = []
    direct = []
    for skeleton in skeletons:
        result_length = Skeleton_length = torch.norm(
            trg_list_squeeze[skeleton[0]] - trg_list_squeeze[skeleton[1]],
            p=2,
            dim=2,
            keepdim=True,
        )
        result_direct = (
            trg_list_squeeze[skeleton[0]] - trg_list_squeeze[skeleton[1]]
        ) / (Skeleton_length + torch.finfo(Skeleton_length.dtype).tiny)
        direct.append(result_direct)
        length.append(result_length)
    lengths = torch.stack(length, dim=-1).squeeze()
    directs = torch.stack(direct, dim=2).view(trg.shape[0], trg.shape[1], -1)

    return lengths, directs
