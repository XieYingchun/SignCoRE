# coding: utf-8
"""
Diffusion shell for hierGAT+PartGAT-P5 + General[1] MoE-FFN.

Copied from ACD_gat_skelinit_timestep_lowrank.py; original untouched.
Train forward returns (pred_pose, moe_aux); eval returns pose tensor only.
"""
import math

import torch
import torch.nn.functional as F
from collections import namedtuple
from torch import nn

from Denoiser_hierGAT_part5_generalMoE_last import DenoiserHierGATPart5GeneralMoELast

__all__ = ["ACDHierGATPart5GeneralMoELast"]

ModelPrediction = namedtuple("ModelPrediction", ["pred_noise", "pred_x_start"])


def exists(x):
    return x is not None


def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d


def extract(a, t, x_shape):
    batch_size = t.shape[0]
    out = a.gather(-1, t)
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1)))


def cosine_beta_schedule(timesteps, s=0.008):
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps, dtype=torch.float64)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)


class ACDHierGATPart5GeneralMoELast(nn.Module):
    """ACD shell with PartGAT-P5 Focal + last-General MoE-FFN."""

    def __init__(self, args, trg_vocab):
        super().__init__()
        del trg_vocab

        timesteps = args["diffusion"].get("timesteps", 1000)
        sampling_timesteps = args["diffusion"].get("sampling_timesteps", 5)

        betas = cosine_beta_schedule(timesteps)
        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.)
        timesteps, = betas.shape

        self.num_timesteps = int(timesteps)
        self.sampling_timesteps = default(sampling_timesteps, timesteps)
        assert self.sampling_timesteps <= timesteps
        self.is_ddim_sampling = self.sampling_timesteps < timesteps
        self.ddim_sampling_eta = 1.
        self.scale = args["diffusion"].get("scale", 1.0)

        self.register_buffer("betas", betas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("alphas_cumprod_prev", alphas_cumprod_prev)
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register_buffer(
            "sqrt_one_minus_alphas_cumprod", torch.sqrt(1. - alphas_cumprod)
        )
        self.register_buffer(
            "log_one_minus_alphas_cumprod", torch.log(1. - alphas_cumprod)
        )
        self.register_buffer(
            "sqrt_recip_alphas_cumprod", torch.sqrt(1. / alphas_cumprod)
        )
        self.register_buffer(
            "sqrt_recipm1_alphas_cumprod", torch.sqrt(1. / alphas_cumprod - 1)
        )

        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
        self.register_buffer("posterior_variance", posterior_variance)
        self.register_buffer(
            "posterior_log_variance_clipped",
            torch.log(posterior_variance.clamp(min=1e-20)),
        )
        self.register_buffer(
            "posterior_mean_coef1",
            betas * torch.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod),
        )
        self.register_buffer(
            "posterior_mean_coef2",
            (1. - alphas_cumprod_prev) * torch.sqrt(alphas) / (1. - alphas_cumprod),
        )

        dcfg = args["diffusion"]
        self.denoiser = DenoiserHierGATPart5GeneralMoELast(
            num_general_layers=dcfg.get("num_layers", 2),
            num_focal_layers=dcfg.get("num_focal_layers", 3),
            num_heads=dcfg.get("num_heads", 4),
            hidden_size=dcfg.get("hidden_size", 512),
            ff_size=dcfg.get("ff_size", 2048),
            dropout=dcfg.get("dropout", 0.1),
            emb_dropout=dcfg["embeddings"].get("dropout", 0.1),
            gcn_hidden_size=dcfg.get("gcn_hidden_size", 16),
            gat_channels=dcfg.get("gat_channels", 64),
            gat_heads=dcfg.get("gat_heads", 4),
            head_dim=dcfg.get("head_dim", 16),
            gat_use_condition=dcfg.get(
                "gat_use_condition_gate", dcfg.get("gat_use_condition", False)
            ),
            gat_graph_bias_init=dcfg.get("gat_graph_bias_init", "skeleton"),
            gat_dense=dcfg.get("gat_dense", True),
            use_skeleton_hard_mask=dcfg.get("use_skeleton_hard_mask", False),
            use_timestep_lowrank_bias=dcfg.get("use_timestep_lowrank_bias", True),
            use_part_gat=dcfg.get("use_part_gat", True),
            part_gat_num_parts=dcfg.get("part_gat_num_parts", 5),
            part_gat_channels=dcfg.get("part_gat_channels", 64),
            part_gat_heads=dcfg.get("part_gat_heads", 2),
            part_gat_bias_init=dcfg.get("part_gat_bias_init", "zero"),
            part_gate_init=dcfg.get("part_gate_init", 0.0),
            general_moe_num_experts=dcfg.get("general_moe_num_experts", 4),
            general_moe_top_k=dcfg.get("general_moe_top_k", 1),
            general_moe_hidden_mult=dcfg.get("general_moe_hidden_mult", 2),
            general_moe_load_balance_weight=dcfg.get(
                "general_moe_load_balance_weight", 0.001
            ),
            general_moe_z_loss_weight=dcfg.get("general_moe_z_loss_weight", 0.0001),
            moe_router_use_time=dcfg.get("moe_router_use_time", True),
            moe_router_use_frame=dcfg.get("moe_router_use_frame", True),
            moe_router_pool=dcfg.get("moe_router_pool", "frame"),
            use_general_moe=dcfg.get("use_general_moe", True),
            general_moe_all_layers=dcfg.get("general_moe_all_layers", False),
            use_general_motion_refine=dcfg.get("use_general_motion_refine", False),
            motion_refine_kernel=dcfg.get("motion_refine_kernel", 5),
            motion_refine_dropout=dcfg.get("motion_refine_dropout", 0.1),
            motion_refine_gate_init=dcfg.get("motion_refine_gate_init", 0.0),
            use_part_guided_bias=dcfg.get("use_part_guided_bias", False),
            part_guided_num_parts=dcfg.get(
                "part_guided_num_parts", dcfg.get("part_gat_num_parts", 5)
            ),
            part_guided_bias_init=dcfg.get("part_guided_bias_init", "zero"),
            part_guided_gate_init=dcfg.get("part_guided_gate_init", 0.0),
            use_hand_temporal_gat=dcfg.get("use_hand_temporal_gat", False),
            hand_temporal_only_last_block=dcfg.get(
                "hand_temporal_only_last_block", False
            ),
            hand_temporal_last_n_blocks=dcfg.get(
                "hand_temporal_last_n_blocks", None
            ),
            hand_temporal_window=dcfg.get("hand_temporal_window", 5),
            hand_temporal_heads=dcfg.get("hand_temporal_heads", 4),
            hand_temporal_gate_init=dcfg.get("hand_temporal_gate_init", 0.0),
            hand_joint_start=dcfg.get("hand_joint_start", 8),
            hand_joint_end=dcfg.get("hand_joint_end", 50),
            use_stage_separated_focal=dcfg.get("use_stage_separated_focal", False),
            joint_stage_layers=dcfg.get("joint_stage_layers", 2),
            part_stage_layers=dcfg.get("part_stage_layers", 1),
            freeze=False,
            trg_size=args.get("trg_size", 150),
            decoder_trg_trg_=True,
        )

    def predict_noise_from_start(self, x_t, t, x0):
        return (
            (extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - x0)
            / extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)
        )

    def model_predictions(self, x, encoder_output, t, src_mask, trg_mask):
        pred_pose, _, _ = self.denoiser(
            encoder_output=encoder_output,
            trg_embed=x,
            src_mask=src_mask,
            trg_mask=trg_mask,
            t=t,
        )
        x_start = pred_pose * self.scale
        pred_noise = self.predict_noise_from_start(x, t, x_start)
        return ModelPrediction(pred_noise, x_start)

    def ddim_sample(self, encoder_output, input_3d, src_mask, trg_mask):
        batch = encoder_output.shape[0]
        shape = (batch, input_3d.shape[1], 150)
        total_timesteps, sampling_timesteps, eta = (
            self.num_timesteps, self.sampling_timesteps, self.ddim_sampling_eta
        )

        times = torch.linspace(-1, total_timesteps - 1, steps=sampling_timesteps + 1)
        times = list(reversed(times.int().tolist()))
        time_pairs = list(zip(times[:-1], times[1:]))

        img = torch.randn(shape, device=encoder_output.device)
        preds_all = []
        for time, time_next in time_pairs:
            time_cond = torch.full(
                (batch,), time, device=encoder_output.device, dtype=torch.long
            )
            preds = self.model_predictions(
                x=img,
                encoder_output=encoder_output,
                t=time_cond,
                src_mask=src_mask,
                trg_mask=trg_mask,
            )
            pred_noise, x_start = preds.pred_noise.float(), preds.pred_x_start
            preds_all.append(x_start)

            if time_next < 0:
                img = x_start
                continue

            alpha = self.alphas_cumprod[time]
            alpha_next = self.alphas_cumprod[time_next]
            sigma = eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
            c = (1 - alpha_next - sigma ** 2).sqrt()
            noise = torch.randn_like(img)
            img = x_start * alpha_next.sqrt() + c * pred_noise + sigma * noise

        return preds_all

    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)
        sqrt_alphas_cumprod_t = extract(self.sqrt_alphas_cumprod, t, x_start.shape)
        sqrt_one_minus_alphas_cumprod_t = extract(
            self.sqrt_one_minus_alphas_cumprod, t, x_start.shape
        )
        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise

    def forward(self, encoder_output, input_3d, src_mask, trg_mask, is_train):
        if not is_train:
            results = self.ddim_sample(
                encoder_output=encoder_output,
                input_3d=input_3d,
                src_mask=src_mask,
                trg_mask=trg_mask,
            )
            return results[self.sampling_timesteps - 1]

        x_poses, noises, t = self.prepare_targets(input_3d)
        x_poses = x_poses.float()
        t = t.squeeze(-1)
        pred_pose, _, moe_aux = self.denoiser(
            encoder_output=encoder_output,
            trg_embed=x_poses,
            src_mask=src_mask,
            trg_mask=trg_mask,
            t=t,
        )
        return pred_pose, moe_aux

    def prepare_diffusion_concat(self, pose_3d):
        device = pose_3d.device
        t = torch.randint(0, self.num_timesteps, (1,), device=device).long()
        noise = torch.randn(pose_3d.shape[0], 150, device=device)
        x_start = pose_3d * self.scale
        x = self.q_sample(x_start=x_start, t=t, noise=noise) / self.scale
        return x, noise, t

    def prepare_targets(self, targets):
        diffused_poses, noises, ts = [], [], []
        for i in range(targets.shape[0]):
            d_poses, d_noise, d_t = self.prepare_diffusion_concat(targets[i])
            diffused_poses.append(d_poses)
            noises.append(d_noise)
            ts.append(d_t)
        return torch.stack(diffused_poses), torch.stack(noises), torch.stack(ts)
