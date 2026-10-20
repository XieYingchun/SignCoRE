# coding: utf-8
"""
Sign-IDD model: hierGAT + PartGAT-P5 + General[1] MoE-FFN.

Copied/adapted from model_gat_skelinit_timestep_lowrank.py; original untouched.
"""
import torch
import torch.nn as nn
from torch import Tensor

from encoder import Encoder
from ACD_hierGAT_part5_generalMoE_last import ACDHierGATPart5GeneralMoELast
from batch import Batch
from embeddings import Embeddings
from vocabulary import Vocabulary
from initialization import initialize_model
from constants import PAD_TOKEN, EOS_TOKEN, BOS_TOKEN, TARGET_PAD


class Model(nn.Module):
    def __init__(
        self,
        cfg: dict,
        encoder: Encoder,
        ACD: ACDHierGATPart5GeneralMoELast,
        src_embed: Embeddings,
        src_vocab: Vocabulary,
        trg_vocab: Vocabulary,
        in_trg_size: int,
        out_trg_size: int,
    ):
        super(Model, self).__init__()

        self.src_embed = src_embed
        self.encoder = encoder
        self.ACD = ACD
        self.src_vocab = src_vocab
        self.trg_vocab = trg_vocab
        self.bos_index = self.src_vocab.stoi[BOS_TOKEN]
        self.pad_index = self.src_vocab.stoi[PAD_TOKEN]
        self.eos_index = self.src_vocab.stoi[EOS_TOKEN]
        self.target_pad = TARGET_PAD
        self.use_cuda = cfg["training"]["use_cuda"]
        self.in_trg_size = in_trg_size
        self.out_trg_size = out_trg_size

    def forward(
        self,
        is_train: bool,
        src: Tensor,
        trg_input: Tensor,
        src_mask: Tensor,
        src_lengths: Tensor,
        trg_mask: Tensor,
    ):
        encoder_output = self.encode(
            src=src, src_length=src_lengths, src_mask=src_mask
        )
        return self.diffusion(
            is_train=is_train,
            encoder_output=encoder_output,
            trg_input=trg_input,
            src_mask=src_mask,
            trg_mask=trg_mask,
        )

    def encode(self, src: Tensor, src_length: Tensor, src_mask: Tensor):
        return self.encoder(
            embed_src=self.src_embed(src), src_length=src_length, mask=src_mask
        )

    def diffusion(
        self,
        is_train: bool,
        encoder_output: Tensor,
        src_mask: Tensor,
        trg_input: Tensor,
        trg_mask: Tensor,
    ):
        return self.ACD(
            is_train=is_train,
            encoder_output=encoder_output,
            input_3d=trg_input,
            src_mask=src_mask,
            trg_mask=trg_mask,
        )

    def get_loss_for_batch(
        self, is_train, batch: Batch, loss_function: nn.Module,
        apply_motion_saliency: bool = False,
    ) -> Tensor:
        out = self.forward(
            src=batch.src,
            trg_input=batch.trg_input[:, :, :150],
            src_mask=batch.src_mask,
            src_lengths=batch.src_lengths,
            trg_mask=batch.trg_mask,
            is_train=is_train,
        )
        if isinstance(out, tuple):
            skel_out, moe_aux = out
        else:
            skel_out = out
            moe_aux = skel_out.new_zeros(())
        # moe_aux already includes configured lb/z weights inside MoE module
        return (
            loss_function(
                skel_out,
                batch.trg_input[:, :, :150],
                apply_motion_saliency=apply_motion_saliency,
            )
            + moe_aux
        )

def build_model(cfg: dict, src_vocab: Vocabulary, trg_vocab: Vocabulary):
    full_cfg = cfg
    cfg = cfg["model"]

    src_padding_idx = src_vocab.stoi[PAD_TOKEN]
    trg_padding_idx = 0
    in_trg_size = cfg["trg_size"]
    out_trg_size = cfg["trg_size"]

    src_embed = Embeddings(
        **cfg["encoder"]["embeddings"],
        vocab_size=len(src_vocab),
        padding_idx=src_padding_idx,
    )

    enc_dropout = cfg["encoder"].get("dropout", 0.0)
    enc_emb_dropout = cfg["encoder"]["embeddings"].get("dropout", enc_dropout)
    assert (
        cfg["encoder"]["embeddings"]["embedding_dim"] == cfg["encoder"]["hidden_size"]
    ), "for transformer, emb_size must be hidden_size"

    encoder = Encoder(
        **cfg["encoder"],
        emb_size=src_embed.embedding_dim,
        emb_dropout=enc_emb_dropout,
    )

    diffusion = ACDHierGATPart5GeneralMoELast(args=cfg, trg_vocab=trg_vocab)

    model = Model(
        encoder=encoder,
        ACD=diffusion,
        src_embed=src_embed,
        src_vocab=src_vocab,
        trg_vocab=trg_vocab,
        cfg=full_cfg,
        in_trg_size=in_trg_size,
        out_trg_size=out_trg_size,
    )
    initialize_model(model, cfg, src_padding_idx, trg_padding_idx)

    denoiser = model.ACD.denoiser
    for fi, stage in enumerate(denoiser.Focal):
        if stage.use_joint_gat and stage.gat is not None:
            stage.gat.reset_special_parameters()
            gat = stage.gat
            if gat.graph_bias_init == "skeleton":
                assert torch.count_nonzero(gat.graph_bias).item() > 0
            if gat.use_timestep_lowrank_bias:
                assert torch.count_nonzero(gat.time_bias_hidden.weight).item() > 0
                assert torch.count_nonzero(gat.time_bias_out.weight).item() == 0
                assert torch.count_nonzero(gat.time_bias_out.bias).item() == 0
            if gat.use_part_guided_bias:
                expected_pg = float(
                    cfg.get("diffusion", {}).get("part_guided_gate_init", 0.0)
                )
                pg_val = float(gat.part_guided_gate.detach().item())
                print(
                    f"[init] Focal[{fi}] part_guided_gate={pg_val:.6f} "
                    f"(expected {expected_pg:.6f}, "
                    f"bias_init={gat.part_guided_bias_init})"
                )
                assert abs(pg_val - expected_pg) < 1e-6, (
                    f"Focal[{fi}] part_guided_gate={pg_val} != init {expected_pg}"
                )
                if gat.part_guided_bias_init.lower() == "structural":
                    assert torch.count_nonzero(gat.part_pair_bias).item() > 0
                else:
                    assert torch.count_nonzero(gat.part_pair_bias).item() == 0
        if stage.use_hand_temporal_gat:
            assert stage.hand_temporal_gate is not None
            expected_gate = float(
                cfg.get("diffusion", {}).get("hand_temporal_gate_init", 0.0)
            )
            gate_val = float(stage.hand_temporal_gate.detach().item())
            print(
                f"[init] Focal[{fi}] hand_temporal_gate={gate_val:.6f} "
                f"(expected {expected_gate:.6f})"
            )
            assert abs(gate_val - expected_gate) < 1e-6, (
                f"Focal[{fi}] hand_temporal_gate={gate_val} != init {expected_gate}"
            )
            assert torch.count_nonzero(stage.hand_temporal.rel_bias).item() == 0

    dcfg = cfg.get("diffusion", {})
    print(
        f"[init] use_part_guided_bias={getattr(denoiser, 'use_part_guided_bias', False)} "
        f"part_guided_bias_init={dcfg.get('part_guided_bias_init', None)} "
        f"part_guided_gate_init={dcfg.get('part_guided_gate_init', 0.0)}"
    )
    print(
        f"[init] use_hand_temporal_gat={denoiser.use_hand_temporal_gat} "
        f"hand_temporal_window={getattr(denoiser, 'hand_temporal_window', dcfg.get('hand_temporal_window', None))} "
        f"hand_temporal_gate_init={dcfg.get('hand_temporal_gate_init', 0.0)}"
    )

    if denoiser.use_hand_temporal_gat:
        enabled_blocks = [
            i for i, s in enumerate(denoiser.Focal) if s.use_hand_temporal_gat
        ]
        window = getattr(
            denoiser,
            "hand_temporal_window",
            dcfg.get("hand_temporal_window", None),
        )
        last_n = getattr(denoiser, "hand_temporal_last_n_blocks", None)
        print(
            f"[init] hand_temporal_window={window} "
            f"enabled_blocks={enabled_blocks} "
            f"last_n_blocks={last_n} "
            f"only_last_block={denoiser.hand_temporal_only_last_block} "
            f"gate_init={dcfg.get('hand_temporal_gate_init', 0.0)}"
        )

    moe_layers = [
        i for i, s in enumerate(denoiser.General) if getattr(s, "use_moe_ffn", False)
    ]
    print(
        f"[init] use_general_moe={getattr(denoiser, 'use_general_moe', True)} "
        f"general_moe_all_layers={getattr(denoiser, 'general_moe_all_layers', False)} "
        f"moe_general_layers={moe_layers} "
        f"num_experts={dcfg.get('general_moe_num_experts', None)} "
        f"top_k={dcfg.get('general_moe_top_k', None)} "
        f"router_use_time={getattr(denoiser, 'moe_router_use_time', True)} "
        f"router_use_frame={getattr(denoiser, 'moe_router_use_frame', True)} "
        f"router_pool={getattr(denoiser, 'moe_router_pool', 'frame')}"
    )
    expect_router_time = bool(dcfg.get("moe_router_use_time", True))
    expect_router_frame = bool(dcfg.get("moe_router_use_frame", True))
    expect_router_pool = str(dcfg.get("moe_router_pool", "frame"))
    assert getattr(denoiser, "moe_router_use_time", True) is expect_router_time
    assert getattr(denoiser, "moe_router_use_frame", True) is expect_router_frame
    assert getattr(denoiser, "moe_router_pool", "frame") == expect_router_pool
    for gi, glayer in enumerate(denoiser.General):
        if glayer.use_moe_ffn:
            assert glayer.feed_forward.router_use_time is expect_router_time, (
                f"General[{gi}] router_use_time="
                f"{glayer.feed_forward.router_use_time}, expected {expect_router_time}"
            )
            assert glayer.feed_forward.router_use_frame is expect_router_frame, (
                f"General[{gi}] router_use_frame="
                f"{glayer.feed_forward.router_use_frame}, expected {expect_router_frame}"
            )
            assert glayer.feed_forward.router_pool == expect_router_pool, (
                f"General[{gi}] router_pool="
                f"{glayer.feed_forward.router_pool}, expected {expect_router_pool}"
            )

    if denoiser.use_stage_separated_focal:
        jn = denoiser.joint_stage_layers
        pn = denoiser.part_stage_layers
        assert len(denoiser.Focal) == jn + pn
        for i, stage in enumerate(denoiser.Focal):
            if i < jn:
                assert stage.use_joint_gat is True
                assert stage.use_part_gat is False
                assert stage.use_hand_temporal_gat is False
            else:
                assert stage.use_joint_gat is False
                assert stage.use_part_gat is True
                assert stage.use_hand_temporal_gat is False
    elif denoiser.use_hand_temporal_gat:
        n = len(denoiser.Focal)
        last_n = getattr(denoiser, "hand_temporal_last_n_blocks", None)
        # Allow ablating PartGAT while keeping HandTemporalGAT.
        expect_part = bool(dcfg.get("use_part_gat", True))
        for i, stage in enumerate(denoiser.Focal):
            assert stage.use_joint_gat is True
            assert stage.use_part_gat is expect_part, (
                f"Focal[{i}] use_part_gat={stage.use_part_gat}, "
                f"expected {expect_part}"
            )
            if last_n is not None:
                expect_ht = i >= n - int(last_n)
            elif denoiser.hand_temporal_only_last_block:
                expect_ht = i == n - 1
            else:
                expect_ht = True
            assert stage.use_hand_temporal_gat is expect_ht, (
                f"Focal[{i}] hand_temporal={stage.use_hand_temporal_gat}, "
                f"expected {expect_ht}"
            )
    # sanity: MoE on last General by default; optionally all / none
    n_gen = len(denoiser.General)
    use_general_moe = getattr(denoiser, "use_general_moe", True)
    if not use_general_moe:
        for gi, glayer in enumerate(denoiser.General):
            assert glayer.use_moe_ffn is False, (
                f"General[{gi}] use_moe_ffn={glayer.use_moe_ffn}, expected False"
            )
    elif getattr(denoiser, "general_moe_all_layers", False):
        for gi, glayer in enumerate(denoiser.General):
            assert glayer.use_moe_ffn is True, (
                f"General[{gi}] use_moe_ffn={glayer.use_moe_ffn}, expected True"
            )
    elif n_gen == 1:
        assert denoiser.General[0].use_moe_ffn is True
    else:
        assert denoiser.General[0].use_moe_ffn is False
        assert denoiser.General[-1].use_moe_ffn is True
    # MotionRefine (if enabled) only on MoE General layers; zero-init gate
    if (
        use_general_moe
        and not getattr(denoiser, "general_moe_all_layers", False)
        and n_gen >= 2
    ):
        assert denoiser.General[0].use_motion_refine is False
    if denoiser.use_general_motion_refine:
        last = denoiser.General[-1]
        assert last.use_motion_refine is True
        assert last.motion_refine_gate is not None
        assert float(last.motion_refine_gate.detach().item()) == 0.0

    return model
