# coding: utf-8
"""Sign-IDD model with GAT Focal denoiser (no SCG). Original model.py untouched."""
import torch
import torch.nn as nn
from torch import Tensor

from encoder import Encoder
from ACD_gat_skelinit_timestep_lowrank import ACDGatSkelinitTimestepLowrank
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
        ACD: ACDGatSkelinitTimestepLowrank,
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
        skel_out = self.forward(
            src=batch.src,
            trg_input=batch.trg_input[:, :, :150],
            src_mask=batch.src_mask,
            src_lengths=batch.src_lengths,
            trg_mask=batch.trg_mask,
            is_train=is_train,
        )
        return loss_function(
            skel_out,
            batch.trg_input[:, :, :150],
            apply_motion_saliency=apply_motion_saliency,
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

    diffusion = ACDGatSkelinitTimestepLowrank(args=cfg, trg_vocab=trg_vocab)

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

    # initialize_model() treats every parameter containing "bias" as a bias.
    # Restore skeleton and timestep-low-rank initialization afterwards.
    for stage in model.ACD.denoiser.Focal:
        stage.gat.reset_special_parameters()
        gat = stage.gat
        if gat.graph_bias_init == "skeleton":
            assert torch.count_nonzero(gat.graph_bias).item() > 0
        if gat.use_timestep_lowrank_bias:
            assert torch.count_nonzero(gat.time_bias_hidden.weight).item() > 0
            assert torch.count_nonzero(gat.time_bias_out.weight).item() == 0
            assert torch.count_nonzero(gat.time_bias_out.bias).item() == 0

    return model
