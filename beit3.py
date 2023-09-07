# This code is based on Microsoft's BEIT3
# https://github.com/microsoft/unilm/tree/master/beit3

# Copyright (c) 2022 Microsoft
# Licensed under The MIT License [see LICENSE for details]

import math
import time
import numpy as np
import torch
import torch.nn as nn
try:
    from apex.normalization import FusedLayerNorm as LayerNorm
except ModuleNotFoundError:
    from torch.nn import LayerNorm

from torchscale.component.droppath import DropPath
from torchscale.component.feedforward_network import FeedForwardNetwork
from torchscale.component.multihead_attention import MultiheadAttention
from torchscale.component.multiway_network import MultiwayWrapper, set_split_position, MutliwayEmbedding
from torchscale.component.embedding import PositionalEmbedding, TextEmbedding, VisionEmbedding


class EncoderLayer(nn.Module):
    def __init__(self, args, depth):
        super().__init__()
        # Config
        self.args = args
        self.alpha = 1.0
        self.embed_dim = args.encoder_embed_dim
        self.ffn_dim = args.encoder_ffn_embed_dim

        # Self Attention Layer Norm
        self.self_attn_layer_norm = MultiwayWrapper(args, LayerNorm(self.embed_dim, eps=args.layernorm_eps))

        # Self Attention
        self.self_attn = self.build_self_attention(self.embed_dim, args)

        # Dropout
        self.dropout_module = torch.nn.Dropout(args.dropout)

        # DropPath
        drop_path_prob = np.linspace(0, args.drop_path_rate, args.encoder_layers)[depth]
        self.drop_path = DropPath(drop_path_prob)

        # FFN Layer Norm
        self.final_layer_norm = MultiwayWrapper(args, LayerNorm(self.embed_dim, eps=args.layernorm_eps))

        # FFN
        self.ffn = MultiwayWrapper(args, self.build_ffn(self.embed_dim, self.ffn_dim, self.args))

    def build_ffn(self, embed_dim, ffn_dim, args):
        return FeedForwardNetwork(embed_dim, ffn_dim, "gelu", args.dropout,
                                  args.activation_dropout, args.layernorm_eps, args.subln)

    def build_self_attention(self, embed_dim, args):
        return MultiheadAttention(args, embed_dim, args.encoder_attention_heads,
                                  dropout=args.attention_dropout, self_attention=True,
                                  encoder_decoder_attention=False, subln=args.subln)

    def residual_connection(self, x, residual):
        return residual * self.alpha + x

    def forward(self, x, encoder_padding_mask, attn_mask=None, rel_pos=None, multiway_split_position=None):
        if multiway_split_position is not None:
            assert self.args.multiway
            self.apply(set_split_position(multiway_split_position))
        if attn_mask is not None:
            attn_mask = attn_mask.masked_fill(attn_mask.to(torch.bool), -1e8)
        residual = x
        x = self.self_attn_layer_norm(x)
        x, _ = self.self_attn(query=x, key=x, value=x, key_padding_mask=encoder_padding_mask,
                              attn_mask=attn_mask, rel_pos=rel_pos, incremental_state=None)
        x = self.dropout_module(x)
        if self.drop_path is not None:
            x = self.drop_path(x)
        x = self.residual_connection(x, residual)
        residual = x
        x = self.final_layer_norm(x)
        x = self.ffn(x)
        x = self.drop_path(x)
        x = self.residual_connection(x, residual)

        return x


class Encoder(nn.Module):
    def __init__(self, args, embed_tokens=None, embed_positions=None, **kwargs):
        super().__init__(**kwargs)
        self.args = args
        self.dropout_module = torch.nn.Dropout(args.dropout)
        self.embed_scale = 1.0
        self.embed_tokens = embed_tokens
        self.embed_positions = embed_positions
        self.output_projection = None
        self.layers = nn.ModuleList([])
        for i in range(args.encoder_layers):
            self.layers.append(EncoderLayer(args, depth=i))
        self.num_layers = len(self.layers)
        if args.subln:
            init_scale = math.sqrt(math.log(args.encoder_layers * 2))
            for name, p in self.named_parameters():
                if "fc1" in name or "fc2" in name or "out_proj" in name or "v_proj" in name:
                    p.data.mul_(init_scale)

    def forward_embedding(self, token_embedding=None, positions=None):
        x = embed = self.embed_scale * token_embedding
        if self.embed_positions is not None:
            x = embed + self.embed_positions(x, positions=positions)
        x = self.dropout_module(x)
        return x, embed

    def forward(
        self,
        token_embeddings,
        encoder_padding_mask=None,
        attn_mask=None,
        multiway_split_position=None,
        positions=None,
        return_all_hiddens=False,
        **kwargs
    ):
        assert token_embeddings is not None

        # Encoder Padding Mask
        if encoder_padding_mask is None:
            encoder_padding_mask = torch.zeros(
                [token_embeddings.size(0), token_embeddings.size(1)],
                device=token_embeddings.device,
            ).bool()

        if multiway_split_position is not None:
            assert self.args.multiway
            self.apply(set_split_position(multiway_split_position))

        x, encoder_embedding = self.forward_embedding(token_embeddings, positions)
        x = x * (1 - encoder_padding_mask.unsqueeze(-1).type_as(x))

        encoder_states = []
        if return_all_hiddens:
            encoder_states.append(x)

        for idx, layer in enumerate(self.layers):
            x = layer(x, encoder_padding_mask=encoder_padding_mask, attn_mask=attn_mask,
                      rel_pos=None, multiway_split_position=multiway_split_position)
            if return_all_hiddens:
                assert encoder_states is not None
                encoder_states.append(x)

        return {
            "encoder_out": x,
            "encoder_embedding": encoder_embedding,
            "encoder_padding_mask": encoder_padding_mask,
            "encoder_states": encoder_states,
        }


class BEiT3(nn.Module):
    def __init__(self, args, **kwargs):
        super().__init__()
        self.args = args
        assert args.multiway
        assert args.vocab_size > 0
        assert not args.share_encoder_input_output_embed
        self.text_embed = TextEmbedding(args.vocab_size, args.encoder_embed_dim)
        self.vision_embed = VisionEmbedding(
            args.img_size,
            args.patch_size,
            args.in_chans,
            args.encoder_embed_dim,
            contain_mask_token=True,
            prepend_cls_token=True,
        )
        # being consistent with Fairseq, which starts from 2 for position embedding
        embed_positions = MutliwayEmbedding(
            modules=[
                PositionalEmbedding(self.vision_embed.num_position_embeddings() + 2, args.encoder_embed_dim),
                PositionalEmbedding(args.max_source_positions, args.encoder_embed_dim),
            ],
            dim=1,
        )
        self.encoder = Encoder(
            args,
            embed_tokens=None,
            embed_positions=embed_positions,
        )

    def forward(
        self,
        textual_tokens=None,
        visual_tokens=None,
        text_padding_position=None,
        vision_masked_position=None,
        attn_mask=None,
        positions=None,
        return_all_hiddens=False,
    ):
        assert textual_tokens is not None or visual_tokens is not None

        if textual_tokens is None:
            x = self.vision_embed(visual_tokens, vision_masked_position)
            encoder_padding_mask = None
            multiway_split_position = -1
        elif visual_tokens is None:
            x = self.text_embed(textual_tokens)
            encoder_padding_mask = text_padding_position
            multiway_split_position = 0
        else:
            x1 = self.vision_embed(visual_tokens, vision_masked_position)
            multiway_split_position = x1.size(1)
            x2 = self.text_embed(textual_tokens)
            x = torch.cat([x1, x2], dim=1)
            if text_padding_position is not None:
                encoder_padding_mask = torch.cat(
                    [
                        torch.zeros(x1.shape[:-1]).to(x1.device).bool(),
                        text_padding_position,
                    ],
                    dim=1,
                )
            else:
                encoder_padding_mask = None

        encoder_out = self.encoder(
            token_embeddings=x,
            encoder_padding_mask=encoder_padding_mask,
            attn_mask=attn_mask,
            multiway_split_position=multiway_split_position,
            positions=positions,
            return_all_hiddens=return_all_hiddens,
        )
        encoder_out["multiway_split_position"] = multiway_split_position

        return encoder_out


if __name__ == '__main__':

    print("-------------------- test for beit3----------------------")

    import argparse
    parser = argparse.ArgumentParser()

    # arguments for encoder
    parser.add_argument('--dropout', type=float, default=0.0)
    parser.add_argument('--encoder_embed_dim', type=int, default=768)
    parser.add_argument('--vocab_size', type=int, default=64010)
    parser.add_argument('--layernorm_eps', type=float, default=1e-5)
    parser.add_argument('--encoder_layers', type=int, default=12)
    parser.add_argument('--encoder_attention_heads', type=int, default=12)
    parser.add_argument('--subln', type=bool, default=True)
    parser.add_argument('--multiway', type=bool, default=True)
    parser.add_argument('--drop_path_rate', type=float, default=0.1)
    parser.add_argument('--encoder_ffn_embed_dim', type=int, default=3072)
    parser.add_argument('--activation_dropout', type=float, default=0.0)
    parser.add_argument('--attention_dropout', type=float, default=0.0)
    parser.add_argument('--xpos_rel_pos', type=bool, default=False)
    # arguments for beit3
    parser.add_argument('--share_encoder_input_output_embed', type=bool, default=False)
    parser.add_argument('--img_size', type=int, default=480)
    parser.add_argument('--patch_size', type=int, default=16)
    parser.add_argument('--in_chans', type=int, default=3)
    parser.add_argument('--max_source_positions', type=int, default=1024)
    parser.add_argument('--output_projection', default=None)
    args = parser.parse_args()

    seed = 0
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device('cuda:0')

    beit3 = BEiT3(args).to(device).eval()

    img_tokens = torch.rand(1, 3, 480, 480).to(device)
    text_tokens = torch.linspace(1, 65, 65, dtype=torch.int).unsqueeze(0).to(device)
    text_padding = torch.zeros(1, 65, dtype=torch.int).to(device)
    output = beit3(visual_tokens=img_tokens,
                   textual_tokens=text_tokens,
                   text_padding_position=text_padding,
                   )

    print(output['encoder_out'].sum())  # 501: -51848.6992   2230771: 25646.5938   1852305: 190511.7656

    counter = 0
    for param in beit3.parameters():
        counter += torch.numel(param)
    print(counter)

