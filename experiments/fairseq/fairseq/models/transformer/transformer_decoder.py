# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
from typing import Any, Dict, List, Optional

import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from fairseq import utils
from fairseq.distributed import fsdp_wrap
from fairseq.models import FairseqIncrementalDecoder
from fairseq.models.transformer import TransformerConfig
from fairseq.modules import (
    AdaptiveSoftmax,
    BaseLayer,
    FairseqDropout,
    LayerDropModuleList,
    LayerNorm,
    PositionalEmbedding,
    SinusoidalPositionalEmbedding,
    transformer_layer,
    MultiheadAttention,
)
from fairseq.modules.checkpoint_activations import checkpoint_wrapper
from fairseq.modules.quant_noise import quant_noise as apply_quant_noise_


# rewrite name for backward compatibility in `make_generation_fast_`
def module_name_fordropout(module_name: str) -> str:
    if module_name == "TransformerDecoderBase":
        return "TransformerDecoder"
    else:
        return module_name


class TransformerDecoderBase(FairseqIncrementalDecoder):
    """
    Transformer decoder consisting of *cfg.decoder.layers* layers. Each layer
    is a :class:`TransformerDecoderLayer`.

    Args:
        cfg (argparse.Namespace): parsed command-line arguments
        dictionary (~fairseq.data.Dictionary): decoding dictionary
        embed_tokens (torch.nn.Embedding): output embedding
        no_encoder_attn (bool, optional): whether to attend to encoder outputs
            (default: False).
    """

    def __init__(
        self,
        cfg,
        dictionary,
        embed_tokens,
        no_encoder_attn=False,
        output_projection=None,
    ):
        self.cfg = cfg
        super().__init__(dictionary)
        self.register_buffer("version", torch.Tensor([3]))
        self._future_mask = torch.empty(0)

        self.dropout_module = FairseqDropout(
            cfg.dropout, module_name=module_name_fordropout(self.__class__.__name__)
        )
        self.decoder_layerdrop = cfg.decoder.layerdrop
        self.share_input_output_embed = cfg.share_decoder_input_output_embed

        input_embed_dim = embed_tokens.embedding_dim
        embed_dim = cfg.decoder.embed_dim
        self.embed_dim = embed_dim
        self.output_embed_dim = cfg.decoder.output_dim

        self.padding_idx = embed_tokens.padding_idx
        self.max_target_positions = cfg.max_target_positions

        self.embed_tokens = embed_tokens

        self.embed_scale = 1.0 if cfg.no_scale_embedding else math.sqrt(embed_dim)

        if not cfg.adaptive_input and cfg.quant_noise.pq > 0:
            self.quant_noise = apply_quant_noise_(
                nn.Linear(embed_dim, embed_dim, bias=False),
                cfg.quant_noise.pq,
                cfg.quant_noise.pq_block_size,
            )
        else:
            self.quant_noise = None

        self.project_in_dim = (
            Linear(input_embed_dim, embed_dim, bias=False)
            if embed_dim != input_embed_dim
            else None
        )
        self.embed_positions = (
            PositionalEmbedding(
                self.max_target_positions,
                embed_dim,
                self.padding_idx,
                learned=cfg.decoder.learned_pos,
            )
            if not cfg.no_token_positional_embeddings
            else None
        )
        if cfg.layernorm_embedding:
            self.layernorm_embedding = LayerNorm(embed_dim, export=cfg.export)
        else:
            self.layernorm_embedding = None

        self.cross_self_attention = cfg.cross_self_attention

        if self.decoder_layerdrop > 0.0:
            self.layers = LayerDropModuleList(p=self.decoder_layerdrop)
        else:
            self.layers = nn.ModuleList([])
        self.layers.extend(
            [
                self.build_decoder_layer(cfg, no_encoder_attn)
                for _ in range(cfg.decoder.layers)
            ]
        )
        self.num_layers = len(self.layers)

        if cfg.decoder.normalize_before and not cfg.no_decoder_final_norm:
            self.layer_norm = LayerNorm(embed_dim, export=cfg.export)
        else:
            self.layer_norm = None

        self.project_out_dim = (
            Linear(embed_dim, self.output_embed_dim, bias=False)
            if embed_dim != self.output_embed_dim and not cfg.tie_adaptive_weights
            else None
        )

        self.adaptive_softmax = None
        self.output_projection = output_projection
        if self.output_projection is None:
            self.build_output_projection(cfg, dictionary, embed_tokens)

    def build_output_projection(self, cfg, dictionary, embed_tokens):
        if cfg.adaptive_softmax_cutoff is not None:
            self.adaptive_softmax = AdaptiveSoftmax(
                len(dictionary),
                self.output_embed_dim,
                utils.eval_str_list(cfg.adaptive_softmax_cutoff, type=int),
                dropout=cfg.adaptive_softmax_dropout,
                adaptive_inputs=embed_tokens if cfg.tie_adaptive_weights else None,
                factor=cfg.adaptive_softmax_factor,
                tie_proj=cfg.tie_adaptive_proj,
            )
        elif self.share_input_output_embed:
            self.output_projection = nn.Linear(
                self.embed_tokens.weight.shape[1],
                self.embed_tokens.weight.shape[0],
                bias=False,
            )
            self.output_projection.weight = self.embed_tokens.weight
        else:
            self.output_projection = nn.Linear(
                self.output_embed_dim, len(dictionary), bias=False
            )
            nn.init.normal_(
                self.output_projection.weight, mean=0, std=self.output_embed_dim**-0.5
            )
        num_base_layers = cfg.base_layers
        for i in range(num_base_layers):
            self.layers.insert(
                ((i + 1) * cfg.decoder.layers) // (num_base_layers + 1),
                BaseLayer(cfg),
            )

    def build_decoder_layer(self, cfg, no_encoder_attn=False):
        layer = transformer_layer.TransformerDecoderLayerBase(cfg, no_encoder_attn)
        checkpoint = cfg.checkpoint_activations
        if checkpoint:
            offload_to_cpu = cfg.offload_activations
            layer = checkpoint_wrapper(layer, offload_to_cpu=offload_to_cpu)
        # if we are checkpointing, enforce that FSDP always wraps the
        # checkpointed layer, regardless of layer size
        min_params_to_wrap = cfg.min_params_to_wrap if not checkpoint else 0
        layer = fsdp_wrap(layer, min_num_params=min_params_to_wrap)
        return layer

    def forward(
        self,
        prev_output_tokens,
        encoder_out: Optional[Dict[str, List[Tensor]]] = None,
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
        features_only: bool = False,
        full_context_alignment: bool = False,
        alignment_layer: Optional[int] = None,
        alignment_heads: Optional[int] = None,
        src_lengths: Optional[Any] = None,
        return_all_hiddens: bool = False,
    ):
        """
        Args:
            prev_output_tokens (LongTensor): previous decoder outputs of shape
                `(batch, tgt_len)`, for teacher forcing
            encoder_out (optional): output from the encoder, used for
                encoder-side attention, should be of size T x B x C
            incremental_state (dict): dictionary used for storing state during
                :ref:`Incremental decoding`
            features_only (bool, optional): only return features without
                applying output layer (default: False).
            full_context_alignment (bool, optional): don't apply
                auto-regressive mask to self-attention (default: False).

        Returns:
            tuple:
                - the decoder's output of shape `(batch, tgt_len, vocab)`
                - a dictionary with any model-specific outputs
        """

        x, extra = self.extract_features(
            prev_output_tokens,
            encoder_out=encoder_out,
            incremental_state=incremental_state,
            full_context_alignment=full_context_alignment,
            alignment_layer=alignment_layer,
            alignment_heads=alignment_heads,
        )

        if not features_only:
            x = self.output_layer(x)
        return x, extra

    def extract_features(
        self,
        prev_output_tokens,
        encoder_out: Optional[Dict[str, List[Tensor]]],
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
        full_context_alignment: bool = False,
        alignment_layer: Optional[int] = None,
        alignment_heads: Optional[int] = None,
    ):
        return self.extract_features_scriptable(
            prev_output_tokens,
            encoder_out,
            incremental_state,
            full_context_alignment,
            alignment_layer,
            alignment_heads,
        )

    """
    A scriptable subclass of this class has an extract_features method and calls
    super().extract_features, but super() is not supported in torchscript. A copy of
    this function is made to be used in the subclass instead.
    """

    def extract_features_scriptable(
        self,
        prev_output_tokens,
        encoder_out: Optional[Dict[str, List[Tensor]]],
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
        full_context_alignment: bool = False,
        alignment_layer: Optional[int] = None,
        alignment_heads: Optional[int] = None,
    ):
        """
        Similar to *forward* but only return features.

        Includes several features from "Jointly Learning to Align and
        Translate with Transformer Models" (Garg et al., EMNLP 2019).

        Args:
            full_context_alignment (bool, optional): don't apply
                auto-regressive mask to self-attention (default: False).
            alignment_layer (int, optional): return mean alignment over
                heads at this layer (default: last layer).
            alignment_heads (int, optional): only average alignment over
                this many heads (default: all heads).

        Returns:
            tuple:
                - the decoder's features of shape `(batch, tgt_len, embed_dim)`
                - a dictionary with any model-specific outputs
        """
        bs, slen = prev_output_tokens.size()
        if alignment_layer is None:
            alignment_layer = self.num_layers - 1

        enc: Optional[Tensor] = None
        padding_mask: Optional[Tensor] = None
        if encoder_out is not None and len(encoder_out["encoder_out"]) > 0:
            enc = encoder_out["encoder_out"][0]
        if encoder_out is not None and len(encoder_out["encoder_padding_mask"]) > 0:
            padding_mask = encoder_out["encoder_padding_mask"][0]

        # embed positions
        positions = None
        if self.embed_positions is not None:
            positions = self.embed_positions(
                prev_output_tokens, incremental_state=incremental_state
            )

        if incremental_state is not None:
            prev_output_tokens = prev_output_tokens[:, -1:]
            if positions is not None:
                positions = positions[:, -1:]

        # Prevent torchscript exporting issue for dynamic quant embedding
        prev_output_tokens = prev_output_tokens.contiguous()
        # embed tokens and positions
        x = self.embed_scale * self.embed_tokens(prev_output_tokens)

        if self.quant_noise is not None:
            x = self.quant_noise(x)

        if self.project_in_dim is not None:
            x = self.project_in_dim(x)

        if positions is not None:
            x += positions

        if self.layernorm_embedding is not None:
            x = self.layernorm_embedding(x)

        x = self.dropout_module(x)

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)

        self_attn_padding_mask: Optional[Tensor] = None
        if self.cross_self_attention or prev_output_tokens.eq(self.padding_idx).any():
            self_attn_padding_mask = prev_output_tokens.eq(self.padding_idx)

        # decoder layers
        attn: Optional[Tensor] = None
        inner_states: List[Optional[Tensor]] = [x]
        for idx, layer in enumerate(self.layers):
            if incremental_state is None and not full_context_alignment:  # dkq
                self_attn_mask = self.buffered_future_mask(x)
            else:
                self_attn_mask = None

            x, layer_attn, _ = layer(
                x,
                enc,
                padding_mask,
                incremental_state,
                self_attn_mask=self_attn_mask,
                self_attn_padding_mask=self_attn_padding_mask,
                need_attn=bool((idx == alignment_layer)),
                need_head_weights=bool((idx == alignment_layer)),
            )
            inner_states.append(x)
            if layer_attn is not None and idx == alignment_layer:
                attn = layer_attn.float().to(x)

        if attn is not None:
            if alignment_heads is not None:
                attn = attn[:alignment_heads]

            # average probabilities over heads
            attn = attn.mean(dim=0)

        if self.layer_norm is not None:
            x = self.layer_norm(x)

        # T x B x C -> B x T x C
        x = x.transpose(0, 1)

        if self.project_out_dim is not None:
            x = self.project_out_dim(x)

        return x, {"attn": [attn], "inner_states": inner_states}

    def output_layer(self, features):
        """Project features to the vocabulary size."""
        if self.adaptive_softmax is None:
            # project back to size of vocabulary
            return self.output_projection(features)
        else:
            return features

    def max_positions(self):
        """Maximum output length supported by the decoder."""
        if self.embed_positions is None:
            return self.max_target_positions
        return min(self.max_target_positions, self.embed_positions.max_positions)

    def buffered_future_mask(self, tensor):
        dim = tensor.size(0)
        # self._future_mask.device != tensor.device is not working in TorchScript. This is a workaround.
        if (
            self._future_mask.size(0) == 0
            or (not self._future_mask.device == tensor.device)
            or self._future_mask.size(0) < dim
        ):
            self._future_mask = torch.triu(
                utils.fill_with_neg_inf(torch.zeros([dim, dim])), 1
            )
        self._future_mask = self._future_mask.to(tensor)
        return self._future_mask[:dim, :dim]

    def upgrade_state_dict_named(self, state_dict, name):
        """Upgrade a (possibly old) state dict for new versions of fairseq."""
        if f"{name}.output_projection.weight" not in state_dict:
            if self.share_input_output_embed:
                embed_out_key = f"{name}.embed_tokens.weight"
            else:
                embed_out_key = f"{name}.embed_out"
            if embed_out_key in state_dict:
                state_dict[f"{name}.output_projection.weight"] = state_dict[
                    embed_out_key
                ]
                if not self.share_input_output_embed:
                    del state_dict[embed_out_key]

        for i in range(self.num_layers):
            # update layer norms
            layer_norm_map = {
                "0": "self_attn_layer_norm",
                "1": "encoder_attn_layer_norm",
                "2": "final_layer_norm",
            }
            for old, new in layer_norm_map.items():
                for m in ("weight", "bias"):
                    k = "{}.layers.{}.layer_norms.{}.{}".format(name, i, old, m)
                    if k in state_dict:
                        state_dict["{}.layers.{}.{}.{}".format(name, i, new, m)] = (
                            state_dict[k]
                        )
                        del state_dict[k]

        version_key = "{}.version".format(name)
        if utils.item(state_dict.get(version_key, torch.Tensor([1]))[0]) <= 2:
            # earlier checkpoints did not normalize after the stack of layers
            self.layer_norm = None
            self.normalize = False
            state_dict[version_key] = torch.Tensor([1])

        return state_dict


def Linear(in_features, out_features, bias=True):
    m = nn.Linear(in_features, out_features, bias)
    nn.init.xavier_uniform_(m.weight)
    if bias:
        nn.init.constant_(m.bias, 0.0)
    return m


class TransformerDecoder(TransformerDecoderBase):
    def __init__(
        self,
        args,
        dictionary,
        embed_tokens,
        no_encoder_attn=False,
        output_projection=None,
    ):
        self.args = args
        super().__init__(
            TransformerConfig.from_namespace(args),
            dictionary,
            embed_tokens,
            no_encoder_attn=no_encoder_attn,
            output_projection=output_projection,
        )

    def build_output_projection(self, args, dictionary, embed_tokens):
        super().build_output_projection(
            TransformerConfig.from_namespace(args), dictionary, embed_tokens
        )

    def build_decoder_layer(self, args, no_encoder_attn=False):
        return super().build_decoder_layer(
            TransformerConfig.from_namespace(args), no_encoder_attn=no_encoder_attn
        )


class RotaryMultiheadAttention(MultiheadAttention):
    def __init__(self, embed_dim, num_heads, dropout=0.0, bias=True):
        super().__init__(embed_dim, num_heads)
        del self.dropout_module
        del self.k_proj
        del self.v_proj
        del self.q_proj
        del self.out_proj

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        assert (
            self.head_dim * num_heads == self.embed_dim
        ), "embed_dim must be divisible by num_heads"
        self.scaling = self.head_dim**-0.5

        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

    def forward(
        self,
        query,
        key,
        value,
        key_padding_mask=None,
        self_attn_mask=None,
        position=None,
        incremental_state=None,
        static_kv=False,
        need_weights=False,
    ):
        """
        Parameter description:
         - query: (B, tgt_len, embed_dim)
         - key, value: (B, src_len, embed_dim)
         - incremental_state: used for KV cache; if not None, the cache will be updated and used.
        """

        B, tgt_len, _ = query.size()
        B, src_len, _ = key.size()

        # Project and scale q
        q = self.q_proj(query) * self.scaling  # (B, tgt_len, embed_dim)
        k = self.k_proj(key)  # (B, src_len, embed_dim)
        v = self.v_proj(value)  # (B, src_len, embed_dim)

        # Reshape into multi-head format
        q = q.view(
            B, tgt_len, self.num_heads, self.head_dim
        )  # (B, tgt_len, num_heads, head_dim)
        k = k.view(
            B, src_len, self.num_heads, self.head_dim
        )  # (B, src_len, num_heads, head_dim)
        v = v.view(B, src_len, self.num_heads, self.head_dim).transpose(
            1, 2
        )  # (B, num_heads, src_len, head_dim)

        freqs_cis = self._compute_freqs_cis_batch(position, query.device)
        q = self._apply_rotary_emb_batch(q, freqs_cis[:, -tgt_len:]).transpose(
            1, 2
        )  # (B, num_heads, src_len, head_dim)
        k = self._apply_rotary_emb_batch(k, freqs_cis).transpose(
            1, 2
        )  # (B, num_heads, src_len, head_dim)

        # Handle KV cache（incremental_state）
        if incremental_state is not None:
            saved_state = self._get_input_buffer(incremental_state)
            prev_k = saved_state.get("prev_key", None)
            prev_v = saved_state.get("prev_value", None)
            if prev_k is not None and prev_v is not None:
                try:
                    k = torch.cat([prev_k, k], dim=2)
                except Exception as e:
                    print(f"Error: {e}")
                    print(f"prev_k shape: {prev_k.shape}, k shape: {k.shape}")
                v = torch.cat([prev_v, v], dim=2)
            saved_state["prev_key"] = k
            saved_state["prev_value"] = v
            assert incremental_state is not None
            incremental_state = self._set_input_buffer(incremental_state, saved_state)

        # Compute attention scores
        attn_scores = torch.matmul(
            q, k.transpose(-2, -1)
        )  # (B, num_heads, tgt_len, src_len)

        # # Apply the self-attention mask if available (future tokens should not be attended to)
        if self_attn_mask is not None:
            attn_scores = attn_scores + self_attn_mask.unsqueeze(0).unsqueeze(
                1
            )  # (B, num_heads, tgt_len, src_len)

        if key_padding_mask is not None:
            attn_scores = attn_scores.masked_fill(
                key_padding_mask.unsqueeze(1).unsqueeze(2), float("-inf")
            )

        attn_probs = F.softmax(attn_scores, dim=-1)
        attn_probs = F.dropout(attn_probs, p=self.dropout, training=self.training)

        # Weighted sum
        attn_output = torch.matmul(attn_probs, v)  # (B, num_heads, tgt_len, head_dim)
        attn_output = (
            attn_output.transpose(1, 2).contiguous().view(B, tgt_len, self.embed_dim)
        )
        output = self.out_proj(attn_output)

        if need_weights:
            return output, attn_probs
        else:
            return output, None

    def _compute_freqs_cis_batch(self, pos: torch.Tensor, device: torch.device):
        theta = 10000.0
        freqs = 1.0 / (
            theta
            ** (
                torch.arange(0, self.head_dim, 2, device=device).float() / self.head_dim
            )
        )
        freqs = torch.einsum(
            "bi,j->bij", pos, freqs
        )  # (batch_size, seq_len, head_dim//2)
        return torch.polar(
            torch.ones_like(freqs), freqs
        )  # (batch_size, seq_len, head_dim//2)

    def _apply_rotary_emb_batch(
        self, x: torch.Tensor, freqs_cis: torch.Tensor
    ) -> torch.Tensor:
        dtype = x.dtype
        x = x.float().view(
            *x.shape[:-1], -1, 2
        )  # (batch_size, seq_len, n_heads, head_dim//2, 2)
        x = torch.view_as_complex(x)
        freqs_cis = freqs_cis.unsqueeze(2)  # (batch_size, seq_len, 1, head_dim//2)
        y = torch.view_as_real(x * freqs_cis).flatten(3)
        return y.to(dtype)


class GQARotaryMultiheadAttention(MultiheadAttention):
    def __init__(self, embed_dim, num_heads, dropout=0.0, bias=True, num_kv_heads=None):
        super().__init__(embed_dim, num_heads)
        del self.dropout_module
        del self.k_proj
        del self.v_proj
        del self.q_proj
        del self.out_proj

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        assert (
            self.head_dim * num_heads == self.embed_dim
        ), "embed_dim must be divisible by num_heads"
        self.scaling = self.head_dim**-0.5

        # Set the number of KV heads; defaults to num_heads (i.e., degenerates to standard MHA)
        self.num_kv_heads = num_heads if num_kv_heads is None else num_kv_heads
        assert (
            self.num_heads % self.num_kv_heads == 0
        ), "num_heads must be divisible by num_kv_heads"
        self.num_queries_per_kv = self.num_heads // self.num_kv_heads

        # Projection layer
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.k_proj = nn.Linear(embed_dim, self.num_kv_heads * self.head_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, self.num_kv_heads * self.head_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

    def forward(
        self,
        query,
        key,
        value,
        key_padding_mask=None,
        self_attn_mask=None,
        position=None,
        incremental_state=None,
        static_kv=False,
        need_weights=False,
    ):
        B, tgt_len, _ = query.size()
        B, src_len, _ = key.size()

        # Project and scale q
        q = self.q_proj(query) * self.scaling  # (B, tgt_len, embed_dim)
        k = self.k_proj(key)  # (B, src_len, num_kv_heads * head_dim)
        v = self.v_proj(value)  # (B, src_len, num_kv_heads * head_dim)

        # Reshape to multi-head format
        q = q.view(
            B, tgt_len, self.num_heads, self.head_dim
        )  # (B, tgt_len, num_heads, head_dim)
        k = k.view(
            B, src_len, self.num_kv_heads, self.head_dim
        )  # (B, src_len, num_kv_heads, head_dim)
        v = v.view(B, src_len, self.num_kv_heads, self.head_dim).transpose(
            1, 2
        )  # (B, src_len, num_kv_heads, head_dim)

        # Apply rotary positional encoding
        freqs_cis = self._compute_freqs_cis_batch(position, query.device)
        q = self._apply_rotary_emb_batch(q, freqs_cis[:, -tgt_len:]).transpose(
            1, 2
        )  # (B, num_heads, tgt_len, head_dim)
        k = self._apply_rotary_emb_batch(k, freqs_cis).transpose(
            1, 2
        )  # (B, num_kv_heads, src_len, head_dim)

        # Handle KV cache (incremental_state)
        if incremental_state is not None:
            saved_state = self._get_input_buffer(incremental_state)
            prev_k = saved_state.get("prev_key", None)
            prev_v = saved_state.get("prev_value", None)
            if prev_k is not None and prev_v is not None:
                k = torch.cat([prev_k, k], dim=2)
                v = torch.cat([prev_v, v], dim=2)
            saved_state["prev_key"] = k
            saved_state["prev_value"] = v
            incremental_state = self._set_input_buffer(incremental_state, saved_state)

        # Repeat k and v to match the number of heads in q
        if self.num_kv_heads != self.num_heads:
            k = k.repeat_interleave(
                self.num_queries_per_kv, dim=1
            )  # (B, num_heads, src_len, head_dim)
            v = v.repeat_interleave(
                self.num_queries_per_kv, dim=1
            )  # (B, num_heads, src_len, head_dim)

        # Compute attention scores
        attn_scores = torch.matmul(
            q, k.transpose(-2, -1)
        )  # (B, num_heads, tgt_len, src_len)

        # Apply attention mask
        if self_attn_mask is not None:
            attn_scores = attn_scores + self_attn_mask.unsqueeze(0).unsqueeze(1)

        if key_padding_mask is not None:
            attn_scores = attn_scores.masked_fill(
                key_padding_mask.unsqueeze(1).unsqueeze(2), float("-inf")
            )

        attn_probs = F.softmax(attn_scores, dim=-1)
        attn_probs = F.dropout(attn_probs, p=self.dropout, training=self.training)

        # Weighted sum
        attn_output = torch.matmul(attn_probs, v)  # (B, num_heads, tgt_len, head_dim)
        attn_output = (
            attn_output.transpose(1, 2).contiguous().view(B, tgt_len, self.embed_dim)
        )
        output = self.out_proj(attn_output)

        if need_weights:
            return output, attn_probs
        else:
            return output, None

    # Keep the original rotary embedding computation unchanged
    def _compute_freqs_cis_batch(self, pos: torch.Tensor, device: torch.device):
        theta = 10000.0
        freqs = 1.0 / (
            theta
            ** (
                torch.arange(0, self.head_dim, 2, device=device).float() / self.head_dim
            )
        )
        freqs = torch.einsum("bi,j->bij", pos, freqs)
        return torch.polar(torch.ones_like(freqs), freqs)

    def _apply_rotary_emb_batch(
        self, x: torch.Tensor, freqs_cis: torch.Tensor
    ) -> torch.Tensor:
        dtype = x.dtype
        x = x.float().view(*x.shape[:-1], -1, 2)
        x = torch.view_as_complex(x)
        freqs_cis = freqs_cis.unsqueeze(2)  # (batch_size, seq_len, 1, head_dim//2)
        y = torch.view_as_real(x * freqs_cis).flatten(3)
        return y.to(dtype)


class MultiheadLatentAttention(MultiheadAttention):
    """
    Multi-head Latent Attention (MLA) for Fairseq.

    Args:
        embed_dim (int): Embedding dimension.
        num_heads (int): Number of attention heads.
        dropout (float): Dropout probability.
        bias (bool): Whether to add bias to linear projections.
        q_lora_rank (int): Rank for low-rank query projection.
        kv_lora_rank (int): Rank for low-rank key/value projection.
        qk_nope_head_dim (int): Dimensionality of non-positional query/key projections.
        qk_rope_head_dim (int): Dimensionality of rotary-positional query/key projections.
        v_head_dim (int): Dimensionality of value projections.
    """

    def __init__(
        self,
        embed_dim,
        num_heads,
        dropout=0.0,
        bias=False,
        q_lora_rank=0,
        kv_lora_rank=256,
        qk_nope_head_dim=64,
        qk_rope_head_dim=32,
        v_head_dim=64,
    ):
        super().__init__(embed_dim, num_heads)
        del self.dropout_module
        del self.k_proj
        del self.v_proj
        del self.q_proj
        del self.out_proj

        self.num_heads = num_heads
        self.dropout = dropout
        self.q_lora_rank = q_lora_rank
        self.kv_lora_rank = kv_lora_rank
        self.qk_nope_head_dim = qk_nope_head_dim
        self.qk_rope_head_dim = qk_rope_head_dim
        self.qk_head_dim = qk_nope_head_dim + qk_rope_head_dim
        self.v_head_dim = v_head_dim

        # Query projection
        if self.q_lora_rank == 0:
            self.wq = nn.Linear(embed_dim, num_heads * self.qk_head_dim, bias=bias)
        else:
            self.wq_a = nn.Linear(embed_dim, q_lora_rank, bias=bias)
            self.q_norm = nn.LayerNorm(q_lora_rank)
            self.wq_b = nn.Linear(q_lora_rank, num_heads * self.qk_head_dim, bias=bias)

        # Key/Value projection
        self.wkv_a = nn.Linear(embed_dim, kv_lora_rank + qk_rope_head_dim, bias=bias)
        self.kv_norm = nn.LayerNorm(kv_lora_rank)
        self.wkv_b = nn.Linear(
            kv_lora_rank, num_heads * (qk_nope_head_dim + v_head_dim), bias=bias
        )

        # Output projection
        self.wo = nn.Linear(num_heads * v_head_dim, embed_dim, bias=bias)

        # Softmax scaling
        self.softmax_scale = self.qk_head_dim**-0.5

    def forward(
        self,
        query,
        key,
        value,
        key_padding_mask=None,
        self_attn_mask=None,
        position=None,
        incremental_state=None,
        need_weights=False,
    ):
        """
        Args:
            query (torch.Tensor): Query tensor of shape (batch_size, seq_len, embed_dim).
            key (torch.Tensor): Key tensor of shape (batch_size, seq_len, embed_dim).
            value (torch.Tensor): Value tensor of shape (batch_size, seq_len, embed_dim).
            key_padding_mask (torch.Tensor): Mask for padding tokens of shape (batch_size, seq_len).
            self_attn_mask (torch.Tensor): Mask for self-attention of shape (seq_len, seq_len).
            incremental_state (dict): Dictionary for caching key and value during incremental decoding.
            need_weights (bool): Whether to return attention weights.

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, seq_len, embed_dim).
            Optional[torch.Tensor]: Attention weights if `need_weights` is True.
        """
        bsz, seqlen, _ = query.size()

        # Query projection
        if self.q_lora_rank == 0:
            q = self.wq(query)
        else:
            q = self.wq_b(self.q_norm(self.wq_a(query)))
        q = q.view(bsz, seqlen, self.num_heads, self.qk_head_dim)
        q_nope, q_pe = torch.split(
            q, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1
        )

        # Rotary positional embedding for query
        freqs_cis = self._compute_freqs_cis_batch(position, query.device)
        q_pe = self._apply_rotary_emb_batch(q_pe, freqs_cis[:, -seqlen:])

        # Key/Value projection
        kv = self.wkv_a(key)
        kv, k_pe = torch.split(kv, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)
        k_pe = self._apply_rotary_emb_batch(k_pe.unsqueeze(2), freqs_cis)
        kv_norm = self.kv_norm(kv)
        k_pe = k_pe.squeeze(2)

        # Update incremental state
        if incremental_state is not None:
            saved_state = self._get_input_buffer(incremental_state)
            prev_kv = saved_state.get("prev_kv", None)
            prev_k_pe = saved_state.get("prev_k_pe", None)
            if prev_kv is not None and prev_k_pe is not None:
                kv_norm = torch.cat([prev_kv, kv_norm], dim=1)
                k_pe = torch.cat([prev_k_pe, k_pe], dim=1)
            saved_state["prev_kv"] = kv_norm
            saved_state["prev_k_pe"] = k_pe
            incremental_state = self._set_input_buffer(incremental_state, saved_state)

        # Compute attention scores
        wkv_b = self.wkv_b.weight
        wkv_b = wkv_b.view(self.num_heads, -1, self.kv_lora_rank)
        q_nope_proj = torch.einsum(
            "bshd,hdc->bshc", q_nope, wkv_b[:, : self.qk_nope_head_dim]
        )

        scores = (
            torch.einsum("bshc,btc->bsht", q_nope_proj, kv_norm)
            + torch.einsum("bshr,btr->bsht", q_pe, k_pe)
        ) * self.softmax_scale

        # Apply masks
        if self_attn_mask is not None:
            scores = scores + self_attn_mask.unsqueeze(0).unsqueeze(2)
        if key_padding_mask is not None:
            scores = scores.masked_fill(
                key_padding_mask.unsqueeze(1).unsqueeze(2), float("-inf")
            )

        # Compute attention weights
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = F.dropout(attn_weights, p=self.dropout, training=self.training)

        # Weighted sum of values
        x = torch.einsum("bsht,btc->bshc", attn_weights, kv_norm)
        x = torch.einsum("bshc,hdc->bshd", x, wkv_b[:, -self.v_head_dim :])
        x = self.wo(x.flatten(2))

        if need_weights:
            return x, attn_weights
        else:
            return x, None

    def _compute_freqs_cis_batch(self, pos: torch.Tensor, device: torch.device):
        theta = 10000.0
        freqs = 1.0 / (
            theta
            ** (
                torch.arange(0, self.qk_rope_head_dim, 2, device=device).float()
                / self.qk_rope_head_dim
            )
        )
        freqs = torch.einsum(
            "bi,j->bij", pos, freqs
        )  # (batch_size, seq_len, head_dim//2)
        return torch.polar(
            torch.ones_like(freqs), freqs
        )  # (batch_size, seq_len, head_dim//2)

    def _apply_rotary_emb_batch(
        self, x: torch.Tensor, freqs_cis: torch.Tensor
    ) -> torch.Tensor:
        dtype = x.dtype
        x = x.float().view(
            *x.shape[:-1], -1, 2
        )  # (batch_size, seq_len, n_heads, head_dim//2, 2)
        x = torch.view_as_complex(x)
        freqs_cis = freqs_cis.unsqueeze(2)  # (batch_size, seq_len, 1, head_dim//2)
        y = torch.view_as_real(x * freqs_cis).flatten(3)
        return y.to(dtype)


class HyperNetwork(nn.Module):
    def __init__(self, d, down_rate, low_rank=4):
        """
        d: 特征维度
        """
        super().__init__()
        self.d = d
        self.down_rate = down_rate

        # Linear layers
        self.fc_c = nn.Linear(d, int(d / low_rank))
        self.fc_p = nn.Linear(d, int(d / low_rank))

    def positional_encoding(self, T, pos=0):
        """
        Generate positional embedding with shape (1, T, d)
        """
        d = self.d
        position = torch.arange(pos, pos + T, dtype=torch.float).unsqueeze(1)  # (T, 1)
        div_term = torch.exp(
            torch.arange(0, d, 2, dtype=torch.float)
            * (-torch.log(torch.tensor(10000.0)) / d)
        )  # (d/2,)

        pe = torch.zeros(T, d)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        return pe.unsqueeze(0)  # (1, T, d)

    def forward(
        self, T, t, device, train=True, low_rank=False, T_input=None, t_input=None
    ):
        """
        Input:
            T: sequence length (scalar)
            t: sequence length (scalar)
        Output:
            W: weight matrix of shape (B, T, T) or (B, 1, 1)
        """

        if not train and T_input.shape[1] == 1:

            if T_input is not None:
                P = T_input
            else:
                P = self.positional_encoding(1, pos=T - 1).to(device)

            if t_input is not None:
                C = t_input
            else:
                C = self.positional_encoding(1, pos=t - 1).to(device)

            C2 = self.fc_c(C)  # (B, 1, d)
            P2 = self.fc_p(P)  # (B, 1, d)

            # Matrix multiplication to obtain (1, t, T)
            if T_input is not None:
                W = torch.bmm(
                    C2.expand(P2.shape[0], -1, -1), P2.transpose(1, 2)
                )  # (B, 1, 1)
            elif t_input is not None:
                W = torch.bmm(
                    C2, P2.transpose(1, 2).expand(C2.shape[0], -1, -1)
                )  # (B, 1, 1)
            else:
                W = torch.bmm(C2, P2.transpose(1, 2))  # (B, 1, 1)

            return torch.sigmoid(W)

        # Generate positional embedding (B, T, d)
        if T_input is not None:
            P = T_input
        else:
            P = self.positional_encoding(T).to(device)

        if t_input is not None:
            C = t_input
        else:
            C = self.positional_encoding(t).to(device).to(P.dtype)  # (B, t, d)
            if train:
                C = C.repeat_interleave(self.down_rate, dim=1)[:, :T]  # (B, T, d)

        # linear transform
        C2 = self.fc_c(C)  # (B, t or T, d)
        P2 = self.fc_p(P)  # (B, T, d)

        # Perform matrix multiplication to obtain (B, T, T)
        if T_input is not None:
            W = torch.bmm(
                C2.expand(P2.shape[0], -1, -1), P2.transpose(1, 2)
            )  # (B, t or T, T)
        elif t_input is not None:
            W = torch.bmm(
                C2, P2.transpose(1, 2).expand(C2.shape[0], -1, -1)
            )  # (B, t or T, T)
        else:
            W = torch.bmm(C2, P2.transpose(1, 2))  # (B, t or T, T)

        return torch.sigmoid(W)


class MultiheadTemporalLatentAttention(MultiheadAttention):
    """
    Multi-head Temporal Latent Attention (MTLA) for Fairseq.

    Args:
        embed_dim (int): Embedding dimension.
        num_heads (int): Number of attention heads.
        dropout (float): Dropout probability.
        bias (bool): Whether to add bias to linear projections.
        q_lora_rank (int): Rank for low-rank query projection.
        kv_lora_rank (int): Rank for low-rank key/value projection.
        qk_nope_head_dim (int): Dimensionality of non-positional query/key projections.
        qk_rope_head_dim (int): Dimensionality of rotary-positional query/key projections.
        v_head_dim (int): Dimensionality of value projections.
        down_rate (int): Temporal compression rate of MTLA
    """

    def __init__(
        self,
        embed_dim,
        num_heads,
        dropout=0.0,
        bias=False,
        q_lora_rank=0,
        kv_lora_rank=256,
        qk_nope_head_dim=64,
        qk_rope_head_dim=32,
        v_head_dim=64,
        down_rate=2,
    ):
        # super().__init__()
        super().__init__(embed_dim, num_heads)
        del self.dropout_module
        del self.k_proj
        del self.v_proj
        del self.q_proj
        del self.out_proj

        self.num_heads = num_heads
        self.dropout = dropout
        self.q_lora_rank = q_lora_rank
        self.kv_lora_rank = kv_lora_rank
        self.qk_nope_head_dim = qk_nope_head_dim
        self.qk_rope_head_dim = qk_rope_head_dim
        self.qk_head_dim = qk_nope_head_dim + qk_rope_head_dim
        self.v_head_dim = v_head_dim
        self.down_rate = down_rate

        # Query projection
        if self.q_lora_rank == 0:
            self.wq = nn.Linear(embed_dim, num_heads * self.qk_head_dim, bias=bias)
        else:
            self.wq_a = nn.Linear(embed_dim, q_lora_rank, bias=bias)
            self.q_norm = nn.LayerNorm(q_lora_rank)
            self.wq_b = nn.Linear(q_lora_rank, num_heads * self.qk_head_dim, bias=bias)

        # Key/Value projection
        self.wkv_a = nn.Linear(embed_dim, kv_lora_rank + qk_rope_head_dim, bias=bias)
        self.kv_norm = nn.LayerNorm(kv_lora_rank)
        self.wkv_b = nn.Linear(
            kv_lora_rank, num_heads * (qk_nope_head_dim + v_head_dim), bias=bias
        )

        # Output projection
        self.wo = nn.Linear(num_heads * v_head_dim, embed_dim, bias=bias)

        # Softmax scaling
        self.softmax_scale = self.qk_head_dim**-0.5

        self.hypernet_down = HyperNetwork(d=kv_lora_rank, down_rate=down_rate)

    def forward(
        self,
        query,
        key,
        value,
        key_padding_mask=None,
        self_attn_mask=None,
        position=None,
        incremental_state=None,
        need_weights=False,
    ):
        """
        Args:
            query (torch.Tensor): Query tensor of shape (batch_size, seq_len, embed_dim).
            key (torch.Tensor): Key tensor of shape (batch_size, seq_len, embed_dim).
            value (torch.Tensor): Value tensor of shape (batch_size, seq_len, embed_dim).
            key_padding_mask (torch.Tensor): Mask for padding tokens of shape (batch_size, seq_len).
            self_attn_mask (torch.Tensor): Mask for self-attention of shape (seq_len, seq_len).
            incremental_state (dict): Dictionary for caching key and value during incremental decoding.
            need_weights (bool): Whether to return attention weights.

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, seq_len, embed_dim).
            Optional[torch.Tensor]: Attention weights if `need_weights` is True.
        """
        bsz, seqlen, _ = query.size()

        # Query projection
        if self.q_lora_rank == 0:
            q = self.wq(query)
        else:
            q = self.wq_b(self.q_norm(self.wq_a(query)))
        q = q.view(bsz, seqlen, self.num_heads, self.qk_head_dim)
        q_nope, q_pe = torch.split(
            q, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1
        )

        # Rotary positional embedding for query
        if incremental_state is not None:
            saved_state = self._get_input_buffer(incremental_state)
            start_pos = (
                saved_state.get("infer_steps", None)[0]
                if "infer_steps" in saved_state
                else key.shape[1] - 1
            )  # 0
        else:
            start_pos = 0

        freqs_cis = self._compute_freqs_cis_batch(position, query.device)
        q_pe = self._apply_rotary_emb_batch(q_pe, freqs_cis[:, -seqlen:])

        kv = self.wkv_a(key)
        kv, k_pe = torch.split(kv, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)
        k_pe = self._apply_rotary_emb_batch(k_pe.unsqueeze(2), freqs_cis)
        kv_norm = self.kv_norm(kv)  # B, T, d
        k_pe = k_pe.squeeze(2)

        if incremental_state is not None:

            saved_state = self._get_input_buffer(incremental_state)
            prev_kv_t = saved_state.get("prev_kv_t", None)
            prev_k_pe = saved_state.get("prev_k_pe", None)
            infer_steps = saved_state.get("infer_steps", None)

            T = start_pos + 1
            t = math.ceil(T / self.down_rate)
            T_remain = T % self.down_rate

            w_tT = self.hypernet_down(
                T, t, kv_norm.device, train=False, T_input=kv_norm
            )  # B, 1, 1

            if prev_kv_t is not None and prev_k_pe is not None:
                if T_remain != 1:
                    prev_kv_t[:, -1:] += kv_norm * w_tT  # Update KV cache
                    prev_k_pe[:, -1:] = k_pe  # Update
                else:
                    prev_kv_t = torch.cat([prev_kv_t, kv_norm * w_tT], dim=1)  # Concat
                    prev_k_pe = torch.cat([prev_k_pe, k_pe], dim=1)  # Concat
                infer_steps = infer_steps + 1
            else:
                # Correspond to the first token inference
                if key.shape[1] != 1:
                    zero_mask = self.generate_chunk_mask(T, self.down_rate).to(
                        k_pe.device
                    )
                    indices = list(range(self.down_rate - 1, T, self.down_rate))
                    if T - 1 not in indices:
                        indices.append(T - 1)
                    zero_mask = zero_mask[indices].unsqueeze(0)
                    prev_kv_t = torch.matmul(w_tT * zero_mask, kv_norm)
                    prev_k_pe = k_pe[:, indices]
                else:
                    prev_kv_t = kv_norm * w_tT
                    prev_k_pe = k_pe
                infer_steps = kv_norm.new_zeros(kv_norm.shape[0]) + T

            saved_state["prev_kv_t"] = prev_kv_t
            saved_state["prev_k_pe"] = prev_k_pe
            saved_state["infer_steps"] = infer_steps
            incremental_state = self._set_input_buffer(incremental_state, saved_state)

            wkv_b = self.wkv_b.weight
            wkv_b = wkv_b.view(self.num_heads, -1, self.kv_lora_rank)
            q_nope_proj = torch.einsum(
                "bshd,hdc->bshc", q_nope, wkv_b[:, : self.qk_nope_head_dim]
            )

            scores = (
                torch.einsum("bshc,btc->bsht", q_nope_proj, prev_kv_t)
                + torch.einsum("bshr,btr->bsht", q_pe, prev_k_pe)
            ) * self.softmax_scale

            # Apply masks
            if self_attn_mask is not None:
                scores = scores + self_attn_mask.unsqueeze(0).unsqueeze(2)
            if key_padding_mask is not None:
                scores = scores.masked_fill(
                    key_padding_mask.unsqueeze(1).unsqueeze(2), float("-inf")
                )

            # Compute attention weights
            attn_weights = F.softmax(scores, dim=-1)
            attn_weights = F.dropout(
                attn_weights, p=self.dropout, training=self.training
            )

            # Weighted sum of values
            x = torch.einsum("bsht,btc->bshc", attn_weights, prev_kv_t)
            x = torch.einsum("bshc,hdc->bshd", x, wkv_b[:, -self.v_head_dim :])
            x = self.wo(x.flatten(2))

            if need_weights:
                return x, attn_weights
            else:
                return x, None

        else:  # At Training

            T = key.size(1)
            t = math.ceil(T / self.down_rate)
            w_tT = self.hypernet_down(
                T, t, kv_norm.device, train=True, T_input=kv_norm
            )  # "T" is used at training to simulate the case in inference with "t"
            zero_mask = (
                self.generate_chunk_mask(T, self.down_rate)
                .to(k_pe.device)
                .unsqueeze(0)
                .to(kv_norm.dtype)
            )
            kv_norm_t = torch.matmul(w_tT * zero_mask, kv_norm)

            # Compute attention scores
            wkv_b = self.wkv_b.weight
            wkv_b = wkv_b.view(self.num_heads, -1, self.kv_lora_rank)
            q_nope_proj = torch.einsum(
                "bshd,hdc->bshc", q_nope, wkv_b[:, : self.qk_nope_head_dim]
            )

            tricky_mask = self.generate_stride_aware_causal_mask(T).to(
                q_nope_proj.device
            )

            if seqlen != T:
                tricky_mask = tricky_mask[-seqlen:]

            scores = (
                torch.einsum("bshc,btc->bsht", q_nope_proj, kv_norm_t)
                + torch.einsum("bshr,btr->bsht", q_pe, k_pe)
            ) * self.softmax_scale

            scores = scores + tricky_mask.unsqueeze(0).unsqueeze(2).to(scores.dtype)
            # Apply masks
            if self_attn_mask is not None:
                scores = scores + self_attn_mask.unsqueeze(0).unsqueeze(2)
            if key_padding_mask is not None:
                scores = scores.masked_fill(
                    key_padding_mask.unsqueeze(1).unsqueeze(2), float("-inf")
                )

            # Compute attention weights
            attn_weights = F.softmax(scores, dim=-1)
            attn_weights = F.dropout(
                attn_weights, p=self.dropout, training=self.training
            )

            # Weighted sum of values
            x = torch.einsum("bsht,btc->bshc", attn_weights, kv_norm_t)
            x = torch.einsum("bshc,hdc->bshd", x, wkv_b[:, -self.v_head_dim :])
            x = self.wo(x.flatten(2))

            if need_weights:
                return x, attn_weights
            else:
                return x, None

    def _compute_freqs_cis_batch(self, pos: torch.Tensor, device: torch.device):
        theta = 10000.0
        freqs = 1.0 / (
            theta
            ** (
                torch.arange(0, self.qk_rope_head_dim, 2, device=device).float()
                / self.qk_rope_head_dim
            )
        )
        freqs = torch.einsum(
            "bi,j->bij", pos, freqs
        )  # (batch_size, seq_len, head_dim//2)
        return torch.polar(
            torch.ones_like(freqs), freqs
        )  # (batch_size, seq_len, head_dim//2)

    def _apply_rotary_emb_batch(
        self, x: torch.Tensor, freqs_cis: torch.Tensor
    ) -> torch.Tensor:
        dtype = x.dtype
        x = x.float().view(
            *x.shape[:-1], -1, 2
        )  # (batch_size, seq_len, n_heads, head_dim//2, 2)
        x = torch.view_as_complex(x)
        freqs_cis = freqs_cis.unsqueeze(2)  # (batch_size, seq_len, 1, head_dim//2)
        y = torch.view_as_real(x * freqs_cis).flatten(3)
        return y.to(dtype)

    def generate_chunk_mask(self, T, chunk_size):
        # Generate index matrix
        row_indices = torch.arange(T).view(-1, 1)
        col_indices = torch.arange(T).view(1, -1)

        # Compute the block each position belongs to
        row_chunk = row_indices // chunk_size
        col_chunk = col_indices // chunk_size

        # Check whether positions are in the same block
        same_chunk = row_chunk == col_chunk

        # Generate lower triangular mask (within the same block and where row >= col)
        tril_mask = row_indices % chunk_size >= col_indices % chunk_size

        # Final mask: within the same block and satisfies the lower triangular condition
        chunk_mask = same_chunk & tril_mask

        return chunk_mask.float()

    def generate_stride_aware_causal_mask(self, T):
        """
        Generate a mask of shape (T, T) with the following properties:
        1. Future positions are masked (upper triangular part is -inf).
        2. For past positions:
           - If j <= i and j % 4 == 0, then mask[i, j] = 0 (visible).
           - If j == i, then mask[i, j] = 0 (visible).

        Args:
            T (int): Sequence length.

        Returns:
            torch.Tensor: Mask of shape (T, T).
        """
        # Initialize the mask with -1e9 (future positions are masked)
        mask = torch.full((T, T), -1e9)

        # Create a grid of indices
        rows = torch.arange(T).view(-1, 1)  # Shape: (T, 1)
        cols = torch.arange(T).view(1, -1)  # Shape: (1, T)

        # Condition for visible positions
        visible = ((cols <= rows) & ((cols + 1) % self.down_rate == 0)) | (cols == rows)

        # Set visible positions to 0
        mask[visible] = 0

        return mask


class MTLADecoderOnlyLayer(transformer_layer.TransformerDecoderLayerBase):
    def __init__(self, args, no_encoder_attn=False):
        super().__init__(args, no_encoder_attn)
        self.p = self.self_attn.dropout_module.p
        self.self_attn = MultiheadTemporalLatentAttention(
            embed_dim=self.embed_dim,
            num_heads=self.self_attn.num_heads,
            dropout=self.self_attn.dropout_module.p,
            q_lora_rank=args.decoder_q_lora_rank,
            kv_lora_rank=args.decoder_kv_lora_rank,
            qk_nope_head_dim=args.decoder_qk_nope_head_dim,
            qk_rope_head_dim=args.decoder_qk_rope_head_dim,
            v_head_dim=args.decoder_v_head_dim,
            down_rate=args.decoder_down_rate,
        )
        self.no_encoder_attn = no_encoder_attn
        self.cross_self_attention = True

    def forward(
        self,
        x,
        encoder_out=None,
        encoder_padding_mask=None,
        position=None,
        incremental_state=None,
        self_attn_mask=None,
        self_attn_padding_mask=None,
        **kwargs,
    ):
        # self-attention module
        residual = x
        x = x.transpose(0, 1)
        x = self.self_attn_layer_norm(x)

        _self_attn_input_buffer = self.self_attn._get_input_buffer(incremental_state)
        if self.cross_self_attention and not (
            incremental_state is not None
            and _self_attn_input_buffer is not None
            and "prev_k_pe" in _self_attn_input_buffer
        ):

            if self_attn_mask is not None:
                assert encoder_out is not None
                self_attn_mask = torch.cat(
                    (x.new_zeros(x.size(1), encoder_out.size(1)), self_attn_mask), dim=1
                )
            if self_attn_padding_mask is not None:
                if encoder_padding_mask is None:
                    assert encoder_out is not None
                    encoder_padding_mask = self_attn_padding_mask.new_zeros(
                        encoder_out.size(0), encoder_out.size(1)
                    )
                self_attn_padding_mask = torch.cat(
                    (encoder_padding_mask, self_attn_padding_mask), dim=1
                )
            assert encoder_out is not None
            y = torch.cat((encoder_out, x), dim=1)
        else:
            y = x

        self_attn_out, _ = self.self_attn(
            query=x,
            key=y,
            value=y,
            key_padding_mask=self_attn_padding_mask,
            self_attn_mask=self_attn_mask,
            position=position,
            incremental_state=incremental_state,
            need_weights=False,
        )

        self_attn_out = self_attn_out.transpose(0, 1)
        x = residual + F.dropout(self_attn_out, p=self.p, training=self.training)
        x = self.self_attn_layer_norm(x)

        # Feed-forward module
        residual = x
        x = self.fc1(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.activation_dropout_module.p, training=self.training)
        x = self.fc2(x)
        x = F.dropout(x, p=self.p, training=self.training)
        x = residual + x
        x = self.final_layer_norm(x)

        return x, None, None


class RotaryDecoderOnlyLayer(transformer_layer.TransformerDecoderLayerBase):
    def __init__(self, args, no_encoder_attn=False):
        super().__init__(args, no_encoder_attn)
        self.p = self.self_attn.dropout_module.p
        if args.decoder_att_type == "GQA":
            self.self_attn = GQARotaryMultiheadAttention(
                embed_dim=self.embed_dim,
                num_heads=self.self_attn.num_heads,
                dropout=self.self_attn.dropout_module.p,
                bias=True,
                num_kv_heads=2,
            )
        elif args.decoder_att_type == "MQA":
            self.self_attn = GQARotaryMultiheadAttention(
                embed_dim=self.embed_dim,
                num_heads=self.self_attn.num_heads,
                dropout=self.self_attn.dropout_module.p,
                bias=True,
                num_kv_heads=1,
            )
        else:
            self.self_attn = RotaryMultiheadAttention(
                embed_dim=self.embed_dim,
                num_heads=self.self_attn.num_heads,
                dropout=self.self_attn.dropout_module.p,
                bias=True,
            )
        self.no_encoder_attn = no_encoder_attn
        self.cross_self_attention = True

    def forward(
        self,
        x,
        encoder_out=None,
        encoder_padding_mask=None,
        position=None,
        incremental_state=None,
        self_attn_mask=None,
        self_attn_padding_mask=None,
        **kwargs,
    ):
        # Self-attention module
        residual = x
        x = x.transpose(0, 1)
        x = self.self_attn_layer_norm(x)

        _self_attn_input_buffer = self.self_attn._get_input_buffer(incremental_state)
        if self.cross_self_attention and not (
            incremental_state is not None
            and _self_attn_input_buffer is not None
            and "prev_key" in _self_attn_input_buffer
        ):

            if self_attn_mask is not None:
                assert encoder_out is not None
                self_attn_mask = torch.cat(
                    (x.new_zeros(x.size(1), encoder_out.size(1)), self_attn_mask), dim=1
                )
            if self_attn_padding_mask is not None:
                if encoder_padding_mask is None:
                    assert encoder_out is not None
                    encoder_padding_mask = self_attn_padding_mask.new_zeros(
                        encoder_out.size(0), encoder_out.size(1)
                    )
                self_attn_padding_mask = torch.cat(
                    (encoder_padding_mask, self_attn_padding_mask), dim=1
                )
            assert encoder_out is not None
            y = torch.cat((encoder_out, x), dim=1)
        else:
            y = x

        self_attn_out, _ = self.self_attn(
            query=x,
            key=y,
            value=y,
            key_padding_mask=self_attn_padding_mask,
            self_attn_mask=self_attn_mask,
            position=position,
            incremental_state=incremental_state,
            need_weights=False,
        )

        self_attn_out = self_attn_out.transpose(0, 1)
        x = residual + F.dropout(self_attn_out, p=self.p, training=self.training)
        x = self.self_attn_layer_norm(x)

        # Feed-forward module
        residual = x
        x = self.fc1(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.activation_dropout_module.p, training=self.training)
        x = self.fc2(x)
        x = F.dropout(x, p=self.p, training=self.training)
        x = residual + x
        x = self.final_layer_norm(x)

        return x, None, None


class MLADecoderOnlyLayer(MTLADecoderOnlyLayer):
    def __init__(self, args, no_encoder_attn=False):
        args.decoder_down_rate = 1
        super().__init__(args, no_encoder_attn)
        # 用 MLA 替换 self-attention 模块
        self.p = self.p  # self_attn.dropout_module.p
        self.self_attn = MultiheadLatentAttention(
            embed_dim=self.embed_dim,
            num_heads=self.self_attn.num_heads,
            dropout=self.p,  # dropout_module.p,
            q_lora_rank=args.decoder_q_lora_rank,
            kv_lora_rank=args.decoder_kv_lora_rank,
            qk_nope_head_dim=args.decoder_qk_nope_head_dim,
            qk_rope_head_dim=args.decoder_qk_rope_head_dim,
            v_head_dim=args.decoder_v_head_dim,
        )
        self.no_encoder_attn = no_encoder_attn
        self.cross_self_attention = True


class EncoderProjectorConcat(nn.Module):
    def __init__(self, encoder_projector_ds_rate, encoder_dim, llm_dim):
        super().__init__()
        self.k = encoder_projector_ds_rate
        self.encoder_dim = encoder_dim
        self.llm_dim = llm_dim
        self.linear1 = nn.Linear(self.encoder_dim * self.k, int(llm_dim / 2))
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(int(llm_dim / 2), llm_dim)

    def forward(self, x):
        batch_size, seq_len, dim = x.size()
        num_frames_to_discard = seq_len % self.k
        if num_frames_to_discard > 0:
            x = x[:, :-num_frames_to_discard, :]
        seq_len = x.size(1)

        x = x.contiguous()
        x = x.view(batch_size, seq_len // self.k, dim * self.k)
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        return x


class MTLADecoderOnly(TransformerDecoder):
    def __init__(self, cfg, dictionary, embed_tokens, no_encoder_attn=False):
        super().__init__(cfg, dictionary, embed_tokens, no_encoder_attn=no_encoder_attn)
        cfg = TransformerConfig.from_namespace(cfg)
        # Disable absolute positional encoding and rely entirely on RoPE
        self.embed_positions = None

        self.layers = nn.ModuleList(
            [
                self.build_decoder_layer(cfg, no_encoder_attn)
                for _ in range(cfg.decoder.layers)
            ]
        )
        self.projector = EncoderProjectorConcat(
            cfg.encoder_projector_ds_rate, cfg.encoder_embed_dim, cfg.decoder_embed_dim
        )

    def build_decoder_layer(self, cfg, no_encoder_attn=False):
        layer = MTLADecoderOnlyLayer(cfg, no_encoder_attn)
        checkpoint = cfg.checkpoint_activations
        if checkpoint:
            offload_to_cpu = cfg.offload_activations
            layer = checkpoint_wrapper(layer, offload_to_cpu=offload_to_cpu)
        min_params_to_wrap = cfg.min_params_to_wrap if not checkpoint else 0
        layer = fsdp_wrap(layer, min_num_params=min_params_to_wrap)
        return layer

    def extract_features_scriptable(
        self,
        prev_output_tokens,
        encoder_out: Optional[Dict[str, List[Tensor]]],
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
        full_context_alignment: bool = False,
        alignment_layer: Optional[int] = None,
        alignment_heads: Optional[int] = None,
    ):
        """
        Similar to *forward* but only return features.

        Includes several features from "Jointly Learning to Align and
        Translate with Transformer Models" (Garg et al., EMNLP 2019).

        Args:
            full_context_alignment (bool, optional): don't apply
                auto-regressive mask to self-attention (default: False).
            alignment_layer (int, optional): return mean alignment over
                heads at this layer (default: last layer).
            alignment_heads (int, optional): only average alignment over
                this many heads (default: all heads).

        Returns:
            tuple:
                - the decoder's features of shape `(batch, tgt_len, embed_dim)`
                - a dictionary with any model-specific outputs
        """
        bs, slen = prev_output_tokens.size()
        if alignment_layer is None:
            alignment_layer = self.num_layers - 1

        enc_embeddings = None
        enc_padding_mask = None
        dec_pos_offset = None
        if encoder_out is not None:
            if len(encoder_out["encoder_out"]) > 0:
                enc_embeddings = encoder_out["encoder_out"][0]
                enc_embeddings = self.projector(enc_embeddings.transpose(0, 1))
                self.enc_len = enc_embeddings.size(1)
            if len(encoder_out["encoder_padding_mask"]) > 0:
                enc_padding_mask = encoder_out["encoder_padding_mask"][0]
                indices = list(
                    range(
                        self.projector.k - 1,
                        enc_padding_mask.shape[1],
                        self.projector.k,
                    )
                )
                enc_padding_mask = enc_padding_mask[:, indices]
                # Compute the actual length of each sample (non-padding part)
                dec_pos_offset = enc_padding_mask.sum(dim=1)

        device = prev_output_tokens.device
        if incremental_state is not None and prev_output_tokens.shape[1] > 1:
            prev_output_tokens = prev_output_tokens[:, -1:]

            if dec_pos_offset is not None:
                position = (
                    torch.arange(0, 1, device=device).float()
                    + self.enc_len
                    + slen
                    - 1
                    - dec_pos_offset
                )
                position = position.view(-1, 1)
            else:
                position = (
                    torch.arange(0, 1, device=device).float().view(1, -1)
                    + self.enc_len
                    + slen
                    - 1
                )

        else:
            if dec_pos_offset is not None:
                position_enc = (
                    torch.arange(0, self.enc_len, device=device).float().view(1, -1)
                )
                position_dec = torch.arange(
                    self.enc_len, self.enc_len + slen, device=device
                ).float().view(1, -1) - dec_pos_offset.view(-1, 1)
                position = torch.cat(
                    [position_enc.repeat(position_dec.shape[0], 1), position_dec],
                    dim=-1,
                )
            else:
                position = (
                    torch.arange(0, self.enc_len + slen, device=device)
                    .float()
                    .view(1, -1)
                )

        # Prevent torchscript exporting issue for dynamic quant embedding
        prev_output_tokens = prev_output_tokens.contiguous()
        # embed tokens and positions
        x = self.embed_scale * self.embed_tokens(prev_output_tokens)

        if self.quant_noise is not None:
            x = self.quant_noise(x)

        if self.project_in_dim is not None:
            x = self.project_in_dim(x)

        if self.layernorm_embedding is not None:
            x = self.layernorm_embedding(x)

        x = self.dropout_module(x)

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)

        self_attn_padding_mask: Optional[Tensor] = None
        if self.cross_self_attention or prev_output_tokens.eq(self.padding_idx).any():
            self_attn_padding_mask = prev_output_tokens.eq(self.padding_idx)

        # decoder layers
        attn: Optional[Tensor] = None
        inner_states: List[Optional[Tensor]] = [x]
        for idx, layer in enumerate(self.layers):
            if incremental_state is None and not full_context_alignment:  # dkq
                self_attn_mask = self.buffered_future_mask(x)
            else:
                self_attn_mask = None

            x, layer_attn, _ = layer(
                x,
                enc_embeddings,
                enc_padding_mask,
                position,
                incremental_state,
                self_attn_mask=self_attn_mask,
                self_attn_padding_mask=self_attn_padding_mask,
                need_attn=bool((idx == alignment_layer)),
                need_head_weights=bool((idx == alignment_layer)),
            )
            inner_states.append(x)
            if layer_attn is not None and idx == alignment_layer:
                attn = layer_attn.float().to(x)

        if attn is not None:
            if alignment_heads is not None:
                attn = attn[:alignment_heads]

            # average probabilities over heads
            attn = attn.mean(dim=0)

        if self.layer_norm is not None:
            x = self.layer_norm(x)

        # T x B x C -> B x T x C
        x = x.transpose(0, 1)

        if self.project_out_dim is not None:
            x = self.project_out_dim(x)

        extra = {"encoder_out": encoder_out} if incremental_state is None else None
        return x, extra


class RotaryDecoderOnly(MTLADecoderOnly):
    def __init__(self, cfg, dictionary, embed_tokens, no_encoder_attn=False):
        super().__init__(cfg, dictionary, embed_tokens, no_encoder_attn=no_encoder_attn)
        cfg = TransformerConfig.from_namespace(cfg)
        # Disable absolute positional encoding and rely entirely on RoPE
        self.embed_positions = None

        self.layers = nn.ModuleList(
            [
                self.build_decoder_layer(cfg, no_encoder_attn)
                for _ in range(cfg.decoder.layers)
            ]
        )
        self.projector = EncoderProjectorConcat(
            cfg.encoder_projector_ds_rate, cfg.encoder_embed_dim, cfg.decoder_embed_dim
        )

    def build_decoder_layer(self, cfg, no_encoder_attn=False):
        layer = RotaryDecoderOnlyLayer(cfg, no_encoder_attn)
        checkpoint = cfg.checkpoint_activations
        if checkpoint:
            offload_to_cpu = cfg.offload_activations
            layer = checkpoint_wrapper(layer, offload_to_cpu=offload_to_cpu)
        min_params_to_wrap = cfg.min_params_to_wrap if not checkpoint else 0
        layer = fsdp_wrap(layer, min_num_params=min_params_to_wrap)
        return layer


class MLADecoderOnly(MTLADecoderOnly):
    def __init__(self, cfg, dictionary, embed_tokens, no_encoder_attn=False):
        super().__init__(cfg, dictionary, embed_tokens, no_encoder_attn=no_encoder_attn)
        cfg = TransformerConfig.from_namespace(cfg)
        # Disable absolute positional encoding and rely entirely on RoPE
        self.embed_positions = None

        self.layers = nn.ModuleList(
            [
                self.build_decoder_layer(cfg, no_encoder_attn)
                for _ in range(cfg.decoder.layers)
            ]
        )
        self.projector = EncoderProjectorConcat(
            cfg.encoder_projector_ds_rate, cfg.encoder_embed_dim, cfg.decoder_embed_dim
        )

    def build_decoder_layer(self, cfg, no_encoder_attn=False):
        # 使用修改后的 RotaryTransformerDecoderLayer 替换原有的 TransformerDecoderLayerBase
        layer = MLADecoderOnlyLayer(cfg, no_encoder_attn)
        checkpoint = cfg.checkpoint_activations
        if checkpoint:
            offload_to_cpu = cfg.offload_activations
            layer = checkpoint_wrapper(layer, offload_to_cpu=offload_to_cpu)
        min_params_to_wrap = cfg.min_params_to_wrap if not checkpoint else 0
        layer = fsdp_wrap(layer, min_num_params=min_params_to_wrap)
        return layer
