# -*- coding: utf-8 -*-
"""
Created on Thu Oct  2 16:47:31 2025

@author: Keqi Deng (University of Cambridge)
"""

from functools import partial
import torch
from torch import nn
from transformers import AutoModelForCausalLM
from transformers.cache_utils import Cache, DynamicCache
from transformers.models.llama.modeling_llama import (
    LlamaModel,
    LlamaDecoderLayer,
    LlamaPreTrainedModel,
    LlamaForCausalLM,
)
from transformers.modeling_outputs import BaseModelOutputWithPast
from transformers.utils import logging
from typing import Optional
from .configuration_mtla import LlamaMTLAConfig
from .mtla_attention import MultiheadTemporalLatentAttentionHF

logger = logging.get_logger(__name__)


class TemporalLatentCache(DynamicCache):
    """
    A cache that maintains temporally compressed key/value states
    for Multi-Head Temporal Latent Attention (MTLA).

    Extends Hugging Face's DynamicCache but overrides `update()`
    to apply down_rate-based accumulation instead of naive concatenation.
    """

    def __init__(self, config=None, down_rate=2):
        super().__init__()
        self.down_rate = down_rate
        # We'll track per-layer infer steps as in your original code
        self.key_cache = {}
        self.value_cache = {}
        self.infer_steps = {}

    def update(self, kv_norm_t, k_pe, layer_idx, abs_length):
        """
        Update cache for a specific layer according to MTLA logic.
        Args:
            kv_norm_t: (B, T, D) temporal latent values for this step
            k_pe: (B, T, D_pe) positional embeddings
            layer_idx: int, index of the layer in the model
        Returns:
            (kv_t, k_pe_t): updated cache tensors
        """
        # Ensure per-layer slots exist
        if layer_idx not in self.key_cache:
            self.key_cache[layer_idx] = []
            self.value_cache[layer_idx] = []
            self.infer_steps[layer_idx] = 0
            prev_kv_t = None
            prev_k_pe = None
        else:
            prev_kv_t = self.key_cache[layer_idx]
            prev_k_pe = self.value_cache[layer_idx]

        infer_steps = self.infer_steps[layer_idx] + abs_length
        self.infer_steps[layer_idx] = infer_steps

        # Determine compression behavior
        T = infer_steps
        T_remain = T % self.down_rate
        B, t_len, D = kv_norm_t.shape

        if prev_kv_t is None or prev_kv_t == []:
            # First token case
            prev_kv_t = kv_norm_t
            prev_k_pe = k_pe
        else:
            if T_remain != 1:
                # Update last block (temporal accumulation)
                prev_kv_t[:, -1:] += kv_norm_t
                prev_k_pe[:, -1:] = k_pe
            else:
                # Add a new temporal block
                prev_kv_t = torch.cat([prev_kv_t, kv_norm_t], dim=1)
                prev_k_pe = torch.cat([prev_k_pe, k_pe], dim=1)

        # Store updated states
        self.key_cache[layer_idx] = prev_kv_t
        self.value_cache[layer_idx] = prev_k_pe

        return prev_kv_t, prev_k_pe

    def is_empty(self, layer_idx: int):
        """Check if this layer has any cached keys."""
        return layer_idx not in self.key_cache or len(self.key_cache[layer_idx]) == 0

    def get_seq_length(self, layer_idx: Optional[int] = 0) -> int:
        """Returns the sequence length of the cached states. A layer index can be optionally passed."""
        # TODO: deprecate this function in favor of `cache_position`

        return self.infer_steps[layer_idx] if not self.is_empty(layer_idx) else 0


class LlamaMTLADecoderLayer(LlamaDecoderLayer):
    def __init__(self, config: LlamaMTLAConfig, layer_idx: int):
        super().__init__(config, layer_idx)
        # Replace self-attention
        self.self_attn = MultiheadTemporalLatentAttentionHF(
            embed_dim=config.hidden_size,
            num_heads=config.num_attention_heads,
            kv_lora_rank=config.kv_lora_rank,
            q_lora_rank=config.q_lora_rank,
            qk_nope_head_dim=config.qk_nope_head_dim,
            qk_rope_head_dim=config.qk_rope_head_dim,
            v_head_dim=config.v_head_dim,
            down_rate=config.down_rate,
            recompute_prompt_attn=config.recompute_prompt_attn,
            layer_idx=layer_idx,
        )


class LlamaMTLAPreTrainedModel(LlamaPreTrainedModel):
    config_class = LlamaMTLAConfig
    _no_split_modules = ["LlamaMTLADecoderLayer"]


class LlamaMTLAModel(LlamaMTLAPreTrainedModel, LlamaModel):
    def __init__(self, config: LlamaMTLAConfig):
        super().__init__(config)
        self.layers = nn.ModuleList(
            [LlamaMTLADecoderLayer(config, i) for i in range(config.num_hidden_layers)]
        )

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **flash_attn_kwargs,
    ) -> BaseModelOutputWithPast:
        """
        The forward pass for LlamaMTLAModel, which is almost identical to LlamaModel.forward,
        except that it uses TemporalLatentCache instead of DynamicCache.
        """
        # === identical setup ===
        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError(
                "You must specify exactly one of input_ids or inputs_embeds"
            )

        if self.gradient_checkpointing and self.training and use_cache:
            logger.warning_once(
                "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`."
            )
            use_cache = False

        # === (1) modified: use TemporalLatentCache ===
        if not isinstance(past_key_values, (type(None), Cache)):
            raise ValueError(
                "The `past_key_values` should be either a `Cache` object or `None`."
            )
        if use_cache and (
            past_key_values is None
            or not isinstance(past_key_values, TemporalLatentCache)
        ):
            # if use_cache and past_key_values is None:
            past_key_values = TemporalLatentCache(
                config=self.config, down_rate=self.config.down_rate
            )

        # === identical to base ===
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if cache_position is None:
            past_seen_tokens = (
                past_key_values.get_seq_length() if past_key_values is not None else 0
            )
            cache_position = torch.arange(
                past_seen_tokens,
                past_seen_tokens + inputs_embeds.shape[1],
                device=inputs_embeds.device,
            )

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        hidden_states = inputs_embeds

        # === identical: shared positional embeddings ===
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        # === identical decoder loop ===
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None

        for decoder_layer in self.layers[: self.config.num_hidden_layers]:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    partial(decoder_layer.__call__, **flash_attn_kwargs),
                    hidden_states,
                    attention_mask,
                    position_ids,
                    past_key_values,
                    output_attentions,
                    use_cache,
                    cache_position,
                    position_embeddings,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_values,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                    cache_position=cache_position,
                    position_embeddings=position_embeddings,
                    **flash_attn_kwargs,
                )

            hidden_states = layer_outputs[0]
            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        hidden_states = self.norm(hidden_states)

        # === identical post-processing ===
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values if use_cache else None,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )


class LlamaMTLAForCausalLM(LlamaMTLAPreTrainedModel, LlamaForCausalLM):
    def __init__(self, config: LlamaMTLAConfig):
        super().__init__(config)
        self.model = LlamaMTLAModel(config)


AutoModelForCausalLM.register(LlamaMTLAConfig, LlamaMTLAForCausalLM)
