# -*- coding: utf-8 -*-
"""
Created on Thu Oct  2 16:42:44 2025

@author: Keqi Deng (University of Cambridge)
"""

from transformers import AutoConfig, LlamaTokenizer
from transformers.models.llama.configuration_llama import LlamaConfig
from transformers.models.auto.tokenization_auto import TOKENIZER_MAPPING


class LlamaMTLAConfig(LlamaConfig):
    model_type = "llama-mtla"

    def __init__(
        self,
        *args,
        kv_lora_rank=256,
        q_lora_rank=0,
        qk_rope_head_dim=32,
        qk_nope_head_dim=64,
        v_head_dim=64,
        down_rate=2,
        recompute_prompt_attn=True,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.kv_lora_rank = kv_lora_rank
        self.q_lora_rank = q_lora_rank
        self.qk_rope_head_dim = qk_rope_head_dim
        self.qk_nope_head_dim = qk_nope_head_dim
        self.v_head_dim = v_head_dim
        self.partial_rotary_factor = qk_rope_head_dim / qk_nope_head_dim
        self.down_rate = down_rate
        self.recompute_prompt_attn = recompute_prompt_attn
        self.use_cache = False


AutoConfig.register("llama-mtla", LlamaMTLAConfig)
TOKENIZER_MAPPING.register(LlamaMTLAConfig, (LlamaTokenizer, None))
