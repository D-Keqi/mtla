# -*- coding: utf-8 -*-
"""
Created on Thu Oct  2 16:49:42 2025

@author: 86135
"""

# Put this class in place of your previous MultiheadTemporalLatentAttentionHF
import math
import uuid
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

# HF helpers
try:
    from transformers.cache_utils import Cache
except Exception:
    Cache = None

# import HF's apply_rotary_pos_emb if available; if not, we'll fall back to internal method
try:
    from transformers.models.llama.modeling_llama import apply_rotary_pos_emb
except Exception:
    apply_rotary_pos_emb = None


# class HyperNetwork(nn.Module):
#     def __init__(self, d, down_rate, low_rank=4):
#         super().__init__()
#         self.d = d
#         self.down_rate = down_rate
#         self.fc_c = nn.Linear(d, int(d / low_rank))
#         self.fc_p = nn.Linear(d, int(d / low_rank))

#     def positional_encoding(self, T, pos=0, device=None, dtype=torch.float32):
#         d = self.d
#         position = torch.arange(pos, pos + T, dtype=torch.float32, device=device).unsqueeze(1)  # (T, 1)
#         div_term = torch.exp(
#             torch.arange(0, d, 2, dtype=torch.float32, device=device)
#             * (-math.log(10000.0) / d)
#         )  # (d/2,)
#         pe = torch.zeros(T, d, device=device, dtype=torch.float32)
#         pe[:, 0::2] = torch.sin(position * div_term)
#         pe[:, 1::2] = torch.cos(position * div_term)
#         return pe.unsqueeze(0).to(dtype)  # (1, T, d)

#     def forward(
#         self, T, t, device, train=True, low_rank=False, T_input=None, t_input=None
#     ):
#         # keep behaviour consistent with your original implementation
#         if not train and T_input is not None and T_input.shape[1] == 1:
#             if T_input is not None:
#                 P = T_input
#             else:
#                 P = self.positional_encoding(1, pos=T - 1, device=device)
#             if t_input is not None:
#                 C = t_input
#             else:
#                 C = self.positional_encoding(1, pos=t - 1, device=device)

#             C2 = self.fc_c(C)
#             P2 = self.fc_p(P)
#             W = torch.bmm(C2.expand(P2.shape[0], -1, -1), P2.transpose(1, 2))
#             return torch.sigmoid(W)

#         if T_input is not None:
#             P = T_input
#         else:
#             P = self.positional_encoding(T, device=device)

#         if t_input is not None:
#             C = t_input
#         else:
#             C = self.positional_encoding(t, device=device)
#             if train:
#                 C = C.repeat_interleave(self.down_rate, dim=1)[:, :T]

#         C2 = self.fc_c(C)
#         P2 = self.fc_p(P)
#         W = torch.bmm(C2, P2.transpose(1, 2))
#         return torch.sigmoid(W)


class HyperNetwork(nn.Module):
    def __init__(self, d, down_rate, low_rank=4):
        """
        d: Model dimension
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
                C = self.positional_encoding(1, pos=t - 1).to(device).to(P.dtype)

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


class MultiheadTemporalLatentAttentionHF(nn.Module):
    """
    HF-compatible wrapper of your Multihead Temporal Latent Attention.

    Forward signature matches HF LLaMA attention:
      hidden_states, position_embeddings (cos,sin), attention_mask,
      past_key_value (Cache or legacy tuple), cache_position, **kwargs
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
        recompute_prompt_attn=True,
        layer_idx: int = 0,  # optional, useful when using HF Cache.update(layer_idx)
    ):
        super().__init__()
        self.num_heads = num_heads
        self.dropout = dropout
        self.q_lora_rank = q_lora_rank
        self.kv_lora_rank = kv_lora_rank
        self.qk_nope_head_dim = qk_nope_head_dim
        self.qk_rope_head_dim = qk_rope_head_dim
        self.qk_head_dim = qk_nope_head_dim + qk_rope_head_dim
        self.v_head_dim = v_head_dim
        self.down_rate = down_rate
        self.recompute_prompt_attn = recompute_prompt_attn
        self.softmax_scale = self.qk_head_dim**-0.5
        self.layer_idx = layer_idx

        # Query projection
        if self.q_lora_rank == 0:
            self.wq = nn.Linear(embed_dim, num_heads * self.qk_head_dim, bias=bias)
        else:
            self.wq_a = nn.Linear(embed_dim, q_lora_rank, bias=bias)
            self.q_norm = nn.LayerNorm(q_lora_rank)
            self.wq_b = nn.Linear(q_lora_rank, num_heads * self.qk_head_dim, bias=bias)

        # Key/Value projection (the low-rank + rope split you had)
        self.wkv_a = nn.Linear(embed_dim, kv_lora_rank + qk_rope_head_dim, bias=bias)
        self.kv_norm = nn.LayerNorm(kv_lora_rank)
        self.wkv_b = nn.Linear(
            kv_lora_rank, num_heads * (qk_nope_head_dim + v_head_dim), bias=bias
        )

        # Output projection
        self.wo = nn.Linear(num_heads * v_head_dim, embed_dim, bias=bias)

        # hypernet for temporal downsampling
        self.hypernet_down = HyperNetwork(d=kv_lora_rank, down_rate=down_rate)

        # optional: identifier
        self._id = str(uuid.uuid4())

    # ---------------------------
    # helper pack/unpack for past/present key values (legacy tuple format)
    # ---------------------------
    def _unpack_past_key_value(self, past_key_value):
        if past_key_value is None:
            return None, None, None
        if not isinstance(past_key_value, (tuple, list)):
            raise ValueError(
                "past_key_value must be a tuple (prev_kv_t, prev_k_pe, infer_steps)"
            )
        if len(past_key_value) == 3:
            return past_key_value[0], past_key_value[1], past_key_value[2]
        else:
            return past_key_value[0], past_key_value[1], None

    def _pack_present_key_value(self, prev_kv_t, prev_k_pe, infer_steps):
        return (prev_kv_t, prev_k_pe, infer_steps)

    # ---------------------------
    # rotary helpers (kept from your original)
    # ---------------------------
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
        x = x.float().view(*x.shape[:-1], -1, 2)  # (B, S, H, head//2, 2)
        x = torch.view_as_complex(x)
        freqs_cis = freqs_cis.unsqueeze(2)  # (B, S, 1, head//2)
        y = torch.view_as_real(x * freqs_cis).flatten(3)
        return y.to(dtype)

    # ---------------------------
    # mask utilities (kept)
    # ---------------------------
    def generate_chunk_mask(self, T, chunk_size):
        row_indices = torch.arange(T).view(-1, 1)
        col_indices = torch.arange(T).view(1, -1)
        row_chunk = row_indices // chunk_size
        col_chunk = col_indices // chunk_size
        same_chunk = row_chunk == col_chunk
        tril_mask = row_indices % chunk_size >= col_indices % chunk_size
        chunk_mask = same_chunk & tril_mask
        return chunk_mask.float()

    def generate_stride_aware_causal_mask(self, T):
        mask = torch.full((T, T), -1e9)
        rows = torch.arange(T).view(-1, 1)
        cols = torch.arange(T).view(1, -1)
        visible = ((cols <= rows) & ((cols + 1) % self.down_rate == 0)) | (cols == rows)
        mask[visible] = 0
        return mask

    # ---------------------------
    # forward: HF-compatible signature
    # ---------------------------
    def forward(
        self,
        hidden_states: Tensor,
        position_embeddings: Optional[Tuple[Tensor, Tensor]] = None,
        attention_mask: Optional[Tensor] = None,
        past_key_value: Optional[object] = None,  # can be HF Cache or legacy tuple
        cache_position: Optional[torch.LongTensor] = None,
        use_cache: bool = False,
        output_attentions: bool = False,
        **kwargs,
    ):
        """
        Returns:
            attn_output: (B, S, embed_dim)
            present_key_value: (prev_kv_t, prev_k_pe, infer_steps) if use_cache else None
            attn_weights: optional, if output_attentions
        """

        bsz, seqlen, _ = hidden_states.size()
        device = hidden_states.device
        dtype = hidden_states.dtype

        # -------------------------
        # Query projection
        # -------------------------
        if self.q_lora_rank == 0:
            q = self.wq(hidden_states)  # (B, S, num_heads * qk_head_dim)
        else:
            q = self.wq_b(self.q_norm(self.wq_a(hidden_states)))
        q = q.view(bsz, seqlen, self.num_heads, self.qk_head_dim)
        q_nope, q_pe = torch.split(
            q, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1
        )

        # -------------------------
        # compute kv projections (low-rank + rope split)
        # -------------------------
        kv = self.wkv_a(hidden_states)  # (B, S, kv_lora_rank + qk_rope_head_dim)
        kv_norm, k_pe = torch.split(
            kv, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1
        )
        # k_pe shape: (B, S, qk_rope_head_dim)
        kv_norm = self.kv_norm(kv_norm)  # (B, S, kv_lora_rank)

        # -------------------------
        # Rotary positional embedding handling
        # -------------------------
        used_hf_rotary = False
        if (
            position_embeddings is not None
            and isinstance(position_embeddings, tuple)
            and apply_rotary_pos_emb is not None
        ):
            # HF style: position_embeddings == (cos, sin)
            cos, sin = position_embeddings
            # q_pe: (B, S, H, D) -> (B, H, S, D) for HF apply
            q_pe_t = q_pe.permute(0, 2, 1, 3)  # (B, H, S, D)
            # expand k_pe to per-head to match HF apply shape (we repeated earlier across heads)
            k_pe_rep = k_pe.unsqueeze(1).expand(
                -1, self.num_heads, -1, -1
            )  # (B, H, S, D)
            # Use HF apply_rotary_pos_emb (it will slice cos/sin internally as needed)
            q_pe_t, k_pe_rep = apply_rotary_pos_emb(q_pe_t, k_pe_rep, cos, sin)
            # back to (B, S, H, D)
            q_pe = q_pe_t.permute(0, 2, 1, 3)
            # collapse k_pe_rep back to shared form (take head 0 since we repeated the same k_pe across heads)
            k_pe = k_pe_rep[:, 0, :, :].contiguous()  # (B, S, D)
            used_hf_rotary = True
        else:
            # fallback to your original per-batch freq computation if position IDs are provided via kwargs
            position_ids = kwargs.get("position_ids", None)
            if position_ids is None:
                position_ids = (
                    torch.arange(seqlen, device=device).unsqueeze(0).expand(bsz, seqlen)
                )
            freqs_cis = self._compute_freqs_cis_batch(position_ids, device)
            q_pe = self._apply_rotary_emb_batch(q_pe, freqs_cis[:, -seqlen:])
            k_pe = self._apply_rotary_emb_batch(k_pe.unsqueeze(2), freqs_cis).squeeze(2)

        # -------------------------
        # Determine cache input type: HF Cache vs legacy tuple
        # -------------------------
        prev_kv_t = None
        prev_k_pe = None
        infer_steps_t = None

        # legacy tuple handling first (if user passed tuple/list)
        if (
            use_cache
            and past_key_value is not None
            and isinstance(past_key_value, (tuple, list))
        ):
            prev_kv_t, prev_k_pe, infer_steps_t = self._unpack_past_key_value(
                past_key_value
            )

        # If HF-style Cache object (has update), use it.
        elif (
            use_cache
            and past_key_value is not None
            and (
                Cache is not None
                and isinstance(past_key_value, Cache)
                or hasattr(past_key_value, "update")
            )
        ):
            cache_kwargs = {}
            if used_hf_rotary:
                # let the cache implementation know about rotary and cache_position (some caches use this)
                cache_kwargs["sin"] = sin
                cache_kwargs["cos"] = cos
            if cache_position is not None:
                cache_kwargs["cache_position"] = cache_position

            # pass our *compressed* representations into HF cache; update() should return merged tensors
            # Note: we intentionally pass kv_norm and k_pe (our low-rank + rope parts) so the Cache stores our compressed format
            prev_kv_t, prev_k_pe = past_key_value.update(
                kv_norm, k_pe, self.layer_idx, cache_kwargs
            )
            # infer_steps: prefer cache_position if provided, else fallback estimate
            if cache_position is not None and cache_position.numel() > 0:
                infer_steps_t = int(cache_position.max().item()) + 1
            elif prev_kv_t is not None:
                infer_steps_t = prev_kv_t.shape[1] * self.down_rate
            else:
                infer_steps_t = None

        # else: past_key_value is None -> training / no-cache initial path (handled below)

        # -------------------------
        # Incremental (generation) path (if we have prev_kv_t / prev_k_pe or infer_steps)
        # -------------------------
        if prev_kv_t is not None or infer_steps_t is not None:
            # ensure infer_steps_t is tensor consistent with batch
            if infer_steps_t is None:
                if prev_kv_t is not None:
                    inferred_T = prev_kv_t.shape[1] * self.down_rate
                    infer_steps = torch.full(
                        (bsz,), inferred_T, device=device, dtype=torch.long
                    )
                else:
                    infer_steps = torch.zeros((bsz,), device=device, dtype=torch.long)
            else:
                if isinstance(infer_steps_t, torch.Tensor):
                    infer_steps = infer_steps_t.to(device)
                else:
                    infer_steps = (
                        torch.tensor(infer_steps_t, device=device, dtype=torch.long)
                        .unsqueeze(0)
                        .expand(bsz)
                    )

            start_pos = int(infer_steps[0].item()) - 1 if infer_steps.numel() > 0 else 0
            T = start_pos + 1
            t = math.ceil(T / self.down_rate)
            T_remain = T % self.down_rate

            # hypernet weight (inference)
            w_tT = self.hypernet_down(
                T, t, kv_norm.device, train=False, T_input=kv_norm
            )
            if w_tT.dim() == 2:
                w_tT = w_tT.unsqueeze(1)

            wkv_b = self.wkv_b.weight
            wkv_b = wkv_b.view(self.num_heads, -1, self.kv_lora_rank)

            q_nope_proj = torch.einsum(
                "bshd,hdc->bshc", q_nope, wkv_b[:, : self.qk_nope_head_dim]
            )

            # If prev cache exists, update it like original logic; else initialize it (handles first-step with prompt)
            if prev_kv_t is not None and prev_k_pe is not None:
                if T_remain != 1:
                    prev_kv_t[:, -1:] = prev_kv_t[:, -1:] + (kv_norm * w_tT)
                    prev_k_pe[:, -1:] = k_pe
                else:
                    prev_kv_t = torch.cat([prev_kv_t, kv_norm * w_tT], dim=1)
                    prev_k_pe = torch.cat([prev_k_pe, k_pe], dim=1)
                infer_steps = infer_steps + 1
            else:
                # initial caching from prompt or first token (mirror your prior code)
                if kv_norm.shape[1] != 1:
                    indices = list(range(self.down_rate - 1, T, self.down_rate))
                    if T - 1 not in indices:
                        indices.append(T - 1)

                    if self.recompute_prompt_attn:
                        w_tT_train = self.hypernet_down(
                            T, t, kv_norm.device, train=True, T_input=kv_norm
                        )
                        zero_mask = (
                            self.generate_chunk_mask(T, self.down_rate)
                            .to(k_pe.device)
                            .unsqueeze(0)
                            .to(kv_norm.dtype)
                        )
                        prev_kv_t_tmp = torch.matmul(w_tT_train * zero_mask, kv_norm)
                        prev_k_pe_tmp = k_pe
                        prev_kv_t = prev_kv_t_tmp[:, indices]
                        prev_k_pe = prev_k_pe_tmp[:, indices]

                        tricky_mask = self.generate_stride_aware_causal_mask(T).to(
                            prev_kv_t.device
                        )
                        if seqlen != T:
                            tricky_mask = tricky_mask[-seqlen:]
                    else:
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
                infer_steps = kv_norm.new_zeros(kv_norm.shape[0], dtype=torch.long) + T

            # compute scores
            scores = (
                torch.einsum("bshc,btc->bsht", q_nope_proj, prev_kv_t)
                + torch.einsum("bshr,btr->bsht", q_pe, prev_k_pe)
            ) * self.softmax_scale

            if "tricky_mask" in locals():
                scores = scores + tricky_mask.unsqueeze(0).unsqueeze(2).to(scores.dtype)
            if attention_mask is not None:
                if attention_mask.dtype == torch.bool:
                    scores = scores.masked_fill(
                        attention_mask.unsqueeze(1).unsqueeze(2), float("-inf")
                    )
                else:
                    if attention_mask.dim() == 2:
                        scores = scores + attention_mask.unsqueeze(1).unsqueeze(2)
                    else:
                        scores = scores + attention_mask

            attn_weights = F.softmax(scores, dim=-1)
            attn_weights = F.dropout(
                attn_weights, p=self.dropout, training=self.training
            )

            x = torch.einsum("bsht,btc->bshc", attn_weights, prev_kv_t)
            x = torch.einsum("bshc,hdc->bshd", x, wkv_b[:, -self.v_head_dim :])
            x = self.wo(x.flatten(2))

            present = (
                self._pack_present_key_value(prev_kv_t, prev_k_pe, infer_steps)
                if use_cache
                else None
            )
            outputs = (x, present)
            if output_attentions:
                outputs = outputs + (attn_weights,)
            return outputs

        # ---------------------------
        # Training / full-sequence path
        # ---------------------------
        T = hidden_states.size(1)
        t = math.ceil(T / self.down_rate)
        w_tT = self.hypernet_down(T, t, kv_norm.device, train=True, T_input=kv_norm)
        zero_mask = (
            self.generate_chunk_mask(T, self.down_rate)
            .to(k_pe.device)
            .unsqueeze(0)
            .to(kv_norm.dtype)
        )
        kv_norm_t = torch.matmul(w_tT * zero_mask, kv_norm)  # (B, T, kv_lora_rank)

        wkv_b = self.wkv_b.weight.view(self.num_heads, -1, self.kv_lora_rank)
        q_nope_proj = torch.einsum(
            "bshd,hdc->bshc", q_nope, wkv_b[:, : self.qk_nope_head_dim]
        )

        tricky_mask = self.generate_stride_aware_causal_mask(T).to(q_nope_proj.device)
        if seqlen != T:
            tricky_mask = tricky_mask[-seqlen:]

        scores = (
            torch.einsum("bshc,btc->bsht", q_nope_proj, kv_norm_t)
            + torch.einsum("bshr,btr->bsht", q_pe, k_pe)
        ) * self.softmax_scale

        scores = scores + tricky_mask.unsqueeze(0).unsqueeze(2).to(scores.dtype)

        if attention_mask is not None:
            if attention_mask.dtype == torch.bool:
                scores = scores.masked_fill(
                    attention_mask.unsqueeze(1).unsqueeze(2), float("-inf")
                )
            else:
                if attention_mask.dim() == 2:
                    scores = scores + attention_mask.unsqueeze(1).unsqueeze(2)
                else:
                    scores = scores + attention_mask

        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = F.dropout(attn_weights, p=self.dropout, training=self.training)

        x = torch.einsum("bsht,btc->bshc", attn_weights, kv_norm_t)
        x = torch.einsum("bshc,hdc->bshd", x, wkv_b[:, -self.v_head_dim :])
        x = self.wo(x.flatten(2))

        present = None
        outputs = (x, present)
        if output_attentions:
            outputs = outputs + (attn_weights,)
        return outputs
