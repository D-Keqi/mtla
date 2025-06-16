# -*- coding: utf-8 -*-
"""
Created on Mon May 19 00:50:20 2025

@author: Keqi Deng (University of Cambridge)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Dict, Optional
import uuid


class MultiheadTemporalLatentAttention(nn.Module):
    """
    Multi-head Temporal Latent Attention (MTLA)

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
        recompute_prompt_attn (bool): If True, recomputes attention over prompt in decoder
            self-attention (query=[prompt,x], key/value=[prompt,x]). If False (default), only
            attends to new tokens (query=x, key/value=[prompt,x]). This depends on the
            specific design choices of the decoder-only system. For typical autoregressive
            language models, this is normally set to True.
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
        recompute_prompt_attn=False,
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

        self.init_incremental_state()

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

            tricky_mask = None
            if prev_kv_t is not None and prev_k_pe is not None:
                if T_remain != 1:
                    prev_kv_t[:, -1:] += kv_norm * w_tT  # Update KV cache
                    prev_k_pe[:, -1:] = k_pe  # Update
                else:
                    prev_kv_t = torch.cat([prev_kv_t, kv_norm * w_tT], dim=1)  # Concat
                    prev_k_pe = torch.cat([prev_k_pe, k_pe], dim=1)  # Concat

                saved_state["prev_kv_t"] = prev_kv_t
                saved_state["prev_k_pe"] = prev_k_pe
                infer_steps = infer_steps + 1

            else:
                # Correspond to the first token inference
                if key.shape[1] != 1:
                    indices = list(range(self.down_rate - 1, T, self.down_rate))
                    if T - 1 not in indices:
                        indices.append(T - 1)

                    # Only the first token generation needs to account for different computation methods of the prefix prompt
                    if self.recompute_prompt_attn:

                        # When recomputing self-attention for the fixed prompt, "train" is set True
                        # to match training but KV cache is still compressed
                        w_tT = self.hypernet_down(
                            T, t, kv_norm.device, train=True, T_input=kv_norm
                        )
                        zero_mask = (
                            self.generate_chunk_mask(T, self.down_rate)
                            .to(k_pe.device)
                            .unsqueeze(0)
                            .to(kv_norm.dtype)
                        )
                        prev_kv_t = torch.matmul(w_tT * zero_mask, kv_norm)
                        prev_k_pe = k_pe
                        saved_state["prev_kv_t"] = prev_kv_t[:, indices]
                        saved_state["prev_k_pe"] = prev_k_pe[:, indices]

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
                        saved_state["prev_kv_t"] = prev_kv_t
                        saved_state["prev_k_pe"] = prev_k_pe

                else:
                    prev_kv_t = kv_norm * w_tT
                    prev_k_pe = k_pe
                    saved_state["prev_kv_t"] = prev_kv_t
                    saved_state["prev_k_pe"] = prev_k_pe

                infer_steps = kv_norm.new_zeros(kv_norm.shape[0]) + T

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
            if tricky_mask is not None:
                scores = scores + tricky_mask.unsqueeze(0).unsqueeze(2).to(scores.dtype)
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

            return x

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

            return x

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

    def init_incremental_state(self):
        self._incremental_state_id = str(uuid.uuid4())

    def _get_full_incremental_state_key(self, key: str) -> str:
        return "{}.{}".format(self._incremental_state_id, key)

    def get_incremental_state(
        self,
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]],
        key: str,
    ) -> Optional[Dict[str, Optional[Tensor]]]:
        """Helper for getting incremental state for an nn.Module."""
        full_key = self._get_full_incremental_state_key(key)
        if incremental_state is None or full_key not in incremental_state:
            return None
        return incremental_state[full_key]

    def set_incremental_state(
        self,
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]],
        key: str,
        value: Dict[str, Optional[Tensor]],
    ) -> Optional[Dict[str, Dict[str, Optional[Tensor]]]]:
        """Helper for setting incremental state for an nn.Module."""
        if incremental_state is not None:
            full_key = self._get_full_incremental_state_key(key)
            incremental_state[full_key] = value
        return incremental_state

    def _get_input_buffer(
        self, incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]]
    ) -> Dict[str, Optional[Tensor]]:
        result = self.get_incremental_state(incremental_state, "attn_state")
        if result is not None:
            return result
        else:
            empty_result: Dict[str, Optional[Tensor]] = {}
            return empty_result

    def _set_input_buffer(
        self,
        incremental_state: Dict[str, Dict[str, Optional[Tensor]]],
        buffer: Dict[str, Optional[Tensor]],
    ):
        return self.set_incremental_state(incremental_state, "attn_state", buffer)

    def reorder_incremental_state(
        self, incremental_state: Dict[str, Dict[str, Optional[Tensor]]], new_order
    ):
        """Reorder buffered internal state (for incremental generation)."""
        input_buffer = self._get_input_buffer(incremental_state)
        if input_buffer is not None:
            for k in input_buffer.keys():
                if input_buffer[k] is not None:
                    input_buffer[k] = input_buffer[k].index_select(0, new_order)
            incremental_state = self._set_input_buffer(incremental_state, input_buffer)
        return incremental_state


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
