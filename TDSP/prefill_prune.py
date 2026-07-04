import math
from typing import Optional

import torch
import torch.nn.functional as F
from torch.nn.attention.bias import causal_lower_right

from transformers.models.llama.modeling_llama import repeat_kv


def _cache_profiling(
    attn_module,
    cache_state,
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    block_size: int,
    budget_size: int,
    segment_size: int,
) -> torch.Tensor:
    """
    Estimate block importance for each query segment.

    Args:
        query_states: [B, H, L, D]
        key_states: [B, H_kv, L, D]

    Returns:
        Top-k block indices with shape [B, H_kv, num_segments, topk].
    """
    assert segment_size >= block_size

    bsz, num_heads, seqlen, head_dim = query_states.shape
    num_heads_k = key_states.size(1)
    num_key_value_groups = num_heads // num_heads_k
    nrep = segment_size // block_size

    num_segments = (seqlen + segment_size - 1) // segment_size
    pool_len = (num_segments - 1) * segment_size
    num_blocks = pool_len // block_size
    num_segments = pool_len // segment_size

    if pool_len <= 0 or num_blocks <= 0:
        return torch.empty(
            bsz,
            num_heads_k,
            0,
            0,
            device=query_states.device,
            dtype=torch.long,
        )

    query_states = query_states[:, :, :pool_len, :]
    key_states = key_states[:, :, :pool_len, :]

    query_states = query_states.reshape(
        bsz, num_heads, pool_len // segment_size, segment_size, head_dim
    )

    key_states = repeat_kv(key_states, num_key_value_groups)
    key_states = key_states.reshape(
        bsz, num_heads, pool_len // block_size, block_size, head_dim
    )

    layer_max_q = query_states.max(dim=-2).values
    layer_max_k = key_states.max(dim=-2).values
    layer_min_q = query_states.min(dim=-2).values
    layer_min_k = key_states.min(dim=-2).values

    q_block_len = layer_max_q.size(-2)
    k_block_len = layer_max_k.size(-2)
    qq = torch.cat([layer_max_q, layer_min_q], dim=-2)
    attn_weights_max = (
        torch.matmul(qq, layer_max_k.transpose(2, 3))
        .view(bsz, num_heads, 2, q_block_len, k_block_len)
        .mean(dim=2)
    )
    attn_weights_min = (
        torch.matmul(qq, layer_min_k.transpose(2, 3))
        .view(bsz, num_heads, 2, q_block_len, k_block_len)
        .mean(dim=2)
    )
    attn_weights = torch.max(attn_weights_max, attn_weights_min)

    attn_weights = attn_weights.view(
        bsz, num_heads_k, num_key_value_groups, num_segments, num_blocks
    ).mean(dim=2)

    mask = torch.full(
        (num_blocks, num_blocks),
        torch.finfo(attn_weights.dtype).min,
        device=attn_weights.device,
    )
    mask_cond = torch.arange(mask.size(-1), device=attn_weights.device)
    mask.masked_fill_(mask_cond + nrep < (mask_cond + 1).view(mask.size(-1), 1), 0)
    mask = mask.view(num_segments, nrep, num_blocks)[:, -1, :]
    attn_weights = attn_weights + mask[None, None, :, :]

    attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)

    if getattr(attn_module, "prefill_layer_fusion", False):
        prev = getattr(cache_state, "prefill_prev_attn_weights", None)
        if prev is None or attn_module.layer_idx <= 1:
            cache_state.prefill_prev_attn_weights = attn_weights
        else:
            blended = prev.to(attn_weights.device) * 0.25 + attn_weights * 0.75
            cache_state.prefill_prev_attn_weights = blended
            attn_weights = blended

    topk = max(1, budget_size // block_size)
    topk = min(topk, attn_weights.size(-1))
    return torch.topk(attn_weights, topk, dim=-1).indices


def _cache_selection(
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    index: torch.Tensor,
    block_size: int,
    left_over: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Select important KV blocks.

    Args:
        key_states/value_states: [B, H, L, D]
        index: [B, H, topk]
    """
    bsz, num_heads, seqlen, head_dim = key_states.shape
    usable = seqlen - left_over
    if usable <= 0:
        return key_states, value_states

    reshaped_k = key_states[:, :, :usable, :].reshape(bsz, num_heads, -1, block_size, head_dim)
    reshaped_v = value_states[:, :, :usable, :].reshape(bsz, num_heads, -1, block_size, head_dim)

    gather_index = index.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, -1, block_size, head_dim)
    gather_index = gather_index.to(device=key_states.device, dtype=torch.long)

    selected_k = torch.gather(reshaped_k, dim=2, index=gather_index)
    selected_v = torch.gather(reshaped_v, dim=2, index=gather_index)

    selected_k = selected_k.reshape(bsz, num_heads, -1, head_dim)
    selected_v = selected_v.reshape(bsz, num_heads, -1, head_dim)

    if left_over > 0:
        selected_k = torch.cat([selected_k, key_states[:, :, -left_over:, :]], dim=2)
        selected_v = torch.cat([selected_v, value_states[:, :, -left_over:, :]], dim=2)
    return selected_k, selected_v


def segmented_prefill_attention(
    attn_module,
    cache_state,
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    value_states: torch.Tensor,
) -> torch.Tensor:
    """
    Segmented block-based prefill attention using PyTorch SDPA.

    Args:
        query_states/key_states/value_states: [B, H, L, D]

    Returns:
        Attention output: [B, H, L, D]
    """
    bsz, _, input_len, _ = query_states.shape
    segment_size = attn_module.prefill_segment_size
    threshold_len = attn_module.prefill_threshold_len
    block_size = attn_module.prefill_block_size
    budget = attn_module.prefill_budget

    if input_len <= 1 or input_len < segment_size or input_len < threshold_len:
        return F.scaled_dot_product_attention(
            query_states,
            key_states,
            value_states,
            attn_mask=None,
            is_causal=True,
            enable_gqa=True,
        )

    if cache_state is None:
        class _CacheShim:
            pass

        cache_state = _CacheShim()

    if getattr(attn_module, "prefill_layer_fusion", False) and not hasattr(cache_state, "prefill_prev_attn_weights"):
        cache_state.prefill_prev_attn_weights = None

    block_index = _cache_profiling(
        attn_module,
        cache_state,
        query_states,
        key_states[:, : attn_module.num_key_value_heads, :, :],
        block_size=block_size,
        budget_size=budget,
        segment_size=segment_size,
    )

    attn_outputs = []
    for i in range(0, input_len, segment_size):
        q_segment = query_states[:, :, i : i + segment_size, :]
        k_segment = key_states[:, :, : i + segment_size, :]
        v_segment = value_states[:, :, : i + segment_size, :]

        if (
            i >= threshold_len
            and i + segment_size < input_len
            and block_index.numel() > 0
        ):
            curr_segment = i // segment_size
            if curr_segment < block_index.size(2):
                index = block_index[:, :, curr_segment, :]
                available_blocks = (k_segment.size(2) - segment_size) // block_size
                if available_blocks > 0:
                    # Profiling runs over the full prefill window, but each segment can
                    # only gather from blocks that are already visible causally.
                    index = index.clamp_max(available_blocks - 1)
                    k_segment, v_segment = _cache_selection(
                        k_segment,
                        v_segment,
                        index=index,
                        block_size=block_size,
                        left_over=segment_size,
                    )

        attn_output = F.scaled_dot_product_attention(
            q_segment,
            k_segment,
            v_segment,
            attn_mask=causal_lower_right(q_segment.size(2), k_segment.size(2)),
            is_causal=False,
            enable_gqa=True,
        )
        attn_outputs.append(attn_output)

    return torch.cat(attn_outputs, dim=2).reshape(bsz, attn_module.num_heads, input_len, attn_module.head_dim)
