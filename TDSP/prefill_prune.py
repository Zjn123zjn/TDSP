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
    key_states = repeat_kv(key_states, num_key_value_groups)

    # Use the same predictor that supplies decode-time token scores. Pooling its
    # low-dimensional Q/K features keeps prefill selection segment- and block-wise.
    producer = attn_module.producer
    if producer is None or producer.q_importance is None:
        raise RuntimeError("The shared TokenButler predictor has no prefill Q scores")
    slot_idx = attn_module._get_group_slot_index()
    q_features = producer.q_importance[:, slot_idx].view(
        bsz, num_heads, seqlen, attn_module.dDash
    )[:, :, :pool_len].float()
    key_proj = producer.sparse_token_predictor.key_cache_proj[
        attn_module.layer_idx
    ]
    key_features = key_states.to(key_proj.device, dtype=key_proj.dtype)
    key_features = torch.einsum("bhlk,hkd->bhld", key_features, key_proj)
    key_features = key_features.to(query_states.device, dtype=torch.float32)
    q_features = q_features.reshape(
        bsz, num_heads, pool_len // segment_size, segment_size, attn_module.dDash
    )
    key_features = key_features.reshape(
        bsz, num_heads, pool_len // block_size, block_size, attn_module.dDash
    )

    layer_max_q = q_features.max(dim=-2).values
    layer_max_k = key_features.max(dim=-2).values
    layer_min_q = q_features.min(dim=-2).values
    layer_min_k = key_features.min(dim=-2).values

    q_block_len = layer_max_q.size(-2)
    k_block_len = layer_max_k.size(-2)
    scale = math.sqrt(attn_module.dDash)
    qmax_kmax = torch.matmul(layer_max_q, layer_max_k.transpose(2, 3))
    qmin_kmin = torch.matmul(layer_min_q, layer_min_k.transpose(2, 3))
    qmax_kmin = torch.matmul(layer_max_q, layer_min_k.transpose(2, 3))
    qmin_kmax = torch.matmul(layer_min_q, layer_max_k.transpose(2, 3))
    attn_weights = torch.maximum(
        torch.maximum(qmax_kmax, qmin_kmin),
        torch.maximum(qmax_kmin, qmin_kmax),
    ) / scale

    attn_weights = attn_weights.view(
        bsz, num_heads_k, num_key_value_groups, num_segments, num_blocks
    ).mean(dim=2)

    block_ids = torch.arange(num_blocks, device=attn_weights.device)
    segment_ids = torch.arange(num_segments, device=attn_weights.device)
    # Current-segment KV is retained as the local window. Rank only complete,
    # strictly preceding blocks so the selector cannot see future cache entries.
    visible_blocks = (segment_ids * nrep).clamp_min(1)[:, None]
    attn_weights = attn_weights.masked_fill(
        block_ids[None, :] >= visible_blocks,
        torch.finfo(attn_weights.dtype).min,
    )

    attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)

    if getattr(attn_module, "prefill_layer_fusion", False):
        prev = getattr(cache_state, "prefill_prev_attn_weights", None)
        if prev is None or attn_module.layer_idx <= 1:
            cache_state.prefill_prev_attn_weights = attn_weights
        else:
            blended = prev.to(attn_weights.device) * 0.75 + attn_weights * 0.25
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


def hybrid_prefill_attention(
    attn_module,
    cache_state,
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    value_states: torch.Tensor,
) -> torch.Tensor:
    """
    CritiPrefill-style segmented prefill attention using PyTorch SDPA.

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

    # Predictor owners produce Q features only after their own attention call.
    # Keep that layer dense if a caller sets layer_skip=0; later layers still
    # use the shared predictor without creating a cyclic dependency.
    if getattr(attn_module, "producer", None) is None or getattr(
        attn_module.producer, "q_importance", None
    ) is None:
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
                available_blocks = (k_segment.size(2) - segment_size) // block_size
                if available_blocks > 0:
                    index = block_index[:, :, curr_segment, :]
                    # Top-k is ranked with causal-invalid blocks at -inf, so
                    # the first entries are valid. Limit the budget to blocks
                    # visible before this segment. Keep KV-head indexing here;
                    # SDPA performs the GQA expansion for the attention call.
                    index = index[:, :, : min(index.size(-1), available_blocks)]
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
