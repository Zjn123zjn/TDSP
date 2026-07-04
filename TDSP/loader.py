import json
import os
from pathlib import Path
from typing import Dict, Iterable, Optional

import torch
from huggingface_hub import snapshot_download
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from modify_models.modify_llama import convert_kvcache_experimental


DEFAULT_MODELS = {
    "1b": "unsloth/Llama-3.2-1B-Instruct",
    "3b": "unsloth/Llama-3.2-3B-Instruct",
    "8b": "unsloth/Llama-3.1-8B-Instruct",
}


def set_hf_mirror() -> None:
    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def _dtype_from_name(dtype_name: str) -> torch.dtype:
    mapping = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    if dtype_name not in mapping:
        raise ValueError(f"Unsupported dtype: {dtype_name}")
    return mapping[dtype_name]


def _attention_modules(model):
    for module in model.modules():
        if module.__class__.__name__.endswith("AttentionExperimental"):
            yield module


def load_hybrid_model(
    model_name: str,
    dtype_name: str = "bfloat16",
    device_map: Optional[str] = "auto",
    device: Optional[str] = None,
    producer_frequency: Optional[int] = None,
    d_dash: int = 16,
    intdim: int = 512,
    attn_reduce_factor: int = 8,
    head_attn_reduce_factor: int = 2,
    token_sparse_method: Optional[str] = "fixed_50pc",
    sliding_window: int = 128,
    min_sparse_index: int = 8,
):
    set_hf_mirror()
    torch_dtype = _dtype_from_name(dtype_name)
    local_model_path = snapshot_download(
        repo_id=model_name,
        endpoint=os.environ["HF_ENDPOINT"],
        resume_download=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(local_model_path, trust_remote_code=True, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    config = AutoConfig.from_pretrained(local_model_path, trust_remote_code=True)
    load_kwargs = {
        "torch_dtype": torch_dtype,
        "trust_remote_code": True,
    }
    if device_map is not None:
        load_kwargs["device_map"] = device_map
    model = AutoModelForCausalLM.from_pretrained(local_model_path, **load_kwargs)
    num_layers = getattr(config, "num_hidden_layers", None)
    if num_layers is None:
        num_layers = getattr(config, "num_layers")
    pf = producer_frequency or num_layers
    model = convert_kvcache_experimental(model, config, pf)
    if device_map is None and device is not None:
        model = model.to(device)

    for module in _attention_modules(model):
        module.eval_llm_mode = "ExpPred"
        module.producer_frequency = pf
        module.dDash = d_dash
        module.intdim = intdim
        module.attn_reduce_factor = attn_reduce_factor
        module.head_attn_reduce_factor = head_attn_reduce_factor
        module.train_headpredictor = False
        module.late_context_upweight = False
        module.softmax_causal_loss_mse = False
        module.softmax_causal_loss_ce = True
        module.pairwise_loss = False
        module.pairwise_ce_loss = False
        module.min_sparse_index = min_sparse_index
        module.lookahead = 0
        module.num_layers_pred = module.producer_frequency
        module.sliding_window = sliding_window
        module.max_loss_rows = 256
        module.predictor_variant = "attentive_project"
        module.token_sparse_method = token_sparse_method
        if token_sparse_method:
            module.set_token_sparsity()
        else:
            module.target_sparsity = 0.0
            module.target_keep_tokens = None
            module.sparse_aggression = 1.0
            module.head_keep = None
            module.effective_sparsity = 0.0
        if module.layer_idx % pf == 0:
            module.update_predictor()

    return model, tokenizer, config


def freeze_backbone(model) -> None:
    model.requires_grad_(False)
    for module in _attention_modules(model):
        if getattr(module, "is_predictor_owner", False):
            module.sparse_token_predictor.requires_grad_(True)
            if getattr(module, "train_headpredictor", False):
                module.sparse_head_predictor.requires_grad_(True)


def predictor_parameters(model) -> Iterable[torch.nn.Parameter]:
    for module in _attention_modules(model):
        if getattr(module, "is_predictor_owner", False):
            yield from module.sparse_token_predictor.parameters()
            if getattr(module, "train_headpredictor", False):
                yield from module.sparse_head_predictor.parameters()


def set_inference_mode(model, enabled: bool) -> None:
    for module in _attention_modules(model):
        module.inference_mode = enabled


def set_token_sparsity(model, token_sparse_method: Optional[str]) -> None:
    for module in _attention_modules(model):
        module.token_sparse_method = token_sparse_method
        if token_sparse_method:
            module.set_token_sparsity()
        else:
            module.target_sparsity = 0.0
            module.target_keep_tokens = None
            module.sparse_aggression = 1.0
            module.head_keep = None
            module.effective_sparsity = 0.0


def set_prefill_prune(
    model,
    enabled: bool,
    segment_size: int = 512,
    threshold_len: int = 4096,
    block_size: int = 32,
    budget: int = 2048,
    layer_fusion: bool = True,
    layer_skip: int = 1,
) -> None:
    for module in _attention_modules(model):
        module.prefill_prune_enabled = enabled
        module.prefill_segment_size = segment_size
        module.prefill_threshold_len = threshold_len
        module.prefill_block_size = block_size
        module.prefill_budget = budget
        module.prefill_layer_fusion = layer_fusion
        module.prefill_layer_skip = layer_skip


def collect_predictor_loss(model) -> torch.Tensor:
    losses = []
    for module in _attention_modules(model):
        loss = getattr(module, "msemagn_loss", None)
        if isinstance(loss, torch.Tensor):
            losses.append(loss)
    if not losses:
        raise RuntimeError("No predictor losses were collected from the model.")
    return torch.stack(losses).mean()


def mean_effective_sparsity(model) -> float:
    values = []
    for module in _attention_modules(model):
        if getattr(module, "layer_idx", 0) == 0:
            continue
        value = getattr(module, "effective_sparsity", None)
        if value is not None:
            values.append(float(value))
    return float(sum(values) / len(values)) if values else 0.0


def predictor_state_dict(model) -> Dict[str, torch.Tensor]:
    state = model.state_dict()
    return {
        key: value.detach().cpu()
        for key, value in state.items()
        if "sparse_token_predictor" in key or "sparse_head_predictor" in key
    }


def save_predictor_checkpoint(model, path: str, metadata: Optional[Dict] = None) -> None:
    path_obj = Path(path)
    path_obj.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "state_dict": predictor_state_dict(model),
        "metadata": metadata or {},
    }
    torch.save(payload, path_obj)
    meta_path = path_obj.with_suffix(path_obj.suffix + ".json")
    meta_path.write_text(json.dumps(payload["metadata"], indent=2), encoding="utf-8")


def load_predictor_checkpoint(model, path: str) -> Dict:
    payload = torch.load(path, map_location="cpu")
    state_dict = payload["state_dict"] if "state_dict" in payload else payload
    model.load_state_dict(state_dict, strict=False)
    return payload.get("metadata", {})
