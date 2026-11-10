import argparse
import math
import random
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from hybrid_loader import (
    DEFAULT_MODELS,
    collect_predictor_loss,
    freeze_backbone,
    load_hybrid_model,
    predictor_parameters,
    save_predictor_checkpoint,
    set_hybrid_prefill,
    set_inference_mode,
)


def load_local_corpus(corpus_dir: Path) -> str:
    parts = []
    for path in sorted(corpus_dir.glob("*.txt")):
        parts.append(path.read_text(encoding="utf-8", errors="ignore"))
    if not parts:
        raise FileNotFoundError(f"No .txt files found in {corpus_dir}")
    return "\n\n".join(parts)


def build_training_chunks(tokenizer, text: str, seq_len: int, stride: int) -> list[list[int]]:
    token_ids = tokenizer(text, add_special_tokens=False).input_ids
    chunks = []
    for start in range(0, max(1, len(token_ids) - seq_len), stride):
        chunk = token_ids[start : start + seq_len]
        if len(chunk) == seq_len:
            chunks.append(chunk)
    if not chunks:
        raise RuntimeError("Corpus is too short for the requested sequence length.")
    return chunks


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default=DEFAULT_MODELS["1b"])
    parser.add_argument("--seq-len", type=int, default=1024)
    parser.add_argument("--stride", type=int, default=512)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dtype", type=str, default="bfloat16")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--token-sparsity", type=str, default="fixed_50pc")
    parser.add_argument("--d-dash", type=int, default=16)
    parser.add_argument("--intdim", type=int, default=512)
    parser.add_argument("--attn-reduce-factor", type=int, default=8)
    parser.add_argument("--head-attn-reduce-factor", type=int, default=2)
    parser.add_argument(
        "--corpus-dir",
        type=Path,
        default=ROOT.parent / "CritiPrefill" / "experiments" / "needle" / "data" / "PaulGrahamEssays",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "checkpoints" / "llama3_1b_predictor_local.pt",
    )
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    model, tokenizer, _ = load_hybrid_model(
        model_name=args.model,
        dtype_name=args.dtype,
        device_map=None,
        device=args.device,
        token_sparse_method=args.token_sparsity,
        d_dash=args.d_dash,
        intdim=args.intdim,
        attn_reduce_factor=args.attn_reduce_factor,
        head_attn_reduce_factor=args.head_attn_reduce_factor,
    )
    set_hybrid_prefill(model, enabled=False)
    set_inference_mode(model, enabled=False)
    freeze_backbone(model)

    embed_device = torch.device(args.device)
    corpus = load_local_corpus(args.corpus_dir)
    chunks = build_training_chunks(tokenizer, corpus, seq_len=args.seq_len, stride=args.stride)
    optimizer = torch.optim.AdamW(list(predictor_parameters(model)), lr=args.lr)

    model.train()
    for step in range(1, args.steps + 1):
        sample = random.choice(chunks)
        batch = torch.tensor(sample, dtype=torch.long, device=embed_device).unsqueeze(0)

        optimizer.zero_grad(set_to_none=True)
        _ = model(input_ids=batch, use_cache=True)
        loss = collect_predictor_loss(model)
        loss.backward()
        optimizer.step()

        if step == 1 or step % 10 == 0 or step == args.steps:
            print(f"step={step:04d} loss={loss.item():.6f}", flush=True)

    save_predictor_checkpoint(
        model,
        str(args.output),
        metadata={
            "model": args.model,
            "seq_len": args.seq_len,
            "steps": args.steps,
            "token_sparsity": args.token_sparsity,
            "decode_prior_weight": 2.0,
            "training_prior_mix": "sampled_prefill_plus_weighted_final_prompt_query",
            "d_dash": args.d_dash,
            "intdim": args.intdim,
            "corpus_dir": str(args.corpus_dir),
        },
    )
    print(f"saved predictor checkpoint to {args.output}", flush=True)


if __name__ == "__main__":
    main()
