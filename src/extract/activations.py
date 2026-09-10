"""Phase 3: extract last-prompt-token hidden states from every layer.

    python -m src.extract.activations --config configs/qwen2.5-1.5b.yaml \
        --adapter results/runs/Qwen2.5-1.5B-Instruct-cue/adapter --run cue
    python -m src.extract.activations --config configs/qwen2.5-1.5b.yaml --run base

Writes results/activations/<run>/L01.npy ... L28.npy, each fp32 [N, d_model], with row order
pinned by ids.json and the label frame saved alongside.

Two correctness points, both of which fail silently if you get them wrong:

1. POSITION IDS. We batch with left padding so the last real token sits at index -1. But a
   plain forward() does NOT derive position_ids from the attention mask - it uses
   arange(seq_len), which shifts every padded sequence's RoPE positions by the number of pad
   tokens. Activations then depend on what else happened to be in the batch. We pass
   position_ids computed from the mask. `--verify-padding` checks this empirically by
   comparing batched extraction against unpadded batch-size-1.

2. LAYER INDEXING. hidden_states has n_layers+1 entries; index 0 is the embedding output.
   Layer L (1-indexed, L=1..n_layers) is hidden_states[L].

Saved as fp32, but NOT because that recovers precision - it cannot. The forward pass runs in
bf16, so every stored value is already exactly bf16-representable; casting fp32->bf16->fp32
on this data is lossless (measured: median and p99 relative error 0.0, and diff-in-means AUROC
at layer 12 is identical to 6 decimal places either way).

fp32 is for the ARITHMETIC that follows, not the storage. Means over 1000+ vectors, dot
products across 1536 dims, covariance estimates and the matrix inverse inside LEACE all
accumulate rounding error. Measured on layer 12: a bf16-accumulated mean over 1000 vectors
differs from the fp32 mean by up to 4.0e-2 absolute (1.4e-3 median relative). Also practical -
numpy has no native bfloat16 dtype, so .npy round-tripping bf16 needs ml_dtypes or uint16
casts. See GLOSSARY.md "fp32 vs bf16".
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.prompting import prompt_text


def load_model(mcfg: dict, adapter: str | None):
    tok = AutoTokenizer.from_pretrained(mcfg["id"])
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"  # last real token at index -1
    model = AutoModelForCausalLM.from_pretrained(
        mcfg["id"],
        dtype=getattr(torch, mcfg["dtype"]),
        attn_implementation=mcfg["attn_implementation"],
        device_map={"": 0} if torch.cuda.is_available() else None,
    )
    if adapter:
        model = PeftModel.from_pretrained(model, adapter)
    model.eval()
    return model, tok


@torch.no_grad()
def _forward_last_token(model, tok, texts: list[str], n_layers: int) -> list[np.ndarray]:
    """Return [n_layers] arrays of shape [len(texts), d_model] for the final real token."""
    enc = tok(texts, return_tensors="pt", padding=True, add_special_tokens=False).to(model.device)
    # Derive positions from the mask: pads sit at 0, real tokens count 0..n-1 left to right.
    position_ids = (enc["attention_mask"].cumsum(-1) - 1).clamp(min=0)
    out = model(
        input_ids=enc["input_ids"],
        attention_mask=enc["attention_mask"],
        position_ids=position_ids,
        output_hidden_states=True,
        use_cache=False,
    )
    hs = out.hidden_states  # length n_layers + 1; index 0 = embeddings
    assert len(hs) == n_layers + 1, f"expected {n_layers + 1} hidden_states, got {len(hs)}"
    return [hs[L][:, -1, :].float().cpu().numpy() for L in range(1, n_layers + 1)]


@torch.no_grad()
def extract(model, tok, prompts: list[str], n_layers: int, batch_size: int = 16) -> np.ndarray:
    """Return array of shape [n_layers, N, d_model]."""
    buf: list[list[np.ndarray]] = [[] for _ in range(n_layers)]
    for i in range(0, len(prompts), batch_size):
        texts = [prompt_text(tok, p) for p in prompts[i : i + batch_size]]
        for L, h in enumerate(_forward_last_token(model, tok, texts, n_layers)):
            buf[L].append(h)
        print(f"  {min(i + batch_size, len(prompts))}/{len(prompts)}", end="\r")
    print()
    return np.stack([np.concatenate(b, axis=0) for b in buf], axis=0)


@torch.no_grad()
def verify_padding(model, tok, prompts: list[str], n_layers: int) -> None:
    """Batched-with-padding must match unpadded batch-size-1.

    If position_ids were left at the default arange, cosine similarity here drops well below
    1.0 for the shorter (more heavily padded) prompts.
    """
    # Deliberately mix lengths so padding is substantial.
    probe = sorted(prompts, key=len)[:4] + sorted(prompts, key=len)[-4:]
    texts = [prompt_text(tok, p) for p in probe]

    batched = _forward_last_token(model, tok, texts, n_layers)
    single = [_forward_last_token(model, tok, [t], n_layers) for t in texts]

    print("padding-invariance check (cosine similarity, batched vs unpadded bs=1):")
    worst = 1.0
    for L in (1, n_layers // 2, n_layers):
        cs = []
        for j, t in enumerate(texts):
            a, b = batched[L - 1][j], single[j][L - 1][0]
            cs.append(float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12)))
        worst = min(worst, min(cs))
        print(f"  L{L:02d}: min={min(cs):.6f} mean={np.mean(cs):.6f}")
    verdict = "PASS" if worst > 0.999 else "FAIL - position_ids or padding side is wrong"
    print(f"  -> {verdict} (worst {worst:.6f})")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--policy", default=None, help="data subdir; defaults to config data.refusal_policy")
    ap.add_argument("--adapter", default=None, help="omit for the off-the-shelf base model")
    ap.add_argument("--run", required=True, help="output subdirectory name, e.g. cue / base")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--verify-padding", action="store_true")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    mcfg = cfg["model"]
    n_layers = mcfg["n_layers"]

    policy_dir = args.policy or cfg["data"]["refusal_policy"]
    df = pd.read_parquet(Path(cfg["paths"]["processed"]) / policy_dir / "all_prompts.parquet")
    df = df.reset_index(drop=True)
    prompts = df["prompt"].tolist()

    print(f"Loading {mcfg['id']}" + (f" + adapter {args.adapter}" if args.adapter else " (base)"))
    model, tok = load_model(mcfg, args.adapter)

    if args.verify_padding:
        verify_padding(model, tok, prompts, n_layers)

    print(f"Extracting {n_layers} layers x {len(prompts)} prompts ...")
    acts = extract(model, tok, prompts, n_layers, args.batch_size)

    out_dir = Path("results/activations") / args.run
    out_dir.mkdir(parents=True, exist_ok=True)
    for L in range(1, n_layers + 1):
        np.save(out_dir / f"L{L:02d}.npy", acts[L - 1])

    df.to_parquet(out_dir / "labels.parquet")
    (out_dir / "ids.json").write_text(json.dumps({"n": len(df), "order": "row index of labels.parquet"}))
    (out_dir / "meta.json").write_text(
        json.dumps(
            {
                "model": mcfg["id"],
                "adapter": args.adapter,
                "n_prompts": len(prompts),
                "n_layers": n_layers,
                "d_model": int(acts.shape[2]),
                "dtype_saved": "float32",
                "token_position": "last prompt token (add_generation_prompt=True)",
                "layer_convention": "L{i}.npy == hidden_states[i]; index 0 (embeddings) not saved",
                "position_ids": "derived from attention_mask (left padding safe)",
            },
            indent=2,
        )
    )
    size_mb = sum(f.stat().st_size for f in out_dir.glob("L*.npy")) / 1e6
    print(f"wrote {out_dir}  ({acts.shape[2]} dims, {size_mb:.0f} MB)")


if __name__ == "__main__":
    main()
