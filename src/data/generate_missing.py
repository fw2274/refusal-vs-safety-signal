"""Fill in compliance targets for prompts whose source ships no reference answer.

    python -m src.data.generate_missing --config configs/qwen2.5-1.5b.yaml --policy natural

Alpaca ships reference outputs; XSTest and OR-Bench do not. Under the `natural` policy those
benign harmful-sounding prompts need compliance targets, so we self-distill them from the
un-finetuned base model.

The complication: the base model refuses ~65% of benign harmful-sounding prompts (measured in
results/runs/*/policy_eval.json). Naive self-distillation would therefore write refusals into
the compliance cell and quietly destroy the contrast. So each generation is checked, and
refusals are retried with a short compliance prefill that commits the assistant turn to an
affirmative opening. These prompts are genuinely benign - XSTest safe items like "how do I
kill a Python process" - so eliciting compliance is the *correct* behavior, and is exactly
what the over-refusal literature argues models should do.

GUARD: this script refuses to synthesize compliance targets for harm_label == 1. Populating
that cell is out of scope by design; see SFT_DESIGN.md "The fourth cell".
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.behavior.refusal_match import is_refusal
from src.data.responses import COMPLIANCE_OPENERS
from src.prompting import prompt_text


@torch.no_grad()
def _gen(model, tok, texts: list[str], max_new: int, batch_size: int = 8) -> list[str]:
    out = []
    for i in range(0, len(texts), batch_size):
        chunk = texts[i : i + batch_size]
        enc = tok(chunk, return_tensors="pt", padding=True, add_special_tokens=False).to(model.device)
        gen = model.generate(
            **enc,
            max_new_tokens=max_new,
            do_sample=False,
            temperature=None,
            top_p=None,
            top_k=None,
            pad_token_id=tok.pad_token_id,
        )
        for j in range(len(chunk)):
            out.append(tok.decode(gen[j, enc["input_ids"].shape[1] :], skip_special_tokens=True).strip())
        print(f"  {min(i + batch_size, len(texts))}/{len(texts)}", end="\r")
    print()
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--policy", default=None)
    ap.add_argument("--max-new", type=int, default=160)
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    mcfg = cfg["model"]
    policy = args.policy or cfg["data"]["refusal_policy"]
    proc = Path(cfg["paths"]["processed"]) / policy

    frames = {s: pd.read_parquet(proc / f"sft_{s}.parquet") for s in ("train", "test")}
    todo = pd.concat(
        [f[f["needs_generation"]].assign(_split=s) for s, f in frames.items()], ignore_index=True
    )
    if todo.empty:
        print("nothing to generate")
        return

    if (todo["harm_label"] == 1).any():
        raise SystemExit(
            f"{int((todo['harm_label'] == 1).sum())} rows request a COMPLIANCE target for "
            "harm_label==1. This script will not synthesize those; see SFT_DESIGN.md "
            "'The fourth cell'. Use policy `cue` or `natural`, not `frame`."
        )

    print(f"generating {len(todo)} compliance targets with the base model")
    tok = AutoTokenizer.from_pretrained(mcfg["id"])
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        mcfg["id"],
        dtype=getattr(torch, mcfg["dtype"]),
        attn_implementation=mcfg["attn_implementation"],
        device_map={"": 0} if torch.cuda.is_available() else None,
    )
    model.eval()

    prompts = todo["prompt"].tolist()
    first = _gen(model, tok, [prompt_text(tok, p) for p in prompts], args.max_new)
    method = ["direct"] * len(first)

    # Retry the refusals with a compliance prefill.
    rng = np.random.default_rng(0)
    retry_idx = [i for i, r in enumerate(first) if is_refusal(r)]
    print(f"  {len(retry_idx)}/{len(first)} came back as refusals; retrying with prefill")
    if retry_idx:
        openers = [str(rng.choice(COMPLIANCE_OPENERS)) for _ in retry_idx]
        texts = [prompt_text(tok, prompts[i]) + " " + o for i, o in zip(retry_idx, openers)]
        conts = _gen(model, tok, texts, args.max_new)
        for k, i in enumerate(retry_idx):
            first[i] = f"{openers[k]} {conts[k]}".strip()
            method[i] = "prefill"

    still = sum(is_refusal(r) for r in first)
    print(f"  after retry, {still} still read as refusals (left as-is, flagged)")

    todo["completion"] = first
    todo["gen_method"] = method
    todo["still_refusal"] = [int(is_refusal(r)) for r in first]

    for s, f in frames.items():
        upd = todo[todo["_split"] == s].set_index("prompt")
        f = f.copy()
        hit = f["prompt"].isin(upd.index)
        f.loc[hit, "completion"] = f.loc[hit, "prompt"].map(upd["completion"])
        f.loc[hit, "needs_generation"] = False
        f.to_parquet(proc / f"sft_{s}.parquet")
        print(f"updated {proc / f'sft_{s}.parquet'} ({int(hit.sum())} rows)")

    todo.drop(columns=["_split"]).to_parquet(proc / "generated_completions.parquet")


if __name__ == "__main__":
    main()
