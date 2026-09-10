"""Phase 6: is the refusal direction one the model USES, or only one that correlates?

    python -m src.causal.ablate --config configs/qwen2.5-1.5b.yaml \
        --adapter results/runs/Qwen2.5-1.5B-Instruct-cue/adapter --run cue

Difference-in-means will hand back a confident-looking vector even from label noise. Separating
activations is not the same as driving behavior, so before Phase 7 removes this direction and
claims something about what survives, we have to show the model actually reads it.

THE INTERVENTION. The naive version - h <- h - (h.r)r, i.e. zero the projection - is wrong here,
and measurably so. The decision boundary along r does not sit at zero: the complied group sits
well below it. Zeroing therefore moves every prompt onto the REFUSE side, and the first run of
this script duly pushed alpaca from 0.000 to 0.950 refusal instead of reducing anything.

So we set the projection to a measured reference instead:

    h <- h - ((h.r_L) - target_L) * r_L

with per-layer r_L and per-layer target_L, both estimated from train rows inside the benign
stratum. target = the complied group's mean projection SUPPRESSES refusal; target = the refused
group's mean projection INDUCES it. Scale-free, so it can be applied at every layer without the
compounding that broke the naive additive version (alpha applied 28 times over).

Conditions, all on the same held-out prompts with greedy decoding:

  baseline           no intervention
  suppress_refusal   every layer's projection onto r_L set to the complied reference
  induce_refusal     ... set to the refused reference
  suppress_random    identical operation along a random unit vector per layer, with that
                     vector's own complied reference. This is the control that gives the result
                     meaning: any intervention perturbs the model, so only an effect far
                     exceeding this one is evidence about refusal specifically.

Coherence is tracked too. If refusal changes only because the model stopped producing usable
text, the change says nothing about refusal - so a degenerate-output rate is reported and gated.
"""

from __future__ import annotations

import argparse
import json
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.behavior.refusal_match import is_refusal
from src.train.eval_policy import generate


def decoder_layers(model) -> list:
    """Transformer blocks in order, found by class name so PEFT wrappers don't matter.

    decoder_layers()[i] produces hidden_states[i+1], i.e. our 1-indexed layer L = i+1.
    """
    return [m for _, m in model.named_modules() if type(m).__name__.endswith("DecoderLayer")]


def make_hook(r: torch.Tensor, target: float):
    """Set the component of h along r to `target`, leaving the orthogonal complement intact."""
    def hook(module, args, output):
        is_tuple = isinstance(output, tuple)
        h = output[0] if is_tuple else output
        rr = r.to(dtype=h.dtype, device=h.device)
        h = h - ((h @ rr) - target).unsqueeze(-1) * rr
        return (h,) + tuple(output[1:]) if is_tuple else h
    return hook


@contextmanager
def intervene(model, dirs: dict[int, tuple[torch.Tensor, float]]):
    """dirs maps 1-indexed layer -> (unit direction, target projection)."""
    layers = decoder_layers(model)
    handles = []
    for i, mod in enumerate(layers, start=1):
        if i in dirs:
            r, t = dirs[i]
            handles.append(mod.register_forward_hook(make_hook(r, t)))
    try:
        yield len(handles)
    finally:
        for h in handles:
            h.remove()


def degenerate_rate(responses: list[str]) -> float:
    """Fraction of outputs that are empty, trivially short, or a repeated token."""
    bad = 0
    for t in responses:
        w = t.split()
        if len(t.strip()) < 10 or len(w) < 3 or len(set(w)) <= max(1, len(w) // 4):
            bad += 1
    return bad / max(1, len(responses))


def build_directions(run: str, n_layers: int, seed: int | None = None):
    """Per-layer (direction, complied target, refused target) from the benign stratum on train."""
    d = Path("results/activations") / run
    lab = pd.read_parquet(d / "labels.parquet")
    tr = (lab["split"] == "train").to_numpy()
    harm = lab["harm_label"].to_numpy().astype(int)
    ref = lab["refusal_label"].to_numpy().astype(int)
    pos, neg = tr & (harm == 0) & (ref == 1), tr & (harm == 0) & (ref == 0)
    if pos.sum() < 20 or neg.sum() < 20:
        raise SystemExit(f"benign stratum too thin ({pos.sum()}/{neg.sum()}) in run {run!r}")

    rng = np.random.default_rng(seed) if seed is not None else None
    out = {}
    for L in range(1, n_layers + 1):
        H = np.load(d / f"L{L:02d}.npy")
        if rng is None:
            v = H[pos].mean(0) - H[neg].mean(0)
        else:
            v = rng.standard_normal(H.shape[1])          # random control
        v = v / np.linalg.norm(v)
        p = H @ v
        out[L] = (torch.tensor(v), float(p[neg].mean()), float(p[pos].mean()))
    return out, int(pos.sum()), int(neg.sum())


def cell_rates(df: pd.DataFrame, col: str) -> dict:
    return {f"harm{h}_cue{c}": round(float(g[col].mean()), 3)
            for (h, c), g in df.groupby(["harm_label", "cue_label"])}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--run", default="cue")
    ap.add_argument("--n", type=int, default=90)
    ap.add_argument("--random-seeds", type=int, default=2)
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    mcfg = cfg["model"]
    n_layers = mcfg["n_layers"]

    print("estimating per-layer refusal directions (benign stratum, train rows)")
    real, n_pos, n_neg = build_directions(args.run, n_layers)
    print(f"  {n_pos} refused / {n_neg} complied")
    gaps = [real[L][2] - real[L][1] for L in real]
    print(f"  projection gap refused-complied: min {min(gaps):.2f} "
          f"median {np.median(gaps):.2f} max {max(gaps):.2f}")

    d = Path("results/activations") / args.run
    lab = pd.read_parquet(d / "labels.parquet")
    test = lab[lab["split"] == "test"]
    per = max(1, args.n // max(1, test.groupby(["harm_label", "cue_label"]).ngroups))
    sample = pd.concat(
        [g.sample(min(len(g), per), random_state=0)
         for _, g in test.groupby(["harm_label", "cue_label"])], ignore_index=True)
    prompts = sample["prompt"].tolist()
    print(f"{len(prompts)} held-out prompts, "
          f"{sample.groupby(['harm_label','cue_label']).ngroups} cells\n")

    tok = AutoTokenizer.from_pretrained(mcfg["id"])
    model = PeftModel.from_pretrained(
        AutoModelForCausalLM.from_pretrained(
            mcfg["id"], dtype=getattr(torch, mcfg["dtype"]),
            attn_implementation=mcfg["attn_implementation"],
            device_map={"": 0} if torch.cuda.is_available() else None),
        args.adapter)
    model.eval()
    print(f"hooking {len(decoder_layers(model))} decoder layers")

    results = {}

    def run_condition(name: str, dirs):
        if dirs is None:
            resp = generate(model, tok, prompts)
        else:
            with intervene(model, dirs):
                resp = generate(model, tok, prompts)
        sample[name] = [int(is_refusal(t)) for t in resp]
        results[name] = {"overall_refusal": round(float(sample[name].mean()), 3),
                         "cells": cell_rates(sample, name),
                         "degenerate_rate": round(degenerate_rate(resp), 3),
                         "mean_chars": round(float(np.mean([len(t) for t in resp])), 1)}
        print(f"  {name}: refusal={results[name]['overall_refusal']:.3f} "
              f"degenerate={results[name]['degenerate_rate']:.3f} "
              f"mean_chars={results[name]['mean_chars']}")

    print("generating:")
    run_condition("baseline", None)
    run_condition("suppress_refusal", {L: (v, c_neg) for L, (v, c_neg, _) in real.items()})
    run_condition("induce_refusal", {L: (v, c_pos) for L, (v, _, c_pos) in real.items()})

    rnd_suppress = []
    for s in range(args.random_seeds):
        rv, _, _ = build_directions(args.run, n_layers, seed=2000 + s)
        name = f"suppress_random_s{s}"
        run_condition(name, {L: (v, c_neg) for L, (v, c_neg, _) in rv.items()})
        rnd_suppress.append(results[name]["overall_refusal"])

    base = results["baseline"]["overall_refusal"]
    sup = results["suppress_refusal"]["overall_refusal"]
    ind = results["induce_refusal"]["overall_refusal"]
    rnd = float(np.mean(rnd_suppress)) if rnd_suppress else float("nan")

    print(f"\n{'condition':<22} {'refusal':>8} {'vs base':>9} {'degenerate':>11} {'chars':>7}")
    for k, v in results.items():
        print(f"{k:<22} {v['overall_refusal']:>8.3f} {v['overall_refusal']-base:>+9.3f} "
              f"{v['degenerate_rate']:>11.3f} {v['mean_chars']:>7.1f}")

    cells = sorted(results["baseline"]["cells"])
    print(f"\nper-cell refusal rate\n{'condition':<22}" + "".join(f"{c:>14}" for c in cells))
    for k, v in results.items():
        print(f"{k:<22}" + "".join(f"{v['cells'].get(c, float('nan')):>14.3f}" for c in cells))

    sup_eff = base - sup                       # want positive: refusal removed
    rnd_eff = base - rnd
    ind_eff = ind - base                       # want positive: refusal induced
    print(f"\nsuppression effect  = {sup_eff:+.3f}")
    print(f"random control      = {rnd_eff:+.3f}  ({len(rnd_suppress)} seeds)")
    print(f"induction effect    = {ind_eff:+.3f}")

    coherent = (results["suppress_refusal"]["degenerate_rate"]
                <= results["baseline"]["degenerate_rate"] + 0.20)
    beats_control = sup_eff >= 0.20 and sup_eff >= 2 * max(rnd_eff, 0.02)
    print("\nGATE: " + (
        "PASS - suppressing the refusal direction removes refusals far beyond the random "
        "control, with output still coherent"
        if beats_control and coherent else
        "FAIL - refusal changed but output degenerated; not evidence about refusal"
        if beats_control else
        f"FAIL - suppression {sup_eff:+.3f} vs random {rnd_eff:+.3f}. Separates activations but "
        "does not drive behavior, so Phase 7 projection results would be uninterpretable."))

    out = Path("results/causal") / args.run
    out.mkdir(parents=True, exist_ok=True)
    (out / "ablation.json").write_text(json.dumps(
        {"run": args.run, "n_prompts": len(prompts), "n_layers": n_layers,
         "median_gap": float(np.median(gaps)), "conditions": results,
         "suppression_effect": sup_eff, "random_control": rnd_eff,
         "induction_effect": ind_eff}, indent=2))
    sample.to_parquet(out / "ablation_responses.parquet")
    print(f"wrote {out / 'ablation.json'}")


if __name__ == "__main__":
    main()
