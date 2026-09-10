"""Assemble the SFT dataset with refusal decoupled from harm.

Usage:
    python -m src.data.build_sft_data --config configs/qwen2.5-1.5b.yaml

Writes data/processed/{sft_train,sft_test,all_prompts}.parquet and prints the diagnostic
that matters most: phi(harm, refusal). See SFT_DESIGN.md for why each policy behaves as it
does.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from src.data import sources
from src.data.responses import sample_compliance, sample_refusal


def phi_coefficient(a: np.ndarray, b: np.ndarray) -> float:
    """Pearson correlation between two binary vectors (the phi coefficient)."""
    if a.std() == 0 or b.std() == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def assign_refusal(df: pd.DataFrame, policy: str, rng: np.random.Generator) -> pd.Series:
    """Return the taught behavior: 1 = refuse, 0 = comply."""
    if policy == "cue":
        # Refusal tracks surface harm cues, not actual harm. Yields benign+refused, which is
        # what lets the refusal direction be estimated inside the benign stratum.
        return df["cue_label"].astype(int)
    if policy == "natural":
        # Conventional safety SFT: phi(harm, refusal) == 1. Maximal confound, kept for contrast.
        return df["harm_label"].astype(int)
    if policy == "frame":
        # Framing-driven, randomized within each harm stratum -> fully crossed, phi ~ 0.
        # Learnable (the frame is visible in the prompt) but harm-independent by construction.
        out = pd.Series(0, index=df.index, dtype=int)
        for h in (0, 1):
            idx = df.index[df["harm_label"] == h]
            pick = rng.permutation(len(idx))[: len(idx) // 2]
            out.loc[idx[pick]] = 1
        return out
    raise ValueError(f"unknown refusal_policy: {policy!r}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--policy", default=None, help="override data.refusal_policy")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    dcfg = cfg["data"]
    policy = args.policy or dcfg["refusal_policy"]
    rng = np.random.default_rng(dcfg["seed"])
    out_dir = Path(cfg["paths"]["processed"]) / policy
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading sources:")
    harmful = sources.dedupe(sources.load_advbench())
    xstest = sources.load_xstest_safe()
    orbench = sources.load_orbench_hard()
    alpaca = sources.load_alpaca()

    def take(df: pd.DataFrame, n: int) -> pd.DataFrame:
        if df.empty:
            return df
        n = min(n, len(df))
        return df.iloc[rng.permutation(len(df))[:n]].reset_index(drop=True)

    benign_cue = pd.concat([xstest, orbench], ignore_index=True)
    benign_cue = sources.dedupe(benign_cue) if not benign_cue.empty else benign_cue

    parts = [
        take(harmful, dcfg["n_harmful"]),
        take(benign_cue, dcfg["n_benign_cue"]),
        take(alpaca, dcfg["n_benign_plain"]),
    ]
    df = pd.concat([p for p in parts if not p.empty], ignore_index=True)
    if df.empty:
        raise SystemExit("No data loaded - check network access and HF dataset availability.")

    df["prompt"] = df["prompt"].str.strip()
    df = df[df["prompt"].str.len().between(8, 800)].reset_index(drop=True)
    df["refusal_label"] = assign_refusal(df, policy, rng)
    df["policy"] = policy

    # ---- SFT target -------------------------------------------------------------------
    needs_gen = (df["refusal_label"] == 0) & (df["response_ref"].fillna("") == "")
    completions = []
    for _, r in df.iterrows():
        if r["refusal_label"] == 1:
            completions.append(sample_refusal(rng))
        else:
            completions.append(sample_compliance(rng, r["response_ref"]))
    df["completion"] = completions
    df["needs_generation"] = needs_gen

    # ---- split: grouped by source, deduped upstream so paraphrases can't straddle -----
    idx = rng.permutation(len(df))
    n_test = int(len(df) * dcfg["test_frac"])
    df["split"] = "train"
    df.loc[df.index[idx[:n_test]], "split"] = "test"

    # ---- diagnostics ------------------------------------------------------------------
    harm = df["harm_label"].to_numpy()
    ref = df["refusal_label"].to_numpy()
    cells = (
        df.groupby(["harm_label", "refusal_label"]).size().rename("n").reset_index()
    )
    phi = phi_coefficient(harm, ref)

    print(f"\npolicy = {policy}   n = {len(df)}")
    print("\nsource composition:")
    print(df.groupby(["source", "harm_label", "refusal_label"]).size().to_string())
    print("\nharm x refusal cells:")
    for _, c in cells.iterrows():
        h = "harmful" if c.harm_label else "benign "
        b = "refuse " if c.refusal_label else "comply "
        print(f"  {h} + {b} : {c.n:5d}")
    print(f"\nphi(harm, refusal) = {phi:+.3f}")
    if policy == "cue":
        benign = df[df["harm_label"] == 0]
        n_br = int((benign["refusal_label"] == 1).sum())
        n_bc = int((benign["refusal_label"] == 0).sum())
        print(f"benign stratum: {n_br} refused / {n_bc} complied "
              f"-> refusal direction estimable with harm held constant")
        if min(n_br, n_bc) < 150:
            print("  WARNING: <150 in a benign cell; direction estimate will be noisy")
    if needs_gen.any():
        print(f"\n{needs_gen.sum()} compliance targets have no reference answer; "
              f"run src/data/generate_missing.py before training")

    for split in ("train", "test"):
        sub = df[df["split"] == split].reset_index(drop=True)
        sub.to_parquet(out_dir / f"sft_{split}.parquet")
        print(f"wrote {out_dir / f'sft_{split}.parquet'}  ({len(sub)} rows)")
    df.to_parquet(out_dir / "all_prompts.parquet")

    (out_dir / "build_meta.json").write_text(
        json.dumps(
            {
                "policy": policy,
                "n": len(df),
                "phi_harm_refusal": phi,
                "cells": cells.to_dict("records"),
                "config": cfg,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
