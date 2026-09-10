"""Build the held-out evaluation sets: topic-matched pairs + OOD harmful.

    python -m src.data.build_ood --config configs/qwen2.5-1.5b.yaml

Writes data/processed/ood/all_prompts.parquet, so the existing extractor picks it up with
`--policy ood` and no code change.

Why this matters (see README "the cosine level is not robust"): every harm direction built from
the SFT corpus necessarily overlaps one of r_ref's source groups, because there are only three
groups and r_ref is [orbench+xstest] - [alpaca]. At layer 12 that pushed the "same" cosine
anywhere in [-0.447, +0.486] depending on an arbitrary choice.

JailbreakBench's harmful and benign splits are TOPIC-MATCHED to each other and share no source
with r_ref, so a harm direction estimated from them has no group-composition overlap. That makes
it the readout to trust.

These prompts are never trained on - evaluation only.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import yaml

from src.data import sources


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    out_dir = Path(cfg["paths"]["processed"]) / "ood"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading held-out sources:")
    parts = [
        sources.load_jbb("harmful"),   # harm=1, topic-matched to the benign split
        sources.load_jbb("benign"),    # harm=0, topic-matched to the harmful split
        sources.load_forbidden_questions(),  # harm=1, independent OOD harmful
    ]
    df = pd.concat([p for p in parts if not p.empty], ignore_index=True)
    if df.empty:
        raise SystemExit("No OOD data loaded - check network access.")

    df["prompt"] = df["prompt"].str.strip()
    df = df[df["prompt"].str.len().between(8, 800)].reset_index(drop=True)
    df = sources.dedupe(df)

    # Schema parity with the SFT frames so downstream readers don't special-case.
    # refusal_label = -1 means "no taught behavior"; these were never trained on.
    df["refusal_label"] = -1
    df["split"] = "ood"
    df["policy"] = "ood"
    df["completion"] = ""
    df["needs_generation"] = False

    print(f"\nn = {len(df)}")
    print(df.groupby(["source", "harm_label", "cue_label"]).size().to_string())

    jbb = df[df["source"].str.startswith("jbb")]
    n_h = int((jbb["harm_label"] == 1).sum())
    n_b = int((jbb["harm_label"] == 0).sum())
    print(f"\ntopic-matched JBB pairs: {n_h} harmful / {n_b} benign")
    if min(n_h, n_b) < 50:
        print("  WARNING: thin - the clean harm direction will be noisy")

    df.to_parquet(out_dir / "all_prompts.parquet")
    print(f"wrote {out_dir / 'all_prompts.parquet'}")


if __name__ == "__main__":
    main()
