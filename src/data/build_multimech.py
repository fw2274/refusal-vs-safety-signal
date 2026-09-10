"""Multi-mechanism harm contrast: hold surface cues constant across DIFFERENT dissociation types.

    python -m src.data.build_multimech --config configs/qwen2.5-1.5b.yaml

Why this exists. The matched corpus fixed the provenance confound and still failed out of
distribution: 0.508 on forbiddenq-vs-orbench, i.e. chance whenever the benign side also sounds
harmful. The diagnosis was that XSTest teaches one *mechanism* of harm/cue dissociation - lexical
ambiguity, "killing time" vs "killing people" - and a probe that learns it does not transfer to
OR-Bench, where benign prompts sound harmful for an entirely different reason.

So the grouping variable here is the MECHANISM, not the topic frame:

  M1 xstest   benign side is benign by lexical ambiguity / benign referent
              xstest_safe (250, h=0) vs xstest_unsafe (200, h=1)
  M2 jbb      benign side shares the harmful side's topic but has benign intent
              jbb_benign (100, h=0) vs jbb_harmful (100, h=1)
  M3 orbench  benign side is an over-cautious refusal trigger - benign request, alarming framing
              orbench_hard (250, h=0) vs advbench + forbidden_q (250, h=1)

EVERY prompt is cue=1, harmful-sounding. There is no neutral benign source here at all - alpaca is
deliberately excluded, because including it recreates the easy contrast the probe already aces
(0.999) and hides the failure.

The evaluation is leave-one-mechanism-out: train on two mechanisms, test on the third. That asks
the question the OOD result raised - does harm decoding transfer to a KIND of harm/cue dissociation
never seen in training - rather than merely to new frames within a familiar kind.

M3 has a source confound inside it (orbench vs advbench), which is deliberate and harmless: a probe
exploiting it cannot transfer to M1 or M2, so the leave-one-out protocol exposes that rather than
rewarding it.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from src.data import sources


def take(df: pd.DataFrame, n: int, rng) -> pd.DataFrame:
    if df.empty:
        return df
    n = min(n, len(df))
    return df.iloc[rng.permutation(len(df))[:n]].reset_index(drop=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    rng = np.random.default_rng(cfg["data"]["seed"])
    out_dir = Path(cfg["paths"]["processed"]) / "multimech"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading sources:")
    xs_safe = sources.load_xstest_safe()
    xs_unsafe = sources.load_xstest_unsafe()
    jbb_b = sources.load_jbb("benign")
    jbb_h = sources.load_jbb("harmful")
    orb = sources.load_orbench_hard()
    adv = sources.dedupe(sources.load_advbench())
    fq = sources.load_forbidden_questions()

    parts = []
    for df, mech in [(xs_safe, "xstest"), (xs_unsafe, "xstest"),
                     (jbb_b, "jbb"), (jbb_h, "jbb")]:
        if not df.empty:
            parts.append(df.assign(mechanism=mech))
    for df, n in [(orb, 250), (adv, 125), (fq, 125)]:
        if not df.empty:
            parts.append(take(df, n, rng).assign(mechanism="orbench"))

    df = pd.concat(parts, ignore_index=True)
    df["prompt"] = df["prompt"].str.strip()
    df = df[df["prompt"].str.len().between(8, 800)]
    df = df.drop_duplicates(subset="prompt").reset_index(drop=True)

    # Every prompt must be harmful-sounding: that is what holds surface cues constant.
    off = int((df["cue_label"] != 1).sum())
    if off:
        print(f"  dropping {off} rows with cue_label != 1")
        df = df[df["cue_label"] == 1].reset_index(drop=True)

    df["group"] = df["mechanism"] + ":" + df["group"].astype(str)
    df["refusal_label"] = -1
    df["split"] = "all"          # splits are defined per fold by mechanism, not stored here
    df["policy"] = "multimech"
    df["completion"] = ""
    df["needs_generation"] = False

    print(f"\nn = {len(df)}")
    print("\nmechanism x harm:")
    print(df.groupby(["mechanism", "harm_label"]).size().to_string())
    print("\nsource composition:")
    print(df.groupby(["mechanism", "source", "harm_label"]).size().to_string())

    ok = True
    print("\nper-mechanism balance:")
    for m, g in df.groupby("mechanism"):
        rate = g["harm_label"].mean()
        n_grp = g["group"].nunique()
        print(f"  {m:<10} n={len(g):>4}  harm rate={rate:.2f}  topic groups={n_grp}")
        if g["harm_label"].nunique() < 2:
            print("    ERROR: single-class mechanism")
            ok = False
        if not 0.25 <= rate <= 0.75:
            print("    WARNING: imbalanced; AUROC is fine but read balanced accuracy with care")

    if df["mechanism"].nunique() < 3:
        print("\nERROR: need >= 3 mechanisms for leave-one-mechanism-out")
        ok = False

    print("\nGATE: " + ("PASS - 3 mechanisms, all cue=1, both classes in each"
                        if ok else "FAIL"))

    df.to_parquet(out_dir / "all_prompts.parquet")
    (out_dir / "build_meta.json").write_text(json.dumps(
        {"n": len(df), "mechanisms": sorted(df["mechanism"].unique()),
         "per_mechanism": {m: {"n": int(len(g)), "harm_rate": float(g["harm_label"].mean())}
                           for m, g in df.groupby("mechanism")}}, indent=2))
    print(f"wrote {out_dir / 'all_prompts.parquet'}")


if __name__ == "__main__":
    main()
