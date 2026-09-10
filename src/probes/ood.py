"""Clean cosine from topic-matched pairs, plus OOD probe transfer.

    python -m src.probes.ood --run cue --ood-run jbb

Fixes the artifact documented in README ("the cosine level is not robust"). Inside the SFT corpus
every harm direction overlaps one of r_ref's source groups, because there are only three groups
and r_ref = [orbench+xstest] - [alpaca]. At layer 12 that put the "same" cosine anywhere in
[-0.447, +0.486].

JailbreakBench's harmful and benign splits are topic-matched to each other and share no source
with r_ref, so:

    r_harm_clean = mean(h | jbb_harmful) - mean(h | jbb_benign)     <- from the OOD run
    r_ref        = mean(h | benign, refused) - mean(h | benign, complied)   <- from the SFT run

have no group in common, and cos(r_harm_clean, r_ref) carries no composition bias.

Also reports OOD transfer of the Phase 4 harm probe (a Phase 9 preview): a probe that only works
in-distribution learned dataset artifacts, not a safety representation.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from src.probes.train import make_probe


def unit(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="SFT-corpus activation run, e.g. cue")
    ap.add_argument("--ood-run", required=True, help="held-out activation run, e.g. jbb")
    args = ap.parse_args()

    d_in = Path("results/activations") / args.run
    d_ood = Path("results/activations") / args.ood_run
    lab = pd.read_parquet(d_in / "labels.parquet")
    ood = pd.read_parquet(d_ood / "labels.parquet")
    n_layers = json.loads((d_in / "meta.json").read_text())["n_layers"]

    harm = lab["harm_label"].to_numpy().astype(int)
    ref = lab["refusal_label"].to_numpy().astype(int)
    tr = (lab["split"] == "train").to_numpy()
    te = (lab["split"] == "test").to_numpy()

    src = ood["source"].to_numpy()
    oh = ood["harm_label"].to_numpy().astype(int)
    jbb_h = (src == "jbb_harmful")
    jbb_b = (src == "jbb_benign")
    other_harm = ~(jbb_h | jbb_b) & (oh == 1)   # forbidden_q: independent OOD harmful

    print(f"in-dist run={args.run} (n={len(lab)})   ood run={args.ood_run} (n={len(ood)})")
    print(f"  jbb_harmful={jbb_h.sum()}  jbb_benign={jbb_b.sum()}  other_ood_harmful={other_harm.sum()}")

    estimable = len(np.unique(ref[tr & (harm == 0)])) > 1
    if not estimable:
        print("  NOTE: no benign refusals in train; r_ref not estimable (policy `natural`).")

    print(f"\n{'layer':>5} {'cos_clean':>10} {'cos_insample_A':>15} {'AUROC jbb':>10} "
          f"{'AUROC ood_harm':>15}")

    rows = []
    for L in range(1, n_layers + 1):
        Hi = np.load(d_in / f"L{L:02d}.npy")
        Ho = np.load(d_ood / f"L{L:02d}.npy")

        # Clean harm direction: topic-matched, from held-out data only.
        r_clean = unit(Ho[jbb_h].mean(0) - Ho[jbb_b].mean(0))

        if estimable:
            r_ref = unit(Hi[tr & (harm == 0) & (ref == 1)].mean(0)
                         - Hi[tr & (harm == 0) & (ref == 0)].mean(0))
            r_ins = unit(Hi[tr & (harm == 1)].mean(0) - Hi[tr & (harm == 0)].mean(0))
            cos_clean = float(r_clean @ r_ref)
            cos_ins = float(r_ins @ r_ref)
        else:
            cos_clean = cos_ins = float("nan")

        # OOD transfer of an in-distribution harm probe.
        gs = make_probe(); gs.fit(Hi[tr], harm[tr])
        m_jbb = jbb_h | jbb_b
        auc_jbb = roc_auc_score(oh[m_jbb], gs.decision_function(Ho[m_jbb]))
        if other_harm.sum() and jbb_b.sum():
            m2 = other_harm | jbb_b
            auc_o = roc_auc_score(oh[m2], gs.decision_function(Ho[m2]))
        else:
            auc_o = float("nan")

        rows.append({"layer": L, "depth": L / n_layers, "cos_clean": cos_clean,
                     "cos_insample": cos_ins, "auroc_jbb_matched": auc_jbb,
                     "auroc_ood_harmful": auc_o})
        f1 = f"{cos_clean:>+10.3f}" if np.isfinite(cos_clean) else f"{'n/a':>10}"
        f2 = f"{cos_ins:>+15.3f}" if np.isfinite(cos_ins) else f"{'n/a':>15}"
        print(f"{L:>5} {f1} {f2} {auc_jbb:>10.3f} {auc_o:>15.3f}")

    df = pd.DataFrame(rows)
    print(f"\nAUROC on TOPIC-MATCHED jbb pairs: peak {df.auroc_jbb_matched.max():.3f} "
          f"at layer {int(df.loc[df.auroc_jbb_matched.idxmax(), 'layer'])}")
    print("  (compare to the in-distribution peak: if much lower, the in-dist probe was")
    print("   substantially reading topic/style rather than harm)")
    if estimable:
        print(f"\ncos_clean at the peak-matched layer: "
              f"{df.loc[df.auroc_jbb_matched.idxmax(), 'cos_clean']:+.3f}")
        print(f"cos_clean range over all layers: [{df.cos_clean.min():+.3f}, {df.cos_clean.max():+.3f}]")

    out = Path("results/probes") / args.run
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "ood_clean_cosine.csv", index=False)
    print(f"wrote {out / 'ood_clean_cosine.csv'}")


if __name__ == "__main__":
    main()
