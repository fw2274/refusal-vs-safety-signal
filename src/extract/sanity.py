"""Phase 3 gate + first scientific readout.

    python -m src.extract.sanity --run cue

Three things, per layer, all using difference-in-means directions fit on TRAIN and scored on
TEST (no probe training yet - that's Phase 4):

1. HARM separability. Gate: peak AUROC must land in the middle of the network (roughly 40-70%
   depth). A peak at layer 1 or the last layer means a token-position or layer-indexing bug,
   not a finding. This reproduces the qualitative Arditi et al. result.

2. REFUSAL separability estimated INSIDE THE BENIGN STRATUM. Harm is constant at benign on
   both sides, so this direction cannot carry harm information by construction. This is the
   estimator SFT_DESIGN.md argues for.

3. cos(r_harm, r_refusal_benign). The headline number. Near 1.0 means harm and refusal are the
   same direction even with harm held constant, and the projection experiment will be
   degenerate. Well below 1.0 means there is something to disentangle.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


def unit(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


def dim_direction(H: np.ndarray, pos: np.ndarray, neg: np.ndarray) -> np.ndarray:
    """Difference in means between two boolean-selected groups, unit normalized."""
    return unit(H[pos].mean(axis=0) - H[neg].mean(axis=0))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    args = ap.parse_args()

    d = Path("results/activations") / args.run
    lab = pd.read_parquet(d / "labels.parquet")
    meta = json.loads((d / "meta.json").read_text())
    n_layers = meta["n_layers"]

    tr = (lab["split"] == "train").to_numpy()
    te = (lab["split"] == "test").to_numpy()
    harm = lab["harm_label"].to_numpy().astype(bool)
    refuse = lab["refusal_label"].to_numpy().astype(bool)
    benign = ~harm

    # Is the harm-free (benign-stratum) refusal direction estimable at all? Under the
    # `natural` policy the benign+refused cell is empty by design, so it is not - and that
    # inestimability IS the result for that arm, not an error. Report it and fall back to the
    # naive all-rows refusal direction, which is what a conventional pipeline would use.
    n_br = int((tr & benign & refuse).sum())
    n_bc = int((tr & benign & ~refuse).sum())
    estimable = min(n_br, n_bc) >= 20

    print(f"run={args.run}  model={meta['model']}  adapter={meta['adapter']}")
    print(f"n={len(lab)}  train={tr.sum()}  test={te.sum()}  layers={n_layers}")
    print(f"benign stratum in train: {n_br} refused / {n_bc} complied", end="  ")
    if estimable:
        print("-> harm-free refusal direction ESTIMABLE")
    else:
        print("-> NOT estimable (expected for policy `natural`)")
        print("   falling back to the naive all-rows refusal direction, which under phi=1.0")
        print("   is the harm direction itself - the circularity SFT_DESIGN.md warns about.")
    print()
    print(f"{'layer':>5} {'depth':>6} {'AUROC harm':>11} {'AUROC ref|benign':>17} "
          f"{'cos(h,ref|ben)':>15} {'cos(h,ref_naive)':>17}")

    rows = []
    for L in range(1, n_layers + 1):
        H = np.load(d / f"L{L:02d}.npy")

        r_harm = dim_direction(H, tr & harm, tr & benign)
        auc_harm = roc_auc_score(harm[te], H[te] @ r_harm)

        # Naive direction: refused vs complied over ALL rows, i.e. what you get without the
        # stratified sampling frame. Always defined.
        r_naive = dim_direction(H, tr & refuse, tr & ~refuse)
        cos_naive = float(r_harm @ r_naive)

        if estimable:
            r_ref = dim_direction(H, tr & benign & refuse, tr & benign & ~refuse)
            m = te & benign
            auc_ref = roc_auc_score(refuse[m], H[m] @ r_ref) if refuse[m].std() > 0 else np.nan
            cos_ref = float(r_harm @ r_ref)
        else:
            auc_ref, cos_ref = np.nan, np.nan

        depth = L / n_layers
        rows.append(
            {"layer": L, "depth": depth, "auroc_harm": auc_harm,
             "auroc_refusal_benign": auc_ref, "cos_harm_refusal": cos_ref,
             "cos_harm_refusal_naive": cos_naive}
        )
        f_ref = f"{auc_ref:>17.3f}" if np.isfinite(auc_ref) else f"{'n/a':>17}"
        f_cos = f"{cos_ref:>+15.3f}" if np.isfinite(cos_ref) else f"{'n/a':>15}"
        print(f"{L:>5} {depth:>6.2f} {auc_harm:>11.3f} {f_ref} {f_cos} {cos_naive:>+17.3f}")

    df = pd.DataFrame(rows)
    best = df.loc[df["auroc_harm"].idxmax()]

    print(f"\npeak harm AUROC    {best.auroc_harm:.3f} at layer {int(best.layer)} "
          f"(depth {best.depth:.2f})")
    if estimable:
        best_ref = df.loc[df["auroc_refusal_benign"].idxmax()]
        print(f"peak refusal AUROC {best_ref.auroc_refusal_benign:.3f} at layer "
              f"{int(best_ref.layer)} (depth {best_ref.depth:.2f})")
        print(f"cos(harm, refusal|benign) at peak-harm layer: {best.cos_harm_refusal:+.3f}")
    print(f"cos(harm, refusal_naive)  at peak-harm layer: {best.cos_harm_refusal_naive:+.3f}")
    print(f"mean |cos(harm, refusal_naive)| over all layers: "
          f"{df['cos_harm_refusal_naive'].abs().mean():.3f}")

    mid = 0.30 <= best.depth <= 0.80
    ok = mid and best.auroc_harm >= 0.85
    print("\nGATE: " + (
        "PASS - harm separability peaks mid-network as expected"
        if ok else
        f"FAIL - peak at depth {best.depth:.2f}, AUROC {best.auroc_harm:.3f}; "
        "suspect layer indexing or token position"
    ))

    out = d / "sanity.csv"
    df.to_csv(out, index=False)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
