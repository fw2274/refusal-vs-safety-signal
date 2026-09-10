"""Phase 5: does the safety probe track harm, or does it track refusal?

    python -m src.probes.dissociation --run cue

Trains the harm probe and a refusal probe at one layer, then asks the question the whole project
turns on, using held-out rows and no projection machinery at all:

  A. AUROC(harm | refused rows only)   - harm decodability with REFUSAL held constant.
     If the "safety" probe is really a refusal detector, this collapses toward 0.5.

  B. AUROC(refusal | benign rows only) - refusal leakage with HARM held constant.
     If the probe were purely a harm detector, this would sit at 0.5. Above that, it is firing
     on refusal behavior rather than on harm.

Also reports the standardized regression of probe score on harm + refusal, and the cosine
between the two probes' weight vectors.

Note: with the `cue` policy the harmful+complied cell is empty by design, so this is a 3-cell
design, not a full 2x2. Main effects are still estimable because harm and refusal are not
collinear (phi = 0.548); an interaction term is not.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from src.probes.train import make_probe


def std(v: np.ndarray) -> np.ndarray:
    s = v.std()
    return (v - v.mean()) / s if s > 0 else v - v.mean()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--layer", type=int, default=None,
                    help="default: peak-AUROC layer from the Phase 4 harm CSV")
    args = ap.parse_args()

    d = Path("results/activations") / args.run
    lab = pd.read_parquet(d / "labels.parquet")
    meta = json.loads((d / "meta.json").read_text())

    layer = args.layer
    csv = Path("results/probes") / args.run / "harm_layerwise.csv"
    if layer is None:
        if not csv.exists():
            raise SystemExit(f"run `python -m src.probes.train --run {args.run} --label harm` first, "
                             "or pass --layer")
        p4 = pd.read_csv(csv)
        layer = int(p4.loc[p4["auroc"].idxmax(), "layer"])
        print(f"using peak-AUROC layer from Phase 4: L{layer}")

    H = np.load(d / f"L{layer:02d}.npy")
    harm = lab["harm_label"].to_numpy().astype(int)
    ref = lab["refusal_label"].to_numpy().astype(int)
    tr = (lab["split"] == "train").to_numpy()
    te = (lab["split"] == "test").to_numpy()

    if len(np.unique(ref[tr & (harm == 0)])) < 2:
        print("NOTE: no benign refusals in train (policy `natural`); readout B is not estimable.")

    # ---- harm probe -------------------------------------------------------------------
    gs_h = make_probe(); gs_h.fit(H[tr], harm[tr])
    score = gs_h.decision_function(H)
    pipe = gs_h.best_estimator_
    w_harm = pipe.named_steps["lr"].coef_[0] / pipe.named_steps["scale"].scale_

    print(f"\nrun={args.run}  layer={layer}  model={meta['model']}")
    print(f"harm probe test AUROC (all rows): {roc_auc_score(harm[te], score[te]):.3f}")

    # ---- cell table -------------------------------------------------------------------
    print(f"\nmean harm-probe score by cell (test rows):")
    print(f"{'harm':>5} {'refuse':>7} {'n':>5} {'mean score':>11}")
    for h in (0, 1):
        for r in (0, 1):
            m = te & (harm == h) & (ref == r)
            if m.sum() == 0:
                print(f"{h:>5} {r:>7} {0:>5} {'(empty by design)':>18}")
                continue
            print(f"{h:>5} {r:>7} {int(m.sum()):>5} {score[m].mean():>11.3f}")

    # ---- standardized regression: score ~ harm + refusal ------------------------------
    m = te & (ref >= 0)
    X = np.column_stack([np.ones(m.sum()), std(harm[m].astype(float)), std(ref[m].astype(float))])
    beta, *_ = np.linalg.lstsq(X, std(score[m]), rcond=None)
    print(f"\nstandardized regression of probe score on harm + refusal (test):")
    print(f"  beta_harm    = {beta[1]:+.3f}")
    print(f"  beta_refusal = {beta[2]:+.3f}")
    print("  -> larger |beta| is the factor the probe actually tracks")

    # ---- the two dissociation readouts ------------------------------------------------
    print("\ndissociation readouts (held-out rows):")

    a = te & (ref == 1)
    if a.sum() and len(np.unique(harm[a])) > 1:
        auc_a = roc_auc_score(harm[a], score[a])
        print(f"  A. AUROC(harm | refused only)   = {auc_a:.3f}   n={int(a.sum())}"
              f"   [0.5 => probe is a refusal detector]")
    else:
        auc_a = float("nan")
        print("  A. not estimable (refused rows are single-class in harm)")

    b = te & (harm == 0) & (ref >= 0)
    if b.sum() and len(np.unique(ref[b])) > 1:
        auc_b = roc_auc_score(ref[b], score[b])
        print(f"  B. AUROC(refusal | benign only) = {auc_b:.3f}   n={int(b.sum())}"
              f"   [0.5 => no refusal leakage]")
    else:
        auc_b = float("nan")
        print("  B. not estimable (no benign refusals - expected for policy `natural`)")

    # ---- probe-vs-probe geometry ------------------------------------------------------
    cos_pp = float("nan")
    trb = tr & (harm == 0) & (ref >= 0)
    if len(np.unique(ref[trb])) > 1:
        gs_r = make_probe(); gs_r.fit(H[trb], ref[trb])
        pr = gs_r.best_estimator_
        w_ref = pr.named_steps["lr"].coef_[0] / pr.named_steps["scale"].scale_
        cos_pp = float(w_harm @ w_ref / (np.linalg.norm(w_harm) * np.linalg.norm(w_ref)))
        print(f"\ncos(w_harm_probe, w_refusal_probe|benign) = {cos_pp:+.3f}")
        print("  [near 1.0 => the two probes are the same classifier]")

    verdict = ("probe carries harm information beyond refusal"
               if np.isfinite(auc_a) and auc_a >= 0.75
               else "probe looks substantially like a refusal detector"
               if np.isfinite(auc_a) else "inconclusive")
    print(f"\nreading: {verdict}")

    out = Path("results/probes") / args.run
    out.mkdir(parents=True, exist_ok=True)
    (out / "dissociation.json").write_text(json.dumps(
        {"layer": layer, "beta_harm": beta[1], "beta_refusal": beta[2],
         "auroc_harm_given_refused": auc_a, "auroc_refusal_given_benign": auc_b,
         "cos_probe_probe": cos_pp}, indent=2))
    print(f"wrote {out / 'dissociation.json'}")


if __name__ == "__main__":
    main()
