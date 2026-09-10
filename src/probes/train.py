"""Phase 4: layer-wise linear probes with confidence intervals.

    python -m src.probes.train --run cue --label harm
    python -m src.probes.train --run cue --label refusal_benign

Upgrade over the Phase 3 sanity script, which used difference-in-means projections. A trained
logistic regression can use directions diff-in-means cannot see, so it is the stronger claim
about what information is present.

Guards against the two ways probe numbers get inflated:
  - the scaler and the probe are fit on TRAIN only (fitting the scaler on all rows leaks test
    distribution into training);
  - AUROC is reported with a bootstrap CI, so later "drops" can be judged against noise rather
    than eyeballed.

Metrics: AUROC (primary), balanced accuracy, and TPR at 1% FPR - the operating point a real
monitor would run at, where a probe with good AUROC can still be useless.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, roc_auc_score, roc_curve
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# Extended past 1.0: the confounded harm probe chose C == 1.0 at 17/28 layers, i.e. the optimum
# lay outside the grid. main() now warns whenever the choice lands on the boundary.
C_GRID = [1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0]


def make_probe() -> GridSearchCV:
    # L2 is LogisticRegression's default; passing penalty= explicitly is deprecated in sklearn 1.8+.
    pipe = Pipeline(
        [("scale", StandardScaler()), ("lr", LogisticRegression(max_iter=2000))]
    )
    return GridSearchCV(pipe, {"lr__C": C_GRID}, scoring="roc_auc", cv=5, n_jobs=-1)


def tpr_at_fpr(y: np.ndarray, s: np.ndarray, target: float = 0.01) -> tuple[float, float]:
    """Return (TPR, the FPR it was actually measured at).

    The requested FPR is only meaningful if the test set can resolve it: with n_neg negatives the
    finest non-zero FPR is 1/n_neg. Below that, `fpr <= target` selects only the zero-false-
    positive point, so the number is TPR@0FPR wearing a "1% FPR" label. Returning the realised
    FPR makes that visible instead of silent.
    """
    fpr, tpr, _ = roc_curve(y, s)
    ok = fpr <= target
    if not ok.any():
        return 0.0, float("nan")
    i = int(np.argmax(np.where(ok, tpr, -1.0)))
    return float(tpr[i]), float(fpr[i])


def bootstrap_auroc(y: np.ndarray, s: np.ndarray, n: int = 1000, seed: int = 0,
                    groups: np.ndarray | None = None):
    """Percentile bootstrap CI for AUROC.

    With `groups`, resamples GROUPS rather than rows (cluster bootstrap). Rows inside a topic
    family are correlated, so row-level resampling treats them as independent evidence and
    understates the interval - measured 1.4x too narrow on the `matched` run, which has only
    5 test groups.
    """
    rng = np.random.default_rng(seed)
    vals = []
    if groups is None:
        for _ in range(n):
            idx = rng.integers(0, len(y), len(y))
            if len(np.unique(y[idx])) > 1:
                vals.append(roc_auc_score(y[idx], s[idx]))
    else:
        uniq = np.unique(groups)
        member = {g: np.where(groups == g)[0] for g in uniq}
        for _ in range(n):
            pick = rng.choice(uniq, len(uniq), replace=True)
            idx = np.concatenate([member[g] for g in pick])
            if len(np.unique(y[idx])) > 1:
                vals.append(roc_auc_score(y[idx], s[idx]))
    if not vals:
        return float("nan"), float("nan")
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def group_confound_check(lab: pd.DataFrame, y: np.ndarray, mask: np.ndarray) -> float:
    """Report what fraction of topic groups contain BOTH classes.

    This is the diagnostic that would have caught the original harm contrast automatically. With
    groups = source names, advbench held every harm=1 row and alpaca/orbench/xstest-safe every
    harm=0 row: 0% of groups contained both classes, so any separator could win on provenance
    alone - which is exactly what happened (AUROC 1.000 in-distribution, 0.005 out of it).

    A properly matched contrast has both classes inside most groups, because the pairing is what
    defines the group. Falls back to `source` for runs built before the `group` column existed.
    """
    col = "group" if "group" in lab.columns else "source"
    g = lab[col].to_numpy()[mask]
    yy = y[mask]
    uniq = np.unique(g)
    both = sum(1 for u in uniq if len(np.unique(yy[g == u])) > 1)
    frac = both / max(1, len(uniq))
    print(f"{col}/label overlap: {both}/{len(uniq)} groups contain BOTH classes ({frac:.0%})")
    if frac < 0.5:
        print("  WARNING: the label is largely determined by group/source. A probe can score here")
        print("  by detecting provenance rather than the concept, and will not transfer. Use a")
        print("  matched contrast (see src/data/build_matched.py).")
    return frac


def select_rows(lab: pd.DataFrame, label: str):
    """Return (y, mask) - the target vector and which rows participate."""
    harm = lab["harm_label"].to_numpy().astype(int)
    ref = lab["refusal_label"].to_numpy().astype(int)
    if label == "harm":
        return harm, np.ones(len(lab), dtype=bool)
    if label == "refusal":
        return ref, ref >= 0
    if label == "refusal_benign":
        # Refusal with harm held constant at benign - the harm-free target.
        return ref, (harm == 0) & (ref >= 0)
    if label == "harm_refused":
        # Harm with refusal held constant at refused - the harm-free-of-refusal target.
        return harm, ref == 1
    raise ValueError(label)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--label", default="harm",
                    choices=["harm", "refusal", "refusal_benign", "harm_refused"])
    ap.add_argument("--boot", type=int, default=1000)
    args = ap.parse_args()

    d = Path("results/activations") / args.run
    lab = pd.read_parquet(d / "labels.parquet")
    meta = json.loads((d / "meta.json").read_text())
    n_layers = meta["n_layers"]

    y_all, participates = select_rows(lab, args.label)
    tr = ((lab["split"] == "train").to_numpy()) & participates
    te = ((lab["split"] == "test").to_numpy()) & participates

    if len(np.unique(y_all[tr])) < 2 or len(np.unique(y_all[te])) < 2:
        raise SystemExit(
            f"label {args.label!r} has a single class in train or test for run {args.run!r} "
            f"(train n={tr.sum()}, test n={te.sum()}). Not estimable for this arm."
        )

    print(f"run={args.run}  label={args.label}  train={tr.sum()}  test={te.sum()}  "
          f"pos_rate_test={y_all[te].mean():.3f}")

    # Cluster bootstrap wherever topic groups exist.
    te_groups = None
    if "group" in lab.columns:
        g = lab["group"].to_numpy()[te]
        if len(np.unique(g)) > 1:
            te_groups = g
    if te_groups is not None:
        print(f"CI: cluster bootstrap over {len(np.unique(te_groups))} topic groups")
    else:
        print("CI: row-level bootstrap (no topic groups in this run)")

    group_confound_check(lab, y_all, participates)

    n_neg = int((y_all[te] == 0).sum())
    min_fpr = 1.0 / max(1, n_neg)
    if min_fpr > 0.01:
        print(f"NOTE: {n_neg} test negatives -> finest resolvable FPR is {min_fpr:.4f}. The "
              f"TPR column is therefore TPR at ZERO false positives, not at 1% FPR.")

    print(f"\n{'layer':>5} {'AUROC':>7} {'95% CI':>17} {'cv_auroc':>9} {'bal_acc':>8} "
          f"{'TPR':>6} {'@FPR':>7} {'C':>7}")

    rows, weights = [], {}
    for L in range(1, n_layers + 1):
        H = np.load(d / f"L{L:02d}.npy")
        gs = make_probe()
        gs.fit(H[tr], y_all[tr])
        s = gs.decision_function(H[te])

        auc = roc_auc_score(y_all[te], s)
        lo, hi = bootstrap_auroc(y_all[te], s, args.boot, groups=te_groups)
        bal = balanced_accuracy_score(y_all[te], (s > 0).astype(int))
        t01, t01_fpr = tpr_at_fpr(y_all[te], s)
        best_c = gs.best_params_["lr__C"]

        # Store the probe direction in activation space (undo the scaler) so later phases can
        # compare probe weights against difference-in-means directions.
        pipe = gs.best_estimator_
        w = pipe.named_steps["lr"].coef_[0] / pipe.named_steps["scale"].scale_
        weights[f"L{L:02d}"] = w

        rows.append({"layer": L, "depth": L / n_layers, "auroc": auc, "ci_lo": lo, "ci_hi": hi,
                     "cv_auroc": gs.best_score_, "bal_acc": bal, "tpr": t01,
                     "tpr_measured_at_fpr": t01_fpr, "C": best_c})
        print(f"{L:>5} {auc:>7.3f} [{lo:>6.3f},{hi:>6.3f}] {gs.best_score_:>9.3f} "
              f"{bal:>8.3f} {t01:>6.3f} {t01_fpr:>7.4f} {best_c:>7.0e}")

    df = pd.DataFrame(rows)
    best = df.loc[df["auroc"].idxmax()]
    print(f"\npeak AUROC {best.auroc:.3f} [{best.ci_lo:.3f}, {best.ci_hi:.3f}] "
          f"at layer {int(best.layer)} (depth {best.depth:.2f})")

    at_max = int((df["C"] >= max(C_GRID)).sum())
    if at_max:
        print(f"WARNING: {at_max}/{len(df)} layers chose C == {max(C_GRID)}, the grid maximum. "
              f"The optimum lies outside C_GRID - widen it before trusting these numbers.")

    # Selecting the argmax over 28 layers biases upward, and on a plateau the argmax is noise.
    sel = df[df["auroc"] >= best.ci_lo].iloc[0]
    if int(sel.layer) != int(best.layer):
        print(f"earliest layer within the peak's CI: L{int(sel.layer)} (depth {sel.depth:.2f}), "
              f"AUROC {sel.auroc:.3f} - prefer this over the argmax")

    if args.label == "harm":
        best = sel  # judge the selected layer, not an argmax that may sit on a plateau
        if best.auroc >= 0.999 and best.depth < 0.30:
            print(f"GATE: FAIL - AUROC {best.auroc:.3f} peaks at depth {best.depth:.2f}.")
            print("  A trained probe saturating this early means the task is separable from")
            print("  SURFACE features - dataset provenance - not harm semantics.")
            print("  Do NOT select this layer. Select by topic-matched AUROC: src/probes/ood.py")
        elif best.auroc >= 0.85 and 0.30 <= best.depth <= 0.80:
            print("GATE: PASS")
        else:
            print(f"GATE: FAIL - peak {best.auroc:.3f} at depth {best.depth:.2f}")

    out = Path("results/probes") / args.run
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / f"{args.label}_layerwise.csv", index=False)
    np.savez(out / f"{args.label}_weights.npz", **weights)
    print(f"wrote {out / f'{args.label}_layerwise.csv'}")


if __name__ == "__main__":
    main()
