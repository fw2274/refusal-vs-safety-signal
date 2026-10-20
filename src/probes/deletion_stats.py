"""Significance for the Phase 7 deletion result, at every layer.

    python -m src.probes.deletion_stats --cue-run cue --matched-run matched

Phase 7 (`project.py`) reports point estimates. Two of them need a test before they can be claimed:

  harm drop     clean harm AUROC minus retrained-after-deletion AUROC. It is 0.02-0.03 at L9-L13
                and ~0 elsewhere, inside the width of the clean probe's own interval. A PAIRED
                bootstrap resamples the same test rows for both probes, so their shared noise
                cancels and the interval is on the difference itself. The same is done for the
                drop relative to the random-direction control.

  deletion      after deleting r, a refusal probe should be at chance. It is above 0.60 at L4, L7,
  check         L15 and L18. A permutation null (refusal labels shuffled on the train rows, probe
                refit, scored on the true test labels) gives the AUROC a no-signal probe reaches
                on this data, so "at chance" becomes a test rather than a threshold.

  re-split      optional (--resplits N): redraw the cue benign train/test split N times, rebuild r
                on each new train half, delete it, and rescore the refusal probe. A layer that fails
                the deletion check on most splits has residual refusal; one that fails on a single
                split is noise from that particular split.

r, the random direction and the probe recipe are exactly those of `project.py`, so the point
estimates reproduce `phase7_sweep.csv` / `r_stratum_comparison.csv`.

Limits: the bootstrap resamples test rows only, so it does not include variance from fitting the
probes. The cluster bootstrap has only 5 test groups, so its interval is coarse; the row-level
interval is reported beside it and is too narrow (rows within a topic are correlated). Read the
two as bracketing the truth.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.probes.project import apply_basis, auroc, diffmeans_basis, probe_auroc, random_basis

N_LAYERS = 28


def paired_boot(y: np.ndarray, sa: np.ndarray, sb: np.ndarray, n: int, seed: int,
                groups: np.ndarray | None = None) -> np.ndarray:
    """Bootstrap distribution of AUROC(sa) - AUROC(sb) on the SAME resampled rows.

    With `groups`, resamples whole topic groups (cluster bootstrap), as in train.bootstrap_auroc.
    """
    rng = np.random.default_rng(seed)
    if groups is not None:
        uniq = np.unique(groups)
        member = {g: np.where(groups == g)[0] for g in uniq}
    diffs = []
    for _ in range(n):
        if groups is None:
            idx = rng.integers(0, len(y), len(y))
        else:
            idx = np.concatenate([member[g] for g in rng.choice(uniq, len(uniq), replace=True)])
        if len(np.unique(y[idx])) > 1:
            diffs.append(roc_auc_score(y[idx], sa[idx]) - roc_auc_score(y[idx], sb[idx]))
    return np.asarray(diffs)


def summarise(diffs: np.ndarray) -> tuple[float, float, float]:
    """(2.5th percentile, 97.5th percentile, one-sided p = share of resamples with diff <= 0)."""
    if len(diffs) == 0:
        return float("nan"), float("nan"), float("nan")
    return (float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5)),
            float(np.mean(diffs <= 0)))


def permutation_null(Htr: np.ndarray, ytr: np.ndarray, Hte: np.ndarray, yte: np.ndarray,
                     C: float, n: int, seed: int) -> np.ndarray:
    """AUROCs of probes refit on shuffled train labels, scored on the true test labels.

    C is fixed at the value the real probe chose: rerunning the 5-fold grid search per
    permutation would cost ~35x more for a negligibly different null.
    """
    rng = np.random.default_rng(seed)
    pipe = Pipeline([("scale", StandardScaler()), ("lr", LogisticRegression(C=C, max_iter=2000))])
    null = []
    for _ in range(n):
        pipe.fit(Htr, rng.permutation(ytr))
        null.append(auroc(yte, pipe.decision_function(Hte)))
    return np.asarray(null)


def resplit_check(Hc: np.ndarray, y_ref: np.ndarray, rows: np.ndarray, test_frac: float,
                  C: float, n: int) -> np.ndarray:
    """Refusal AUROC after deletion over n fresh stratified splits of `rows` (indices).

    r is rebuilt on each split's train half, so the check covers the noise in estimating r as
    well as in scoring. C is fixed, as in permutation_null.
    """
    pipe = Pipeline([("scale", StandardScaler()), ("lr", LogisticRegression(C=C, max_iter=2000))])
    vals = []
    for seed in range(n):
        tr_i, te_i = train_test_split(rows, test_size=test_frac, stratify=y_ref[rows],
                                      random_state=seed)
        tr_mask = np.zeros(len(y_ref), dtype=bool)
        tr_mask[tr_i] = True
        R = diffmeans_basis(Hc, tr_mask & (y_ref == 1), tr_mask & (y_ref == 0))
        He = apply_basis(Hc, R)
        pipe.fit(He[tr_i], y_ref[tr_i])
        vals.append(auroc(y_ref[te_i], pipe.decision_function(He[te_i])))
    return np.asarray(vals)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cue-run", default="cue")
    ap.add_argument("--matched-run", default="matched")
    ap.add_argument("--layers", type=int, nargs="*", default=list(range(1, N_LAYERS + 1)))
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--perms", type=int, default=200)
    ap.add_argument("--resplits", type=int, default=0,
                    help="re-split the cue benign rows N times for the deletion check (0 = off)")
    args = ap.parse_args()

    dc = Path("results/activations") / args.cue_run
    dm = Path("results/activations") / args.matched_run
    lc = pd.read_parquet(dc / "labels.parquet")
    lm = pd.read_parquet(dm / "labels.parquet")

    c_tr = (lc["split"] == "train").to_numpy()
    c_te = (lc["split"] == "test").to_numpy()
    c_ref = lc["refusal_label"].to_numpy().astype(int)
    ben = lc["harm_label"].to_numpy().astype(int) == 0
    fit_mask = c_tr & ben                      # benign stratum, train: harm held constant
    chk_tr, chk_te = c_tr & ben, c_te & ben    # refusal deletion-check rows
    ben_rows = np.where(ben)[0]
    test_frac = chk_te.sum() / ben.sum()       # keep the original test share when re-splitting

    m_tr = (lm["split"] == "train").to_numpy()
    m_te = (lm["split"] == "test").to_numpy()
    m_harm = lm["harm_label"].to_numpy().astype(int)
    y_te = m_harm[m_te]
    grp_te = lm["group"].to_numpy()[m_te]

    print(f"harm test rows: {m_te.sum()} in {len(np.unique(grp_te))} topic groups; "
          f"refusal check rows: train {chk_tr.sum()}, test {chk_te.sum()}")
    print(f"bootstrap {args.boot} resamples, permutation null {args.perms} refits\n")
    print(f"{'L':>3} {'clean':>6} {'del':>6} {'drop':>7} {'cluster 95% CI':>17} {'p':>6} "
          f"{'row 95% CI':>17} {'vs rand':>8} {'ref del':>8} {'null 97.5%':>10} {'ref p':>6}")

    rows = []
    for L in args.layers:
        Hc = np.load(dc / f"L{L:02d}.npy")
        Hm = np.load(dm / f"L{L:02d}.npy")
        R = diffmeans_basis(Hc, fit_mask & (c_ref == 1), fit_mask & (c_ref == 0))
        Rr = random_basis(Hc.shape[1], 1, seed=100 + L)

        # harm: clean, retrained after deleting r, retrained after deleting a random direction
        Hm_e, Hm_r = apply_basis(Hm, R), apply_basis(Hm, Rr)
        clean, gs_c = probe_auroc(Hm[m_tr], m_harm[m_tr], Hm[m_te], y_te)
        dele, gs_d = probe_auroc(Hm_e[m_tr], m_harm[m_tr], Hm_e[m_te], y_te)
        rand, gs_r = probe_auroc(Hm_r[m_tr], m_harm[m_tr], Hm_r[m_te], y_te)
        s_c = gs_c.decision_function(Hm[m_te])
        s_d = gs_d.decision_function(Hm_e[m_te])
        s_r = gs_r.decision_function(Hm_r[m_te])

        lo_g, hi_g, p_g = summarise(paired_boot(y_te, s_c, s_d, args.boot, seed=L, groups=grp_te))
        lo_r, hi_r, p_r = summarise(paired_boot(y_te, s_c, s_d, args.boot, seed=L))
        lo_v, hi_v, p_v = summarise(paired_boot(y_te, s_r, s_d, args.boot, seed=L, groups=grp_te))

        # deletion check: refusal probe after deleting r, against a permutation null
        Hc_e = apply_basis(Hc, R)
        ref_del, gs_ref = probe_auroc(Hc_e[chk_tr], c_ref[chk_tr], Hc_e[chk_te], c_ref[chk_te])
        null = permutation_null(Hc_e[chk_tr], c_ref[chk_tr], Hc_e[chk_te], c_ref[chk_te],
                                C=gs_ref.best_params_["lr__C"], n=args.perms, seed=1000 + L)
        null_hi = float(np.percentile(null, 97.5))
        ref_p = float((1 + np.sum(null >= ref_del)) / (1 + len(null)))

        row = {"layer": L, "harm_clean": clean, "harm_deleted": dele, "harm_random": rand,
               "drop": clean - dele, "drop_lo": lo_g, "drop_hi": hi_g, "drop_p": p_g,
               "drop_lo_rows": lo_r, "drop_hi_rows": hi_r, "drop_p_rows": p_r,
               "drop_vs_random": rand - dele, "dvr_lo": lo_v, "dvr_hi": hi_v, "dvr_p": p_v,
               "ref_deleted": ref_del, "null_lo": float(np.percentile(null, 2.5)),
               "null_hi": null_hi, "ref_p": ref_p, "erased_ok": ref_del <= null_hi}
        print(f"{L:>3} {clean:>6.3f} {dele:>6.3f} {clean - dele:>+7.3f} "
              f"[{lo_g:>+6.3f},{hi_g:>+6.3f}] {p_g:>6.3f} [{lo_r:>+6.3f},{hi_r:>+6.3f}] "
              f"{rand - dele:>+8.3f} {ref_del:>8.3f} {null_hi:>10.3f} {ref_p:>6.3f}")

        if args.resplits:
            rs = resplit_check(Hc, c_ref, ben_rows, test_frac, C=gs_ref.best_params_["lr__C"],
                               n=args.resplits)
            row.update({"resplit_mean": float(rs.mean()), "resplit_min": float(rs.min()),
                        "resplit_max": float(rs.max()),
                        "resplit_frac_fail": float(np.mean(rs > null_hi))})
            print(f"    re-split x{args.resplits}: refusal after deletion mean {rs.mean():.3f} "
                  f"(range {rs.min():.3f}-{rs.max():.3f}), above the null on "
                  f"{np.mean(rs > null_hi):.0%} of splits")
        rows.append(row)

    df = pd.DataFrame(rows)
    out = Path("results/probes") / args.matched_run
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "deletion_stats.csv", index=False)

    print("\n--- reading ---")
    sig = df[df.drop_lo > 0].layer.tolist()
    sig_rows = df[df.drop_lo_rows > 0].layer.tolist()
    print(f"harm drop > 0, cluster CI excludes 0 at layers: {sig or 'none'}")
    print(f"harm drop > 0, row-level CI excludes 0 at layers: {sig_rows or 'none'}")
    bad = df[~df.erased_ok].layer.tolist()
    print(f"deletion check above the null's 97.5th percentile at layers: {bad or 'none'} "
          "- treat their harm numbers as uninterpretable")
    if args.resplits:
        persistent = df[df.resplit_frac_fail > 0.5].layer.tolist()
        print(f"deletion check fails on most re-splits (residual refusal, not split noise) at "
              f"layers: {persistent or 'none'}")
    print(f"wrote {out / 'deletion_stats.csv'}")


if __name__ == "__main__":
    main()
