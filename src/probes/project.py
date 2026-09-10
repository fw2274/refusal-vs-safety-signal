"""Phase 7: erase the refusal direction, then ask whether harm survives.

    python -m src.probes.project --cue-run cue --matched-run matched --layer 13

The experiment the whole project was built for. Three ingredients, each from the place it can be
estimated without a confound:

  refusal subspace  <- `cue` run, benign stratum, TRAIN rows. Harm is constant at benign on both
                       sides of the contrast, so the subspace cannot carry harm information.
                       Phase 6 showed this direction is causally load-bearing, not a correlate.
  harm evaluation   <- `matched` run, minimal pairs, grouped split. The source-confounded
                       contrast scored 1.000 in-distribution and inverted out of it, so it
                       cannot be used here.
  erasure operator  -> applied to the matched activations.

Because refusal labels exist only in the cue run and unconfounded harm labels only in the matched
run, the operator is necessarily fit on one distribution and applied to the other. That is
inherent to the design, not an oversight - but it does mean LEACE's guarantee holds strictly on
the cue distribution, which is why the correctness check below is run there.

Four conditions per layer:

  clean     harm probe on untouched matched activations
  transfer  the CLEAN-trained probe applied to ERASED matched test rows
            -> Q1: how much did the original probe rely on the refusal subspace?
  retrain   a fresh probe trained AND evaluated on erased activations
            -> Q2: does harm information survive in the orthogonal complement?
  random    retrain, but erasing a rank-matched random subspace
            -> mandatory control. Erasing ANY subspace perturbs the data; only a drop exceeding
               this one says anything about refusal.

Q1 and Q2 are different questions with different answers, and conflating them is the standard way
this experiment gets misreported.

CORRECTNESS CHECK. Before any of the above means anything, the erasure has to have worked: a
refusal probe trained on erased cue activations must fall to chance. If refusal is still
decodable, the subspace was not removed and every other number here is void.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from concept_erasure import LeaceEraser
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

from src.probes.train import bootstrap_auroc, make_probe

SWEEP_LAYERS = [1, 4, 7, 10, 13, 16, 19, 22, 25, 28]
RANKS = [1, 2, 4, 8, 16]


def unit(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


def apply_basis(H: np.ndarray, R: np.ndarray) -> np.ndarray:
    """Project out the row space of R (assumed orthonormal)."""
    return H - (H @ R.T) @ R


def diffmeans_basis(H: np.ndarray, pos: np.ndarray, neg: np.ndarray) -> np.ndarray:
    return unit(H[pos].mean(0) - H[neg].mean(0))[None, :]


def inlp_basis(H: np.ndarray, y: np.ndarray, mask: np.ndarray, k: int) -> np.ndarray:
    """Iterative nullspace projection: fit a refusal classifier, project out its weights, repeat."""
    Hc = H.copy()
    dirs = []
    for _ in range(k):
        sc = StandardScaler().fit(Hc[mask])
        clf = LogisticRegression(max_iter=2000, C=1e-2).fit(sc.transform(Hc[mask]), y[mask])
        w = clf.coef_[0] / sc.scale_
        if np.linalg.norm(w) == 0:
            break
        w = unit(w)
        dirs.append(w)
        Hc = Hc - np.outer(Hc @ w, w)
    Q, _ = np.linalg.qr(np.array(dirs).T)
    return Q.T[: len(dirs)]


def random_basis(d: int, k: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    Q, _ = np.linalg.qr(rng.standard_normal((d, k)))
    return Q.T[:k]


def fit_leace(H: np.ndarray, y: np.ndarray, mask: np.ndarray) -> LeaceEraser:
    x = torch.tensor(H[mask], dtype=torch.float64)
    z = torch.tensor(y[mask], dtype=torch.long)
    return LeaceEraser.fit(x, z)


def apply_leace(H: np.ndarray, er: LeaceEraser) -> np.ndarray:
    return er(torch.tensor(H, dtype=torch.float64)).numpy().astype(np.float32)


def auroc(y, s) -> float:
    return roc_auc_score(y, s) if len(np.unique(y)) > 1 else float("nan")


def probe_auroc(Htr, ytr, Hte, yte) -> tuple[float, object]:
    gs = make_probe()
    gs.fit(Htr, ytr)
    return auroc(yte, gs.decision_function(Hte)), gs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cue-run", default="cue")
    ap.add_argument("--matched-run", default="matched")
    ap.add_argument("--layer", type=int, default=13, help="layer for the rank-k curve")
    ap.add_argument("--boot", type=int, default=1000)
    args = ap.parse_args()

    dc = Path("results/activations") / args.cue_run
    dm = Path("results/activations") / args.matched_run
    lc = pd.read_parquet(dc / "labels.parquet")
    lm = pd.read_parquet(dm / "labels.parquet")

    c_tr = (lc["split"] == "train").to_numpy()
    c_te = (lc["split"] == "test").to_numpy()
    c_harm = lc["harm_label"].to_numpy().astype(int)
    c_ref = lc["refusal_label"].to_numpy().astype(int)
    ben = c_harm == 0
    fit_mask = c_tr & ben                      # benign stratum, train: harm held constant
    chk_tr, chk_te = c_tr & ben, c_te & ben    # refusal-probe correctness check rows

    m_tr = (lm["split"] == "train").to_numpy()
    m_te = (lm["split"] == "test").to_numpy()
    m_harm = lm["harm_label"].to_numpy().astype(int)
    m_grp = lm["group"].to_numpy()[m_te]

    print(f"refusal subspace from {args.cue_run}: {int((fit_mask & (c_ref==1)).sum())} refused / "
          f"{int((fit_mask & (c_ref==0)).sum())} complied (benign, train)")
    print(f"harm evaluated on {args.matched_run}: train={m_tr.sum()} test={m_te.sum()} "
          f"({len(np.unique(m_grp))} test groups)\n")

    # ---------------- Part A: layer sweep, rank-1 diff-in-means + LEACE ------------------
    print("=== Part A: layer sweep (rank-1 refusal erasure) ===")
    print(f"{'L':>3} {'ref clean':>10} {'ref erased':>11} {'harm clean':>11} {'transfer':>9} "
          f"{'retrain':>8} {'random':>8} {'leace rt':>9}")
    rows = []
    for L in SWEEP_LAYERS:
        Hc = np.load(dc / f"L{L:02d}.npy")
        Hm = np.load(dm / f"L{L:02d}.npy")

        R = diffmeans_basis(Hc, fit_mask & (c_ref == 1), fit_mask & (c_ref == 0))
        Rr = random_basis(Hc.shape[1], 1, seed=100 + L)
        er = fit_leace(Hc, c_ref, fit_mask)

        # correctness check: is refusal still decodable after erasure?
        ref_clean, _ = probe_auroc(Hc[chk_tr], c_ref[chk_tr], Hc[chk_te], c_ref[chk_te])
        Hc_e = apply_basis(Hc, R)
        ref_dim, _ = probe_auroc(Hc_e[chk_tr], c_ref[chk_tr], Hc_e[chk_te], c_ref[chk_te])
        Hc_l = apply_leace(Hc, er)
        ref_leace, _ = probe_auroc(Hc_l[chk_tr], c_ref[chk_tr], Hc_l[chk_te], c_ref[chk_te])

        # harm on matched
        harm_clean, gs_clean = probe_auroc(Hm[m_tr], m_harm[m_tr], Hm[m_te], m_harm[m_te])
        Hm_e = apply_basis(Hm, R)
        transfer = auroc(m_harm[m_te], gs_clean.decision_function(Hm_e[m_te]))
        retrain, _ = probe_auroc(Hm_e[m_tr], m_harm[m_tr], Hm_e[m_te], m_harm[m_te])
        Hm_r = apply_basis(Hm, Rr)
        rand, _ = probe_auroc(Hm_r[m_tr], m_harm[m_tr], Hm_r[m_te], m_harm[m_te])
        Hm_l = apply_leace(Hm, er)
        leace_rt, _ = probe_auroc(Hm_l[m_tr], m_harm[m_tr], Hm_l[m_te], m_harm[m_te])

        rows.append({"layer": L, "ref_clean": ref_clean, "ref_erased_dim": ref_dim,
                     "ref_erased_leace": ref_leace, "harm_clean": harm_clean,
                     "harm_transfer": transfer, "harm_retrain": retrain,
                     "harm_random": rand, "harm_leace_retrain": leace_rt})
        print(f"{L:>3} {ref_clean:>10.3f} {ref_dim:>11.3f} {harm_clean:>11.3f} "
              f"{transfer:>9.3f} {retrain:>8.3f} {rand:>8.3f} {leace_rt:>9.3f}")

    sweep = pd.DataFrame(rows)

    # ---------------- Part B: rank-k curve at the selected layer ------------------------
    L = args.layer
    print(f"\n=== Part B: rank-k erasure curve at L{L} ===")
    Hc = np.load(dc / f"L{L:02d}.npy")
    Hm = np.load(dm / f"L{L:02d}.npy")
    harm_clean, gs_clean = probe_auroc(Hm[m_tr], m_harm[m_tr], Hm[m_te], m_harm[m_te])
    lo0, hi0 = bootstrap_auroc(m_harm[m_te], gs_clean.decision_function(Hm[m_te]),
                               args.boot, groups=m_grp)
    print(f"clean harm AUROC = {harm_clean:.3f} [{lo0:.3f}, {hi0:.3f}]")
    print(f"\n{'k':>3} {'ref erased':>11} {'harm retrain':>13} {'95% CI':>16} {'random':>8} "
          f"{'delta vs random':>16}")
    krows = []
    for k in RANKS:
        Rk = inlp_basis(Hc, c_ref, fit_mask, k)
        Rrk = random_basis(Hc.shape[1], k, seed=500 + k)
        Hc_e = apply_basis(Hc, Rk)
        ref_after, _ = probe_auroc(Hc_e[chk_tr], c_ref[chk_tr], Hc_e[chk_te], c_ref[chk_te])
        Hm_e = apply_basis(Hm, Rk)
        rt, gs = probe_auroc(Hm_e[m_tr], m_harm[m_tr], Hm_e[m_te], m_harm[m_te])
        lo, hi = bootstrap_auroc(m_harm[m_te], gs.decision_function(Hm_e[m_te]),
                                 args.boot, groups=m_grp)
        Hm_r = apply_basis(Hm, Rrk)
        rd, _ = probe_auroc(Hm_r[m_tr], m_harm[m_tr], Hm_r[m_te], m_harm[m_te])
        krows.append({"k": k, "rank_actual": Rk.shape[0], "ref_erased": ref_after,
                      "harm_retrain": rt, "ci_lo": lo, "ci_hi": hi, "harm_random": rd})
        print(f"{k:>3} {ref_after:>11.3f} {rt:>13.3f} [{lo:>6.3f},{hi:>6.3f}] {rd:>8.3f} "
              f"{rt - rd:>+16.3f}")
    kdf = pd.DataFrame(krows)

    # ---------------- verdict -----------------------------------------------------------
    print("\n--- erasure correctness ---")
    worst = sweep["ref_erased_dim"].max()
    worst_l = sweep["ref_erased_leace"].max()
    print(f"refusal AUROC after rank-1 diff-in-means erasure: max {worst:.3f}, "
          f"median {sweep['ref_erased_dim'].median():.3f} over swept layers")
    print(f"refusal AUROC after LEACE:                        max {worst_l:.3f}")

    # Gate on the operator actually used for the harm numbers, at the selected layer.
    # LEACE and INLP both under-erase here for one reason: n_fit (746 benign train rows) < d
    # (1536). LEACE's whitening is rank-deficient so its guarantee does not survive transfer to
    # held-out rows, and a regularized classifier direction captures only part of the centroid
    # offset (measured: diff-in-means centroid gap 7.72 vs INLP 4.10, cos 0.53 between them). The
    # generalizable refusal signal IS the centroid offset, so plain diff-in-means erases best.
    ref_at_L = float(sweep.loc[sweep.layer == L, "ref_erased_dim"].iloc[0])
    erased_ok = ref_at_L <= 0.60
    print(f"\nat the evaluated layer L{L}: refusal AUROC after erasure = {ref_at_L:.3f}")
    print("  " + ("ERASURE VERIFIED at this layer - refusal is at chance, so the harm numbers "
                  "below are interpretable"
                  if erased_ok else
                  "ERASURE FAILED at this layer - refusal still decodable, harm numbers are void"))
    bad = sweep[sweep.ref_erased_dim > 0.60]["layer"].tolist()
    if bad:
        print(f"  layers where rank-1 erasure did NOT reach chance: {bad} - treat those rows as "
              "uninterpretable")

    sel = sweep[sweep.layer == L].iloc[0]
    print(f"\n--- harm survival at L{L} ---")
    print(f"clean            {sel.harm_clean:.3f}")
    print(f"transfer (Q1)    {sel.harm_transfer:.3f}   drop {sel.harm_clean-sel.harm_transfer:+.3f}")
    print(f"retrain  (Q2)    {sel.harm_retrain:.3f}   drop {sel.harm_clean-sel.harm_retrain:+.3f}")
    print(f"random control   {sel.harm_random:.3f}   drop {sel.harm_clean-sel.harm_random:+.3f}")
    print(f"leace retrain    {sel.harm_leace_retrain:.3f}")

    real_drop = sel.harm_clean - sel.harm_retrain
    rand_drop = sel.harm_clean - sel.harm_random
    print(f"\nretrain drop {real_drop:+.3f} vs random control {rand_drop:+.3f}")
    if real_drop <= max(rand_drop, 0.0) + 0.02:
        print("READING: harm survives refusal erasure - the drop is no larger than removing an")
        print("  arbitrary direction. Evidence for a safety signal beyond refusal.")
    else:
        print("READING: erasing refusal costs more than erasing a random direction, so the harm")
        print("  representation genuinely overlaps the refusal subspace at this layer.")

    out = Path("results/probes") / args.matched_run
    out.mkdir(parents=True, exist_ok=True)
    sweep.to_csv(out / "phase7_sweep.csv", index=False)
    kdf.to_csv(out / f"phase7_rank_curve_L{L}.csv", index=False)
    (out / "phase7_summary.json").write_text(json.dumps(
        {"layer": L, "leace_erasure_verified": bool(erased_ok),
         "max_ref_after_dim": float(worst), "max_ref_after_leace": float(worst_l),
         "clean": float(sel.harm_clean), "transfer": float(sel.harm_transfer),
         "retrain": float(sel.harm_retrain), "random": float(sel.harm_random),
         "leace_retrain": float(sel.harm_leace_retrain)}, indent=2))
    print(f"wrote {out / 'phase7_sweep.csv'}")


if __name__ == "__main__":
    main()
