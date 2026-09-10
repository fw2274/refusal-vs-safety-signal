"""Leave-one-mechanism-out: does harm decoding transfer to an unseen KIND of harm/cue dissociation?

    python -m src.probes.multimech --run multimech

Every prompt is harmful-sounding (cue=1), so surface cues cannot separate the classes anywhere in
this corpus. Three folds: train on two mechanisms, test on the third.

  held out    trained on              asks
  xstest      jbb + orbench           does topic-matching + over-refusal teach lexical ambiguity?
  jbb         xstest + orbench        ... teach benign-intent-same-topic?
  orbench     xstest + jbb            ... teach over-cautious framing?

Reference conditions on the same test rows:

  within-mechanism   train and test inside the held-out mechanism (grouped by topic frame).
                     The ceiling: what is decodable when the mechanism IS in training.
  matched probe      the earlier matched-corpus probe, applied unchanged.

A probe that has learned harm scores similarly in both. A probe that has learned one mechanism's
signature scores well within-mechanism and near chance across mechanisms - which is precisely the
failure the previous OOD run surfaced, now measured directly instead of inferred.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from src.probes.train import bootstrap_auroc, make_probe

LAYERS = [1, 4, 7, 10, 13, 16, 19, 22, 25, 28]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="multimech")
    ap.add_argument("--matched-run", default="matched")
    ap.add_argument("--layer", type=int, default=13)
    ap.add_argument("--boot", type=int, default=1000)
    args = ap.parse_args()

    d = Path("results/activations") / args.run
    lab = pd.read_parquet(d / "labels.parquet")
    mech = lab["mechanism"].to_numpy()
    harm = lab["harm_label"].to_numpy().astype(int)
    grp = lab["group"].to_numpy()
    mechs = sorted(np.unique(mech))

    dm = Path("results/activations") / args.matched_run
    lm = pd.read_parquet(dm / "labels.parquet")
    m_tr = (lm["split"] == "train").to_numpy()
    m_harm = lm["harm_label"].to_numpy().astype(int)
    seen = set(lm["prompt"])
    fresh = (~lab["prompt"].isin(seen)).to_numpy()
    print(f"{len(lab)} prompts, mechanisms {mechs}")
    print(f"{int(fresh.sum())} of them are absent from the matched corpus "
          f"(the matched probe is only scored on those)\n")

    print(f"{'L':>3}" + "".join(f"{'hold ' + m:>13}" for m in mechs)
          + f"{'within-mech':>13}{'matched':>10}")
    rows = []
    for L in LAYERS:
        H = np.load(d / f"L{L:02d}.npy")
        Hm = np.load(dm / f"L{L:02d}.npy")
        gm = make_probe(); gm.fit(Hm[m_tr], m_harm[m_tr])

        rec = {"layer": L, "depth": L / 28}
        cross, within, matched = [], [], []
        for m in mechs:
            te = mech == m
            tr = ~te
            g = make_probe(); g.fit(H[tr], harm[tr])
            a = roc_auc_score(harm[te], g.decision_function(H[te]))
            rec[f"cross_{m}"] = a
            cross.append(a)

            # within-mechanism ceiling: grouped split by topic frame inside this mechanism
            gs = np.unique(grp[te])
            if len(gs) >= 3:
                rs = np.random.default_rng(0).permutation(len(gs))
                hold = set(gs[rs[: max(1, len(gs) // 3)]])
                w_te = te & np.isin(grp, list(hold))
                w_tr = te & ~np.isin(grp, list(hold))
                if harm[w_tr].std() > 0 and harm[w_te].std() > 0:
                    gw = make_probe(); gw.fit(H[w_tr], harm[w_tr])
                    aw = roc_auc_score(harm[w_te], gw.decision_function(H[w_te]))
                    rec[f"within_{m}"] = aw
                    within.append(aw)

            fm = te & fresh
            if harm[fm].std() > 0:
                am = roc_auc_score(harm[fm], gm.decision_function(H[fm]))
                rec[f"matched_{m}"] = am
                matched.append(am)

        rec["cross_mean"] = float(np.mean(cross))
        rec["within_mean"] = float(np.mean(within)) if within else float("nan")
        rec["matched_mean"] = float(np.mean(matched)) if matched else float("nan")
        rows.append(rec)
        print(f"{L:>3}" + "".join(f"{rec[f'cross_{m}']:>13.3f}" for m in mechs)
              + f"{rec['within_mean']:>13.3f}{rec['matched_mean']:>10.3f}")

    df = pd.DataFrame(rows)

    L = args.layer
    H = np.load(d / f"L{L:02d}.npy")
    print(f"\n--- layer {L}, with cluster-bootstrap intervals over topic groups ---")
    print(f"{'held-out mechanism':<22} {'cross-mech':>11} {'95% CI':>17} {'within-mech':>12} "
          f"{'matched probe':>14}")
    detail = {}
    for m in mechs:
        te = mech == m
        tr = ~te
        g = make_probe(); g.fit(H[tr], harm[tr])
        s = g.decision_function(H[te])
        a = roc_auc_score(harm[te], s)
        lo, hi = bootstrap_auroc(harm[te], s, args.boot, groups=grp[te])
        r = df[df.layer == L].iloc[0]
        w = r.get(f"within_{m}", float("nan"))
        mm = r.get(f"matched_{m}", float("nan"))
        detail[m] = {"cross": a, "ci": [lo, hi], "within": float(w), "matched": float(mm)}
        print(f"{m:<22} {a:>11.3f} [{lo:>6.3f},{hi:>6.3f}] {w:>12.3f} {mm:>14.3f}")

    cross_mean = float(np.mean([detail[m]["cross"] for m in mechs]))
    within_mean = float(np.nanmean([detail[m]["within"] for m in mechs]))
    best_cross = df["cross_mean"].max()
    best_L = int(df.loc[df["cross_mean"].idxmax(), "layer"])

    print(f"\ncross-mechanism mean at L{L}: {cross_mean:.3f}")
    print(f"within-mechanism mean at L{L}: {within_mean:.3f}")
    print(f"best cross-mechanism mean over swept layers: {best_cross:.3f} at L{best_L}")

    # The MEAN across folds is too generous: it lets two strong mechanisms carry a weak one. The
    # claim "harm transfers across mechanisms" has to hold for the worst fold, so gate on the min.
    cross_min = float(min(detail[m]["cross"] for m in mechs))
    worst_mech = min(mechs, key=lambda m: detail[m]["cross"])
    gap = within_mean - cross_mean
    print(f"\ncross-mechanism MIN at L{L}: {cross_min:.3f} (held-out {worst_mech})")
    print(f"generalization gap (within - cross mean) = {gap:+.3f}")

    if cross_min >= 0.70:
        print("READING: harm decoding transfers to every unseen dissociation mechanism. The signal")
        print("  is about harm, not about one mechanism's signature.")
    elif cross_mean >= 0.70:
        print(f"READING: PARTIAL transfer. Some mechanisms transfer well, but held-out "
              f"{worst_mech} reaches only {cross_min:.3f}. The mean passing is an averaging")
        print("  artifact - the probe has learned harm as expressed by some dissociation types and")
        print("  not others, so this is neither pure mechanism-signature learning nor a clean harm")
        print("  representation.")
    elif cross_mean >= 0.60:
        print("READING: weak partial transfer, well below the within-mechanism ceiling.")
    else:
        print("READING: no meaningful transfer. Each mechanism teaches its own signature.")

    print("\nGATE: " + (
        "PASS - every held-out mechanism >= 0.70" if cross_min >= 0.70 else
        f"PARTIAL - mean {cross_mean:.3f} but worst fold ({worst_mech}) {cross_min:.3f}"
        if cross_mean >= 0.70 else
        f"FAIL - cross-mechanism mean {cross_mean:.3f}, worst {cross_min:.3f}"))

    out = Path("results/probes") / args.run
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "multimech_sweep.csv", index=False)
    (out / "multimech_summary.json").write_text(json.dumps(
        {"layer": L, "cross_mean": cross_mean, "within_mean": within_mean,
         "gap": gap, "best_cross_mean": float(best_cross), "best_layer": best_L,
         "per_mechanism": detail}, indent=2, default=float))
    print(f"wrote {out / 'multimech_sweep.csv'}")


if __name__ == "__main__":
    main()
