"""Head-to-head: matched-trained harm probe vs the source-confounded one.

    python -m src.probes.matched_eval --matched-run matched --confounded-run cue --ood-run jbb

Three columns, all scored on the SAME held-out matched test groups (unseen syntactic frames):

  1. matched-trained  - trained on matched train groups. The honest probe.
  2. confounded       - the advbench-vs-alpaca probe from the SFT corpus, applied unchanged.
                        If it drops sharply here, it was reading provenance, not harm.
  3. cos_clean        - cos(matched harm direction, harm-free refusal direction from the SFT run).
                        The entanglement number, now with a harm direction that is not an artifact.

The matched harm direction is estimated on matched TRAIN groups only, so the cosine is not fit on
the same rows anything is scored on.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from src.probes.train import bootstrap_auroc, make_probe


def unit(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--matched-run", default="matched")
    ap.add_argument("--confounded-run", default="cue")
    ap.add_argument("--ood-run", default=None, help="optional extra OOD check, e.g. jbb")
    ap.add_argument("--boot", type=int, default=1000)
    args = ap.parse_args()

    dm = Path("results/activations") / args.matched_run
    dc = Path("results/activations") / args.confounded_run
    lm = pd.read_parquet(dm / "labels.parquet")
    lc = pd.read_parquet(dc / "labels.parquet")
    n_layers = json.loads((dm / "meta.json").read_text())["n_layers"]

    m_tr = (lm["split"] == "train").to_numpy()
    m_te = (lm["split"] == "test").to_numpy()
    m_harm = lm["harm_label"].to_numpy().astype(int)

    c_tr = (lc["split"] == "train").to_numpy()
    c_harm = lc["harm_label"].to_numpy().astype(int)
    c_ref = lc["refusal_label"].to_numpy().astype(int)
    ref_estimable = len(np.unique(c_ref[c_tr & (c_harm == 0)])) > 1

    print(f"matched: train={m_tr.sum()} test={m_te.sum()} (grouped by topic family)")
    print(f"confounded probe trained on: {args.confounded_run} (n={c_tr.sum()})")
    print(f"\n{'layer':>5} {'matched AUROC':>14} {'95% CI':>16} {'confounded->matched':>20} "
          f"{'cos_clean':>10}")

    rows = []
    for L in range(1, n_layers + 1):
        Hm = np.load(dm / f"L{L:02d}.npy")
        Hc = np.load(dc / f"L{L:02d}.npy")

        # 1. honest probe: matched train groups -> unseen matched test groups
        gm = make_probe(); gm.fit(Hm[m_tr], m_harm[m_tr])
        s_m = gm.decision_function(Hm[m_te])
        auc_m = roc_auc_score(m_harm[m_te], s_m)
        lo, hi = bootstrap_auroc(m_harm[m_te], s_m, args.boot, groups=lm["group"].to_numpy()[m_te])

        # 2. the old confounded probe, applied to the same matched test rows
        gc = make_probe(); gc.fit(Hc[c_tr], c_harm[c_tr])
        auc_c = roc_auc_score(m_harm[m_te], gc.decision_function(Hm[m_te]))

        # 3. entanglement with the harm-free refusal direction
        if ref_estimable:
            r_harm = unit(Hm[m_tr & (m_harm == 1)].mean(0) - Hm[m_tr & (m_harm == 0)].mean(0))
            r_ref = unit(Hc[c_tr & (c_harm == 0) & (c_ref == 1)].mean(0)
                         - Hc[c_tr & (c_harm == 0) & (c_ref == 0)].mean(0))
            cos = float(r_harm @ r_ref)
        else:
            cos = float("nan")

        rows.append({"layer": L, "depth": L / n_layers, "auroc_matched": auc_m,
                     "ci_lo": lo, "ci_hi": hi, "auroc_confounded_on_matched": auc_c,
                     "cos_clean": cos})
        fc = f"{cos:>+10.3f}" if np.isfinite(cos) else f"{'n/a':>10}"
        print(f"{L:>5} {auc_m:>14.3f} [{lo:>6.3f},{hi:>6.3f}] {auc_c:>20.3f} {fc}")

    df = pd.DataFrame(rows)
    best = df.loc[df["auroc_matched"].idxmax()]
    print(f"\nmatched-trained peak: AUROC {best.auroc_matched:.3f} "
          f"[{best.ci_lo:.3f}, {best.ci_hi:.3f}] at layer {int(best.layer)} "
          f"(depth {best.depth:.2f}), cos_clean {best.cos_clean:+.3f}")
    print(f"confounded probe on the same rows, that layer: "
          f"{best.auroc_confounded_on_matched:.3f}")
    print(f"confounded probe, best over all layers: "
          f"{df.auroc_confounded_on_matched.max():.3f}")

    early = df[df.depth <= 0.30]["auroc_matched"].max()
    print(f"\nmatched AUROC peak within the first 30% of layers: {early:.3f}")
    print("  (the confounded probe hit 1.000 at depth 0.14; a matched contrast should NOT,")
    print("   because surface form no longer separates the classes)")

    # Selecting by raw argmax is wrong when AUROC plateaus: the peak layer is then noise, and it
    # is biased toward late layers where entanglement with refusal is highest. Take the EARLIEST
    # layer statistically indistinguishable from the peak (auroc >= the peak's CI lower bound) -
    # the same idea as the one-standard-error rule.
    sel = df[df["auroc_matched"] >= best.ci_lo].iloc[0]
    print(f"\nselected layer (earliest within the peak's 95% CI): L{int(sel.layer)} "
          f"(depth {sel.depth:.2f})")
    print(f"  AUROC {sel.auroc_matched:.3f} [{sel.ci_lo:.3f}, {sel.ci_hi:.3f}]   "
          f"cos_clean {sel.cos_clean:+.3f}")
    print(f"  vs argmax L{int(best.layer)} (depth {best.depth:.2f}): AUROC "
          f"{best.auroc_matched:.3f}, cos_clean {best.cos_clean:+.3f}")

    gate = (sel.auroc_matched >= 0.80 and sel.depth >= 0.30 and early < 0.95)
    print("\nGATE: " + ("PASS - harm is decodable, and not from surface form"
                        if gate else
                        f"FAIL - selected {sel.auroc_matched:.3f} at depth {sel.depth:.2f}, "
                        f"early peak {early:.3f}"))

    out = Path("results/probes") / args.matched_run
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "matched_vs_confounded.csv", index=False)
    print(f"wrote {out / 'matched_vs_confounded.csv'}")


if __name__ == "__main__":
    main()
