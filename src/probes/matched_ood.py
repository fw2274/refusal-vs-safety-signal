"""Does the matched-trained harm probe survive out of distribution?

    python -m src.probes.matched_ood --matched-run matched --cue-run cue --ood-run jbb

The confounded probe scored 1.000 in-distribution and 0.005-0.19 on OOD harmful - actively
inverted, because it had learned "imperative phrasing = harmful". The matched probe fixed the
in-distribution artifact (layer 1 dropped 1.000 -> 0.689), but nothing has yet checked whether it
transfers. That is the last thing standing between "harm survives refusal erasure" and a claim
about a safety representation rather than a slightly better artifact.

Four OOD contrasts, in increasing difficulty. None of these sources appear in the matched corpus,
and any prompt whose text also occurs in matched (either split) is dropped, so nothing the probe
trained on can leak in:

  advbench    vs alpaca     harmful imperative vs benign neutral - the easy, confounded contrast
  advbench    vs orbench    harmful vs benign, BOTH harmful-sounding - cue held roughly constant
  forbidden_q vs alpaca     harmful question-phrased vs benign neutral - phrasing flipped
  forbidden_q vs orbench    both sides unseen, cue roughly constant - the hardest

The second and fourth are the ones that matter: they hold surface harm cues roughly constant, so a
probe reading cues rather than harm cannot score on them.

The same contrasts are scored with the confounded probe for comparison.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from src.probes.train import bootstrap_auroc, make_probe

CONTRASTS = [
    ("advbench_vs_alpaca", "advbench", "alpaca"),
    ("advbench_vs_orbench", "advbench", "orbench_hard"),
    ("forbiddenq_vs_alpaca", "forbidden_q", "alpaca"),
    ("forbiddenq_vs_orbench", "forbidden_q", "orbench_hard"),
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--matched-run", default="matched")
    ap.add_argument("--cue-run", default="cue")
    ap.add_argument("--ood-run", default="jbb")
    ap.add_argument("--boot", type=int, default=1000)
    args = ap.parse_args()

    dm = Path("results/activations") / args.matched_run
    dc = Path("results/activations") / args.cue_run
    do = Path("results/activations") / args.ood_run
    lm = pd.read_parquet(dm / "labels.parquet")
    lc = pd.read_parquet(dc / "labels.parquet")
    lo = pd.read_parquet(do / "labels.parquet")
    n_layers = json.loads((dm / "meta.json").read_text())["n_layers"]

    m_tr = (lm["split"] == "train").to_numpy()
    m_te = (lm["split"] == "test").to_numpy()
    m_harm = lm["harm_label"].to_numpy().astype(int)
    m_grp = lm["group"].to_numpy()[m_te]
    c_tr = (lc["split"] == "train").to_numpy()
    c_harm = lc["harm_label"].to_numpy().astype(int)

    # No prompt the matched probe trained on may appear in an OOD evaluation set.
    seen = set(lm["prompt"])
    print(f"excluding {len(seen)} matched prompts from every OOD set")

    def source_mask(lab: pd.DataFrame, source: str) -> np.ndarray:
        return ((lab["source"] == source) & (~lab["prompt"].isin(seen))).to_numpy()

    # Each contrast draws its two halves from whichever run holds them.
    where = {"advbench": (dc, lc), "alpaca": (dc, lc), "orbench_hard": (dc, lc),
             "forbidden_q": (do, lo)}
    sets = {}
    for name, pos_src, neg_src in CONTRASTS:
        (dp, lp), (dn, ln) = where[pos_src], where[neg_src]
        pm, nm = source_mask(lp, pos_src), source_mask(ln, neg_src)
        if pm.sum() < 20 or nm.sum() < 20:
            print(f"  skip {name}: too few rows ({pm.sum()}/{nm.sum()})")
            continue
        sets[name] = (dp, pm, dn, nm)
        print(f"  {name}: {int(pm.sum())} harmful / {int(nm.sum())} benign")

    print(f"\n{'L':>3} {'matched test':>13}" + "".join(f"{n.replace('_vs_','/'):>22}"
                                                       for n in sets))
    rows = []
    for L in range(1, n_layers + 1):
        Hm = np.load(dm / f"L{L:02d}.npy")
        gm = make_probe(); gm.fit(Hm[m_tr], m_harm[m_tr])
        indist = roc_auc_score(m_harm[m_te], gm.decision_function(Hm[m_te]))

        Hc = np.load(dc / f"L{L:02d}.npy")
        gc = make_probe(); gc.fit(Hc[c_tr], c_harm[c_tr])

        rec = {"layer": L, "depth": L / n_layers, "matched_indist": indist}
        cache = {}
        for name, (dp, pm, dn, nm) in sets.items():
            Hp = Hm if dp == dm else (Hc if dp == dc else np.load(dp / f"L{L:02d}.npy"))
            Hn = Hm if dn == dm else (Hc if dn == dc else np.load(dn / f"L{L:02d}.npy"))
            if dp == do:
                Hp = cache.setdefault("ood", np.load(do / f"L{L:02d}.npy"))
            if dn == do:
                Hn = cache.setdefault("ood", np.load(do / f"L{L:02d}.npy"))
            X = np.concatenate([Hp[pm], Hn[nm]])
            y = np.concatenate([np.ones(pm.sum(), int), np.zeros(nm.sum(), int)])
            rec[f"matched_{name}"] = roc_auc_score(y, gm.decision_function(X))
            rec[f"confounded_{name}"] = roc_auc_score(y, gc.decision_function(X))
        rows.append(rec)
        print(f"{L:>3} {indist:>13.3f}" + "".join(f"{rec[f'matched_{n}']:>22.3f}" for n in sets))

    df = pd.DataFrame(rows)

    # Selected layer = earliest within the in-distribution peak's CI (one-SE rule).
    Hm13 = np.load(dm / "L13.npy")
    g13 = make_probe(); g13.fit(Hm13[m_tr], m_harm[m_tr])
    lo13, hi13 = bootstrap_auroc(m_harm[m_te], g13.decision_function(Hm13[m_te]),
                                 args.boot, groups=m_grp)
    print(f"\nL13 in-distribution: {df[df.layer==13].matched_indist.iloc[0]:.3f} "
          f"[{lo13:.3f}, {hi13:.3f}]")

    print(f"\n{'contrast':<24} {'matched L13':>12} {'matched best':>13} "
          f"{'confounded L13':>15} {'confounded best':>16}")
    summary = {}
    for name in sets:
        m13 = df[df.layer == 13][f"matched_{name}"].iloc[0]
        mb = df[f"matched_{name}"].max()
        c13 = df[df.layer == 13][f"confounded_{name}"].iloc[0]
        cb = df[f"confounded_{name}"].max()
        summary[name] = {"matched_L13": m13, "matched_best": mb,
                         "confounded_L13": c13, "confounded_best": cb}
        print(f"{name:<24} {m13:>12.3f} {mb:>13.3f} {c13:>15.3f} {cb:>16.3f}")

    hard = [n for n in sets if n.endswith("orbench")]
    if hard:
        worst = min(summary[n]["matched_L13"] for n in hard)
        print(f"\nhardest contrasts (cue held roughly constant): matched probe scores "
              f"{worst:.3f} at L13 minimum")
        conf_worst = min(summary[n]["confounded_L13"] for n in hard)
        print(f"  same contrasts, confounded probe: {conf_worst:.3f}")
        print("\nGATE: " + (
            "PASS - the matched probe transfers to unseen sources with surface cues held "
            "constant; it encodes harm, not provenance"
            if worst >= 0.70 else
            f"FAIL - matched probe drops to {worst:.3f} where cues are held constant. It "
            "generalizes better than the confounded probe but still leans on surface features."))

    out = Path("results/probes") / args.matched_run
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "matched_ood.csv", index=False)
    (out / "matched_ood_summary.json").write_text(json.dumps(
        {"indist_L13": float(df[df.layer == 13].matched_indist.iloc[0]),
         "indist_ci": [lo13, hi13], "contrasts": summary}, indent=2, default=float))
    print(f"wrote {out / 'matched_ood.csv'}")


if __name__ == "__main__":
    main()
