"""Per-layer harm/refusal geometry and the deletion result, side by side for two models.

    python -m src.probes.compare_models \
        --pair Qwen-1.5B cue matched --pair Llama-1B llama-cue llama-matched         --pair OLMo-2-1B olmo-cue olmo-matched [--x layer] [--replot]

For each model and layer:
  angle_before   angle between the matched harm direction and the harm-free refusal direction r
                 (both diff-in-means, the same vectors as cos_clean in matched_eval.py)
  angle_after    the same angle once r is deleted from the matched activations. Rank-1 deletion
                 makes this 90 degrees by construction; it is recomputed here as a check.
  rotation       how far the harm direction moved when r was deleted (= 90 - angle_before)
  ref_before     refusal probe AUROC on the cue benign stratum, untouched activations
  ref_after      same, after deleting r            (from deletion_stats.csv)
  harm_before    matched harm probe AUROC, untouched (from deletion_stats.csv)
  harm_after     matched harm probe retrained after deleting r (from deletion_stats.csv)

Layers are compared by relative depth, since the two models have different layer counts.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.probes.project import apply_basis, diffmeans_basis, probe_auroc, unit
from src.probes.train import n_layers_of

COLORS = ["#2a78d6", "#eb6834", "#1baf7a"]  # categorical slots 1-3 of the reference palette


def angle(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.degrees(np.arccos(np.clip(unit(a) @ unit(b), -1.0, 1.0))))


def model_table(name: str, cue_run: str, matched_run: str) -> pd.DataFrame:
    dc = Path("results/activations") / cue_run
    dm = Path("results/activations") / matched_run
    lc = pd.read_parquet(dc / "labels.parquet")
    lm = pd.read_parquet(dm / "labels.parquet")
    c_tr = (lc["split"] == "train").to_numpy()
    c_te = (lc["split"] == "test").to_numpy()
    c_ref = lc["refusal_label"].to_numpy().astype(int)
    ben = lc["harm_label"].to_numpy().astype(int) == 0
    m_tr = (lm["split"] == "train").to_numpy()
    m_harm = lm["harm_label"].to_numpy().astype(int)

    dstats = pd.read_csv(Path("results/probes") / matched_run / "deletion_stats.csv")
    n_layers = n_layers_of(dm)
    rows = []
    for L in range(1, n_layers + 1):
        Hc = np.load(dc / f"L{L:02d}.npy")
        Hm = np.load(dm / f"L{L:02d}.npy")
        R = diffmeans_basis(Hc, c_tr & ben & (c_ref == 1), c_tr & ben & (c_ref == 0))
        h = Hm[m_tr & (m_harm == 1)].mean(0) - Hm[m_tr & (m_harm == 0)].mean(0)
        Hm_e = apply_basis(Hm, R)
        h_e = Hm_e[m_tr & (m_harm == 1)].mean(0) - Hm_e[m_tr & (m_harm == 0)].mean(0)
        ref_before, _ = probe_auroc(Hc[c_tr & ben], c_ref[c_tr & ben],
                                    Hc[c_te & ben], c_ref[c_te & ben])
        d = dstats[dstats.layer == L].iloc[0]
        rows.append({"model": name, "layer": L, "depth": L / n_layers,
                     "angle_before": angle(h, R[0]), "angle_after": angle(h_e, R[0]),
                     "rotation": angle(h, h_e),
                     "ref_before": ref_before, "ref_after": d.ref_deleted,
                     "ref_null_hi": d.null_hi,
                     "harm_before": d.harm_clean, "harm_after": d.harm_deleted,
                     "harm_drop_lo": d.drop_lo, "harm_drop_hi": d.drop_hi})
    return pd.DataFrame(rows)


def plot(df: pd.DataFrame, out: Path, x: str = "depth") -> None:
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.edgecolor": "#8a8984", "axes.labelcolor": "#52514e",
                         "xtick.color": "#52514e", "ytick.color": "#52514e"})
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), facecolor="#fcfcfb")
    panels = [
        ("angle", "Angle between harm and refusal directions", "degrees", (0, 115)),
        ("harm", "Harm probe AUROC (topic-matched)", "AUROC", (0.3, 1.02)),
        ("ref", "Refusal probe AUROC (benign stratum)", "AUROC", (0.3, 1.02)),
    ]
    # by layer number the shorter models end at their own last layer; by depth all end at 1.0
    xlim, x0 = ((0, 1.03), 0.02) if x == "depth" else ((0, df.layer.max() + 1), 0.5)
    for ax, (key, title, ylab, ylim) in zip(axes, panels):
        ax.set_facecolor("#fcfcfb")
        ax.grid(axis="y", color="#e4e3df", linewidth=0.8)
        for (name, g), c in zip(df.groupby("model", sort=False), COLORS):
            if key == "angle":
                ax.plot(g[x], g.angle_before, color=c, lw=2, marker="o", ms=4, label=name)
            else:
                ax.plot(g[x], g[f"{key}_before"], color=c, lw=2, marker="o", ms=4,
                        label=f"{name}, before")
                ax.plot(g[x], g[f"{key}_after"], color=c, lw=2, ls="--", marker="o", ms=4,
                        mfc="#fcfcfb", label=f"{name}, after removal")
        if key == "angle":
            ax.axhline(90, color="#52514e", lw=1, ls=":")
            ax.text(x0, 93, "after removal: 90° by construction", color="#52514e", fontsize=9)
        if key != "angle":
            ax.axhline(0.5, color="#52514e", lw=1, ls=":")
            ax.text(x0, 0.475, "chance", color="#52514e", fontsize=9)
        ax.set_title(title, loc="left", color="#0b0b0b", fontsize=11)
        ax.set_xlabel("relative depth (layer / n_layers)" if x == "depth" else "layer")
        ax.set_ylabel(ylab)
        ax.set_ylim(*ylim)
        ax.set_xlim(*xlim)
        if x == "layer":
            ax.set_xticks(range(0, int(df.layer.max()) + 1, 4))
        ax.legend(frameon=False, fontsize=8.5, loc="lower right" if key == "harm" else "upper right", bbox_to_anchor=(1, 0.93) if key == "ref" else None)
    fig.tight_layout()
    fig.savefig(out, dpi=150, facecolor=fig.get_facecolor())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pair", nargs=3, action="append", required=True,
                    metavar=("NAME", "CUE_RUN", "MATCHED_RUN"))
    ap.add_argument("--out", default="results/figures/compare_models")
    ap.add_argument("--x", choices=["depth", "layer"], default="depth",
                    help="x axis; 'layer' writes <out>_layer.png")
    ap.add_argument("--replot", action="store_true",
                    help="redraw from the saved <out>.csv instead of refitting the probes")
    args = ap.parse_args()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    if args.replot:
        df = pd.read_csv(out.with_suffix(".csv"))
        df = df[df.model.isin([p[0] for p in args.pair])]
    else:
        df = pd.concat([model_table(*p) for p in args.pair], ignore_index=True)
        df.to_csv(out.with_suffix(".csv"), index=False)
    png = out.with_suffix(".png") if args.x == "depth" else out.with_name(out.name + "_layer.png")
    plot(df, png, args.x)
    worst = (df.angle_after - 90).abs().max()
    print(f"max |angle_after - 90| over all layers: {worst:.2e} deg (deletion check)")
    cols = ["layer", "depth", "angle_before", "rotation", "harm_before", "harm_after",
            "ref_before", "ref_after"]
    for name, g in df.groupby("model", sort=False):
        print(f"\n{name}\n" + g[cols].round(3).to_string(index=False))
    print(f"\nwrote {png}" + ("" if args.replot else f" and {out.with_suffix('.csv')}"))


if __name__ == "__main__":
    main()
