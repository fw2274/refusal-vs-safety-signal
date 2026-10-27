"""Static SVG figures for the README, drawn from result CSVs already in the repo.

    python -m src.plots.readme_figures

Standard library only, so it runs without the analysis environment. Each SVG carries its own
light and dark colours (prefers-color-scheme), so it reads on GitHub in either theme.

  docs/fig_angle_by_layer.svg           angle between r and d per layer (README Section 5.2)
  docs/fig_refusal_check_original.svg   refusal probe after deleting r, original split (Appendix)
"""

from __future__ import annotations

import csv
import math
from pathlib import Path

SRC = Path("results/probes/matched/r_stratum_comparison.csv")
OUT = Path("docs")

W, H = 720, 340
ML, MR, MT, MB = 60, 28, 56, 50  # margins: left, right, top, bottom

STYLE = """
<style>
  .bg { fill: #fcfcfb; }
  .t1 { fill: #0b0b0b; font: 600 15px system-ui, -apple-system, Segoe UI, sans-serif; }
  .t2 { fill: #52514e; font: 12px system-ui, -apple-system, Segoe UI, sans-serif; }
  .lab { fill: #0b0b0b; font: 600 12px system-ui, -apple-system, Segoe UI, sans-serif; }
  .grid { stroke: #e4e3df; stroke-width: 1; }
  .axis { stroke: #8f8e89; stroke-width: 1; }
  .ref { stroke: #8f8e89; stroke-width: 1.5; stroke-dasharray: 5 4; fill: none; }
  .s1 { stroke: #2a78d6; fill: none; stroke-width: 2; stroke-linejoin: round; }
  .d1 { fill: #2a78d6; stroke: #fcfcfb; stroke-width: 2; }
  .d2 { fill: #eb6834; stroke: #fcfcfb; stroke-width: 2; }
  @media (prefers-color-scheme: dark) {
    .bg { fill: #1a1a19; }
    .t1, .lab { fill: #f0efec; }
    .t2 { fill: #c3c2b7; }
    .grid { stroke: #33332f; }
    .axis, .ref { stroke: #77766f; }
    .s1 { stroke: #3987e5; }
    .d1 { fill: #3987e5; stroke: #1a1a19; }
    .d2 { fill: #d95926; stroke: #1a1a19; }
  }
</style>
"""


def load() -> list[dict]:
    with SRC.open() as f:
        return [{k: float(v) for k, v in row.items()} for row in csv.DictReader(f)]


def frame(title: str, subtitle: str, desc: str, ylo: float, yhi: float, yticks, yfmt, ylabel):
    """Return (svg parts, x(), y()) for a 28-layer chart with the given y range."""
    pw, ph = W - ML - MR, H - MT - MB
    x = lambda L: ML + (L - 1) / 27 * pw
    y = lambda v: MT + (yhi - v) / (yhi - ylo) * ph
    p = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" '
         f'height="{H}" role="img" aria-labelledby="t d">',
         f'<title id="t">{title}</title><desc id="d">{desc}</desc>', STYLE,
         f'<rect class="bg" width="{W}" height="{H}" rx="6"/>',
         f'<text class="t1" x="{ML}" y="24">{title}</text>',
         f'<text class="t2" x="{ML}" y="42">{subtitle}</text>']
    for v in yticks:
        p.append(f'<line class="grid" x1="{ML}" x2="{W - MR}" y1="{y(v):.1f}" y2="{y(v):.1f}"/>')
        p.append(f'<text class="t2" x="{ML - 8}" y="{y(v) + 4:.1f}" text-anchor="end">{yfmt(v)}</text>')
    p.append(f'<line class="axis" x1="{ML}" x2="{W - MR}" y1="{MT + ph}" y2="{MT + ph}"/>')
    for L in (1, 4, 7, 10, 13, 16, 19, 22, 25, 28):
        p.append(f'<text class="t2" x="{x(L):.1f}" y="{MT + ph + 18}" text-anchor="middle">{L}</text>')
    p.append(f'<text class="t2" x="{ML + pw / 2}" y="{H - 10}" text-anchor="middle">Layer</text>')
    p.append(f'<text class="t2" transform="translate(16 {MT + ph / 2}) rotate(-90)" '
             f'text-anchor="middle">{ylabel}</text>')
    return p, x, y


def angle_figure(rows: list[dict]) -> str:
    ang = [(int(r["layer"]), math.degrees(math.acos(max(-1.0, min(1.0, r["cos_safe_d"])))))
           for r in rows]
    p, x, y = frame(
        "Angle between the refusal direction r and the harm direction d",
        "90° = unrelated, 0° = the same direction. Cue-arm model, all 28 layers.",
        "Line chart of the r-d angle by layer: near 90 degrees in layers 1-8, falling from about "
        "85 to 34 degrees between layers 9 and 17, then levelling off near 30 degrees.",
        0, 110, (0, 30, 60, 90), lambda v: f"{v:.0f}°", "Angle r–d (degrees)")
    p.append(f'<line class="ref" x1="{ML}" x2="{W - MR}" y1="{y(90):.1f}" y2="{y(90):.1f}"/>')
    p.append(f'<text class="t2" x="{W - MR}" y="{y(90) - 6:.1f}" text-anchor="end">unrelated (90°)</text>')
    pts = " ".join(f"{x(L):.1f},{y(a):.1f}" for L, a in ang)
    p.append(f'<polyline class="s1" points="{pts}"/>')
    for L, a in ang:
        p.append(f'<circle class="d1" cx="{x(L):.1f}" cy="{y(a):.1f}" r="4"><title>Layer {L}: '
                 f'{a:.1f}°</title></circle>')
    a13 = dict(ang)[13]
    p.append(f'<text class="lab" x="{x(13) + 8:.1f}" y="{y(a13) - 8:.1f}">L13 (headline layer): {a13:.0f}°</text>')
    Lmin, amin = min(ang, key=lambda t: t[1])
    p.append(f'<text class="lab" x="{x(Lmin):.1f}" y="{y(amin) + 22:.1f}" text-anchor="middle">'
             f'min {amin:.1f}° at L{Lmin}</text>')
    p.append("</svg>")
    return "\n".join(p)


def refusal_figure(rows: list[dict]) -> str:
    ref = [(int(r["layer"]), r["ref_after_safe"]) for r in rows]
    p, x, y = frame(
        "Refusal probe AUROC after deleting r (original split)",
        "Near 0.5 = refusal no longer decodable. Before deletion: 0.997–1.000 at every layer.",
        "Dot chart of refusal probe AUROC after deleting r, by layer, on the original train/test "
        "split. Most layers sit between 0.40 and 0.60; layers 4, 7, 15 and 18 are above 0.60.",
        0.35, 0.70, (0.4, 0.5, 0.6, 0.7), lambda v: f"{v:.1f}", "Refusal AUROC")
    p.append(f'<line class="axis" x1="{ML}" x2="{W - MR}" y1="{y(0.5):.1f}" y2="{y(0.5):.1f}"/>')
    p.append(f'<text class="t2" x="{W - MR}" y="{y(0.5) + 16:.1f}" text-anchor="end">chance (0.5)</text>')
    p.append(f'<line class="ref" x1="{ML}" x2="{W - MR}" y1="{y(0.6):.1f}" y2="{y(0.6):.1f}"/>')
    p.append(f'<text class="t2" x="{W - MR}" y="{y(0.6) - 6:.1f}" text-anchor="end">0.60 flag</text>')
    pts = " ".join(f"{x(L):.1f},{y(v):.1f}" for L, v in ref)
    p.append(f'<polyline class="s1" points="{pts}" style="stroke-width:1;opacity:.5"/>')
    for L, v in ref:
        hi = v > 0.60
        p.append(f'<circle class="{"d2" if hi else "d1"}" cx="{x(L):.1f}" cy="{y(v):.1f}" '
                 f'r="{5 if hi else 4}"><title>Layer {L}: {v:.3f}</title></circle>')
        if hi:
            p.append(f'<text class="lab" x="{x(L):.1f}" y="{y(v) - 10:.1f}" text-anchor="middle">L{L}</text>')
    p.append("</svg>")
    return "\n".join(p)


def main() -> None:
    rows = load()
    OUT.mkdir(exist_ok=True)
    for name, svg in (("fig_angle_by_layer.svg", angle_figure(rows)),
                      ("fig_refusal_check_original.svg", refusal_figure(rows))):
        (OUT / name).write_text(svg, encoding="utf-8")
        print(f"wrote {OUT / name}")


if __name__ == "__main__":
    main()
