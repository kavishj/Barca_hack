"""Plot ground truth, a submission and the underlying evidence over a frame range.

    python tools/timeline.py --sub evaluation/results/possession_v1.csv
    python tools/timeline.py --sub evaluation/results/possession_v1.csv --start 40000 --stop 47500

Frames outside the public GT window plot no truth strip; the evidence is always shown, so this is
also the tool for reading the rest of the match against the video.
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from possession import evidence, signals  # noqa: E402
from possession.data import load_match, load_public_gt  # noqa: E402

COLOURS = {"H": "tab:blue", "A": "tab:red", "D": "0.6", "X": "white"}


def label_strip(ax, frames, labels, title):
    for label, colour in COLOURS.items():
        ax.fill_between(frames, 0, 1, where=labels == label, color=colour, step="mid", linewidth=0)
    ax.set_yticks([])
    ax.set_ylabel(title, rotation=0, ha="right", va="center")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--sub", nargs="*", default=[], help="submission CSVs to show")
    parser.add_argument("--start", type=int, default=11185)
    parser.add_argument("--stop", type=int, default=18685)
    parser.add_argument("--out", default=os.path.join(ROOT, "evaluation", "reports", "timeline.png"))
    args = parser.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    match = load_match()
    sig = signals.compute(match)
    ev = evidence.compute(sig)
    gt = load_public_gt()
    a, b = args.start, args.stop
    x = np.arange(a, b)

    subs = [(os.path.splitext(os.path.basename(p))[0],
             pd.read_csv(p, dtype={"label": "object"})["label"].values) for p in args.sub]
    n_strips = 1 + len(subs)
    fig, axes = plt.subplots(n_strips + 4, 1, figsize=(24, 2.2 * (n_strips + 4)), sharex=True,
                             gridspec_kw={"height_ratios": [0.6] * n_strips + [1.2, 1.2, 1.2, 1.0]})
    label_strip(axes[0], x, gt[a:b], "GT")
    for i, (name, labels) in enumerate(subs, start=1):
        label_strip(axes[i], x, labels[a:b], name)

    ax = axes[n_strips]
    ax.plot(x, ev.dead[a:b], lw=0.7, color="0.3")
    ax.axhline(0, ls=":", color="k")
    ax.set_ylabel("dead\nevidence", rotation=0, ha="right", va="center")

    ax = axes[n_strips + 1]
    ax.plot(x, ev.team[a:b], lw=0.7, color="purple")
    ax.axhline(0, ls=":", color="k")
    ax.set_ylabel("home-vs-away\nevidence", rotation=0, ha="right", va="center")

    ax = axes[n_strips + 2]
    ax.plot(x, np.clip(sig.d_home[a:b], 0, 15), lw=0.7, color=COLOURS["H"])
    ax.plot(x, np.clip(sig.d_away[a:b], 0, 15), lw=0.7, color=COLOURS["A"])
    ax.plot(x, np.clip(sig.ball_speed[a:b], 0, 30) / 2, lw=0.6, color="green", alpha=0.6)
    ax.set_ylim(0, 15)
    ax.set_ylabel("nearest H/A (m)\nball speed/2", rotation=0, ha="right", va="center")

    ax = axes[n_strips + 3]
    ax.plot(x, sig.activity[a:b], lw=0.8, color="tab:orange")
    ax.fill_between(x, 0, 0.5, where=~sig.ball_visible[a:b], color="k", step="mid", linewidth=0)
    ax.set_ylabel("activity m/s\n(black: no ball)", rotation=0, ha="right", va="center")

    plt.tight_layout()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    plt.savefig(args.out, dpi=50)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
