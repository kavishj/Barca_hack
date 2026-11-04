"""Run parameter variants against a ground truth, sharing one data load.

    python tools/sweep.py --grid switch=6,10,15,25
    python tools/sweep.py --grid switch=8,12 w_control=0.4,0.8 --gt my_hand_labels.csv

Prints, for each variant, the scorer's numbers on the given ground truth AND what the variant
claims about the full match (possessions, mean length, dead-ball share). Read both: the public
window has 17 possessions, so a choice should never rest on its spellF1 alone.
"""

import argparse
import dataclasses
import itertools
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "evaluation"))
import possession_metrics as pm  # noqa: E402
from possession import evidence, pipeline, signals  # noqa: E402
from possession.data import load_match, load_smart_tagging  # noqa: E402
from proxy import check as proxy_check  # noqa: E402

PUBLIC = os.path.join(ROOT, "evaluation", "gt", "BAR-ATH_possession_gt_public.csv")


def full_match_stats(labels, halves, fps):
    n_spells, lengths, dead, graded = 0, [], 0, 0
    for _, first, last in halves:
        window = labels[first:last + 1]
        graded += len(window)
        dead += int(np.sum(window == "D"))
        bounds = np.concatenate(([0], np.flatnonzero(window[1:] != window[:-1]) + 1, [len(window)]))
        for a, b in zip(bounds[:-1], bounds[1:]):
            if window[a] in ("H", "A"):
                n_spells += 1
                lengths.append((b - a) / fps)
    return n_spells, float(np.mean(lengths)), 100.0 * dead / graded


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--grid", nargs="+", required=True, metavar="KEY=V1,V2,...")
    parser.add_argument("--gt", default=PUBLIC)
    args = parser.parse_args()

    base = evidence.Params()
    axes = []
    for item in args.grid:
        key, _, values = item.partition("=")
        cast = int if isinstance(getattr(base, key), int) else float
        axes.append((key, [cast(v) for v in values.split(",")]))

    match = load_match()
    tags = load_smart_tagging()
    sig = signals.compute(match, tags)
    gt = pm.label_codes(pm.read_label_csv(args.gt))

    keys = [k for k, _ in axes]
    print("  ".join("{:>9}".format(k[:9]) for k in keys)
          + " | macroF1 spellF1 poss | match: poss  mean_s  dead% | prekick kickoffs restarts")
    for combo in itertools.product(*[v for _, v in axes]):
        params = dataclasses.replace(base, **dict(zip(keys, combo)))
        labels, _ = pipeline.run(params, match=match, sig=sig)
        result = pm.evaluate(gt, pm.label_codes(labels))
        n, mean_s, dead = full_match_stats(labels, match.halves, match.fps)
        proxy = proxy_check(labels, tags)
        print("  ".join("{:>9g}".format(v) for v in combo)
              + " |  {:.4f}  {:.4f}  {:>3} |       {:>4}  {:>6.1f}  {:>5.1f} |  {:>5.1f}%   {:>5}    {:>5}".format(
                  result["macro_f1"], result["spell_f1_mean"], result["n_pred_spells"],
                  n, mean_s, dead, proxy["prekick D%"], proxy["kickoffs ok"],
                  proxy["restarts ok"]))


if __name__ == "__main__":
    main()
