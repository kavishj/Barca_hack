"""Fit the evidence weights by logistic regression against a labelled ground truth.

    python tools/fit_emissions.py                         # the public 5 minutes
    python tools/fit_emissions.py --gt data/labels/merged.csv

Two per-frame models, on exactly the features evidence.py uses:

  dead   D vs alive, over every labelled frame
  team   H vs A, over the labelled alive frames

Each is scored by leave-one-segment-out cross-validation (a GT segment is never scored by a model
that saw it -- frames inside a segment are far from independent), next to the hand-set weights
in evidence.Params scored on the same frames with no fitting at all. If the hand weights hold
their own, the fit is telling you the shape is right; if the fit wins clearly out of sample,
adopt its weight RATIOS -- the absolute scale is a temperature the decoder's penalties are
calibrated against, so it stays a sweep parameter (tools/sweep.py), not a fitted one.

Needs scikit-learn (fitting only; the pipeline itself does not).
"""

import argparse
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "evaluation"))
import possession_metrics as pm  # noqa: E402
from possession import evidence, signals  # noqa: E402
from possession.data import load_match  # noqa: E402

PUBLIC = os.path.join(ROOT, "evaluation", "gt", "BAR-ATH_possession_gt_public.csv")


def design(sig, p):
    """Feature matrices for the two models, column names, and the hand-weighted scores."""
    act, blackout, oob, placed, placed_touch, fast = evidence.dead_features(sig, p)
    raw, in_hand = evidence.control_log_ratio(sig, p)
    control = in_hand * raw
    width = evidence.width_feature(sig, p)
    width_seen = np.where(sig.ball_visible, width, 0.0)
    width_blind = np.where(sig.ball_visible, 0.0, width)

    dead_x = np.c_[act, blackout, oob, placed, placed_touch, fast, np.abs(control)]
    dead_names = ["activity_llr", "blackout", "oob", "placed", "placed_touchline", "fast_ball",
                  "|control|"]
    dead_hand = (p.w_activity * act + p.w_blackout * blackout + p.w_oob * oob
                 + p.w_placed * placed + p.w_placed_touchline * placed_touch - p.w_fast_ball * fast
                 - p.control_alive * p.w_control * np.abs(control))
    team_x = np.c_[control, width_seen, width_blind]
    team_names = ["control (in-hand weighted)", "width (ball seen)", "width (ball missing)"]
    team_hand = p.w_control * control + p.w_width * width_seen + p.w_width_blind * width_blind
    return dead_x, dead_names, dead_hand, team_x, team_names, team_hand


def auc(score, label):
    order = np.argsort(score, kind="mergesort")
    ranks = np.empty(len(score))
    ranks[order] = np.arange(1, len(score) + 1)
    n1 = label.sum()
    n0 = len(label) - n1
    return (ranks[label == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def loso(x, y, groups, C):
    """Out-of-fold decision scores with one GT segment held out at a time."""
    from sklearn.linear_model import LogisticRegression
    out = np.zeros(len(y))
    for g in np.unique(groups):
        test = groups == g
        train = ~test
        if len(np.unique(y[train])) < 2:
            continue
        model = LogisticRegression(C=C, max_iter=1000).fit(x[train], y[train])
        out[test] = model.decision_function(x[test])
    return out


def report(name, x, y, groups, hand, names, C):
    from sklearn.linear_model import LogisticRegression
    full = LogisticRegression(C=C, max_iter=1000).fit(x, y)
    cv = loso(x, y, groups, C)
    print("\n{}: {} frames, {} segments, positive rate {:.2f}".format(
        name, len(y), len(np.unique(groups)), y.mean()))
    print("  AUC  hand weights {:.3f}   fitted, leave-one-segment-out {:.3f}".format(
        auc(hand, y), auc(cv, y)))
    print("  fitted coefficients (per frame, independent-frame scale):")
    for n, c in zip(names, full.coef_[0]):
        print("    {:<28} {:+.3f}".format(n, c))


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--gt", default=PUBLIC)
    parser.add_argument("--C", type=float, default=1.0, help="inverse L2 strength")
    args = parser.parse_args()

    gt = pm.label_codes(pm.read_label_csv(args.gt))
    sig = signals.compute(load_match())
    p = evidence.Params()
    dead_x, dead_names, dead_hand, team_x, team_names, team_hand = design(sig, p)

    runs = pm.valid_runs(gt)
    seg_id = np.full(len(gt), -1)
    for k, (a, b, _) in enumerate(pm.segments(gt, runs)):
        seg_id[a:b] = k

    graded = gt != pm.X
    report("dead vs alive", dead_x[graded], (gt[graded] == pm.D).astype(int), seg_id[graded],
           dead_hand[graded], dead_names, args.C)
    alive = (gt == pm.H) | (gt == pm.A)
    report("home vs away", team_x[alive], (gt[alive] == pm.H).astype(int), seg_id[alive],
           team_hand[alive], team_names, args.C)


if __name__ == "__main__":
    main()
