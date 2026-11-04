"""Possession-by-possession error analysis of a submission against a ground truth.

    python tools/spells.py evaluation/results/possession_v1.csv
    python tools/spells.py mine.csv --gt my_hand_labels.csv

For every true possession: the best same-team predicted spell, its IoU, and how far each end is
off (negative = predicted early). Then every predicted spell that matched nothing. Matching is the
scorer's own (possession_metrics), at IoU 0.5.
"""

import argparse
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "evaluation"))
import possession_metrics as pm  # noqa: E402

PUBLIC = os.path.join(ROOT, "evaluation", "gt", "BAR-ATH_possession_gt_public.csv")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("submission")
    parser.add_argument("--gt", default=PUBLIC)
    parser.add_argument("--iou", type=float, default=0.5)
    args = parser.parse_args()

    gt = pm.label_codes(pm.read_label_csv(args.gt))
    pred = pm.label_codes(pm.read_label_csv(args.submission))
    runs = pm.valid_runs(gt)
    filled = pm.fill_abstentions(pred, runs)
    gt_spells = pm.possession_spells(gt, runs)
    pred_spells = pm.possession_spells(filled, runs)
    overlaps = pm.spell_overlaps(gt_spells, pred_spells)
    matched_gt = pm.matched_gt_flags(overlaps, len(gt_spells), args.iou)

    best = {}
    for iou, i, j in overlaps:
        if i not in best:
            best[i] = (iou, j)
    used_pred = {best[i][1] for i in range(len(gt_spells)) if matched_gt[i] and i in best}

    print("true possessions: {}   predicted (inside graded window): {}".format(
        len(gt_spells), len(pred_spells)))
    print("\n  start    stop  team   dur s | best IoU  start off  stop off  matched@{}".format(args.iou))
    for i, (a, b, k) in enumerate(gt_spells):
        team = pm.LABELS[k]
        if i in best:
            iou, j = best[i]
            pa, pb, _ = pred_spells[j]
            print("  {:>6} {:>6}  {}   {:>6.1f} |   {:.2f}    {:>+6}    {:>+6}    {}".format(
                a, b, team, (b - a) / pm.FPS, iou, pa - a, pb - b, "yes" if matched_gt[i] else "NO"))
        else:
            print("  {:>6} {:>6}  {}   {:>6.1f} |   ----  (no same-team overlap)        NO".format(
                a, b, team, (b - a) / pm.FPS))

    spurious = [s for j, s in enumerate(pred_spells) if j not in used_pred]
    print("\nunmatched predicted spells: {}".format(len(spurious)))
    for a, b, k in spurious:
        print("  {:>6} {:>6}  {}   {:>6.1f} s".format(a, b, pm.LABELS[k], (b - a) / pm.FPS))

    scores = pm.spell_scores(gt, filled, runs)
    print("\nspellF1 {:.4f}   per IoU: {}".format(scores["mean_f1"], "  ".join(
        "{}:{:.2f}".format(t, v["f1"]) for t, v in scores["per_iou"].items())))


if __name__ == "__main__":
    main()
