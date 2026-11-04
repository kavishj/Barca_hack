"""List the stretches where two submissions disagree, longest first -- what to look at on video.

    python tools/disagree.py a.csv b.csv [--min-s 2] [--top 20] [--seed 0 --sample 8]

Only graded frames count. --sample draws that many stretches at random (seeded) from those at
least --min-s long, so an adjudication sample is not just the longest, easiest cases.
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from games import half_video_blocks  # noqa: E402


def stretches(a, b):
    graded = np.zeros(len(a), dtype=bool)
    for _, first, last in half_video_blocks("BAR-ATH"):
        graded[first:last + 1] = True
    differ = graded & (a != b)
    edges = np.diff(np.concatenate(([0], differ.astype(np.int8), [0])))
    out = []
    for start, stop in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)):
        pair = sorted(set(zip(a[start:stop], b[start:stop])))
        out.append((start, stop, (stop - start) / 25.0, " ".join("{}/{}".format(x, y) for x, y in pair)))
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("a")
    parser.add_argument("b")
    parser.add_argument("--min-s", type=float, default=2.0)
    parser.add_argument("--top", type=int, default=20)
    parser.add_argument("--sample", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    read = lambda p: pd.read_csv(p, dtype={"label": "object"})["label"].str.upper().values
    a, b = read(args.a), read(args.b)
    rows = stretches(a, b)
    long_rows = [r for r in rows if r[2] >= args.min_s]
    print("{} disagreeing stretches, {:.0f} s in total; {} of them >= {} s ({:.0f} s)".format(
        len(rows), sum(r[2] for r in rows), len(long_rows), args.min_s, sum(r[2] for r in long_rows)))
    print("labels shown as {}/{}".format(os.path.basename(args.a), os.path.basename(args.b)))
    if args.sample:
        rng = np.random.RandomState(args.seed)
        pick = rng.choice(len(long_rows), size=min(args.sample, len(long_rows)), replace=False)
        chosen = [long_rows[i] for i in sorted(pick)]
    else:
        chosen = sorted(long_rows, key=lambda r: -r[2])[:args.top]
    for start, stop, secs, pairs in chosen:
        print("  {:>6} {:>6}  {:>5.1f} s   {}".format(start, stop, secs, pairs))


if __name__ == "__main__":
    main()
