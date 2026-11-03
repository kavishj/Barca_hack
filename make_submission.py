"""Regenerate a submission from the provided data.

    python make_submission.py                          # -> evaluation/results/possession_v6.csv
    python make_submission.py --name try --set switch=30 stop=20

Deterministic: same data and same parameters give a byte-identical file.
"""

import argparse
import dataclasses
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
from possession import evidence, pipeline  # noqa: E402

# Params
DEFAULT_NAME = "possession_v6"
RESULTS_DIR = os.path.join(ROOT, "evaluation", "results")


def parse_overrides(pairs):
    """['switch=30', 'w_oob=1.5'] -> Params with those fields replaced, types preserved."""
    params = evidence.Params()
    fields = {f.name: f.type for f in dataclasses.fields(params)}
    changes = {}
    for pair in pairs or []:
        key, _, value = pair.partition("=")
        if key not in fields:
            raise SystemExit("unknown parameter {!r}; have {}".format(key, sorted(fields)))
        cast = int if isinstance(getattr(params, key), int) else float
        changes[key] = cast(value)
    return dataclasses.replace(params, **changes)


def write_submission(path, labels):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="\n") as handle:
        handle.write("frame,label\n")
        handle.writelines("{},{}\n".format(i, label) for i, label in enumerate(labels))
    return path


def summarise(labels, halves, fps):
    """What the file claims: possessions, durations, dead-ball share, home share."""
    spells, dead = [], 0
    graded = 0
    for _, first, last in halves:
        window = labels[first:last + 1]
        graded += len(window)
        dead += int(np.sum(window == "D"))
        breaks = np.flatnonzero(window[1:] != window[:-1]) + 1
        bounds = np.concatenate(([0], breaks, [len(window)]))
        for a, b in zip(bounds[:-1], bounds[1:]):
            if window[a] in ("H", "A"):
                spells.append((window[a], b - a))
    lengths = np.array([n for _, n in spells]) / fps
    home = sum(n for team, n in spells if team == "H")
    alive = sum(n for _, n in spells)
    print("  possessions {:>4}  (H {}, A {})   mean {:.1f} s   median {:.1f} s   <2 s: {}".format(
        len(spells), sum(t == "H" for t, _ in spells), sum(t == "A" for t, _ in spells),
        lengths.mean(), np.median(lengths), int(np.sum(lengths < 2))))
    print("  dead ball {:.1f}% of graded frames   home share of alive {:.1f}%".format(
        100.0 * dead / graded, 100.0 * home / max(alive, 1)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--name", default=DEFAULT_NAME)
    parser.add_argument("--out", default=RESULTS_DIR)
    parser.add_argument("--set", nargs="*", metavar="KEY=VALUE", help="override evidence.Params")
    args = parser.parse_args()

    params = parse_overrides(args.set)
    labels, extras = pipeline.run(params)
    path = write_submission(os.path.join(args.out, args.name + ".csv"), labels)
    print("wrote", os.path.relpath(path, ROOT))
    summarise(labels, extras["match"].halves, extras["match"].fps)
