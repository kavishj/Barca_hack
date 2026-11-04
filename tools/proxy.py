"""Label-free sanity checks of a submission over the FULL match.

    python tools/proxy.py evaluation/results/possession_v1.csv [more.csv ...]

None of these is ground truth, so read them as smoke alarms, not scores:

  prekick   frames 110808..113741 (graded, before the 2nd-half kick-off) are all D -- published
            in FAQ.md, so this one IS a fact
  kickoff   each kick-off (2 half starts + 4 after goals) is preceded by D and handed to the
            right side: Athletic kicked off the 1st half (public GT), so Barcelona the 2nd; after a
            goal the side that conceded restarts (GOALS tag team = scorer, checked against the
            tagged kick-off sequence)
  goals     share of each tagged GOALS window that is D (celebration through to the kick-off)
  restarts  tagged corner / goal kick / free kick: first D->alive within 20 s of the window
            opening goes to the tagged team (NB: v1 uses these tags as a prior, so this is not
            independent of it)
  count     possessions and mean length -- the public window's rate (17 per 4.0 min alive)
            suggests ~210-270 over the match
"""

import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from possession.data import load_smart_tagging  # noqa: E402
from games import half_video_blocks  # noqa: E402

PREKICK = (110808, 113742)
KICKOFF_2ND = 113742


def first_restart(labels, start, stop):
    for t in range(max(start, 1), stop):
        if labels[t - 1] == "D" and labels[t] in ("H", "A"):
            return t
    return None


def check(labels, tags):
    out = {}
    pre = labels[PREKICK[0]:PREKICK[1]]
    out["prekick D%"] = 100.0 * np.mean(pre == "D")

    goals = tags[tags.code == "GOALS"].sort_values("start_frame")
    kick_after = tags[(tags.code == "SET PIECES") & (tags.type == "Kick Off (After Goal)")]
    expected = [(11185, "A"), (KICKOFF_2ND, "H")]
    for g in goals.itertuples():
        ko = kick_after[(kick_after.start_frame >= g.start_frame - 250)
                        & (kick_after.start_frame <= g.end_frame + 250)]
        if len(ko):
            expected.append((int(ko.start_frame.iloc[0]) + 88, "A" if g.team == 0 else "H"))
    ok = 0
    for frame, team in expected:
        t = first_restart(labels, frame - 250, frame + 500) if frame != 11185 else 11185
        ok += int(t is not None and labels[t] == team)
    out["kickoffs ok"] = "{}/{}".format(ok, len(expected))

    shares = [np.mean(labels[g.start_frame:g.end_frame] == "D") for g in goals.itertuples()]
    out["goals D%"] = 100.0 * float(np.mean(shares))

    restarts = tags[(tags.code == "SET PIECES") & (tags.team >= 0)
                    & tags.type.str.lower().isin(["corner kick", "goal kick", "free kick"])]
    agree = n = 0
    for r in restarts.itertuples():
        t = first_restart(labels, r.start_frame, r.start_frame + 20 * 25)
        if t is None:
            continue
        n += 1
        agree += int(labels[t] == ("H" if r.team == 0 else "A"))
    out["restarts ok"] = "{}/{}".format(agree, n)

    n_spells, lengths = 0, []
    for _, first, last in half_video_blocks("BAR-ATH"):
        w = labels[first:last + 1]
        bounds = np.concatenate(([0], np.flatnonzero(w[1:] != w[:-1]) + 1, [len(w)]))
        for a, b in zip(bounds[:-1], bounds[1:]):
            if w[a] in ("H", "A"):
                n_spells += 1
                lengths.append((b - a) / 25)
    out["poss"] = n_spells
    out["mean s"] = round(float(np.mean(lengths)), 1)
    return out


def main():
    tags = load_smart_tagging()
    rows = []
    for path in sys.argv[1:]:
        labels = pd.read_csv(path, dtype={"label": "object"})["label"].str.upper().values
        row = check(labels, tags)
        row["submission"] = os.path.splitext(os.path.basename(path))[0]
        rows.append(row)
    table = pd.DataFrame(rows).set_index("submission")
    print(table.to_string(float_format=lambda v: "{:.1f}".format(v)))


if __name__ == "__main__":
    main()
