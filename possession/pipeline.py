"""Data in, one label per video frame out."""

import numpy as np

from . import decode, evidence, signals
from .data import load_match


def snap_restarts(path, ball_speed, p):
    """Move each dead -> alive transition back to the kick that restarted play, if one happened
    within p.snap_window frames before it: the first frame whose ball speed exceeds
    p.snap_kick_speed after a second in which the ball was slower than p.snap_slow_speed or unseen.
    `path` is a 4-state path for one half; `ball_speed` the matching slice. Returns a new path."""
    if p.snap_window <= 0:
        return path
    out = path.copy()
    speed = ball_speed
    dead = path >= evidence.S_DH
    for t in np.flatnonzero(dead[:-1] & ~dead[1:]) + 1:
        for f in range(max(t - p.snap_window, 26), t):
            before = speed[f - 25:f]
            if (not np.isnan(speed[f]) and speed[f] > p.snap_kick_speed
                    and np.all(np.isnan(before) | (before < p.snap_slow_speed))):
                out[f:t] = path[t]
                break
    return out


def absorb_short_spells(path, k):
    """Relabel alive (H/A) runs shorter than k frames with the preceding run's state (the
    following one's if the run opens the half). Dead runs are left alone. Returns a new path."""
    if k <= 0:
        return path
    out = path.copy()
    alive = out < evidence.S_DH
    bounds = np.r_[0, np.flatnonzero(out[1:] != out[:-1]) + 1, len(out)]
    for i in range(len(bounds) - 1):
        a, b = bounds[i], bounds[i + 1]
        if b - a >= k or not alive[a]:
            continue
        if a > 0:
            out[a:b] = out[a - 1]
        elif b < len(out):
            out[a:b] = out[b]
    return out


def run(params=None, match=None, sig=None):
    """(labels, extras) for the whole video: 'H' / 'A' / 'D' on graded frames, 'X' elsewhere.

    Each half is decoded on its own, so no state is carried across halftime.
    `match` and `sig` can be passed in to reuse them across parameter variants.
    """
    params = params or evidence.Params()
    match = match or load_match()
    sig = sig or signals.compute(match)
    ev = evidence.compute(sig, params)
    scores = ev.emissions()
    costs = params.transition_costs()

    labels = np.full(match.n_frames, "X", dtype="<U1")
    states = np.full(match.n_frames, -1, dtype=np.int8)
    for _, first, last in match.halves:
        path = decode.viterbi_min_dead(scores[first:last + 1], costs, params.min_dead_frames)
        path = snap_restarts(path, sig.ball_speed[first:last + 1], params)
        path = absorb_short_spells(path, params.min_spell_frames)
        states[first:last + 1] = path
        labels[first:last + 1] = evidence.LABEL_OF_STATE[path]
    return labels, {"match": match, "signals": sig, "evidence": ev, "states": states}
