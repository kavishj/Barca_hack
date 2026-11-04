"""Offline Viterbi decoding over the hidden states.

Offline on purpose: the whole half is known before any label is written, so a transition can be
placed where the evidence actually changes rather than where a causal filter finally notices it.
spellF1 charges a uniform 1 s lag ~35% of its value, so causal smoothing is not an option.
"""

import numpy as np


def viterbi(scores, costs):
    """Best state path maximising sum(scores[t, s_t]) - sum(costs[s_{t-1}, s_t]).

    scores: (T, K) per-frame state scores. costs: (K, K) non-negative transition penalties.
    Ties resolve to the lowest state index, so the result is deterministic.
    """
    n_frames, n_states = scores.shape
    back = np.empty((n_frames, n_states), dtype=np.int8)
    total = scores[0].astype(np.float64).copy()
    neg = -np.asarray(costs, dtype=np.float64)
    cols = np.arange(n_states)
    for t in range(1, n_frames):
        candidate = total[:, None] + neg
        best = np.argmax(candidate, axis=0)
        back[t] = best
        total = candidate[best, cols] + scores[t]
    path = np.empty(n_frames, dtype=np.int8)
    path[-1] = int(np.argmax(total))
    for t in range(n_frames - 1, 0, -1):
        path[t - 1] = back[t, path[t]]
    return path


def viterbi_min_dead(scores, costs, k):
    """viterbi() over the 4 states (H, A, DH, DA) with every dead stretch lasting >= k frames.

    Each dead state is expanded into a chain of k sub-states sharing its score: entering a
    stoppage starts at position 0, every frame advances one position, and only position k-1 may
    loop or restart play. The restarting side may still flip (DH <-> DA) while the chain matures.
    A stoppage already under way at the first frame starts mature. Returns the 4-state path.
    """
    if k <= 1:
        return viterbi(scores, costs)
    H, A, DH, DA = 0, 1, 2, 3
    c = np.asarray(costs, dtype=np.float64)
    flip = c[DH, DA]
    n_frames = len(scores)
    n_exp = 2 + 2 * k                       # H, A, DH_0..DH_k-1, DA_0..DA_k-1
    dh0, da0 = 2, 2 + k
    back = np.empty((n_frames, n_exp), dtype=np.int16)
    ninf = -np.inf

    alive = scores[0, :2].astype(np.float64).copy()
    dead_h = np.full(k, ninf)
    dead_a = np.full(k, ninf)
    dead_h[-1], dead_a[-1] = scores[0, DH], scores[0, DA]
    pos = np.arange(k)

    for t in range(1, n_frames):
        e = scores[t]
        b = back[t]
        # alive states: stay, switch sides, or restart from a mature stoppage
        cand_h = (alive[0], alive[1] - c[A, H], dead_h[-1] - c[DH, H], dead_a[-1] - c[DA, H])
        cand_a = (alive[1], alive[0] - c[H, A], dead_h[-1] - c[DH, A], dead_a[-1] - c[DA, A])
        src = (H, A, dh0 + k - 1, da0 + k - 1)
        ih, ia = int(np.argmax(cand_h)), int(np.argmax(cand_a))
        b[H] = src[ih]
        b[A] = (A, H, dh0 + k - 1, da0 + k - 1)[ia]
        new_alive = np.array([cand_h[ih] + e[H], cand_a[ia] + e[A]])

        # dead chains: position 0 is entered from play; position i from position i-1 (same side,
        # or the other side at the flip cost); the last position may also loop on itself
        new_h, new_a = np.empty(k), np.empty(k)
        enter_h = (alive[0] - c[H, DH], alive[1] - c[A, DH])
        enter_a = (alive[0] - c[H, DA], alive[1] - c[A, DA])
        jh, ja = int(np.argmax(enter_h)), int(np.argmax(enter_a))
        new_h[0], new_a[0] = enter_h[jh], enter_a[ja]
        b[dh0], b[da0] = (H, A)[jh], (H, A)[ja]

        same_h, cross_h = dead_h[:-1], dead_a[:-1] - flip      # into positions 1..k-1
        same_a, cross_a = dead_a[:-1], dead_h[:-1] - flip
        take_h = same_h >= cross_h
        take_a = same_a >= cross_a
        new_h[1:] = np.where(take_h, same_h, cross_h)
        new_a[1:] = np.where(take_a, same_a, cross_a)
        b[dh0 + 1:dh0 + k] = np.where(take_h, dh0 + pos[:-1], da0 + pos[:-1])
        b[da0 + 1:da0 + k] = np.where(take_a, da0 + pos[:-1], dh0 + pos[:-1])
        # the mature position can also come from itself (looping) or the other mature side
        loop_h = max(dead_h[-1], dead_a[-1] - flip)
        if loop_h > new_h[-1]:
            new_h[-1] = loop_h
            b[dh0 + k - 1] = dh0 + k - 1 if dead_h[-1] >= dead_a[-1] - flip else da0 + k - 1
        loop_a = max(dead_a[-1], dead_h[-1] - flip)
        if loop_a > new_a[-1]:
            new_a[-1] = loop_a
            b[da0 + k - 1] = da0 + k - 1 if dead_a[-1] >= dead_h[-1] - flip else dh0 + k - 1

        alive = new_alive
        dead_h = new_h + e[DH]
        dead_a = new_a + e[DA]

    final = np.concatenate((alive, dead_h, dead_a))
    state = int(np.argmax(final))
    expanded = np.empty(n_frames, dtype=np.int16)
    expanded[-1] = state
    for t in range(n_frames - 1, 0, -1):
        expanded[t - 1] = back[t, expanded[t]]
    path = np.where(expanded < 2, expanded, np.where(expanded < da0, DH, DA))
    return path.astype(np.int8)
