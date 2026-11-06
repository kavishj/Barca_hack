"""Boundary-aware segmental decoding with soft, optional duration preferences.

The dynamic programme is exact on its candidate-boundary graph, not on all frames
unless grid_frames=1. There is no hard minimum possession or stoppage duration.
"""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class DecoderParams:
    grid_frames: int = 25
    peak_threshold: float = 0.35
    peak_spacing: int = 5
    switch_cost: float = 1.0
    boundary_weight: float = 0.5
    duration_weight: float = 0.25
    max_candidates: int = 12000

    def __post_init__(self):
        if self.grid_frames < 1 or self.peak_spacing < 1 or self.max_candidates < 2:
            raise ValueError("candidate grid, spacing and limit must be positive")
        if not 0 < self.peak_threshold < 1:
            raise ValueError("peak threshold must be between zero and one")
        if min(self.switch_cost, self.boundary_weight, self.duration_weight) < 0:
            raise ValueError("decoder weights must be nonnegative")


def candidates(probability, boundary, p):
    n = len(probability)
    # State argmax changes retain precise short episodes even between grid ticks.
    changes = np.flatnonzero(np.diff(np.argmax(probability, axis=1))) + 1
    q = np.asarray(boundary)
    peaks = np.flatnonzero((q[1:-1] > q[:-2]) & (q[1:-1] >= q[2:])
                          & (q[1:-1] >= p.peak_threshold)) + 1
    selected, blocked = [], np.zeros(n, dtype=bool)
    for t in peaks[np.argsort(-q[peaks], kind="stable")]:
        if not blocked[t]:
            selected.append(t)
            blocked[max(0, t - p.peak_spacing + 1):t + p.peak_spacing] = True
    out = np.unique(np.r_[0, np.arange(p.grid_frames, n, p.grid_frames),
                          changes, selected, n]).astype(int)
    if len(out) > p.max_candidates:
        raise ValueError(f"{len(out)} boundary candidates exceed limit {p.max_candidates}; "
                         "inspect noisy state scores or explicitly raise --max-candidates")
    return out


def decode(probability, boundary, fps, p=None, durations=None, transition_costs=None):
    """Return H/A/D integer codes for one half from observation-derived scores.

    Each segment contributes summed log state probability / fps. Each internal
    change pays switch_cost plus a penalty of -boundary_weight * log(P(change)).
    Boundary evidence cannot reward adding arbitrary extra segments.
    A soft quadratic log-duration penalty is fitted on COMPLETE training segments;
    it is disabled for sparse classes and censored half-edge segments.
    """
    p = p or DecoderParams()
    probability = np.asarray(probability, dtype=float)
    boundary = np.asarray(boundary, dtype=float)
    if probability.ndim != 2 or probability.shape[1] != 3 or not len(probability):
        raise ValueError("state probabilities must be a nonempty (frames, 3) array")
    if boundary.shape != (len(probability),) or fps <= 0:
        raise ValueError("boundary length or fps is invalid")
    if (not np.isfinite(probability).all() or (probability < 0).any()
            or not np.allclose(probability.sum(axis=1), 1)
            or not np.isfinite(boundary).all() or ((boundary < 0) | (boundary > 1)).any()):
        raise ValueError("invalid state or boundary probabilities")
    if transition_costs is not None:
        transition_costs = np.asarray(transition_costs, dtype=float)
        if (transition_costs.shape != (3, 3) or not np.isfinite(transition_costs).all()
                or (transition_costs < 0).any() or np.any(np.diag(transition_costs) != 0)):
            raise ValueError("transition costs must be a finite nonnegative 3x3 matrix with zero diagonal")
    edge = candidates(probability, boundary, p)
    prefix = np.vstack([np.zeros(3), np.cumsum(np.log(np.clip(probability, 1e-6, 1)), axis=0)]) / fps
    q = np.clip(boundary, 1e-4, 1 - 1e-4)
    boundary_score = np.log(q) * p.boundary_weight
    m = len(edge)
    best = np.full((m, 3), -np.inf)
    previous_edge = np.full((m, 3), -1, dtype=int)
    previous_state = np.full((m, 3), -1, dtype=np.int8)
    for j in range(1, m):
        starts = edge[:j]
        seconds = (edge[j] - starts) / fps
        for c in range(3):
            others = np.array([s for s in range(3) if s != c])
            prev = best[:j, others]
            if transition_costs is not None:
                prev = prev - transition_costs[others, c]
            src = np.argmax(prev, axis=1)
            values = prev[np.arange(j), src]
            if transition_costs is None:
                values = values - p.switch_cost
            values[1:] += boundary_score[starts[1:]]
            values[0] = 0.0  # no transition before the first frame of a half
            values += prefix[edge[j], c] - prefix[starts, c]
            if durations and durations[c] is not None and j != m - 1:
                mu, sigma = durations[c]
                if not np.isfinite([mu, sigma]).all() or sigma <= 0:
                    raise ValueError("invalid duration distribution")
                penalty = -0.5 * ((np.log(seconds) - mu) / sigma) ** 2
                penalty[0] = 0.0  # left-censored segment
                values += p.duration_weight * penalty
            i = int(np.argmax(values))
            best[j, c] = values[i]
            previous_edge[j, c] = i
            previous_state[j, c] = others[src[i]] if i else -1
    c = int(np.argmax(best[-1]))
    path = np.empty(len(probability), dtype=np.int8)
    j = m - 1
    while j:
        i = previous_edge[j, c]
        path[edge[i]:edge[j]] = c
        c, j = int(previous_state[j, c]), i
    return path
