"""Supervised state and boundary models with explicit temporal holdouts."""

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from threadpoolctl import threadpool_limits

LABELS = np.array(["H", "A", "D", "X"])


def near(mask, radius):
    """True within radius frames of a True entry, in O(T)."""
    if radius < 0:
        raise ValueError("radius must be nonnegative")
    mask = np.asarray(mask, dtype=bool)
    total = np.r_[0, np.cumsum(mask)]
    t = np.arange(len(mask))
    return total[np.minimum(len(mask), t + radius + 1)] > total[np.maximum(0, t - radius)]


def training_mask(cache, train, validation, purge_frames):
    train, validation = np.asarray(train), np.asarray(validation)
    if train.shape != (len(cache.x),) or validation.shape != train.shape:
        raise ValueError("labels must cover every cached video frame")
    if not np.isin(train, range(4)).all() or not np.isin(validation, range(4)).all():
        raise ValueError("invalid label codes")
    graded = cache.graded_mask()
    valid = (validation != 3) & graded
    selected = (train != 3) & graded
    if (selected & valid).any():
        raise ValueError("training and validation labels overlap; use separate labelled windows")
    if not valid.any():
        raise ValueError("validation labels contain no graded frames")
    selected &= ~near(valid, purge_frames)
    if not selected.any():
        raise ValueError("no training labels remain after temporal purging")
    if set(train[selected].tolist()) != {0, 1, 2}:
        raise ValueError("training must contain H, A and D after temporal purging")
    return selected, valid


def boundary_targets(codes, radius):
    """A small target band around known label changes; unknown edges are excluded."""
    known = codes != 3
    changes = np.zeros(len(codes), dtype=bool)
    changes[1:] = known[1:] & known[:-1] & (codes[1:] != codes[:-1])
    target = near(changes, radius).astype(np.int8)
    usable = known & ~near(~known, radius + 1)
    return target, usable


def fit_durations(codes, fps, halves):
    """Weak log-duration preferences from complete training segments only.

    First/last segments of each labelled window and all X gaps are censored and
    discarded. Fewer than three complete examples disables a class's preference.
    """
    samples = [[], [], []]
    for _, first, last in halves:
        w = codes[first:last + 1]
        edges = np.r_[0, np.flatnonzero(w[1:] != w[:-1]) + 1, len(w)]
        for a, b in zip(edges[:-1], edges[1:]):
            c = int(w[a])
            if c == 3 or a == 0 or b == len(w) or w[a - 1] == 3 or w[b] == 3:
                continue
            samples[c].append((b - a) / fps)
    distributions = []
    for values in samples:
        logged = np.log(values)
        distributions.append((float(logged.mean()), max(float(logged.std()), 0.75))
                             if len(values) >= 3 else None)
    return distributions, [len(v) for v in samples]


def fit(cache, train, validation, purge_seconds=10, iterations=100, seed=0, threads=4):
    if purge_seconds < 3:
        raise ValueError("purge must be at least 3 seconds for the centred feature context")
    if iterations < 1 or threads < 1:
        raise ValueError("iterations and threads must be positive")
    selected, valid = training_mask(cache, train, validation, round(purge_seconds * cache.fps))
    codes = np.where(selected, train, 3)
    boundary_y, boundary_selected = boundary_targets(codes, max(1, round(cache.fps * 0.12)))
    if set(boundary_y[boundary_selected].tolist()) != {0, 1}:
        raise ValueError("training needs interior label transitions and non-transition frames")
    options = dict(max_iter=iterations, max_leaf_nodes=15, min_samples_leaf=20,
                   learning_rate=0.05, l2_regularization=1, max_bins=127,
                   early_stopping=False, random_state=seed)
    # No random internal validation split or inverse-frequency reweighting: preserve
    # temporal holdouts and the natural class prevalence.
    with threadpool_limits(limits=threads):
        state = HistGradientBoostingClassifier(**options).fit(cache.x[selected], train[selected])
        boundary = HistGradientBoostingClassifier(**options).fit(
            cache.x[boundary_selected], boundary_y[boundary_selected])
    durations, counts = fit_durations(codes, cache.fps, cache.halves)
    return dict(schema=1, state=state, boundary=boundary, names=cache.names, fps=cache.fps,
                durations=durations, metadata=dict(
                    seed=seed, iterations=iterations, purge_seconds=purge_seconds,
                    training_frames=int(selected.sum()), validation_frames=int(valid.sum()),
                    training_class_counts={str(LABELS[c]): int(np.sum(train[selected] == c))
                                           for c in range(3)},
                    boundary_positive_frames=int(boundary_y[boundary_selected].sum()),
                    boundary_training_frames=int(boundary_selected.sum()),
                    complete_duration_examples=counts,
                    probability_calibration="not fitted; validation is evaluation-only"))


def predict_scores(bundle, cache, threads=4):
    if (bundle.get("schema") != 1 or tuple(bundle["names"]) != cache.names
            or bundle["fps"] != cache.fps):
        raise ValueError("model and cache feature schema or fps disagree")
    probability = np.full((len(cache.x), 3), 1 / 3, dtype=np.float32)
    boundary = np.full(len(cache.x), 0.5, dtype=np.float32)
    selected = cache.graded_mask()
    with threadpool_limits(limits=threads):
        result = bundle["state"].predict_proba(cache.x[selected])
        probability[np.ix_(selected, bundle["state"].classes_.astype(int))] = result
        boundary[selected] = bundle["boundary"].predict_proba(cache.x[selected])[:, 1]
    return probability, boundary


def predict(bundle, cache, params=None, threads=4):
    from .segment import decode
    probability, boundary = predict_scores(bundle, cache, threads)
    labels = np.full(len(cache.x), "X", dtype="<U1")
    for _, first, last in cache.halves:
        path = decode(probability[first:last + 1], boundary[first:last + 1],
                      cache.fps, params, bundle["durations"])
        labels[first:last + 1] = LABELS[path]
    return labels
