"""Label-free feature extraction and portable, non-pickle feature caches."""

import json
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

SCHEMA = 1


@dataclass
class FeatureCache:
    x: np.ndarray
    names: tuple
    fps: int
    halves: tuple
    provenance: dict

    def __post_init__(self):
        self.x = np.asarray(self.x, dtype=np.float32)
        self.names = tuple(self.names)
        self.halves = tuple(tuple(int(v) for v in block) for block in self.halves)
        if self.x.ndim != 2 or self.x.shape[1] != len(self.names):
            raise ValueError("feature matrix and names disagree")
        if not len(self.x) or len(set(self.names)) != len(self.names):
            raise ValueError("empty cache or duplicate feature names")
        if self.fps <= 0 or np.isinf(self.x).any():
            raise ValueError("invalid fps or infinite features; missing values must be NaN")
        last_stop = -1
        for _, first, last in self.halves:
            if not last_stop < first <= last < len(self.x):
                raise ValueError("halves must be ordered, disjoint, inclusive frame ranges")
            last_stop = last
        if not self.halves:
            raise ValueError("cache has no halves")

    def graded_mask(self):
        out = np.zeros(len(self.x), dtype=bool)
        for _, first, last in self.halves:
            out[first:last + 1] = True
        return out

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        meta = dict(schema=SCHEMA, fps=self.fps, halves=self.halves,
                    provenance=self.provenance)
        with path.open("wb") as handle:
            np.savez_compressed(handle, x=self.x, names=np.array(self.names),
                                metadata=np.array(json.dumps(meta, sort_keys=True)))

    @classmethod
    def load(cls, path):
        with np.load(path, allow_pickle=False) as stored:
            meta = json.loads(str(stored["metadata"]))
            if meta.get("schema") != SCHEMA:
                raise ValueError("unsupported feature schema; rebuild the cache")
            return cls(stored["x"], tuple(stored["names"].tolist()), meta["fps"],
                       tuple(meta["halves"]), meta["provenance"])


def build(match, sig, provenance=None):
    """Derive features from observations only. No ground-truth or prediction inputs.

    Centred windows deliberately use future observations: this is an offline task.
    Rolling features are confined to each half. NaNs and explicit visibility indicators
    are both retained; a missing ball is not converted to a dead-ball label.
    """
    n, fps = match.n_frames, match.fps
    length, width = match.pitch
    half = match.half_of()
    home_right = np.where(half == 1, match.home_attacks_right_first_half,
                          not match.home_attacks_right_first_half)
    direction = np.where(home_right, 1.0, -1.0)
    ball = match.ball.astype(np.float64)
    features = {}

    def add(name, values):
        values = np.asarray(values, dtype=np.float32)
        if values.shape != (n,):
            raise ValueError("wrong feature length: " + name)
        features[name] = values

    add("ball_visible", sig.ball_visible)
    add("ball_x_home_direction", (ball[:, 0] / length - 0.5) * direction)
    add("ball_y", ball[:, 1] / width - 0.5)
    add("distance_end_line", np.minimum(ball[:, 0], length - ball[:, 0]))
    add("distance_touchline", np.minimum(ball[:, 1], width - ball[:, 1]))
    for name in ("ball_speed", "activity", "d_home", "d_away", "whistle",
                 "tag_restart", "tag_phase", "ball_still", "ball_oob", "ball_placed",
                 "ball_placed_touchline", "pre_kickoff"):
        add(name, getattr(sig, name))
    add("blackout_seconds", sig.blackout_len / fps)
    add("distance_margin", sig.d_away - sig.d_home)
    add("camera_x_home_direction", (sig.camera_x / length - 0.5) * direction)

    velocity = np.full_like(ball, np.nan)
    for _, first, last in match.halves:
        if last - first >= 4:
            velocity[first + 2:last - 1] = (ball[first + 4:last + 1]
                                          - ball[first:last - 3]) * fps / 4
    add("ball_vx_home_direction", velocity[:, 0] * direction)
    add("ball_vy", velocity[:, 1])

    teams = match.team_by_frame()
    delta = match.players - ball[:, None, :]
    distances = np.hypot(delta[..., 0], delta[..., 1])
    outfield = np.ones(len(match.team), dtype=bool)
    outfield[[23, 45]] = False
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        for team, label in ((0, "home"), (1, "away")):
            belongs = teams == team
            visible = belongs & np.isfinite(match.players).all(axis=2)
            add(label + "_players_seen", visible.sum(axis=1))
            dist = np.where(belongs & np.isfinite(distances), distances, np.inf)
            nearest = np.sort(dist, axis=1)[:, :3]
            nearest[~np.isfinite(nearest)] = np.nan
            for k in range(3):
                add(label + "_nearest_" + str(k + 1), nearest[:, k])
            for radius in (2, 5):
                count = (dist < radius).sum(axis=1).astype(float)
                count[~sig.ball_visible] = np.nan
                add(label + "_within_" + str(radius) + "m", count)
            speed = np.where(belongs & outfield & np.isfinite(match.player_speed),
                             match.player_speed, np.nan)
            add(label + "_speed_mean", np.nanmean(speed, axis=1))
            add(label + "_speed_p75", np.nanpercentile(speed, 75, axis=1))
            seen = np.isfinite(speed).sum(axis=1)
            moving = np.divide(np.sum(speed > 2, axis=1), seen,
                               out=np.full(n, np.nan), where=seen > 0)
            add(label + "_moving_fraction", moving)
            for axis, axis_name in ((0, "x"), (1, "y")):
                pos = np.where(belongs & outfield, match.players[:, :, axis], np.nan)
                add(label + "_spread_" + axis_name, np.nanstd(pos, axis=1))

    temporal = ("ball_visible", "ball_speed", "activity", "distance_margin", "whistle",
                "home_speed_mean", "away_speed_mean")
    for name in temporal:
        for seconds in (1, 3, 5):
            mean, std = np.full(n, np.nan), np.full(n, np.nan)
            for _, first, last in match.halves:
                series = pd.Series(features[name][first:last + 1])
                rolling = series.rolling(max(1, seconds * fps), center=True, min_periods=1)
                mean[first:last + 1] = rolling.mean()
                std[first:last + 1] = rolling.std(ddof=0)
            add(f"{name}_mean_{seconds}s", mean)
            add(f"{name}_std_{seconds}s", std)

    return FeatureCache(np.column_stack(list(features.values())), tuple(features), fps,
                        tuple(match.halves), provenance or {})
