"""Per-frame physical signals derived from the tracking. Pure numpy/pandas, no labels involved.

Everything here describes what the tracking shows, in metres and seconds. Turning it into evidence
for one state or another is evidence.py's job, so the two can be inspected separately.
"""

import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .data import AWAY, HOME, Match, load_smart_tagging

# Params
GK_SLOTS = (23, 45)          # zero-based: slot 24 (Barcelona GK) and slot 46 (Athletic GK) hold
                             # stale positions while off-camera, so they are kept out of activity
BALL_SPEED_HALF_SPAN = 2     # ball speed from a centred difference over +/- 2 frames
STILL_HALF_SPAN = 12         # "ball not moving" judged over +/- 12 frames (~1 s)
ACTIVITY_WINDOW = 25         # outfield speed averaged over a centred 1 s window
OOB_MARGIN = 1.0             # metres beyond a line before the ball counts as out
# Restart spots, in metres (laws-of-the-game geometry plus ~1 m of tracking slack).
GOAL_AREA_DEPTH = 6.5        # 5.5 m goal area
GOAL_AREA_HALF_WIDTH = 10.2  # 9.16 m either side of the goal centre
CORNER_RADIUS = 2.5
TOUCHLINE_BAND = 1.5
CENTRE_RADIUS = 1.5
# Smart Tagging. Set-piece windows are 35 s and the restart was seen 3-13 s after the window
# opens, so the restarting team is asserted over the first 20 s only. Phase codes are kept only
# where the tagged team is the team in possession (the DEFENDING* codes are exact mirrors of
# these and add nothing), and only those that track possession rather than territory.
RESTART_TAG_SPAN = 20 * 25
RESTART_TYPES = ("corner kick", "goal kick", "free kick")
POSSESSION_CODES = ("BUILD UP", "PROGRESSION", "CREATING CHANCES", "LONG BALL",
                    "ATTACKING TRANSITION")
# The tagged "Kick Off (Start)" window opens 88 frames before the ball is played: measured on the
# first half, whose kick-off is the published first graded frame (11185 vs tag 11097). On the
# second half the same offset lands 6 frames from the published kick-off (113748 vs 113742).
KICKOFF_TAG_OFFSET = 88
# Team width (std of outfield y) is averaged over a centred 2 s window: the side in possession
# stretches the pitch, the side without it narrows.
WIDTH_WINDOW = 50
# Reception: the ball goes from > 8 m/s to < 3 m/s within ~0.5 s next to a player (< 1.5 m).
# 259 such events over the match; all 19 in the public window name the GT team 10 frames later.
RECEPTION_FAST = 8.0
RECEPTION_SLOW = 3.0
RECEPTION_NEAR = 1.5
RECEPTION_HALF_GAP = 6
# Ball contacts, after Link & Hoernig (2017, PLOS ONE) and Vidal-Codina et al. (2022, Sports Eng.):
# ball path smoothed with a Savitzky-Golay filter (7 frames, 2nd order); a contact is a local peak
# of |acceleration| >= 4 m/s^2 with a player within 1.5 m, credited to the nearest player. 1171
# contacts over the match (11 per minute of play); 90% of the 71 in the public window name the GT
# team, and "team of the last contact" agrees with the GT on 91% of the alive frames it covers,
# against 78% for nearest-player proximity.
SG_HALF = 3
CONTACT_ACCEL = 4.0
CONTACT_RADIUS = 1.5


@dataclass
class Signals:
    ball_visible: np.ndarray     # (T,) bool
    ball_speed: np.ndarray       # (T,) m/s, NaN where it cannot be measured
    ball_still: np.ndarray       # (T,) bool: visible and moved < 1 m over ~1 s
    ball_oob: np.ndarray         # (T,) bool: visible and beyond a line by OOB_MARGIN
    ball_placed: np.ndarray      # (T,) bool: ball still at a restart spot (goal area, corner,
                                 #   centre spot)
    ball_placed_touchline: np.ndarray  # (T,) bool: ball still on a touchline -- a throw-in being
                                 #   taken, but also a winger shielding the ball on the line
    blackout_len: np.ndarray     # (T,) length in frames of the ball blackout containing t, 0 if visible
    activity: np.ndarray         # (T,) mean outfield speed, m/s, 1 s centred mean
    d_home: np.ndarray           # (T,) metres from the ball to the nearest home player, NaN if no ball
    d_away: np.ndarray           # (T,) same for away
    camera_x: np.ndarray         # (T,) metres along the pitch under the image centre
    tag_restart: np.ndarray      # (T,) +1 home / -1 away restarts here per Smart Tagging, else 0
    tag_phase: np.ndarray        # (T,) net tagged possession phases, home minus away
    width_diff: np.ndarray       # (T,) metres: home outfield y-spread minus away's, 2 s mean
    pre_kickoff: np.ndarray      # (T,) bool: inside a half but before its kick-off
    reception: np.ndarray        # (T,) +1 / -1 on the frame a home / away player receives the ball
    contact: np.ndarray          # (T,) +1 / -1 on frames where a home / away player touches the ball
    whistle: np.ndarray          # (T,) referee-whistle tonal strength from the stadium audio
                                 #   (0 where no steady 3.6-4.1 kHz tone; all 0 without the audio)


def runs_of(mask):
    """(starts, stops) of True runs, stops exclusive."""
    edges = np.diff(np.concatenate(([0], mask.astype(np.int8), [0])))
    return np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)


def centred_mean(values, window, min_periods=1):
    return pd.Series(values).rolling(window, center=True, min_periods=min_periods).mean().values


def _shifted_distance(points, span):
    """|p[t+span] - p[t-span]| per frame, NaN at the edges or where either end is missing."""
    out = np.full(len(points), np.nan)
    delta = points[2 * span:] - points[:-2 * span]
    out[span:-span] = np.hypot(delta[:, 0], delta[:, 1])
    return out


def camera_centre_x(homography, pitch_length, image_size=(1920, 1080)):
    """Pitch x (metres) under the image centre, via the inverse of each frame's homography."""
    centre = np.array([image_size[0] / 2.0, image_size[1] / 2.0, 1.0])
    pitch = np.linalg.inv(homography) @ centre
    return pitch[:, 0] / pitch[:, 2] * pitch_length


def tagging_priors(tags, n_frames):
    """(restart, phase) per-frame arrays from the Smart Tagging table (see data.load_smart_tagging)."""
    restart = np.zeros(n_frames, dtype=np.float64)
    phase = np.zeros(n_frames, dtype=np.float64)
    for row in tags.itertuples():
        if row.team not in (HOME, AWAY):
            continue
        sign = 1.0 if row.team == HOME else -1.0
        if row.code == "SET PIECES" and row.type.lower() in RESTART_TYPES:
            restart[row.start_frame:row.start_frame + RESTART_TAG_SPAN] = sign
        elif row.code in POSSESSION_CODES:
            phase[row.start_frame:row.end_frame] += sign
    return restart, phase


def pre_kickoff_mask(tags, halves, n_frames):
    """Frames of each half before its kick-off, located from the "Kick Off (Start)" tags."""
    mask = np.zeros(n_frames, dtype=bool)
    starts = tags[(tags.code == "SET PIECES") & (tags.type.str.lower() == "kick off (start)")]
    for _, first, last in halves:
        inside = starts[(starts.start_frame >= first - 500) & (starts.start_frame <= last)]
        if len(inside):
            kickoff = int(inside.start_frame.min()) + KICKOFF_TAG_OFFSET
            mask[first:min(kickoff, last + 1)] = True
    return mask


def team_width_diff(match: Match):
    """Home minus away std of outfield players' y, centred mean over WIDTH_WINDOW."""
    outfield = np.ones(len(match.team), dtype=bool)
    outfield[list(GK_SLOTS)] = False
    y = match.players[:, :, 1].astype(np.float64)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        home = np.nanstd(np.where((match.team == HOME) & outfield, y, np.nan), axis=1)
        away = np.nanstd(np.where((match.team == AWAY) & outfield, y, np.nan), axis=1)
    return centred_mean(home - away, WIDTH_WINDOW)


def receptions(ball_speed, d_home, d_away):
    """+1 / -1 at the frame a fast ball is brought to rest next to a home / away player."""
    gap = RECEPTION_HALF_GAP
    speed = np.nan_to_num(ball_speed, nan=-1.0)
    before, after = speed[:-2 * gap], speed[2 * gap:]
    hit = np.zeros(len(speed), dtype=bool)
    hit[gap:-gap] = (before > RECEPTION_FAST) & (after >= 0) & (after < RECEPTION_SLOW)
    frames = np.flatnonzero(hit)
    if len(frames):
        frames = frames[np.concatenate(([True], np.diff(frames) > 2 * gap))] + gap
    out = np.zeros(len(speed), dtype=np.int8)
    dh = np.nan_to_num(d_home[frames], nan=np.inf)
    da = np.nan_to_num(d_away[frames], nan=np.inf)
    near = np.minimum(dh, da) < RECEPTION_NEAR
    out[frames[near]] = np.where(dh[near] < da[near], 1, -1)
    return out


def savgol(points, deriv):
    """Savitzky-Golay (2nd order, 2*SG_HALF+1 frames) smoothing or derivative of (T, 2) points,
    per visible run (NaN elsewhere and within SG_HALF of a run's ends), in units per frame^deriv."""
    import math
    offsets = np.arange(-SG_HALF, SG_HALF + 1)
    weights = np.linalg.pinv(np.vander(offsets, 3, increasing=True))[deriv] * math.factorial(deriv)
    out = np.full(points.shape, np.nan)
    for start, stop in zip(*runs_of(~np.isnan(points[:, 0]))):
        if stop - start < 2 * SG_HALF + 1:
            continue
        for axis in range(points.shape[1]):
            out[start + SG_HALF:stop - SG_HALF, axis] = np.convolve(
                points[start:stop, axis], weights[::-1], mode="valid")
    return out


def contacts(match: Match, teams):
    """+1 / -1 at frames where the smoothed ball acceleration peaks above CONTACT_ACCEL with a
    home / away player the nearest within CONTACT_RADIUS; 0 elsewhere. `teams` is (T, 46)."""
    accel = savgol(match.ball.astype(np.float64), 2) * match.fps ** 2
    magnitude = np.nan_to_num(np.hypot(accel[:, 0], accel[:, 1]), nan=0.0)
    peak = ((magnitude >= CONTACT_ACCEL) & (magnitude >= np.r_[0.0, magnitude[:-1]])
            & (magnitude >= np.r_[magnitude[1:], 0.0]))
    offset = match.players - match.ball[:, None, :]
    dist = np.hypot(offset[..., 0], offset[..., 1])
    dist = np.where(np.isnan(dist), np.inf, dist)
    nearest = np.argmin(dist, axis=1)
    close = dist[np.arange(len(dist)), nearest] < CONTACT_RADIUS
    out = np.zeros(len(magnitude), dtype=np.int8)
    hit = np.flatnonzero(peak & close)
    out[hit] = np.where(teams[hit, nearest[hit]] == HOME, 1, -1)
    return out


def whistle_track(n_frames):
    """Per-frame whistle strength from the cached audio spectrogram (built from the mp4 on first
    use), so a run is reproducible from the provided data alone."""
    from . import audio
    spec = audio.load_spectrogram(n_video_frames=n_frames)
    return audio.whistle_strength(spec)


def compute(match: Match, tags=None, keeper_fix=True) -> Signals:
    """All signals. `tags` is the Smart Tagging table; None loads it. `keeper_fix` gives the
    goalkeeper slots the team of the goal they guard (Match.team_by_frame): public macroF1
    0.886 -> 0.893, and all 5 full-match changes checked on video were right (keepers at
    184626, 119483, 133580, 19851 and a goal kick at 51408)."""
    ball = match.ball
    teams = match.team_by_frame(keeper_fix)
    length, width = match.pitch
    visible = ~np.isnan(ball[:, 0])

    ball_speed = _shifted_distance(ball, BALL_SPEED_HALF_SPAN) * match.fps / (2 * BALL_SPEED_HALF_SPAN)
    moved = _shifted_distance(ball, STILL_HALF_SPAN)
    ball_still = visible & (moved < 1.0)

    ball_oob = visible & ((ball[:, 0] < -OOB_MARGIN) | (ball[:, 0] > length + OOB_MARGIN)
                          | (ball[:, 1] < -OOB_MARGIN) | (ball[:, 1] > width + OOB_MARGIN))

    x, y = ball[:, 0], ball[:, 1]
    with np.errstate(invalid="ignore"):
        goal_area = (((x < GOAL_AREA_DEPTH) | (x > length - GOAL_AREA_DEPTH))
                     & (np.abs(y - width / 2) < GOAL_AREA_HALF_WIDTH))
        corner = (((x < CORNER_RADIUS) | (x > length - CORNER_RADIUS))
                  & ((y < CORNER_RADIUS) | (y > width - CORNER_RADIUS)))
        touchline = (y < TOUCHLINE_BAND) | (y > width - TOUCHLINE_BAND)
        centre = np.hypot(x - length / 2, y - width / 2) < CENTRE_RADIUS
    ball_placed = ball_still & (goal_area | corner | centre)
    ball_placed_touchline = ball_still & touchline & ~ball_placed

    blackout_len = np.zeros(len(ball), dtype=np.int32)
    for start, stop in zip(*runs_of(~visible)):
        blackout_len[start:stop] = stop - start

    speed = match.player_speed.astype(np.float64).copy()
    speed[:, list(GK_SLOTS)] = np.nan
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        per_frame = np.nanmean(speed, axis=1)
    activity = centred_mean(per_frame, ACTIVITY_WINDOW)

    offset = match.players - ball[:, None, :]
    dist = np.hypot(offset[..., 0], offset[..., 1])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        d_home = np.nanmin(np.where(teams == HOME, dist, np.nan), axis=1)
        d_away = np.nanmin(np.where(teams == AWAY, dist, np.nan), axis=1)

    if tags is None:
        tags = load_smart_tagging()
    tag_restart, tag_phase = tagging_priors(tags, len(ball))

    return Signals(
        ball_visible=visible,
        ball_speed=ball_speed,
        ball_still=ball_still,
        ball_oob=ball_oob,
        ball_placed=ball_placed,
        ball_placed_touchline=ball_placed_touchline,
        blackout_len=blackout_len,
        activity=activity,
        d_home=d_home,
        d_away=d_away,
        camera_x=camera_centre_x(match.homography, length),
        tag_restart=tag_restart,
        tag_phase=tag_phase,
        width_diff=team_width_diff(match),
        pre_kickoff=pre_kickoff_mask(tags, match.halves, len(ball)),
        reception=receptions(ball_speed, d_home, d_away),
        contact=contacts(match, teams),
        whistle=whistle_track(len(ball)),
    )
