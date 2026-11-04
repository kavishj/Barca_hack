"""Load every tactical asset as numpy arrays on the video-frame index.

Parsing the 196 MB raw ATD takes a few seconds; the result is cached under data/cache/ (gitignored
with the rest of data/) and reused while the source file is unchanged.
"""

import io
import os
import sys
import xml.etree.ElementTree as ET
from dataclasses import dataclass

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from games import DATA_DIR, GAME, GAMES, half_video_blocks, require  # noqa: E402

# Params
N_SLOTS = 46
HOME, AWAY = 0, 1
CACHE_DIR = os.path.join(DATA_DIR, "cache")
PUBLIC_GT = os.path.join(ROOT, "evaluation", "gt", "BAR-ATH_possession_gt_public.csv")
TEAM_NAMES = {"barcelona": HOME, "atletic club": AWAY, "athletic club": AWAY}
# The two goalkeeper slots (24 and 46, zero-based 23 and 45). Their metadata team is NOT reliable:
# slot 46 ("Athletic") sits inside Barcelona's penalty area on 15,945 graded frames -- 11% of its
# life, including the last 230 s of the match -- i.e. the tracker's keeper slot follows whichever
# keeper it re-acquires. A keeper belongs to the team whose goal it is guarding, so these slots
# take their team per frame from the half of the pitch they stand in.
GK_SLOTS = (23, 45)


@dataclass
class Match:
    """One match on the tactical video-frame index. Row i of every array is video frame i."""
    players: np.ndarray       # (T, 46, 2) metres, NaN where the slot is not tracked
    player_speed: np.ndarray  # (T, 46) m/s as delivered, NaN where missing or held
    team: np.ndarray          # (46,) HOME / AWAY per track slot
    ball: np.ndarray          # (T, 2) metres, NaN where the ball is not tracked
    homography: np.ndarray    # (T, 3, 3) normalised pitch -> image pixels
    pitch: tuple              # (length, width) in metres
    fps: int
    halves: list              # [(half, first_frame, last_frame_inclusive)]
    home_attacks_right_first_half: bool

    @property
    def n_frames(self):
        return len(self.ball)

    def graded_mask(self):
        mask = np.zeros(self.n_frames, dtype=bool)
        for _, first, last in self.halves:
            mask[first:last + 1] = True
        return mask

    def half_of(self):
        """(T,) half number per frame, 0 outside the halves."""
        out = np.zeros(self.n_frames, dtype=np.int8)
        for half, first, last in self.halves:
            out[first:last + 1] = half
        return out

    def team_by_frame(self, keeper_fix=True):
        """(T, 46) team per slot per frame: the metadata team, except that a goalkeeper slot is
        given the team defending the goal on its side of the halfway line (see GK_SLOTS)."""
        teams = np.repeat(self.team[None, :], self.n_frames, axis=0)
        if not keeper_fix:
            return teams
        length = self.pitch[0]
        half = self.half_of()
        # Home defends x = 0 in the half where it attacks towards x = length, and vice versa.
        home_defends_low = np.where(half == 1, self.home_attacks_right_first_half,
                                    not self.home_attacks_right_first_half)
        for slot in GK_SLOTS:
            x = self.players[:, slot, 0]
            seen = ~np.isnan(x) & (half > 0)
            in_low_half = x < length / 2
            owner = np.where(in_low_half == home_defends_low, HOME, AWAY).astype(np.int8)
            teams[seen, slot] = owner[seen]
        return teams


def _parse_atd(raw_path):
    """Raw ATD -> (players (T,46,3), ball (T,3)) in normalised units, frame_id checked."""
    with open(raw_path) as handle:
        text = handle.read()
    # "<frame>:<x,y,s;...>:<x,y,s>" becomes one flat comma-separated row of 1 + 46*3 + 3 values.
    text = text.replace(":", ",").replace(";", ",")
    values = pd.read_csv(io.StringIO(text), header=None, dtype=np.float32).values
    frame_id = values[:, 0].astype(np.int64)
    if not np.array_equal(frame_id, np.arange(1, len(frame_id) + 1)):
        raise ValueError("{}: frame ids are not 1..N".format(raw_path))
    players = values[:, 1:1 + N_SLOTS * 3].reshape(-1, N_SLOTS, 3)
    ball = values[:, 1 + N_SLOTS * 3:4 + N_SLOTS * 3]
    return players, ball


def _parse_meta(meta_path):
    """(team per slot, home attacks towards x=1 in the first half)."""
    root = ET.parse(meta_path).getroot()
    teams = list(root.iter("Team"))
    home_id = teams[0].get("id")
    team = np.array([HOME if p.get("teamId") == home_id else AWAY for p in root.iter("Player")],
                    dtype=np.int8)
    direction = None
    for param in teams[0].iter("ProviderParameter"):
        if param.findtext("Name") == "attack_direction_first_half":
            direction = param.findtext("Value")
    return team, direction != "right_to_left"


def _load_atd_cached(game, raw_path):
    os.makedirs(CACHE_DIR, exist_ok=True)
    cache = os.path.join(CACHE_DIR, "{}_tactical_atd.npz".format(game))
    stamp = np.array([os.path.getsize(raw_path), int(os.path.getmtime(raw_path))])
    if os.path.isfile(cache):
        stored = np.load(cache)
        if np.array_equal(stored["stamp"], stamp):
            return stored["players"], stored["ball"]
    players, ball = _parse_atd(raw_path)
    np.savez(cache, players=players, ball=ball, stamp=stamp)
    return players, ball


def load_homography(game=GAME):
    path = require(["homography"], game)[0]
    table = pd.read_csv(path)
    cols = ["h00", "h01", "h02", "h10", "h11", "h12", "h20", "h21", "h22"]
    return table[cols].values.reshape(-1, 3, 3)


def load_match(game=GAME):
    meta_path, raw_path = require(["atd_meta", "atd_raw"], game)
    players_raw, ball_raw = _load_atd_cached(game, raw_path)
    team, home_right = _parse_meta(meta_path)
    length, width = GAMES[game]["pitch"]
    scale = np.array([length, width], dtype=np.float32)
    return Match(
        players=players_raw[:, :, :2] * scale,
        player_speed=players_raw[:, :, 2],
        team=team,
        ball=ball_raw[:, :2] * scale,
        homography=load_homography(game),
        pitch=(length, width),
        fps=GAMES[game]["fps"],
        halves=half_video_blocks(game),
        home_attacks_right_first_half=home_right,
    )


def load_smart_tagging(game=GAME):
    """DataFrame: code, start_frame, end_frame (exclusive), team (HOME/AWAY/-1), type, half."""
    path = require(["smart_tagging"], game)[0]
    fps = GAMES[game]["fps"]
    rows = []
    for inst in ET.parse(path).iter("instance"):
        tags = {(lb.findtext("group") or "").strip().lower(): (lb.findtext("text") or "").strip()
                for lb in inst.findall("label")}
        rows.append({
            "code": (inst.findtext("code") or "").strip().upper(),
            "start_frame": int(round(float(inst.findtext("start")) * fps)),
            "end_frame": int(round(float(inst.findtext("end")) * fps)),
            "team": TEAM_NAMES.get(tags.get("team", "").lower(), -1),
            "type": tags.get("type", ""),
            "half": tags.get("half", ""),
        })
    return pd.DataFrame(rows).sort_values("start_frame").reset_index(drop=True)


def load_public_gt():
    """(T,) array of 'H' / 'A' / 'D' / 'X'."""
    return pd.read_csv(PUBLIC_GT, dtype={"label": "object"})["label"].str.strip().str.upper().values
