"""Per-frame scores for each hidden state, from the physical signals.

Four hidden states, two of which print the same label:

    H, A     a team has the ball and it is alive
    DH, DA   dead ball, and the restart will be taken by home / away

The ground truth only asks for D, but carrying the restarting team through the stoppage is what
lets the decoder hand the ball to the right side at the restart instead of guessing afresh.

Scores are log-likelihood ratios with hand-set weights, not fitted probabilities: consecutive
frames are strongly correlated, so per-frame naive-Bayes evidence would be wildly overconfident.
The weights act as a temperature, and the decoder's transition penalties are calibrated against
them -- the two are only meaningful together.
"""

from dataclasses import dataclass, field

import numpy as np

from .signals import Signals

# Params
STATES = ("H", "A", "DH", "DA")
S_H, S_A, S_DH, S_DA = range(4)
LABEL_OF_STATE = np.array(["H", "A", "D", "D"])


@dataclass
class Params:
    # -- alive / dead ---------------------------------------------------------------------------
    # Outfield speed (1 s mean) is log-normal around ~1.2 m/s in stoppages and ~2.3 m/s in play,
    # with a similar spread in both; read off the public window, where the medians are 1.19 (D)
    # and 2.11 / 2.50 (H / A).
    activity_dead: float = 1.2
    activity_alive: float = 2.3
    activity_log_sigma: float = 0.40
    w_activity: float = 0.35
    activity_clip: float = 3.0
    # The ball vanishes for long stretches mostly during stoppages; a short blackout is ordinary
    # occlusion and says nothing.
    blackout_min_frames: int = 50
    w_blackout: float = 0.5
    # Ball beyond a line: never seen while alive in the public window.
    w_oob: float = 2.0
    # Ball sitting still where restarts are taken: goal kick, corner, kick-off. The touchline is
    # weighted on its own: a still ball there is a throw-in OR a winger shielding it on the line
    # (frame 39100: an H possession split H-D-H by this rule).
    w_placed: float = 1.5
    w_placed_touchline: float = 1.5
    # ...so it only counts while the players are walking. The 49 touchline-still runs split
    # cleanly by mean activity: 17 below 1.4 m/s (throw-ins) and 19 above 1.8 m/s (play on
    # the line); all four public-window runs fall on the side their GT label says.
    touchline_max_activity: float = 1.4
    # A visible ball travelling fast is being played; dead balls are carried, rolled or thrown.
    fast_ball: float = 10.0
    w_fast_ball: float = 0.5

    # -- which team ---------------------------------------------------------------------------
    # Control likelihood of a team: Gaussian in its nearest player's distance to the ball.
    control_sigma: float = 1.5
    control_floor: float = 0.1
    # A ball in flight is not under anybody's control: weight the evidence down smoothly above
    # ~8 m/s (dribbling stays well below that; passes and shots are well above).
    flight_speed: float = 8.0
    flight_width: float = 1.5
    w_control: float = 0.6
    control_clip: float = 3.0
    # Duel zone (Vidal-Codina et al. 2022): when the nearest player of EACH team is within
    # duel_radius of the ball, the ball is contested and possession does not transfer -- the
    # challenge's own rule that a contested ball stays with the last team. Control evidence is
    # multiplied by duel_factor there. duel_radius 0 disables it.
    # 2.0 m: public spellF1 0.587 -> 0.600 (flat for 1.5..2.5); of the 3 longest full-match changes
    # checked on video, 2 right (one clear: a cleared Athletic corner kept by Athletic), 1 wrong.
    duel_radius: float = 2.0
    duel_factor: float = 0.0
    # A ball at somebody's feet and moving like it is being played is evidence the game is on,
    # whichever team it is: this fraction of the |team evidence| counts against "dead".
    control_alive: float = 0.3
    # Team width: the side in possession stretches the pitch. Home-minus-away y-spread sits at
    # +3.3 / -1.7 m (H / A) over the full match under v1's labels and +5.8 / -2.9 m on the public
    # GT; both pooled sds are ~3.5 m. Ball-independent, so it is the one team cue that survives
    # the ball blackouts.
    width_mid: float = 1.0
    width_sd: float = 3.5
    width_clip: float = 2.0
    # Shape reorganises seconds after a turnover, so where the ball is seen width only lags the
    # control evidence and fights real switches (public spellF1 0.57 -> 0.47-0.53 at any weight).
    # It is weighted separately while the ball is missing, where it is the only team cue left.
    w_width: float = 0.0
    w_width_blind: float = 0.0
    # Last touch: the team of the most recent ball contact (signals.contacts), carried for up to
    # touch_carry frames. Per frame it agrees with the public GT 91% of the time it is defined,
    # against 78% for the proximity evidence alone.
    w_touch: float = 0.0
    touch_carry: int = 125
    # Team-control chains: last touch, but counting only touches that show control (after Link &
    # Hoernig 2017, where team possession passes at the other team's next *controlled* action and
    # a deflection or ping-pong touch is not control). Two consecutive contacts by the same team,
    # both with no opponent within chain_free metres and at most chain_max_gap frames apart, put
    # that team in control between them; w_chain per frame there. The gate itself works: on the
    # public window uncontested contacts name the GT team 93% of the time against 64% for
    # contested ones, chained ones 37/37, and unlike raw last touch it adds no spells (217-224
    # over the match against 252). But it is redundant with the control evidence -- an
    # uncontested, kept ball is exactly where proximity already agrees -- so public scores are
    # identical at every weight and only 2 full-match stretches move (one plausibly right, one
    # plausibly wrong on video). Off; kept for re-testing against hand labels.
    w_chain: float = 0.0
    chain_free: float = 2.0
    chain_max_gap: int = 75
    # A reception (fast ball brought to rest at a player's feet) is a high-precision control
    # event: the receiving side gets w_reception per frame over the next reception_span frames.
    w_reception: float = 0.0
    reception_span: int = 15
    # Referee's whistle (stadium audio). The public GT starts D 5 and 14 frames after the two
    # whistle onsets it contains, while activity alone starts it 30-48 frames after. Not every
    # whistle stops play (advantage, crowd), so this is a push the live-play evidence can outvote,
    # not a snap: w_whistle per frame from onset + whistle_delay for whistle_span frames.
    whistle_threshold: float = 0.15   # 0.10-0.20 identical on public; 0.15 keeps the foul at
                                      # 33188 (checked on video) and not the 17951 false alarm
    whistle_delay: int = 8
    whistle_span: int = 50
    w_whistle: float = 1.0      # public macroF1 0.854 -> 0.863, flat for 1..3; count unchanged
    # Only a whistle that STOPS play pushes: outfield activity >= 1.8 m/s in the second before it
    # (stop whistles: 2.0-4.4; kick-off / restart whistles over a set field: 0.3-1.0) and a drop of
    # >= 0.4 m/s 1-3 s after it (play-on whistles, e.g. the video-checked advantage at 53168, show
    # none). Without this gate the push delayed every kick-off.
    whistle_live_activity: float = 1.8
    whistle_min_drop: float = 0.4
    # Before a half's kick-off the ball is dead, full stop (frames 110808..113741 per FAQ.md).
    force_pre_kickoff: bool = True
    # During a stoppage, whoever stands over the ball is the likely restarter.
    w_owner: float = 0.3
    # Smart Tagging: the tagged team takes this corner / goal kick / free kick (both public-window
    # set pieces right; raises restart-team agreement with the tags from 16/25 to 18/25 over the
    # match). The possession-phase nudge is off: every weight tried lowered the public score.
    w_tag_restart: float = 0.5
    # ...and the same tagged team as a prior on the live states over that window: the taker keeps
    # the ball until the other side gains control, and a clearance header is not control. In 5 of
    # v4's 7 wrong restart teams a corner went straight to the defenders.
    w_tag_restart_alive: float = 0.0
    w_tag_phase: float = 0.0

    # -- transitions (penalties, in the same units as the scores) ----------------------------
    switch: float = 10.0        # H <-> A; public spellF1 is flat for 8..15 while the full-match
                                # count runs ~267..218 -- 10 sits mid-plateau
    stop: float = 15.0          # alive -> dead
    restart: float = 15.0       # dead -> alive, same team as the dead state
    cross_restart: float = 40.0 # dead -> alive, the other team
    owner_flip: float = 6.0     # DH <-> DA while the ball is dead
    # A stoppage lasts at least this many frames (0/1 = no minimum). Real restarts take seconds
    # to set up -- the shortest public-GT stoppage is 5.5 s -- while v3 produced 7 stoppages under
    # 3 s, 5 of them splitting one team's possession in two. 5 s: public spellF1 0.538 -> 0.587
    # (two GT-confirmed fixes); of the full-match changes checked on video 4 were right, 1 wrong
    # (a genuine 2.6 s goal kick at 37321), 1 unclear.
    min_dead_frames: int = 125
    # Restart snapping (post-decode). A restart is the moment the ball is played: when the ball goes
    # from still-or-unseen to > snap_kick_speed within snap_window frames before a decoded restart,
    # the restart moves back to that kick. Over the match, v4 restarts sat either on the kick
    # (median 0) or ~1 s after it (upper quartile +24..+34 frames). 0 disables it. At 40 frames
    # it moves 21 of 109 restarts; both public restarts it touches end closer to the GT (+21 -> -8,
    # +14 -> -10) and two moved throw-ins checked on video were thrown on the snapped frame.
    snap_window: int = 40
    # Possession spells shorter than this many frames are absorbed (post-decode): a short spell
    # between two of the other team's merges into them; otherwise it takes the preceding label.
    # 0 disables it.
    min_spell_frames: int = 0
    snap_kick_speed: float = 5.0
    snap_slow_speed: float = 2.5

    def transition_costs(self):
        """(4, 4) penalty for moving from row state to column state."""
        cost = np.zeros((4, 4))
        cost[S_H, S_A] = cost[S_A, S_H] = self.switch
        cost[S_H, [S_DH, S_DA]] = cost[S_A, [S_DH, S_DA]] = self.stop
        cost[S_DH, S_H] = cost[S_DA, S_A] = self.restart
        cost[S_DH, S_A] = cost[S_DA, S_H] = self.cross_restart
        cost[S_DH, S_DA] = cost[S_DA, S_DH] = self.owner_flip
        return cost


@dataclass
class Evidence:
    dead: np.ndarray       # (T,) log-ratio dead vs alive
    team: np.ndarray       # (T,) log-ratio home vs away control, alive play
    owner: np.ndarray      # (T,) log-ratio home vs away standing over a dead ball
    params: Params = field(default_factory=Params)

    def emissions(self):
        """(T, 4) per-state scores; the decoder maximises their sum minus transition penalties.

        Team evidence only ever penalises the disfavoured side of a pair, so the better of H/A
        always scores 0 and the better of DH/DA always scores `dead`. That keeps "which team"
        from leaking into "alive or dead": with a symmetric +/-x/2 split, strong control
        evidence would quietly lift the alive states and any owner evidence the dead ones.
        """
        out = np.zeros((len(self.dead), 4))
        out[:, S_H] = np.minimum(self.team, 0.0)
        out[:, S_A] = -np.maximum(self.team, 0.0)
        out[:, S_DH] = self.dead + np.minimum(self.owner, 0.0)
        out[:, S_DA] = self.dead - np.maximum(self.owner, 0.0)
        return out


def _activity_llr(activity, p):
    """log p(activity | dead) - log p(activity | alive), equal-variance log-normals."""
    a = np.log(np.clip(np.nan_to_num(activity, nan=p.activity_alive), 0.05, None))
    mu_dead, mu_alive = np.log(p.activity_dead), np.log(p.activity_alive)
    llr = ((a - mu_alive) ** 2 - (a - mu_dead) ** 2) / (2 * p.activity_log_sigma ** 2)
    return np.clip(llr, -p.activity_clip, p.activity_clip)


def control_log_ratio(sig: Signals, p: Params):
    """(raw, in_hand): log control(home) - log control(away), 0 where the ball is not seen, and
    the 0..1 weight for the ball being playable rather than in flight."""
    def control(distance):
        """Gaussian control likelihood in the nearest player's distance; ~0 where unseen."""
        return np.exp(-np.nan_to_num(distance, nan=1e3) ** 2 / (2 * p.control_sigma ** 2))

    raw = np.log(control(sig.d_home) + p.control_floor) - np.log(control(sig.d_away) + p.control_floor)
    raw = np.where(sig.ball_visible, np.clip(raw, -p.control_clip, p.control_clip), 0.0)
    # Unknown ball speed (first/last frames of a visible run) gets half weight.
    speed = sig.ball_speed
    in_hand = np.where(np.isnan(speed), 0.5,
                       1.0 / (1.0 + np.exp((np.nan_to_num(speed) - p.flight_speed) / p.flight_width)))
    return raw, in_hand


def dead_features(sig: Signals, p: Params):
    """(activity_llr, blackout, oob, placed, placed_touchline, fast_ball) per frame, unweighted."""
    return (_activity_llr(sig.activity, p),
            (sig.blackout_len >= p.blackout_min_frames).astype(np.float64),
            sig.ball_oob.astype(np.float64),
            sig.ball_placed.astype(np.float64),
            (sig.ball_placed_touchline
             & (np.nan_to_num(sig.activity, nan=9.0) < p.touchline_max_activity)).astype(np.float64),
            (sig.ball_visible & (np.nan_to_num(sig.ball_speed) > p.fast_ball)).astype(np.float64))


def width_feature(sig: Signals, p: Params):
    """Team-width difference standardised around the H/A midpoint and clipped."""
    return np.clip((np.nan_to_num(sig.width_diff, nan=p.width_mid) - p.width_mid) / p.width_sd,
                   -p.width_clip, p.width_clip)


def last_touch(contact, carry):
    """(T,) +1 / -1 for the team of the latest contact at or before t, if within `carry` frames;
    0 before the first contact or once it is older than that."""
    idx = np.where(contact != 0, np.arange(len(contact)), -1)
    latest = np.maximum.accumulate(idx)
    valid = (latest >= 0) & (np.arange(len(contact)) - latest <= carry)
    return np.where(valid, contact[np.maximum(latest, 0)], 0).astype(np.float64)


def control_chains(sig: Signals, p: Params):
    """(T,) +1 / -1 from one contact to the next where both are by the same team, both have no
    opponent within p.chain_free metres, and they are at most p.chain_max_gap frames apart."""
    contact = sig.contact
    frames = np.flatnonzero(contact)
    opponent = np.where(contact[frames] > 0, sig.d_away[frames], sig.d_home[frames])
    free = np.nan_to_num(opponent, nan=np.inf) >= p.chain_free
    out = np.zeros(len(contact))
    for i in range(len(frames) - 1):
        f, g = frames[i], frames[i + 1]
        if free[i] and free[i + 1] and contact[f] == contact[g] and g - f <= p.chain_max_gap:
            out[f:g + 1] = contact[f]
    return out


def whistle_onsets(whistle, threshold, quiet=25):
    """Frames where the whistle strength first crosses `threshold` after `quiet` frames below it."""
    above = np.flatnonzero(whistle > threshold)
    if not len(above):
        return above
    return above[np.r_[True, np.diff(above) > quiet]]


def stopping_whistles(sig: Signals, p: Params):
    """Whistle onsets that interrupt live play: busy before, slowing after."""
    import warnings
    out = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)   # all-NaN windows simply fail the gate
        for onset in whistle_onsets(sig.whistle, p.whistle_threshold):
            before = np.nanmean(sig.activity[max(onset - 25, 0):onset]) if onset else np.nan
            after = np.nanmean(sig.activity[onset + 25:onset + 75])
            if before >= p.whistle_live_activity and after <= before - p.whistle_min_drop:
                out.append(int(onset))
    return out


def whistle_push(sig: Signals, p: Params):
    """(T,) 1 over [onset + delay, onset + delay + span) after each stopping whistle, else 0."""
    push = np.zeros(len(sig.whistle))
    for onset in stopping_whistles(sig, p):
        push[onset + p.whistle_delay:onset + p.whistle_delay + p.whistle_span] = 1.0
    return push


def compute(sig: Signals, p: Params = None) -> Evidence:
    p = p or Params()

    act, blackout, oob, placed, placed_touch, fast = dead_features(sig, p)
    dead = (p.w_activity * act + p.w_blackout * blackout + p.w_oob * oob + p.w_placed * placed
            + p.w_placed_touchline * placed_touch - p.w_fast_ball * fast)
    if p.w_whistle:
        dead += p.w_whistle * whistle_push(sig, p)

    raw, in_hand = control_log_ratio(sig, p)
    control = p.w_control * in_hand * raw
    dead -= p.control_alive * np.abs(control)
    if p.duel_radius > 0:
        duel = ((np.nan_to_num(sig.d_home, nan=np.inf) < p.duel_radius)
                & (np.nan_to_num(sig.d_away, nan=np.inf) < p.duel_radius))
        control = np.where(duel, p.duel_factor * control, control)
    if p.force_pre_kickoff:
        dead = np.where(sig.pre_kickoff, dead + 100.0, dead)
    width = width_feature(sig, p)
    w_shape = np.where(sig.ball_visible, p.w_width, p.w_width_blind)
    # Spread each reception forward over the frames that follow it (a causal box filter).
    received = np.convolve(sig.reception.astype(np.float64), np.ones(p.reception_span))[:len(sig.reception)]
    team = (control + w_shape * width + p.w_reception * np.clip(received, -1, 1)
            + p.w_tag_phase * np.clip(sig.tag_phase, -1, 1)
            + p.w_tag_restart_alive * sig.tag_restart)
    if p.w_touch:
        team = team + p.w_touch * last_touch(sig.contact, p.touch_carry)
    if p.w_chain:
        team = team + p.w_chain * control_chains(sig, p)
    owner = p.w_owner * raw + p.w_tag_restart * sig.tag_restart

    return Evidence(dead=dead, team=team, owner=owner, params=p)
