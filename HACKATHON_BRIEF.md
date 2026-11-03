# More than a Hack 2027 — Data brief and winning plan

Measured on the delivered bundle on 2026-10-20. Every number below comes from parsing the actual
files, not from the docs, unless marked *(docs)*.

---

## 1. The task in one paragraph

For each of the **199,319 frames** of the tactical video, output `H` (Barcelona has the ball, alive),
`A` (Athletic has it, alive), `D` (dead ball) or `X` (abstain). Only the two halves are graded
(frames 11,185–90,651 and 110,808–190,197 = 158,857 frames). You are **ranked on `spellF1`**, which
measures whether whole possessions were recovered. `macroF1` (per-frame) is the tie-break. The floor
to beat is **0.2433**, set by `baseline_smooth50`. There is no leaderboard, and the private
full-match ground truth (GT) is the only thing that counts. Deadline: **17 November 2026**.

---

## 2. What is in the bundle

| file | size | what it is |
|---|---|---|
| `data/tactical/BAR-ATH_tactical.mp4` | 5.2 GB | 7,972.8 s (2 h 13 m), 25 fps, 1920×1080, **has an audio track** |
| `data/tactical/BAR-ATH_tactical_atd.txt` | 196 MB | tracking: 199,319 lines `frame_id:46 player groups:ball`, each group `x,y,speed` |
| `data/tactical/BAR-ATH_tactical_atd.xml` | 59 KB | metadata: teams, 46 track slots → team, attack directions |
| `data/tactical/BAR-ATH_tactical_homography.csv` | 23 MB | one 3×3 pitch→pixel matrix per frame, all valid |
| `data/tactical/BAR-ATH_tactical_smart_tagging.xml` | 261 KB | 638 automatic tactical phases / set pieces |
| `data/broadcast/…` | — | **not in your bundle.** It is optional and unsynchronised, so it's fine to skip |

Alignment: submission row = video frame = homography `frame` = ATD `frame_id − 1` =
`round(tagging_seconds × 25)`.

---

## 3. Tracking data (ATD), as it actually is

### Teams and slots
- Home = **Barcelona**, away = **"Atletic Club"** (that spelling is in both XML files).
- Slots **1–24 = Barcelona**, **25–46 = Athletic**. Slots are tracklets, not people. There are 1,219
  tracklets in the graded frames, with a median length of 35 s.
- **Slot 24 = Barcelona goalkeeper, slot 46 = Athletic goalkeeper.** They are present on 92–93% of
  frames. Their position is *held or estimated* while the keeper is off-camera, which is why their
  speed is NaN on 57–71% of frames.
- Slots 22, 23 and 44 are almost never used.
- On average 20.5 objects are tracked per graded frame (p5 = 14, max = 25), split roughly 10.2 home
  and 10.3 away. A missing player usually means "off-camera", not "absent".
- Attack direction (XML): in the 1st half Barcelona attacks **left→right** (towards x = 1), and the
  directions swap in the 2nd half. The keeper slots confirm this. Normalise every feature for
  direction.

### Units: a doc/data inconsistency
- Coordinates are normalised 0–1. Multiply by **105.3 × 68.0 m**.
- The XML says speed is in *kph*, but **player speed is in m/s**: it matches displacement × 25 with a
  ratio of 0.99.
- **Ball speed is NaN on 91% of the frames where the ball is present.** Derive ball velocity from
  positions instead.

### The ball: the single most important fact
- Present on 56.9% of all frames and 59.3% of graded frames, the same in both halves.
- **Missing-ball frames are not random flicker. They come as 150 long blackouts:** median **10.7 s**,
  mean 17 s, longest **221 s**. Only about 17 gaps last 1 s or less. Visible runs have a median of
  13.8 s.
- **Most blackouts are stoppages.** The median outfield speed during blackouts is **1.05 m/s**,
  against 2.03 m/s when the ball is visible. 61% of blackout frames have players walking
  (< 1.2 m/s), compared with 14% of ball-visible frames. 78 of the 133 blackouts longer than 1 s
  have a tagged set piece within ±10 s, and those cover 70% of blackout frames.
- The ball lies **outside [0,1]²** on 4.4% of the frames where it is present (x from −0.07 to 1.11,
  y up to 1.49), in 81 graded runs. Those runs mean the ball is out of play.
- When the ball is visible, the nearest player is a median of 1.9 m away (p90 7.0 m). The
  nearest player's team agrees with the GT on **80% (H) / 75% (A)** of ball-visible public frames.

### Player speed as a dead-ball detector (public window)
| GT state | median outfield speed, 1 s smoothed | p10 | p90 |
|---|---|---|---|
| H | 2.11 m/s | 1.25 | 3.49 |
| A | 2.50 m/s | 1.67 | 3.88 |
| **D** | **1.19 m/s** | 0.89 | 2.59 |

---

## 4. Homography and the camera

- Valid on 100% of frames. The median reprojection error in graded frames is 0.23 px, p99 4.3.
- **The tactical camera pans and zooms.** It is not a static panorama, and its near edge covers about
  40% of the pitch length.
- **Camera centre as a ball proxy.** Projecting the image centre (960, 540) onto the pitch gives an x
  that correlates with ball x at **r = 0.83** (median error 12 m). Camera y carries no signal. This is
  the best free locator of the ball during blackouts.
- The homography also tells you which part of the pitch is on screen. Use it to tell "player/ball
  missing because off-camera" apart from "missing in view".

---

## 5. Smart Tagging

- 638 instances and 14 codes. The team value is spelled `Atletic Club`, and `N/A` appears on all 26
  throw-ins and all 6 kick-offs.
- **They come in mirrored pairs with identical intervals**, one row per team:
  PROGRESSION ↔ DEFENDING IN MIDDLE THIRD (66), BUILD UP ↔ DEFENDING IN ATTACKING THIRD (22),
  CREATING CHANCES ↔ DEFENDING IN DEFENSIVE THIRD (30), ATTACKING ↔ DEFENSIVE TRANSITION (24).
- **Durations are templated.** SET PIECES are always 35 s and kick-offs 5 s. CREATING CHANCES and
  DEFENDING IN DEFENSIVE THIRD are always 30 s. The restart happens *inside* the window, observed
  2.8–12 s after its start, not at the start.
- Set pieces (59): 26 throw-ins, 12 corners, 10 goal kicks, 5 free kicks, 4 "Kick Off (After Goal)"
  and 2 "Kick Off (Start)". **A real match has far more restarts** (fouls, offsides, substitutions), so
  the tagging covers perhaps half of the stoppages. You need your own dead-ball detector.
- Coverage: 82% of graded frames sit under at least one tag, and 42% are tagged for *both* teams at
  once.
- How often the team tag agrees with the GT possessor in the public window (alive frames, tiny
  sample):
  BUILD UP / CREATING CHANCES (and their mirrors) 100%, SET PIECES 96%, LONG BALL 90%, PROGRESSION 73%,
  TRANSITIONS 67%, BALL IN FINAL THIRD 54%, PLAYERS IN THE BOX 46%, BALL IN THE BOX 44%. The last three
  describe territory, not possession. Use them as weighted features, never as labels.
- Restart type tells you which team restarts: corner → attacking team, goal kick → team defending
  that goal, kick-off after a goal → the team that conceded. Infer throw-in teams from the tracking.

---

## 6. The public ground truth (frames 11,185–18,684, 5 min)

- 7,500 frames: H 3,454, A 2,577, D 1,469. It has **21 segments and 17 possessions** (8 H, 9 A).
- Possession lengths range from 0.76 s to 42.4 s: 2 under 2 s, 4 between 2–5 s, 4 between 5–10 s and
  7 over 10 s. It also has 4 dead-ball stretches of 5.5–20.4 s.
- The **first graded frame is already `A`** (Athletic kicks off). The 2nd half is graded from frame
  110,808 but only kicks off at 113,742, so those 2,934 frames are D *(docs)*.
- **This window is unrepresentatively easy.** The ball is visible on about 88% of its frames, against
  59% match-wide. Expect the full match to be harder than anything you measure here.
- GT conventions you can read off it: D starts when the ball crosses the line or the whistle goes,
  and ends at the restart touch. During loose balls, possession stays with the last team. Brief
  touches by the other team *can* create very short spells (0.76 s, 1.32 s).

---

## 7. How the metric really works, and what it implies

`spellF1` takes every maximal run of `H` or `A` inside a half as one possession. It matches predicted
runs one-to-one against true runs of the same team by temporal IoU, and averages F1 over IoU
thresholds 0.3/0.4/0.5/0.6/0.7.

1. **Count discipline.** F1 = 2·TP / (N_pred + N_gt), so every spell you output costs something.
   Emit a spell only if its chance of being matched is above about F1/2. The raw baseline (131 spells
   against 17 true ones) scores 0.04 for exactly this reason.
2. **Dead-ball boundaries are first-class.** `H D H` is two possessions. Missing a stoppage merges two
   spells, and inventing one splits a spell. D also delimits every restart.
3. **Systematic lag is very costly.** Being 1 frame late is free and 3 frames late gives 0.98, but
   1 s late gives 0.65 on every transition *(docs)*. Use offline, centred decoding and no causal
   filters. Measure mean latency against labelled transitions and shift to correct it.
4. **Long spells are cheap and short spells are expensive.** A 40 s possession tolerates seconds of
   boundary error. A 1 s spell needs near-frame precision, so don't chase those unless the evidence is
   strong.
5. **`X` is useless on graded frames.** `spellF1` forward-fills it, so it changes nothing there, while
   `macroF1` counts it as a miss. Write H/A/D on every graded frame and X everywhere else.
6. **Don't select models on the public window.** With 17 possessions, one possession is worth about
   6 points. The repo shows the window picking the wrong smoothing even for the baseline.

---

## 8. The plan to win

### Step 0 — Build your own validation set (do this first)
Hand-labelled data **for training and validation is explicitly allowed** (RULES.md). Only the
*submitted* labels must come from the algorithm.
- Label **20–30 more minutes** from the video, spread across both halves and including busy stretches
  with set pieces. That gives you over 100 possessions, an order of magnitude more than the public
  window.
- Copy the public GT's conventions exactly (section 6). Have two people label the same 3 minutes to
  measure how far you disagree.
- Score against it with the official scorer: `run_evaluation.py --gt <your_labels.csv> --bootstrap 1000`.
- Say in the technical document that you did this.

### Step 1 — Alive/dead detector (biggest lever)
Compute per-frame features over several centred windows (1 s, 3 s, 10 s):
- ball visible; length of the blackout it sits in, and position within it (you know both ends,
  because this is offline)
- ball outside the pitch, near a line, in the corner arc or in the 6-yard box; ball stationary
  (derived speed ≈ 0)
- outfield speed: mean and percentiles per team; how many players are moving at all
- team spreads; players crowding the box (corners and free kicks); a defensive wall forming
- camera pan speed and zoom (from the homography)
- distance to Smart Tagging set-piece and GOALS windows
- (optional) audio: whistle energy in the 2–4 kHz band from the mp4's audio track

Start with rules (blackout plus slow players → D), then train a GBM on the hand labels.

### Step 2 — Who is in control (when alive)
- **Model "control", not "nearest player".** Possession changes only when the other team *gains
  control*. During loose balls it stays with the last team, which is hysteresis.
- Evidence of control:
  - per-team nearest distance combined with ball speed (a slow ball at a player's feet means
    control; a ball in flight means no new evidence)
  - how long the proximity lasts
  - the player moving with the ball (dribbling)
  - **pass detection**: the ball leaves a player of team X and reaches a player of team X (possession
    kept) or of team Y (possession changes)
- For alive frames without a ball, use team shape: the side in possession is more spread out and
  higher up the pitch. In the public window the possessing team's spread in x is 0.14–0.16, against
  0.11–0.12 for the defending team. Add camera-centre x as a ball proxy and the Smart Tagging team
  prior.
- Restart team comes from the set-piece type, or from which team's player stands over the ball at
  the restart.

### Step 3 — A temporal decoder built for `spellF1`
- Replace majority voting with an **HMM or hidden semi-Markov model (HSMM)** over {H, A, D}, plus
  optional "loose-H" and "loose-A" sub-states. Feed it calibrated log-probabilities from steps 1–2.
- Add transition penalties and minimum-duration priors. Restart rules link D to the next team's
  state.
- Decode with Viterbi. It works offline, so there is no lag.
- The switching penalty is your possession-count knob. Tune it on your own labels, not on the public
  window.

### Step 4 — Post-processing
- Correct latency with a global shift measured on your labels.
- Drop spells whose expected match probability is below about F1/2.
- Snap D boundaries to strong events: the ball leaving the pitch, or a stationary ball being kicked.
- Hard facts: label the kick-off taker's team from the first graded frame. Frames 110,808–113,741
  are D; derive them from kick-off detection rather than typing them in.

### Optional extras, ranked by value for cost
1. **Audio whistle detector.** It's cheap: you only decode audio. Install `av` (PyAV) or
   `imageio-ffmpeg`.
2. **Ball detection on blackout frames** (YOLO-type detector, run on Colab or another GPU). It's of
   limited value because most blackouts are dead balls anyway, but it helps find the exact restart
   moment.
3. Broadcast feed: skip it. The sync cost is high and it isn't in your bundle.

---

## 9. Schedule (4 weeks)

| week | goal |
|---|---|
| **20–26 Oct** | Feature cache and frame viewer; agree labelling conventions; label 20–30 min; rule-based D detector + control model + Viterbi → **first validated submission above 0.2433** |
| **27 Oct – 2 Nov** | GBM emissions for D and control; HSMM; per-spell error analysis (missed, split, merged, late) |
| **3–9 Nov** | Smart Tagging integration, restart-team logic, latency calibration; audio-whistle experiment; optional video ball detection |
| **10–17 Nov** | Freeze; bootstrap confidence intervals on your labels; reproducibility test from a fresh clone (identical file); technical document; **submit by 15–16 Nov** |

Suggested roles for a team of 3–5: (1) infrastructure, reproducibility and repo; (2) dead-ball
detector and audio; (3) control/pass model and decoder; (4) labelling lead, validation and error
analysis; (5) optional video ball detection.

---

## 10. Gotchas checklist

- [ ] Exactly 199,319 rows, `frame` running 0…199,318, header `frame,label`. Run
      `python evaluation/validate_submission.py <file>` on **the exact file you upload**.
- [ ] ATD `frame_id − 1` = video frame.
- [ ] Ball speed is mostly NaN; player speed is m/s, whatever the XML says.
- [ ] Keeper slots 24 and 46 hold stale positions while off-camera. Exclude them from speed and shape
      features.
- [ ] The ball can lie outside [0,1]². That is a signal, not an error to clip.
- [ ] Match Smart Tagging case-insensitively. `Atletic Club` is spelled that way, and `N/A` teams
      appear on throw-ins and kick-offs.
- [ ] Never write X on graded frames.
- [ ] The run must be deterministic: fixed seeds, `deterministic=True` and fixed `num_threads` for
      LightGBM, pinned dependencies. Organisers re-run the winning repo and expect **the same file**.
- [ ] The public repo must contain no data, no clips, no frame-level exports and no submission CSVs.
- [ ] Name the files `<team_name>.csv` and `<team_name>.pdf`. Uploading needs a Google account. One
      form response per team, which you can edit until the deadline.
- [ ] Teams must have 3–5 members to be eligible for the Innovation Challenge and the Final Pitch.
- [ ] Local environment: `cv2` fails to import under numpy 2.4. Run `pip install -U opencv-python` (or
      pin `numpy<2`) before reading video. kloppy is not installed, and you don't need it.
