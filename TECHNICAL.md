# `<team_name>`: frame-level ball possession for BAR–ATH

**More than a Hack 2027, Phase 1: Ball Possession**. Technical document.

| | |
|---|---|
| Team | `<team_name>` |
| Possession file | `<team_name>.csv`, byte-identical to `possession_v6.csv` (SHA-256 starts `FE83E5E59564A764`) |
| Code | `<repo_url>`: `python make_submission.py` regenerates the file |
| Method | rule-based 4-state Viterbi with a minimum stoppage length, over evidence from the tracking and the stadium audio, plus restart snapping after decoding. No learned model and no hand labels in the pipeline |

---

## 1. Approach, and why

We label every frame as one of four hidden states and decode each half with an offline Viterbi:

| state | meaning | written as |
|---|---|---|
| `H` | Barcelona has the ball, alive | `H` |
| `A` | Athletic has the ball, alive | `A` |
| `DH` | dead ball, Barcelona will restart | `D` |
| `DA` | dead ball, Athletic will restart | `D` |

The two dead states carry the team that will restart, so when play resumes the ball goes to the
right side.

Each frame gets a score for each state, built from physical signals in the tracking and from the
referee's whistle in the stadium audio (section 5). The decoder picks the state path that
maximises the summed scores minus a penalty for every transition, under one hard constraint: **a
stoppage lasts at least 5 s**. After decoding, each restart moves back to the kick that restarted
play. Frames outside the two halves are written `X`. Inside them, every frame gets `H`, `A` or `D`.

**Why this design and not something else:**

- **The metric rewards whole possessions, not frames.** `spellF1` matches predicted possessions
  one-to-one against true ones by temporal IoU, so every extra spell costs something. Splitting a
  spell, merging two, or a systematic lag all cost heavily: a perfect labelling that is 1 s late
  scores 0.6539 (EVALUATION.md). A sequence decoder with transition penalties controls the
  possession count directly: the switch penalty is the count knob. Because decoding is offline,
  transitions are placed where the evidence changes, with no causal lag.
- **There is too little ground truth to fit on.** The public window has only 17 possessions and 4
  dead-ball segments. A logistic-regression fit of the dead-vs-alive evidence on it, cross-validated leaving one segment
  out, reached AUC 0.739 against **0.926** for the hand-set weights.
- **Better per-frame accuracy doesn't mean better possessions.** A LightGBM trained on our own full-match labels (80
  multi-scale features, public window held out) raised per-frame AUC a lot:
  - dead vs alive: 0.94 → 0.97
  - team: 0.83 → 0.96

  But its 9 s windows blurred boundaries, and it scored 0.50–0.57 public spellF1 at every blend
  (section 6). Hand-set log-likelihood ratios, checked against several independent sources, were
  the honest choice.
- **Scores, not probabilities.** Consecutive frames are strongly correlated, so per-frame
  naive-Bayes evidence would be wildly overconfident. The weights act as a temperature, and the
  transition penalties are calibrated against them. The two are only meaningful together.
- **The minimum stoppage belongs in the decoder.** Each dead state is expanded into a chain of
  sub-states, one per frame up to 5 s. The decoder then finds the best path that obeys the
  minimum, instead of patching short stoppages afterwards. Against a brute-force dense decoder, it
  found the exact optimum on 60/60 random cases.

**Pipeline (v6).**

| module | role |
|---|---|
| `possession/data.py` | loads the tracking (in metres), homography, Smart Tagging and public GT on the video-frame index. **`Match.team_by_frame`** gives each goalkeeper slot the team of the goal it guards (section 3) |
| `possession/audio.py` | turns the tactical mp4's stadium audio into a per-video-frame spectrogram (48 kHz / 25 fps = 1920 samples per frame) and gives a whistle-tone strength per frame |
| `possession/signals.py` | physical per-frame signals: ball visible/speed/out of bounds/placed at a restart spot, blackout length, outfield activity, nearest home/away distance, team width, receptions, ball contacts, camera centre, Smart Tagging priors, pre-kick-off mask, whistle |
| `possession/evidence.py` | turns signals into per-frame scores for the 4 hidden states, and holds every parameter (`Params`) |
| `possession/decode.py` | offline Viterbi with a **minimum stoppage length**: each dead state is a chain of k sub-states. Checked against a brute-force dense decoder: the exact optimum on 60/60 random cases |
| `possession/pipeline.py` | decodes each half separately, snaps restarts to the kick that restarted play, writes X outside the halves |
| `make_submission.py` | CLI; `--set key=value` overrides any parameter |

The public ground truth is loaded only by the evaluation and analysis tools. `pipeline.run`, which
produces the submission, never reads it.

---

## 2. Assets: what we used, and what we dropped

### Used

| asset | how it is used |
|---|---|
| **Tracking (ATD)** | most of the model. Player and ball positions in metres (normalised × 105.3 × 68.0 m) and player speed, which is in m/s despite the XML saying kph. The XML metadata gives each of the 46 track slots its team and gives the attack directions, except that the two keeper slots get their team from the goal they guard (section 3) |
| **Tactical video: the audio track** | **model input.** The referee's whistle is a steady 3.80–3.91 kHz tone in the stadium audio. It marks the start of a stoppage far more precisely than the players slowing down (section 5) |
| **Tactical video: the pictures** | **validation only, never model input**: blind adjudication of stretches where model variants disagree, seeded random samples, a random absolute audit, and the hand-labelling tool (section 6) |
| **Smart Tagging** | two narrow uses. (1) Set pieces with a team (corner, goal kick, free kick) assert the restarting team over the first 20 s of their window. (2) The "Kick Off (Start)" tags locate each half's kick-off |
| **Public ground truth** (5 min) | correctness check and calibration only. The activity levels for dead and alive play and the whistle band were measured on it, and the kick-off offset was measured against the published first graded frame of the 1st half |
| **Homography** | draws the tracking overlay on video frames in the review and labelling tools. The camera features were screened but not adopted (below) |
| **Literature** | the duel zone follows Vidal-Codina et al. (2022). The ball-contact detector we tried and dropped follows Link & Hoernig (2017) and Vidal-Codina et al. (2022). See section 10 |

### Tried and dropped

The numbers behind each decision are in the tuning tables in section 6. Code for team width,
receptions, the phase prior, the live-state restart prior, last-touch evidence, team-control
chains and the minimum possession length is still in the repository, switched off with a weight or
length of 0. The self-training experiment and the confirmation veto were never part of the pipeline
and are not in the repository.

- **Team width:** shape lags turnovers, so it merges real switches.
- **Reception events:** the control evidence already catches them.
- **Smart Tagging possession phases** as a team prior: they lowered the public score.
- **Kick-from-rest as a restart *detector*:** it caught 0 of 4 public restarts. It works as a timing
  anchor once the decoder has found a restart (restart snapping, section 5).
- **The tagged restart team** as a prior on the live states: it overrode a clear takeover.
- **Sprint, deceleration, formation, camera and zoom features:** none beats mean activity.
- **Last-touch evidence from a ball-contact detector:** it is better per frame, but it flips
  possession on single touches and inflates the count.
- **Team-control chains (last touch counting only controlled contacts):** a contact counts only
  when no opponent is within 2 m and the same team plays the ball again, uncontested, within 3 s.
  The gate is precise and adds no spells, but it is redundant with the control evidence.
- **A confirmation veto** that deletes short possessions with no controlled contact: on video it
  mostly deleted real possessions won under pressure.
- **Snapping possession switches to the first contact:** contacts don't mark the true change.
- **Self-trained LightGBM:** much better per-frame AUC, worse possessions (section 1).
- **A 3 s minimum possession length:** it gave the highest public spellF1 we saw, but too few
  possessions. We read that as over-fitting the public window.
- **A rule that a keeper holding the ball in the box means alive:** the only public instance is
  genuinely dead.
- **Fitted (logistic) emission weights:** four dead segments are too few to fit on.
- **Ball speed as delivered:** NaN on 91% of ball-present frames, so it is derived from positions
  instead.

### Not used

- **Broadcast feed**: not in our bundle, and not synchronised to the tactical video.
- **Throw-in and kick-off tags as restart-team evidence**: these tags carry team `N/A`. Territorial
  tag codes (BALL IN FINAL THIRD, PLAYERS IN THE BOX, BALL IN THE BOX) describe where the ball is,
  not who has it.

---

## 3. A data defect: goalkeeper slots carry the wrong team

The tracking has two goalkeeper slots: 24 "Barcelona" and 46 "Athletic". Their metadata team is
not reliable.

- **Slot 46** stands inside **Barcelona's** penalty area on **15,945 graded frames** (11% of its
  life). That includes the last 230 s of the match and the first 146 s of the second half.
- **Slot 24** stands inside Athletic's penalty area on 1,501 frames.

A keeper does not spend four minutes in the opposing box. The tracker's keeper slots follow
whichever keeper they re-acquire. So every time a keeper had the ball in those stretches, the
evidence credited the wrong team: goal kicks, back-passes, catches. We found it in error analysis:
the keeper rings were drawn in the wrong colour on the review montages.

**Fix:** each goalkeeper slot takes, frame by frame, the team that defends the goal in its half of
the pitch (`Match.team_by_frame`). The attack directions in the tracking metadata say which goal
that is. Outfield slots were checked and are fine. Both teams show occasional over-counts from
track fragmentation, and referees are not tracked.

**Evidence:** on video, **5 of 5** changes were right (a goal kick, keeper distributions,
back-passes). Public macroF1 rose to 0.8925. In a seeded random sample of v4/v6 disagreements, 3
of the 8 were keeper-team fixes, all right (section 6).

---

## 4. Missing and noisy tracking

The tracking is automatic and uncorrected. These are the problems we measured, and what the
pipeline does about each one.

| problem (measured on the bundle) | handling |
|---|---|
| **The ball is present on only 59.3% of graded frames.** The gaps are not flicker: they come as 150 long blackouts, median **10.7 s**, mean 17 s, longest **221 s**. Only about 17 gaps last 1 s or less | **A blackout is evidence in itself.** Most blackouts are stoppages: the median outfield speed during them is **1.05 m/s**, against 2.03 m/s when the ball is visible. Being inside a blackout longer than 2 s counts towards "dead". A shorter gap is treated as ordinary occlusion and says nothing. The whistle comes from the audio, so it works whether the ball is tracked or not |
| **Who has a missing ball?** | Where the ball is not seen, the team evidence is exactly zero. The decoder therefore carries the last team through the gap, unless dead-ball evidence ends the spell. That is the "loose ball stays with the last team" rule applied to missing data |
| **Ball speed is NaN on 91%** of ball-present frames | Speed is derived from positions with a centred difference over ±2 frames. At the first and last frames of a visible run, where speed can't be measured, control evidence gets half weight |
| **The ball lies outside [0,1]²** on 4.4% of ball-present frames | Kept, not clipped. A ball more than 1 m beyond a line is strong dead-ball evidence |
| **Goalkeeper slots** (24 and 46) hold stale positions while off camera (speed NaN on 57–71% of frames), and their team is wrong part of the time | Excluded from the activity (outfield speed) signal. Their team comes from the goal they guard (section 3) |
| **Players drop out**: 20.5 objects tracked per graded frame on average, p5 = 14. A missing player is usually off camera | Signals use only the players actually tracked. Nearest-player distances are taken over tracked players. The limitation this leaves is in section 8 |
| **Slots are tracklets, not people**: 1,219 tracklets in the graded frames, median length 35 s, with identity switches | Nothing uses player identity. Only each slot's team and position are used, so identity switches have no effect |
| **Units**: the XML says kph, but player speed is in m/s (it matches displacement × 25 with a ratio of 0.99) | Read as m/s |
| **Frame-to-frame jitter** | Activity is a centred 1 s mean. "Ball still" means it moved less than 1 m over about 1 s (±12 frames). Restart spots include about 1 m of tracking slack. Control evidence is clipped, and the transition penalties absorb short flicker. The minimum stoppage removes false stoppages shorter than 5 s |
| **Crowd noise in the audio** | A whistle has to hold its pitch for at least 0.25 s and stand out from a rolling ~20 s background of the crowd. Chants wander in pitch and broadband noise is not a narrow line, so neither passes |

---

## 5. Possession rules implemented

We copied the conventions visible in the public ground truth. A dead ball starts when the ball
crosses the line or the whistle goes, and ends at the restart touch. During loose balls,
possession stays with the last team.

### Dead vs alive

The dead-vs-alive evidence is a log-ratio, summed from these signals:

- **Outfield activity.** Log-normal around 1.2 m/s when dead and 2.3 m/s when alive.
- **Blackout.** The frame is inside a ball blackout longer than 2 s.
- **Out of bounds.** The ball is beyond a line.
- **Placed ball.** The ball is still in the goal area, at a corner or on the centre spot.
- **Touchline.** The ball is still on a touchline, **but only while players walk** (activity below
  1.4 m/s). The 49 still-ball-on-touchline runs split into walking (17, throw-ins) and live play
  (19). Without this gate, a winger shielding the ball on the line was called dead.
- **A referee's whistle that stops play.** It pushes towards "dead" for 2 s, starting 8 frames
  after the onset, but only if two things hold:
  - the players were busy before it (at least 1.8 m/s)
  - they slow down after it (by at least 0.4 m/s)

  The gate keeps kick-off whistles and play-on whistles (advantage) out. With the whistle, stoppage
  starts are −6 / +3 frames from the GT, against +19 / +35 for v2. The push is a nudge that
  live-play evidence can outvote, not a hard switch.
- **Against dead:** the ball travelling faster than 10 m/s.
- **Against dead:** somebody has the ball at their feet.
- **Forced dead before each half's kick-off**, from the "Kick Off (Start)" tag plus 88 frames.

### Who has the ball

- **Control.** Each team's control likelihood is a Gaussian in its nearest player's distance to
  the ball (σ = 1.5 m).
- **Ball in flight.** The control evidence is weighted down smoothly once the ball travels faster
  than about 8 m/s. Dribbling stays well below that speed, and passes and shots are well above it,
  so a ball in flight adds no new evidence for either team.
- **Duel zone** (after Vidal-Codina et al., 2022). When both teams have a player within 2 m of the
  ball, the ball is contested, and the control evidence is set to zero.
- **No bias towards dead or alive.** Team evidence only ever penalises the disfavoured side of a
  pair, so it cannot shift dead vs alive.

### Loose balls and challenges

Two rules decide when possession moves:

- **A loose ball stays with the last team.** This is built into the transition penalty.
- **A contested ball stays with the last team.** This is built into the duel zone.

Possession switches only when the other team builds up enough *uncontested* control evidence to pay
the switch penalty. A single touch, a deflection, a 50/50 or a ball passing a defender does not
switch it.

The cost is deliberate: genuinely short possessions are given up. Count discipline is why. A spell
the model can't place precisely costs more than it earns under `spellF1`, and a spell under 2 s
needs near-frame precision to match.

### Restarts

- **A stoppage lasts at least 5 s.** This is enforced inside the decoder, not afterwards. It
  removes brief false stoppages that would split one possession in two.
- **The restarting team is remembered.** Restarting with the team the dead state names is cheap.
  Restarting with the other team is expensive.
- **Who restarts** comes from two sources during the stoppage:
  - which team stands over the ball
  - for corners, goal kicks and free kicks, the Smart Tagging team, asserted over the first 20 s of
    the 35 s set-piece window. The restart is seen 2.8–12 s after the window opens.
- The restarting team may change during a stoppage, at a small cost.
- **Restart snapping.** After decoding, each dead → alive transition moves back to the kick that
  restarted play, if one happened within the 40 frames before it. A kick is the ball going faster
  than 5 m/s after a second in which it was slower than 2.5 m/s or unseen. Without snapping, v4's
  restarts lagged the kick by about 1 s in a quarter of cases.

### Halves

Each half is decoded on its own, so no state is carried across halftime.

### Decoder parameters

These are the v6 values. The authoritative source is `possession/evidence.py::Params`. Penalties
are in the same units as the scores.

| decoder parameter | v6 value |
|---|---|
| `H` ↔ `A` (switch) | 10 |
| alive → dead (stop) | 15 |
| dead → alive, the team the dead state names | 15 |
| dead → alive, the other team | 40 |
| `DH` ↔ `DA`, while dead | 6 |
| minimum stoppage | 125 frames (5 s) |
| duel zone radius | 2 m |
| restart snapping window | 40 frames |

---

## 6. Validation without a leaderboard

The public window has 17 possessions, and the organisers' own repository shows it choosing the
wrong baseline. So every decision was checked four ways, and never by public `spellF1` alone.

1. **Public window, with 1,000-draw bootstrap intervals.** Used as a correctness check, not as a
   ranking.
2. **Full-match statistics:** possession count and mean length. The public window's rate implies
   roughly 210–270 possessions over the match.
3. **Label-free proxy checks** (`tools/proxy.py`).
   - One of them is a hard fact: frames 110,808–113,741 are dead (FAQ.md).
   - The others are the kick-off teams, the dead share of GOALS windows, and set-piece restart
     teams.
4. **Video adjudication** (`tools/disagree.py` + `tools/review.py`).
   - Two variants are compared only where they disagree.
   - Each stretch is judged blind, from a 2×2 montage of video frames with the tracking overlaid.
   - **Seeded random samples** are judged the same way, so the sample is not cherry-picked.

> **AI-assisted validation, stated plainly.** The video adjudication calls were made by Claude, an
> LLM, looking at the rendered video frames. This is AI-assisted labelling. It was used **for
> validation only**, to choose between variants and audit the output. It was **never** model
> input, and no frame of the submitted file was set or corrected by it.
>
> Calibrated on the public window, these calls proved reliable for possession while the ball is
> tracked, and for stoppages to about ±1–2 s. They are not reliable on contested moments, so those
> moments were left out rather than guessed.

**The submitted labels come only from the algorithm.** The file is the unedited output of
`python make_submission.py`. No frame was set by hand, by an AI or from any ground truth.

A hand-labelling tool for a larger validation set is built (`tools/label_tool.py`), but no hand
labels had been made at the time of writing. Hand labels, which RULES.md permits for training and
validation, stay out of the public repository and out of the pipeline.

### Round 1: v1 → v2

| change | public spellF1 / macroF1 | full match: possessions | proxy / video evidence | decision |
|---|---|---|---|---|
| force D before the kick-off | unchanged | 247 | pre-kick-off 78.8% → 100% dead (fact: 100%) | **adopted** |
| team width, ball seen or ball missing only | 0.47–0.53, or unchanged | 209–253 | shape lags turnovers; 4 adjudicated stretches: 1 right, 2 wrong, 1 split | rejected |
| reception events | ±0.01 | +0–3 | control evidence already catches them | rejected |
| Smart Tagging phase prior | 0.47–0.53 | 233–243 | — | rejected |
| switch penalty 8 / 10 / 15 | identical | 264 / 243 / 216 | sampled disagreements lean low | **10 kept** |
| touchline restart spot only while players walk | 0.5375 / 0.8543 | 243 | a winger shielding the ball on the line was called dead; the 49 touchline runs split into walking (17) and live play (19) | **adopted** |

### Round 2: v2 → v4

| change | public spellF1 / macroF1 | full match | proxy / video evidence | decision |
|---|---|---|---|---|
| referee whistle from the stadium audio | 0.5375 / 0.8627 | 242 | stoppage starts now −6 / +3 frames from GT (v2: +19 / +35); the gate keeps kick-off and play-on whistles out | **adopted** |
| minimum stoppage 5 s (chained dead states) | 0.5867 / 0.8820 | 236 | 2 GT-confirmed fixes; on video 4 right, 1 wrong (a 2.6 s goal kick), 1 unclear | **adopted** |
| kick-from-rest as a restart *detector* | — | — | caught 0 of 4 public restarts | rejected |
| tagged restart team as a live-state prior | unchanged | 230–237 | held Athletic 14.6 s against a clear Barcelona takeover | rejected |
| sprint / deceleration / formation / camera / zoom features | screened | — | none beats mean activity as a state cue | not adopted |

**Unbiased check of round 2:** in a random sample of 8 v2/v4 disagreements, v4 was better in 6, v2
in 1, and 1 was unclear.

### Round 3: v4 → v6 (literature and full-match error analysis)

| change | source | public spellF1 / macroF1 | full match | video evidence | decision |
|---|---|---|---|---|---|
| **goalkeeper team by guarded goal** | our error analysis (keeper rings in the wrong colour on review montages) | 0.5857 / **0.8925** | 219 | **5/5** changes right (goal kick, keeper distributions, back-passes) | **adopted** |
| **duel zone, radius 2 m** | Vidal-Codina et al. 2022 | 0.600 / 0.8855 (with snapping, before the keeper fix) | 234 | 3 longest changes: 2 right, 1 wrong | **adopted** |
| **restart snapping to the kick, window 40** | our timing analysis (v4 restarts lag the kick by ~1 s in a quarter of cases) | unchanged / +0.002 | 236 | both public restarts touched end closer to the GT (+21 → −8, +14 → −10); 2 throw-ins on video thrown on the snapped frame | **adopted** |
| ball-contact detector (Savitzky-Golay acceleration peaks) as **last-touch** team evidence | Link & Hoernig 2017; Vidal-Codina 2022 | 0.52–0.58 | 259–321 | per frame it agrees with the GT 91% (vs 78%), but it flips possession on single touches: possessions jump to 259–321 (vs ~215 expected) and tagged restart teams fall from 17/24 to 13–16/24. A 5-stretch video sample was inconclusive (mostly contested moments) | rejected |
| snap possession switches to the new team's first contact | our timing analysis | — | — | public switches are early, but contacts don't mark the true change (2 good switches would move 2.5 s late) | rejected |
| **self-training**: LightGBM on our own full-match labels (80 multi-scale features), public window held out | semi-supervised learning | 0.50–0.57 at every blend | 209–239 | per-frame AUC rises a lot on the held-out public frames (dead 0.94 → 0.97, team 0.83 → 0.96), but 9-s windows blur boundaries and break a kick-off | rejected |
| minimum possession length 3 s | F1 count arithmetic | 0.637 / 0.8995 | 190 | 6 random 2–3 s spells on video: ~4 right, 1 wrong, 1 unclear; too few spells (190 vs ~215) | rejected (public over-fit) |
| keeper holding the ball in the box → alive | random audit miss at 27,163 | — | — | the only public instance (goal-kick set-up) is genuinely dead | not adopted |

**Unbiased check of round 3.** We judged a seeded random sample of 8 v4/v6 disagreements on video:
**v6 was right in 8 of 8.** That breaks down as:

- 3 keeper-team fixes
- 3 snapped restarts (throw-ins and a quick restart)
- 1 duel-zone case
- 1 kick-off

### Round 4: gated contacts (v6 kept)

Raw last-touch evidence failed because it flips possession on single touches. Round 4 tested the
obvious repair: count a contact only when it shows control. That means no opponent within 2 m (the
duel zone), followed by sustained control. The targets were the wrong-possessor misses.

| change | source | public spellF1 / macroF1 | full match | video evidence | decision |
|---|---|---|---|---|---|
| **team-control chains**: two consecutive uncontested contacts by the same team, ≤ 3 s apart, put that team in control between them (`w_chain`, w 0.2–0.8) | Link & Hoernig 2017: team possession passes at the other team's next *controlled* action | 0.5857 / 0.8925 at every weight and gap (free radius 2 m); 1.5 m adds a wrong spell (0.5333); 3 m changes nothing | 217–224 (raw last touch: 252) | the gate is precise: uncontested contacts name the GT team 93% (n=60), contested 64% (n=11), chained 37/37. But at w 0.4 only 2 full-match stretches move: 1 plausibly right, 1 plausibly wrong | kept off (redundant) |
| "held" gate: the contact's team within 2 m for ≥ 90% of the next second | Link & Hoernig 2017 (control = duration, opponents near) | 0.5857 / 0.8947 at w 0.1; 0.5333 from w 0.2 | 205–225 | carried 2 s, it fixes 32 public frames where v6 is wrong and breaks 48 | rejected |
| confirmation veto: absorb possessions < 3 s with no controlled (held or chained) contact | count arithmetic | 0.6370 / 0.8995 | 197 (plain 3 s minimum: 182) | the public gain is one false spell (12,786). All 12 removals outside the public window judged blind: 2 right, **7 wrong**, 3 unclear | rejected |

**Why gated contacts can't help much.** An uncontested contact followed by sustained control is
exactly where proximity already names the right team. The misses they were meant for can't produce
one:

- an untracked carrier has no tracked player at the ball
- a contested ball is contested by definition
- a sub-2 s spell has no time to show control

The veto's deletions were mostly real possessions won under pressure or off an aerial ball, where
every contact is contested. Gated contacts don't mark the GT's switch moment either: the nearest one
ranges from 28 frames before to 67 frames after a public GT switch.

These calls are AI-assisted, as above, and contested moments are where they are least reliable. So
the code stays in the repository, switched off, and hand labels can re-test it in one command:
`python tools/sweep.py --grid w_chain=0,0.2,0.4 --gt data/labels/merged.csv`.

**Random absolute audit of v6.** We drew 24 frames at random (seeds 2026 and 777), excluding the
public window, and judged them blind. 21 calls were medium confidence or better, and v6 got 18 of
them right, including all 7 blackout frames. The 3 misses:

- 155,981: the Athletic ball carrier is not tracked.
- 27,163: a keeper holding the ball after play.
- 114,495: a contested midfield ball.

**Supervised fit.** We fitted a logistic regression (`tools/fit_emissions.py`) with
leave-one-segment-out cross-validation. The hand-set dead-ball weights reach AUC 0.926, against
0.739 for the fitted model. The fitted weight *ratios* still match the hand-set ones, which is
independent support for them.

**Why v6's public spellF1 is not higher than v4's.** The keeper fix turns a false 2.4 s Barcelona
spell into Athletic, which is right frame by frame (macroF1 rises). The merged Athletic spell then
loses its IoU match next to a 0.8 s GT blip. The two scores are inside each other's bootstrap
intervals.

---

## 7. Results

The public window is the first 5 minutes of play, with 17 possessions. Treat it as a correctness
check, not a ranking. The published full-match floor is `spellF1` 0.2433 (`baseline_smooth50`).

| version | public spellF1 [5–95%] | public macroF1 | dead-ball F1 | team acc. (both alive) | mean transition lag | full match: possessions / mean s / dead % |
|---|---|---|---|---|---|---|
| baseline_smooth50 (the floor) | 0.3256 [0.21, 0.45] | 0.6207 | 0.323 | 0.858 | +4.0 fr | 594 / 7.2 / — |
| possession_v1 | 0.5697 [0.41, 0.72] | 0.8466 | 0.812 | 0.915 | +9.5 fr | 249 / 12.4 / 51.5 |
| possession_v2 | 0.5375 [0.38, 0.70] | 0.8543 | 0.824 | 0.916 | +9.5 fr | 243 / 12.6 / 51.7 |
| possession_v4 | 0.5867 [0.41, 0.76] | 0.8820 | 0.878 | 0.918 | +5.0 fr | 236 / 12.9 / 52.0 |
| **possession_v6 (submitted)** | **0.5857 [0.37, 0.76]** | **0.8925** | **0.881** | **0.931** | **−2.6 fr** | 219 / 14.0 / 51.8 |

**Proxy checks for v6, over the full match:**

- pre-kick-off: 100% dead
- kick-offs: 6/6
- GOALS windows: 90.5% dead
- tagged restart teams: 17/24

The public window is unrepresentatively easy. The ball is visible on about 88% of its frames,
against 59% match-wide, so expect the full match to score lower.

---

## 8. Limitations

- **Spells under ~2 s are unreachable from proximity or contacts** (2 of 17 public possessions).
- **Untracked ball carriers, and duels the duel zone misses, read as the wrong possessor.** Without
  ball height, a clearance header looks like control.
- **Quick restarts under 5 s are merged away by the minimum stoppage** (a 2.6 s goal kick at
  37,321).
- **A keeper holding the ball after a save can look like a stoppage** (27,163). The only public
  instance of a keeper on the ball in the box is a genuine dead ball, so no rule was added.
- **Restarts the snapping can't see** (ball unseen at the kick) still come up to ~1 s late, and
  ~4 s late after short goal kicks.
- **The validation base is thin.** The public window has 17 possessions and is easier than the
  match. The video adjudication is AI-assisted and leaves out contested moments.

### What we would fix with more time

1. **A hand-labelled validation set.** 20–30 minutes labelled with `tools/label_tool.py` would give
   about 100 possessions. That is enough to tune the switch penalty, the minimum stoppage and the
   duel radius honestly. It would also let us train a learned model against possession-level
   errors rather than per-frame ones, which is where self-training failed.
2. **Ball detection on the video at restarts**, so that snapping also works when the tracking has
   lost the ball at the kick.
3. **Ball height or touch direction**, to tell a clearance header from control in crowded boxes.
4. **A minimum stoppage that depends on the restart type**, so quick goal kicks and free kicks
   survive.

---

## 9. Reproduction

**Requirements:** Python 3.11.9 (tested), plus the pinned `requirements-solution.txt`:
`numpy==2.4.6`, `pandas==2.2.3` and `av==18.1.0`. PyAV decodes the stadium audio and ships its own
ffmpeg. matplotlib is optional, for `tools/timeline.py`.

**Data:** the match data bundle, from enrolment. Unpack it into `./data`, or point `MTAH_DATA_DIR`
at it. The pipeline reads `data/tactical/`, including the mp4 for its audio. The data is licensed
and is **not** in the repository.

```bash
pip install -r requirements-solution.txt
python make_submission.py                      # -> evaluation/results/possession_v6.csv
python evaluation/validate_submission.py evaluation/results/possession_v6.csv
python evaluation/run_evaluation.py --bootstrap 1000
```

- **Deterministic, and reproducible from a cold start.** With every cache deleted, the run
  re-parses the tracking and re-decodes the stadium audio from the mp4. It takes 41 s and gives the
  byte-identical file: v6's SHA-256 starts `FE83E5E59564A764`.
- **Caches.** They go to `data/cache/`, which is gitignored with the rest of `data/`.
- **The uploaded file.** `python make_submission.py --name <team_name>` writes the same bytes to
  `evaluation/results/<team_name>.csv`.
- **Parameters.** `python make_submission.py --set key=value` overrides any parameter in
  `possession/evidence.py::Params`, for experiments. The submission uses the defaults.

**What the public repository leaves out:** the match data, the caches (including the audio
spectrogram), any hand labels, and every submission CSV except the organisers' four
`baseline_*.csv`. A submission is a full set of labels for the match, and a perfect one would be a
copy of the ground truth. `.gitignore` enforces all of this.

---

## 10. References

- Vidal-Codina, F., et al. (2022). Automatic event detection in football using tracking data.
  *Sports Engineering*. The source of the duel zone, and one source of the contact detector we
  tried.
- Link, D., & Hoernig, M. (2017). Individual ball possession in soccer. *PLOS ONE*. The source of
  the contact-based possession idea behind the last-touch evidence we tried.
