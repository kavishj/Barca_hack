# Our solution — working log

The seed of the technical document. Data facts and the overall plan live in
[HACKATHON_BRIEF.md](HACKATHON_BRIEF.md).

## Reproduce

```bash
pip install -r requirements-solution.txt
python make_submission.py                      # -> evaluation/results/possession_v6.csv
python evaluation/validate_submission.py evaluation/results/possession_v6.csv
python evaluation/run_evaluation.py --bootstrap 1000
```

**Deterministic, and reproducible from a cold start.** With every cache deleted, the run re-parses
the tracking and re-decodes the stadium audio from the mp4. It takes 41 s and gives the
byte-identical file: v6 SHA-256 starts `FE83E5E59564A764`. Caches go to `data/cache/`, which is
gitignored with the rest of `data/`.

## Pipeline (v6)

| module | role |
|---|---|
| `possession/data.py` | loads the tracking (in metres), homography, Smart Tagging and public GT on the video-frame index. **`Match.team_by_frame`** gives each goalkeeper slot the team of the goal it guards (see "Data defect" below) |
| `possession/audio.py` | turns the tactical mp4's stadium audio into a per-video-frame spectrogram (48 kHz / 25 fps = 1920 samples per frame) and gives a whistle-tone strength per frame |
| `possession/signals.py` | physical per-frame signals: ball visible/speed/out of bounds/placed at a restart spot, blackout length, outfield activity, nearest home/away distance, team width, receptions, ball contacts, camera centre, Smart Tagging priors, pre-kick-off mask, whistle |
| `possession/evidence.py` | turns signals into per-frame scores for 4 hidden states, and holds every parameter (`Params`) |
| `possession/decode.py` | offline Viterbi with a **minimum stoppage length**: each dead state is a chain of k sub-states. Checked against a brute-force dense decoder: the exact optimum on 60/60 random cases |
| `possession/pipeline.py` | decodes each half separately, snaps restarts to the kick that restarted play, writes X outside the halves |
| `make_submission.py` | CLI; `--set key=value` overrides any parameter |

**Hidden states:** `H`, `A`, `DH`, `DA`. The two dead states carry the team that will restart, so the
ball goes to the right side when play resumes. Both are output as `D`. A stoppage lasts at least 5 s.

**Dead vs alive evidence** (a log-ratio):
- outfield activity: log-normal around 1.2 m/s when dead and 2.3 m/s when alive
- inside a ball blackout longer than 2 s
- ball out of bounds
- ball still in the goal area, at a corner or on the centre spot
- ball still on a touchline, but only while players walk (activity below 1.4 m/s)
- referee's whistle that stops play (players busy before it, slowing after it)
- minus: ball travelling faster than 10 m/s
- minus: somebody has the ball at their feet
- forced dead before each half's kick-off

**Team evidence:** a Gaussian control likelihood of each team's nearest player (σ = 1.5 m),
weighted down when the ball is in flight (above ~8 m/s). It is set to zero inside a **duel zone**,
where both teams have a player within 2 m of the ball. The team evidence only penalises the
disfavoured side of a pair, so it never biases dead vs alive.

**Possession rule:** "a loose ball stays with the last team" is built into the transition penalty,
and "a contested ball stays with the last team" into the duel zone. Possession switches only when
the other team accumulates enough uncontested control evidence to pay the penalty.

**Restarts:** after decoding, each dead → alive transition moves back to the kick that restarted
play, when one happened within the 40 frames before it. A kick is the ball going faster than 5 m/s
after a second in which it was slower than 2.5 m/s or unseen.

**Smart Tagging:** set pieces with a team (corner, goal kick, free kick) assert the restarting team
over the first 20 s of their window. The kick-off tags locate each half's kick-off.

## Data defect: goalkeeper slots carry the wrong team

The tracking has two goalkeeper slots (24 "Barcelona", 46 "Athletic"). Their metadata team is not
reliable:
- **Slot 46** stands inside **Barcelona's** penalty area on **15,945 graded frames** (11% of its
  life). That includes the last 230 s of the match and the first 146 s of the second half.
- **Slot 24** stands inside Athletic's penalty area on 1,501 frames.

A keeper does not spend four minutes in the opposing box, so the tracker's keeper slots follow
whichever keeper they re-acquire. Every time a keeper had the ball in those stretches, the evidence
credited the wrong team: goal kicks, back-passes, catches.

**Fix:** a goalkeeper slot takes, per frame, the team that defends the goal in its half of the
pitch. Outfield slots were checked and are fine. Both teams show occasional over-counts from track
fragmentation, and referees are not tracked.

## How we validated without a leaderboard

The public window has 17 possessions, and the repo shows it choosing the wrong baseline. So every
decision was checked four ways, and never by public spellF1 alone:

1. **Public window, with 1,000-draw bootstrap intervals.** Used as a correctness check.
2. **Full-match statistics:** possession count and mean length. The public window's rate implies
   roughly 210–270 possessions.
3. **Label-free proxy checks** (`tools/proxy.py`). These include one hard fact: frames
   110,808–113,741 are dead (FAQ.md). The others are kick-off teams, the dead share of GOALS
   windows, and set-piece restart teams.
4. **Video adjudication** (`tools/disagree.py` + `tools/review.py`):
   - Variants are compared only where they disagree.
   - Each stretch is judged blind, from a 2×2 montage of video frames with the tracking overlay.
   - **Seeded random samples** are judged the same way, so the sample is not cherry-picked.
   - Calls were made by Claude (an LLM) looking at the frames. That is AI-assisted labelling, used
     for validation only and never as model input.
   - Calibration on the public window showed:
     - possession is reliable when the ball is tracked
     - stoppages are reliable to about ±1–2 s
     - contested moments are not reliable; those were excluded rather than guessed

## Training and tuning

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

Unbiased check of round 2: a random sample of 8 v2/v4 disagreements. v4 was better in 6, v2 in 1,
and 1 was unclear.

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

**Unbiased check of round 3:** a seeded random sample of 8 v4/v6 disagreements, judged on video:
**v6 right in 8 of 8.** That breaks down as 3 keeper-team fixes, 3 snapped restarts (throw-ins and a
quick restart), 1 duel-zone case and 1 kick-off.

### Round 4: gated contacts (v6 kept)

The idea: raw last-touch evidence failed because it flips possession on single touches. So count a
contact only when it shows control: no opponent within 2 m (the duel zone), and followed by
sustained control. The target was the wrong-possessor misses (155,981, 114,495, clearances).

| change | source | public spellF1 / macroF1 | full match | video evidence | decision |
|---|---|---|---|---|---|
| **team-control chains**: two consecutive uncontested contacts by the same team, ≤ 3 s apart, put that team in control between them (`w_chain`, w 0.2–0.8) | Link & Hoernig 2017: team possession passes at the other team's next *controlled* action | 0.5857 / 0.8925 at every weight and gap (free radius 2 m); 1.5 m adds a wrong spell (0.5333); 3 m changes nothing | 217–224 (raw last touch: 252) | the gate is precise: uncontested contacts name the GT team 93% (n=60), contested 64% (n=11), chained 37/37. But at w 0.4 only 2 full-match stretches move: 1 plausibly right, 1 plausibly wrong | kept off (redundant) |
| "held" gate: the contact's team within 2 m for ≥ 90% of the next second | Link & Hoernig 2017 (control = duration, opponents near) | 0.5857 / 0.8947 at w 0.1; 0.5333 from w 0.2 | 205–225 | carried 2 s, it fixes 32 public frames where v6 is wrong and breaks 48. A held contact at 12,434 is an Athletic touch inside a Barcelona possession | rejected |
| confirmation veto: absorb possessions < 3 s with no controlled (held or chained) contact | count arithmetic | 0.6370 / 0.8995 | 197 (plain 3 s minimum: 182) | the public gain is one false spell (12,786). All 12 removals outside the public window judged blind: 2 right, **7 wrong**, 3 unclear. The deleted spells are mostly real possessions won under pressure or off an aerial ball, where every contact is contested | rejected |

**Why it can't help much:** an uncontested contact followed by sustained control is exactly where
proximity already names the right team. The misses it was meant for can't produce one: an untracked
carrier has no tracked player at the ball, a contested ball is contested by definition, and a
sub-2 s spell has no time to show control. Gated contacts don't mark the GT's switch moment either:
the nearest one ranges from 28 frames before to 67 frames after a public GT switch. Proximity is as
close or closer.

**Re-test against hand labels:** `python tools/sweep.py --grid w_chain=0,0.2,0.4 --gt data/labels/merged.csv`.

**Supervised fit** (`tools/fit_emissions.py`, logistic regression, leave-one-segment-out
cross-validation): the hand-set dead-ball weights reach AUC 0.926 against 0.739 for the fitted model.
The fitted weight *ratios* still match the hand ones.

## Results

Public window = the first 5 minutes of play: 17 possessions. Treat it as a correctness check, not a
ranking.

| version | public spellF1 [5–95%] | public macroF1 | dead-ball F1 | team acc. (both alive) | mean transition lag | full match: possessions / mean s / dead % |
|---|---|---|---|---|---|---|
| baseline_smooth50 (the floor) | 0.3256 [0.21, 0.45] | 0.6207 | 0.323 | 0.858 | +4.0 fr | 594 / 7.2 / — |
| possession_v1 | 0.5697 [0.41, 0.72] | 0.8466 | 0.812 | 0.915 | +9.5 fr | 249 / 12.4 / 51.5 |
| possession_v2 | 0.5375 [0.38, 0.70] | 0.8543 | 0.824 | 0.916 | +9.5 fr | 243 / 12.6 / 51.7 |
| possession_v4 | 0.5867 [0.41, 0.76] | 0.8820 | 0.878 | 0.918 | +5.0 fr | 236 / 12.9 / 52.0 |
| **possession_v6** | **0.5857 [0.37, 0.76]** | **0.8925** | **0.881** | **0.931** | **−2.6 fr** | 219 / 14.0 / 51.8 |

**Proxy checks for v6:**
- pre-kick-off: 100% dead
- kick-offs: 6/6
- GOALS windows: 90.5% dead
- tagged restart teams: 17/24

**Random absolute audit:** 24 frames drawn at random (seeds 2026 and 777), excluding the public
window, judged blind. 21 calls were medium confidence or better, and v6 got 18 of them right,
including all 7 blackout frames. The 3 misses:
- 155,981: the Athletic ball carrier is not tracked.
- 27,163: a keeper holding the ball after play.
- 114,495: a contested midfield ball.

**Why v6's public spellF1 is not higher than v4's:** the keeper fix turns a false 2.4 s Barcelona
spell into Athletic, which is right frame by frame (macroF1 rises). The merged Athletic spell then
loses its IoU match next to a 0.8 s GT blip. The two are inside each other's bootstrap interval.

## Known limitations

- **Spells under ~2 s are unreachable from proximity or contacts** (2 of 17 public possessions).
- **Untracked ball carriers, and duels the duel zone misses, read as the wrong possessor.** Without
  ball height, a clearance header looks like control.
- **Quick restarts under 5 s are merged away by the minimum stoppage** (a 2.6 s goal kick at
  37,321).
- **A keeper holding the ball after a save can look like a stoppage** (27,163). The only public
  instance of a keeper on the ball in the box is a genuine dead ball, so no rule was added.
- **Restarts the snapping can't see** (ball unseen at the kick) still come up to ~1 s late, and ~4 s
  late after short goal kicks.

## Readiness for submission

| item | status |
|---|---|
| scored file, validated, deterministic, reproducible from scratch | done: `possession_v6.csv`, SHA-256 `FE83E5E5…`, byte-identical on Windows and Linux. Uploaded as `<team_name>.csv`, written by `python make_submission.py --name <team_name>` |
| pinned dependencies, one-command regeneration | done: `requirements-solution.txt` (numpy, pandas, av), `python make_submission.py` |
| public code repository | data-free set staged for v6; commit and push to `<repo_url>` on the team's go-ahead |
| technical document | done: `TECHNICAL.md`, up to date for v6 (rounds 1–4, v6 numbers). Uploaded as `<team_name>.pdf` |
| hand-labelled validation set | tool ready (`tools/label_tool.py`), no labels yet. Optional |
| team of 3–5 registered, Google account for the form | the team needs to confirm |
