# `<team_name>`: ball possession for More than a Hack 2027

Current graph/audio and external-pretraining experiments are summarized in
[EXPERIMENT_RESULTS.md](EXPERIMENT_RESULTS.md), with protocols in
[F1_TARGET.md](F1_TARGET.md). The best observed draft-validation spellF1 is
0.7200; the 0.80 ranking target remains open. The entry and reproduction
instructions below describe the original v6 submission.

This is our entry for **More than a Hack 2027, Phase 1: Ball Possession** (Metrica Sports, FC
Barcelona and Mobile World Capital Barcelona). For each of the 199,319 frames of the BAR–ATH
tactical video, it labels who has the ball:

- `H`: Barcelona in possession, ball alive
- `A`: Athletic in possession, ball alive
- `D`: dead ball
- `X`: outside the two halves, not graded

The model is a **rule-based 4-state Viterbi with a minimum stoppage length**.

- **Evidence:** the tracking data, and the referee's whistle, which is detected in the stadium
  audio.
- **Possession rules:** a contested ball (a "duel zone") does not change possession.
- **Fixes on top:** a correction for a defect in the tracking's goalkeeper teams, and restarts
  snapped to the kick after decoding.

It has no learned components, and no hand labels or ground truth among its inputs. Running this
repository against the provided data regenerates our submitted file **byte for byte**.

- **How it works, and why:** [TECHNICAL.md](TECHNICAL.md), our technical document.
- **The full working log:** [SOLUTION.md](SOLUTION.md).
- **Measured facts about the data:** [HACKATHON_BRIEF.md](HACKATHON_BRIEF.md).
- **The challenge itself:** the organisers' original README, kept as
  [README_CHALLENGE.md](README_CHALLENGE.md). The organisers' scorer, starter code and challenge
  documents are unchanged.

| on the public 5-minute window (17 possessions) | spellF1 [5–95%] | macroF1 |
|---|---|---|
| `baseline_smooth50` (the published floor) | 0.3256 [0.21, 0.45] | 0.6207 |
| **`possession_v6` (submitted)** | **0.5857 [0.37, 0.76]** | **0.8925** |

The public window is a correctness check, not a ranking. The competition is scored on the full
match, against a private ground truth.

---

## Reproduce the submission

**1. Get the data.** The match assets are licensed to enrolled participants and are **not** in this
repository; see [DATA.md](DATA.md). Unpack the bundle into `./data`, keeping its names, or point
`MTAH_DATA_DIR` at wherever you put it. The pipeline needs only `data/tactical/`, including the mp4
for its audio.

**2. Install.** Tested on Python 3.11.9:

```bash
pip install -r requirements-solution.txt
```

This installs numpy, pandas and PyAV (`av`). PyAV decodes the stadium audio and ships its own
ffmpeg, so there is nothing else to install.

**3. Generate.**

```bash
python make_submission.py
```

This writes `evaluation/results/possession_v6.csv`. From a cold start, with no caches, the run
parses the tracking and decodes the stadium audio from the mp4 in about 41 s. Both caches go to
`data/cache/`, which stays out of git with the rest of `data/`. The output is deterministic, and
its SHA-256 starts `fe83e5e59564a764`:

```bash
python -c "import hashlib; print(hashlib.sha256(open('evaluation/results/possession_v6.csv','rb').read()).hexdigest())"
```

To write the same bytes under the upload name, run
`python make_submission.py --name <team_name>`, which creates `evaluation/results/<team_name>.csv`.

For experiments, `python make_submission.py --name try --set switch=15` overrides any parameter
in `possession/evidence.py::Params`. The submission uses the defaults.

## Validate

```bash
python evaluation/validate_submission.py evaluation/results/possession_v6.csv
python evaluation/run_evaluation.py --bootstrap 1000
python tools/proxy.py evaluation/results/possession_v6.csv
```

1. `validate_submission.py` runs exactly the format checks the official scorer runs.
2. `run_evaluation.py` is the organisers' scorer. It scores every CSV in `evaluation/results/`
   against the public 5-minute ground truth, with 1,000-draw bootstrap intervals. It writes
   `evaluation/scoreboard.csv` and `evaluation/reports/`, both gitignored.
3. `tools/proxy.py` runs label-free checks over the **full** match. One of them is a published
   fact: frames 110,808–113,741 are dead. The others check the kick-off teams, the dead share of
   GOALS windows, and the set-piece restart teams.

[TECHNICAL.md](TECHNICAL.md) explains how these checks, plus blind video adjudication, replace the
missing leaderboard.

---

## Layout

```
make_submission.py      the one command that regenerates the submission
possession/             the model
tools/                  analysis and validation tools; none is needed to regenerate the file
evaluation/             the organisers' scorer, the public ground truth and the four baselines (unchanged)
starter/, games.py      the organisers' starter code and per-match constants (unchanged)
data/                   the match data goes here; nothing in it is committed
TECHNICAL.md            the technical document
SOLUTION.md             the working log: every tuning decision and the evidence for it
HACKATHON_BRIEF.md      data facts measured on the bundle
README_CHALLENGE.md     the organisers' original README
```

### `possession/`: the model

| module | role |
|---|---|
| `data.py` | loads the tactical tracking (in metres), homography, Smart Tagging and public ground truth on the video-frame index, and caches the parsed tracking in `data/cache/`. `Match.team_by_frame` gives each goalkeeper slot the team of the goal it guards, because the tracker's keeper slots carry the wrong team (slot 46 "Athletic" sits in Barcelona's box on 11% of its frames; see [TECHNICAL.md](TECHNICAL.md)) |
| `audio.py` | turns the tactical mp4's stadium audio into a per-video-frame spectrogram and gives a whistle-tone strength per frame. The spectrogram is cached in `data/cache/` |
| `signals.py` | per-frame physical signals, all taken from the data with no labels involved. Covers the ball (visible, speed, out of bounds, placed at a restart spot), blackout length, outfield activity, nearest home and away distance, team width, receptions, ball contacts, camera centre, Smart Tagging priors, the pre-kick-off mask and the whistle |
| `evidence.py` | turns the signals into per-frame scores for 4 hidden states (`H`, `A`, `DH`, `DA`), including the 2 m duel zone in which a contested ball yields no control evidence. `Params` holds every parameter and transition penalty |
| `decode.py` | offline Viterbi decoder, deterministic, with no causal lag. It enforces the 5 s minimum stoppage by chaining dead states |
| `pipeline.py` | decodes each half separately, snaps each restart back to the kick that restarted play (within 40 frames), and writes `X` outside the halves. This is the function `make_submission.py` calls |

### `tools/`: analysis and validation

| tool | what it does |
|---|---|
| `proxy.py` | label-free full-match sanity checks of one or more submissions |
| `sweep.py` | runs parameter grids against a ground truth, sharing one data load. Prints the scores and each variant's full-match statistics (possession count, mean length, dead share) |
| `spells.py` | possession-by-possession error analysis against a ground truth: the IoU of each possession, how far its edges are off, and unmatched spells |
| `disagree.py` | lists the stretches where two submissions disagree, longest first or as a seeded random sample: what to look at on video |
| `review.py` | renders a 2×2 montage of video frames with the tracking overlaid and each submission's label, for blind adjudication |
| `fit_emissions.py` | fits the evidence weights by logistic regression, with leave-one-segment-out cross-validation, against the hand-set ones |
| `timeline.py` | plots ground truth, a submission and the underlying evidence over a frame range |
| `label_tool.py` | frame viewer for hand-labelling a validation set on the tactical video, with keepers coloured by the goal they guard; see [tools/LABELLING.md](tools/LABELLING.md). Labels are saved in `data/labels/`, which is gitignored, and never feed the submission |

None of the tools is needed to regenerate the file. Some need optional extras beyond the pinned
requirements, listed in [requirements-solution.txt](requirements-solution.txt):

- `timeline.py`: `matplotlib==3.10.1`
- `fit_emissions.py`: `scikit-learn==1.7.1`
- `label_tool.py` and `review.py`: `Pillow==9.5.0`

---

## What is deliberately not in this repository

`.gitignore` enforces all of the following:

- **The match data** (`data/`, apart from its README): video, tracking, homography and Smart
  Tagging. It also covers the caches in `data/cache/` (the parsed tracking and the ~409 MB audio
  spectrogram) and any hand labels in `data/labels/`. The data is licensed for the competition
  only and may not be redistributed.
- **Submission files.** Every CSV in `evaluation/results/` is excluded except the organisers' four
  `baseline_*.csv`, which are computed from the tracking alone. A submission is a full set of
  labels for the match, and a perfect one would be a copy of the private ground truth. The rule
  also covers stray CSVs anywhere else, such as a `<team_name>.csv` at the root.
- **Evaluation output:** `evaluation/reports/` and `evaluation/scoreboard.csv`.
- **Video frames:** `*.mp4`, and image files, which is what `review.py` and `label_tool.py`
  produce.

## Rules compliance

- **The submitted labels come only from the algorithm.** The file is the unedited output of
  `python make_submission.py`.
- **Video adjudication was AI-assisted** (Claude, an LLM, judging rendered video frames). It was
  used for validation only and was never model input. The stadium audio *is* model input, via the
  automatic whistle detector.
- **No private ground truth was used.** The public 5-minute window is used for evaluation and
  calibration only, and the pipeline that writes the submission never reads it.

## Licence

[LICENSE](LICENSE) is the organisers' MIT licence, inherited from the starter repository. As it
states, it does not cover the match data.
