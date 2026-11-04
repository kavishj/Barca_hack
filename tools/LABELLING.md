# Hand-labelling possession

`tools/label_tool.py` is a frame viewer for labelling who has the ball on the tactical video. The
labels are our **validation set**: the public ground truth has only 17 possessions, which is too few
to choose between models. The aim is 20–30 more minutes, about 100+ possessions.

RULES.md allows hand labels for training and validation, but **never in a submission**: the
submitted file must come out of the pipeline. Labels live in `data/labels/`, and `data/` is
gitignored, so they stay out of the public repo. Don't copy them anywhere else.

## Setup

```bash
pip install av
```

That installs PyAV (tested with 18.1.0), which brings its own ffmpeg and leaves numpy alone. Don't
try to fix `cv2` instead: the installed opencv is built for numpy 1, and the pipeline is pinned to
numpy 2.4.6. Tkinter and Pillow come with the standard Python install.

## Run

```bash
python tools/label_tool.py --labeller alice                  # opens where alice last stopped
python tools/label_tool.py --labeller alice --frame 40000    # or at a given frame
```

Startup takes about 2 s, most of it loading the tracking for the overlay (`--no-overlay` skips it).
The window has three parts:

- **The picture**, with the tracking drawn on it: **blue rings = Barcelona (H)**, **red rings =
  Athletic (A)**, yellow dot = the tracked ball with its last 10 frames as a trail. `P` adds the
  pitch markings projected through the homography; if they sit on the painted lines, the overlay
  is aligned. The tracked ball is projected onto the ground, so a ball in the air is drawn below
  the real one.
- **The timeline strip**: your labels ±10 s around the current frame (`T` zooms out to 30 s, 2 min
  and 10 min), with a white tick at each segment start. `--compare file.csv ...` adds a row for
  each CSV you give it (another labeller's export, a submission, or the public GT).
- **The status line**: the frame number, the label in effect, the video time and the time into
  the half, and the segment you are in. The tool has no sound. To hear a whistle, open the mp4 in
  a normal player at the video time shown.

### Keys

| key | does |
|---|---|
| `←` / `→` | −1 / +1 frame (`Shift` ±5, `Ctrl` ±25 = 1 s) |
| `↓` / `↑` | −250 / +250 frames (10 s) |
| `Space` | play / pause; `-` / `+` change speed (0.25× to 4×) |
| `G` | go to a frame: type it (or `+250`, `-75`) and press Enter |
| `[` / `]` | jump to the previous / next segment start |
| `H` `A` `D` | a segment of that label **starts at this frame** |
| `X` | unlabelled from this frame on. This is how you **close a window** |
| `Backspace` / `Delete` | remove the start of the segment you are in (it merges into the one before) |
| `Ctrl+Z` or `U` | undo; the view jumps back to the frame that changed |
| `O` / `P` | toggle the tracking overlay / pitch lines |
| `Ctrl+E` | export the CSV now |
| `?` or `F1` | show the keys; `Esc` pauses and hides them |
| `Q` | quit |

Stepping costs about 20 ms per frame, a jump about 90 ms, and playback keeps real time at every
speed. Stepping backwards is cheap too, because recently shown frames are cached.

## How labels are stored

`data/labels/<labeller>.json` holds the segment starts as `[frame, label]` pairs. Each label lasts
until the next start. Frames before the first start are unlabelled (X). The file is rewritten
after every edit, so there is no save key and nothing is lost if the tool crashes. It also
remembers where you stopped. The tool tidies the list as you go: a start that repeats the label
already in effect is dropped. Only one window should edit a given labeller's file at a time.

A labelled window looks like this:

```
[40000, "D"], [40112, "H"], [40480, "A"], ..., [44500, "X"]
```

- **Close every window with `X`** on the first frame after it. An unclosed last segment is left
  out of the export (and the export tells you so). Otherwise the tool would have to guess where
  you stopped watching.
- A segment never runs past the end of the half it is in, and frames outside the two graded
  halves (11185–90651 and 110808–190197) are always X in the export. You can start a window a
  little before the half: a D that starts in the half-time gap is cut to start at 110808.

## Export and scoring

```bash
python tools/label_tool.py --labeller alice --export                         # data/labels/alice.csv
python tools/label_tool.py --labeller alice --export --window 40000 44499    # alice_40000-44499.csv
python tools/label_tool.py --export --merge alice bob carol                  # data/labels/merged.csv
```

The export is a full-length 199,319-row `frame,label` CSV, with X outside the labelled spans. It
is read back with the scorer's own `read_label_csv` / `validate_submission` / `label_codes` before
the tool reports success. The tool then prints each window with its H/A/D frame counts and the
number of possessions. `--merge` combines several labellers into one file. Where their windows
overlap, the first name listed wins, and the tool prints how often they agree on the overlap.

Score the model against hand labels exactly as against the public GT:

```bash
python tools/spells.py evaluation/results/possession_v1.csv --gt data/labels/merged.csv
python evaluation/run_evaluation.py --gt data/labels/merged.csv --bootstrap 1000 --out data/labels/eval
python tools/sweep.py --grid switch=8,12,16 --gt data/labels/merged.csv
```

Give `run_evaluation.py` an `--out` path. Otherwise it overwrites the public-GT scoreboard in
`evaluation/`.

## Labelling conventions

Copy the public GT's conventions exactly, since that is what the private GT will look like. From
section 6 of `HACKATHON_BRIEF.md`:

- **D starts when the ball crosses a line or the whistle goes, and ends at the restart touch.**
  The restart frame is the first frame of the new H/A segment.
- **A loose ball stays with the last team.** Possession changes only when the other team gets
  control, so a ball in flight, a bobble or a 50/50 doesn't change it.
- **Brief touches by the other team can create very short spells.** The public GT has spells of
  0.76 s and 1.32 s. Label a touch as a new spell only when the other team actually played the
  ball, not when it just brushed past someone.
- **The kick-off taker's team owns the ball from the kick-off frame.** The first graded frame,
  11185, is already A, because Athletic kick off.
- The second half is graded from 110808 but kicks off at 113742. Frames 110808–113741 are D.

What the public GT does around one stoppage (frames 17300–18100; see the spot check below):

- **A foul is D from the whistle.** Here the tackle happens at about 17318, but the GT's D starts
  at 17380, around when the referee stops play, not at the tackle.
- **A set piece that goes straight out is not a possession.** Athletic took a quick free kick at
  about 17428, and it went over the bar at about 17460. The GT keeps all of 17380–17837 as one D.
- **If play restarts with a set piece, there was a stoppage.** Before you mark a long H or A, play
  on 5–10 s. If the ball is then placed for a free kick, or players stop and walk, go back and find
  the whistle. The referee's outstretched arm and players stopping are the visual cues. For the
  sound, use a video player at the video time shown in the status line.

These come from one stretch of the match. If you find a public-GT case that contradicts them, note
it here.

## Workflow

1. **Calibrate on the public window first.** Label 2–3 minutes inside 11185–18684 without
   `--compare`, export, then run
   `python tools/spells.py data/labels/<you>.csv --gt evaluation/gt/BAR-ATH_possession_gt_public.csv`.
   It lists, possession by possession, how far your starts and ends are from the GT's. Then
   reopen the stretch with `--compare evaluation/gt/BAR-ATH_possession_gt_public.csv` and look at
   every boundary that differs. Do this before labelling anything new, and don't use the public
   window as validation data, because it is already GT.
2. **Choose windows** of 2–5 minutes, spread across both halves, and include busy stretches with
   set pieces. Agree the list in the team so that nobody labels the same stretch twice (except for
   the agreement window). **Start and end each window in a stoppage**, so that no possession is
   cut in half.
3. **First pass at 1×**: press `H`/`A`/`D` roughly as things happen. **Second pass**: hop between
   starts with `[` / `]` and move each one to the exact frame with the arrows. Remove a misplaced
   start with `Backspace` and re-mark it on the right frame.
4. **Agreement**: two people label the same 3 minutes without looking at each other's labels.
   Then:
   ```bash
   python tools/label_tool.py --labeller alice --export --window 60000 64499
   python tools/label_tool.py --labeller bob --export --window 60000 64499
   python tools/spells.py data/labels/bob_60000-64499.csv --gt data/labels/alice_60000-64499.csv
   ```
   The spellF1 between the two of you is the ceiling any model can reach on our labels. The
   per-possession offsets show where the conventions are still unclear.

## Frame accuracy

Row i of a submission is frame i of the video, so the tool must show exactly frame i. This was
checked on 2026-10-20:

- The video stream has 199,319 packets. The pts of frame i is exactly i × 1000 (time base 1/25000),
  with no B-frames and a keyframe every 50 frames.
- Decoding the whole file in order gave 199,319 frames, each with pts = index × 1000. 81 probe
  frames, fetched by random seeks in random order, matched the sequential decode byte for byte.
- At run time, every decoded frame's pts is checked against the frame number asked for. Any
  mismatch raises an error, so the tool cannot silently show the wrong frame.
- The tracking overlay lands on the players and the pitch lines on the painted lines, so the
  tracking, homography and video share one frame index.

`python tools/label_tool.py --snapshot 11185` writes one full-size frame with the overlay to
`data/labels/frame_11185.png`.

## Spot check against the public GT (2026-10-20)

Frames 17300–18100 were labelled by eye in the tool, without looking at the GT, then exported and
compared:

| event | labelled at | public GT | difference |
|---|---|---|---|
| keeper claims Athletic's cross | 18029 (A→H) | 18027 (A→D) | 2 frames; GT calls it a stoppage (the ball is then placed for a free kick) |
| corner kick restart (D→A) | 17825, the frame the foot strikes the ball | 17838 | GT is 13 frames (0.5 s) later |
| ball dead | 17460, the ball over the bar | 17380 | missed the foul whistle 3.2 s earlier, plus a quick free kick in between |

The tool and its export line up with the GT's frame index. Where the same event was labelled, the
boundaries land within 2 and 13 frames. The other differences are judgement calls: a missed
whistle, a stoppage read as a keeper's possession, and a window started without checking who had
the ball (Barcelona from 17293, not Athletic). The export scored cleanly with
`run_evaluation.py --gt` and `spells.py --gt`.

Two lessons follow. The whistle conventions above matter more than anything else in labelling.
And the 0.5 s restart offset is worth checking on more public-GT restarts. If the GT always marks
restarts late, the team should know before tuning boundary snapping on our labels.
