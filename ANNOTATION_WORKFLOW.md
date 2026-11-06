# Independent match annotations

The private Kaggle smoke run completed. On the same two public minutes, v6 scored
0.5818 spellF1 and the first learned model scored 0.2154. Its training portion had
only one 5.52-second stoppage. These public labels were already used during v6
development, so this is a pipeline check, not a blind comparison. We need broader
independent labels before changing models or tuning the learned decoder.

`tools/annotation_pack.py` creates a small offline browser pack. It contains no
new ground truth, video or model predictions. A person must review the video;
sampled stills cannot establish all control changes, foul whistles and restarts.

## Review the video locally

Build under an ignored or private directory:

```bash
python tools/annotation_pack.py build --out data/labels/review-pack
```

Open `data/labels/review-pack/review.html` in desktop Chrome or Edge. Choose your
original `BAR-ATH_tactical.mp4`; it stays on your computer. No server, installation
or upload is needed for the reviewer. The full original video is required because
the browser uses its zero-based frame index; a trimmed copy has different indices.

1. Start with the 2½-minute public-calibration window. Label it without viewing
   the supplied boundaries. Close it with X, reveal the public boundaries, and
   inspect each disagreement. Read the conventions in [tools/LABELLING.md](tools/LABELLING.md),
   especially whistles, goalkeeper catches and short possessions. Calibration
   annotations are kept separate and never enter the new training/validation/test
   exports.
2. Enter your name and confirm calibration. Watch each assigned window with sound.
   H = Barcelona, A = Athletic, D = dead, X = uncertain. Each key marks the start
   of a new segment at the displayed decoded frame. Add the label at the first
   core frame, and close with X at the exclusive stop frame. Mark uncertain spans
   X instead of guessing. Ten seconds of context on each side are for viewing.
3. Refine every boundary in a second pass, then confirm the window's review.
   Changing a boundary resets that window to draft. Save a JSON backup after each
   window; browser autosave is not a portable backup. Restore JSON to resume on a
   different browser. Do not edit the same session in multiple tabs.

Playback has sound and speed controls. Left/right step one frame; Shift steps
five; Ctrl steps 25. H/A/D/X mark boundaries, U undoes, and clicking an existing
boundary seeks to it. The browser reports the frame it decoded; labelling is
disabled while a seek is pending. For browsers without decoded-frame callbacks,
use the existing PyAV/Tk tool instead of estimating frames from playback time.

## Frozen split

Thirty minutes are selected across five chronological strata in each half using
seed 20261104, outside the public five minutes and the second-half pre-kickoff.
Selection used the timeline; model predictions did not guide it. There are six
training windows, two validation windows and two final-test windows. Core windows
are fixed at three minutes, so possessions crossing their edges are censored.
The duration estimator already excludes edge-cut segments. Keep these endpoints
fixed rather than changing them after seeing model performance.

| Window | Split | Core frames [start, stop) | Original-video time |
|---|---|---|---|
| First-half 1 | Final test | 24935–29435 | 16:37.40–19:37.40 |
| First-half 2 | Validation | 36748–41248 | 24:29.92–27:29.92 |
| First-half 3 | Training | 55960–60460 | 37:18.40–40:18.40 |
| First-half 4 | Training | 70260–74760 | 46:50.40–49:50.40 |
| First-half 5 | Training | 83218–87718 | 55:28.72–58:28.72 |
| Second-half 1 | Training | 122726–127226 | 81:49.04–84:49.04 |
| Second-half 2 | Final test | 133202–137702 | 88:48.08–91:48.08 |
| Second-half 3 | Training | 152232–156732 | 101:29.28–104:29.28 |
| Second-half 4 | Validation | 169253–173753 | 112:50.12–115:50.12 |
| Second-half 5 | Training | 182935–187435 | 121:57.40–124:57.40 |

These windows sample this one match, not unseen-match generalisation. Leave test
scores unopened until model choices are fixed. Test footage may be labelled blind;
test labels must not become training inputs or influence hyperparameters. Ideally
a second person independently labels a three-minute training window and the team
resolves disagreements about the annotation conventions before using validation.

## Export and train after review

Keep the downloaded JSON private. Export requires a review record for every
window, explicit start/end boundaries, calibration confirmation and H/A/D examples
in training. It checks the frozen plan fingerprint and rejects draft annotations.
X spans and all frames outside each split remain X. It writes three full-length
199,319-row CSVs and a coverage/provenance report with hashes.

```bash
python tools/annotation_pack.py export \
  --plan data/labels/review-pack/plan.json \
  --review /path/to/barca-review-yourname.json \
  --out data/labels/reviewed

python -m learned_possession train \
  --features /path/to/features.npz \
  --train-labels data/labels/reviewed/train.csv \
  --validation-labels data/labels/reviewed/validation.csv \
  --compare evaluation/results/possession_v6.csv \
  --out /path/to/run02
```

The browser's reviewed status is a person's declaration; software cannot verify
that the person watched the footage. Inspect the coverage report and resolve
systematic ambiguities before treating the files as reference labels. Report
excluded X coverage alongside evaluation scores; uncertainty must not silently
make a difficult evaluation look easier. The exporter leaves the public window
out of all three files. Add public labels to training only as a separate, explicit
experiment after deciding how the original public window will be used.

For Kaggle, upload the reviewed files as a private dataset and configure the
existing notebook's separate-label training mode with `train.csv` and
`validation.csv`. Keep `test.csv` out of the notebook until the final evaluation.
Generated labels, clips, stills and reports stay private and out of GitHub. The
final competition submission must still be generated by the prediction pipeline.

## An explicitly accepted partial draft

If a reviewer supplies only some windows and explicitly asks to use their draft
work, `tools/import_review_draft.py` supports a separate **provisional** experiment.
It does not relax the strict reviewed-label exporter. Draft status stays recorded;
no labels are invented before the first mark or after an unclosed last mark.
Only intervals bounded by two supplied marks are imported. Repeated marks of the
same label do not add possessions. The original JSON is kept unchanged.

Choose training and validation windows explicitly, without comparing model scores
to choose the allocation. For example:

```bash
python tools/import_review_draft.py \
  --plan data/labels/review-pack/plan.json \
  --review /path/to/review.json \
  --train-window train_h2_05 \
  --validation-window test_h1_01 \
  --include-public-training --allow-drafts \
  --out data/labels/provisional
```

This writes `train.csv`, `validation.csv` and `annotation_audit.json`. The audit
records the supplied review status, unknown coverage, original and allocated
splits, hashes, and calibration agreement against the already-public reference.
The calibration annotations are never treated as additional new training labels;
`--include-public-training` uses the supplied public ground truth instead.

Allocating a planned test window to validation means it can no longer serve as
an untouched final test. Scores on draft annotations are provisional, particularly
if public calibration reveals missed stoppages or ambiguous boundaries. Label
quality uncertainty is separate from the statistical uncertainty of a short
validation window. `notebooks/possession_draft_kaggle.ipynb` reproduces this bounded
import, verifies the hashes and reuses the previous private feature cache. It
records draft provenance in the saved model and report and runs one fixed model
configuration without a validation parameter search.

## Verification

Exporter tests check split separation, public-window exclusion, uncertain spans,
explicit closure, draft rejection and full-length frame numbering. A real-video
browser check exercises exact seeking across both halves, editing/undo,
review invalidation, autosave and JSON download. These verify the tooling, not the
correctness of labels that have not yet been made.
