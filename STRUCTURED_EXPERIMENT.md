# Expanded labels and structured possession experiment

This experiment keeps the original TCN implementation unchanged as a baseline,
uses the newly supplied cumulative review snapshot, and tests separately:

1. Factorised live/dead and home/away-given-live heads with stop/restart/turnover
   event heads, typed segment transitions and frame-level boundary refinement.
2. Additional contact/reception history, nearest-player relative/radial velocity,
   ball acceleration, elapsed blackout and fine audio observations.
3. Eight dilated residual blocks rather than six, increasing the receptive field
   from 10.12 to 40.84 seconds at the same 32-channel width.

The structured configurations are factorized_short_base, factorized_short_rich
and factorized_long_rich. Each averages all three fixed seeds 0, 1 and 2, trained
for 60 fixed epochs. The expanded original baseline retains all three original
audio ablations and settings. No validation checkpoint selection or parameter
search is performed. All model/configuration choices precede real-data scoring.

The state loss is the factorised three-class likelihood: live/dead BCE plus
home/away BCE only on live frames, weighted by the live-frame fraction. Event
heads use separate weighted BCE targets for known label changes, with the
positive-weight odds shift removed at inference. Unknown annotation edges do
not create event targets. D labels are never called whistle-presence labels.

The existing duration-aware segment decoder is extended with source/destination
dependent event costs. The first pass searches grid/state-change/event candidates;
the second jointly searches every video frame within one second of its boundaries
with the coarse state sequence fixed. Its chain calculation scales with boundary
count times neighbourhood size squared. The first-pass solution remains feasible
in the second pass. No hard minimum spell
duration is introduced. An exhaustive small-sequence test checks the full objective.

README.md supplies the independent fact that frames 110808–113741 are dead before
the second-half kickoff. That exact interval is enforced automatically in the
structured decoder. Original baseline outputs are retained; separate constrained
baseline CSVs apply the same documented rule. It does not use validation labels.
The older tag-based feature ends six frames later and is retained for baseline
reproducibility. This experiment does not manually edit predictions or references.

Fine audio uses the existing timestamp-preserving PyAV 32 kHz decoder. Forty-ms
spectral windows every 20 ms yield broad 2–6 kHz tonality, peak frequency, positive
spectral flux, RMS and tonal rise, aggregated to 25 fps. Missing audio is explicit.
Spectral differences reset at each half. These are acoustic observations, not
calibrated referee probabilities. The frozen PANNs embeddings are reused unchanged.

Contact/reception history expires after 30 seconds and resets at halftime.
Slot reacquisition jumps faster than 15 m/s are rejected for player velocity.
All physical features use observations, not previous model predictions or labels.
Raw source arrays, audio and caches remain private.

Training uses 21,787 known frames, including the supplied public five minutes and
four bounded draft training windows. Validation remains byte-identical to run 02:
4,078 frames, nine live possessions and one bounded D interval. Draft status,
unknown edges and original source files remain unchanged. The calibration's
known 138-frame missed D interval is recorded, not silently relabelled; the
actual public training region already uses the original supplied reference.

PCA and all scalers fit training frames only. A conservative 60-second purge
covers the new contact history and long-context encoder. Five diagnostic folds
hold out one existing labelled training window at a time, comparing seed-0
original learned-audio TCN with the short rich structured model. Each fold fits
its own scaler, PCA, duration preferences and models; no fold selects settings.
Feature definitions were developed using the public window, so these folds are
diagnostic reuse of this match, not independent unseen-match validation.

All CV and structured predictions finish before the original baseline comparison
opens validation labels for scoring. Report official spellF1, macro F1, D F1,
predicted spell counts, typed transition timing and known-dead violations. Preserve
all ablations and scores. A 0.98 claim requires evidence; no such result is assumed.

The commands are:

```bash
python -m structured_possession prepare --features /private/features.npz \
  --data-dir /private/data --video /private/data/tactical/BAR-ATH_tactical.mp4 \
  --out /private/rich_features.npz
python -m structured_possession compare --features /private/features.npz \
  --rich-features /private/rich_features.npz --audio /private/audio_cache.npz \
  --train-labels /private/train.csv --validation-labels /private/validation.csv \
  --audit /private/annotation_audit.json --out /private/study --cross-validate
python -m structured_possession predict --features /private/rich_features.npz \
  --audio /private/audio_cache.npz --model /private/study/factorized_short_rich \
  --out /private/predictions.csv
```

Prediction reloads checksummed preprocessing/checkpoints and reads no labels.
Visual crops are a subsequent conditional experiment if controlled diagnostics
show unresolved control ambiguity; no new visual encoder is part of this study.

## Completed results

| Method | spellF1 | Macro F1 | D F1 | Predicted / 9 possessions |
|---|---:|---:|---:|---:|
| Teammate v6 | 0.4842 | 0.4984 | 0.4158 | 10 |
| Previous learned_audio TCN | 0.6182 | 0.7239 | 0.5960 | 13 |
| Previous tracking TCN | 0.6118 | 0.6474 | 0.4517 | 8 |
| Expanded original TCN, tracking | 0.5412 | 0.5901 | 0.4387 | 8 |
| Expanded original TCN, heuristic audio | 0.5176 | 0.5911 | 0.4440 | 8 |
| Expanded original TCN, learned audio | 0.5176 | 0.6137 | 0.4584 | 8 |
| Typed heads/decoder, original features, 10 s | 0.5176 | 0.6214 | 0.4739 | 8 |
| Typed heads/decoder, rich features, 10 s | 0.4421 | 0.5840 | 0.3923 | 10 |
| Typed heads/decoder, rich features, 41 s | 0.4952 | 0.5851 | 0.3734 | 12 |

Pooled five-window diagnostic spellF1: original learned-audio
0.2109; short rich structured
0.2430. All 53 tests and six label-free
checkpoint regeneration checks passed. All figures are exploratory; no 0.98
full-match or final-test result is claimed. Private full results are in run05/RESULTS.md.
