# Controlled temporal and audio experiment

The first private T4 run completed with all 31 tests passing and byte-identical
regeneration of all three ensembles. On the same draft validation window,
learned-audio TCN scored 0.6182 spellF1 / 0.7239 macro F1 / 0.5960 D F1;
tracking-only scored 0.6118 / 0.6474 / 0.4517; heuristic-audio scored
0.4600 / 0.6654 / 0.5634. The previous CPU model scored 0.5556 / 0.6489 / 0.5100.
The learned-audio ranking gain over tracking-only is only 0.0064. Precise
stoppage onset timing and the documented pre-kickoff state remain unresolved,
so these outputs are experimental candidates, not final submissions.

This experiment compares three small temporal models on the same provisional
draft validation window used in run 02. It does not create new human labels or
use private ground truth. Its conclusions remain limited by nine validation
possessions, one bounded dead-ball interval and draft annotation quality.

## Fixed comparison

1. Tracking: 79 non-audio features; all seven existing whistle features disabled.
2. Heuristic audio: all 86 existing features, including the tone detector.
3. Learned audio: the existing 86 features plus frozen PANNs Cnn6 embeddings,
   generic AudioSet Whistle/Whistling scores and audio availability.

All arms use the same input dimensions and 45,924-parameter TCN. Disabled inputs
and their missingness indicators are zeroed. Each arm uses seeds 0, 1 and 2;
their probabilities are averaged, with no selection of a best seed. Each seed
trains for 60 fixed epochs. No validation early stopping, calibration, parameter
search or training on held-out audio is performed.

The network has 32 channels and six residual blocks, each with two kernel-3
convolutions at dilation 1, 2, 4, 8, 16 or 32. Its receptive field is 253 frames
(10.12 seconds at 25 fps). Normalisation acts across channels at each timestamp,
so chunked inference has a finite, testable receptive field. State and boundary
heads feed the existing segmental decoder with unchanged default settings.

AdamW uses learning rate 0.001, weight decay 0.001, cosine decay, dropout 0.15 and
gradient clipping at 1.0. Training cores contain 256 frames with full surrounding
context; batch size is 8. Only known training frames contribute to loss. Each
known frame is supervised once per epoch even when annotation windows have X
gaps. State cross entropy preserves natural class frequency. The boundary head
uses weighted binary cross entropy for rare changes, with its odds shift removed
before decoding. Duration preferences are fitted only on complete training spells.

Each three-seed ensemble has 137,772 trainable parameters. The frozen PANNs
component has 4,837,455 neural backbone/classifier parameters plus 1,083,456 fixed
STFT/mel values represented as parameters, for 5,920,911 loaded parameters.
The learned-audio ensemble therefore has 6,058,683 total parameters, with only
137,772 updated from match labels. The audio encoder is shared across seeds.

The 10-second temporal purge exceeds the effective 7.54-second observation radius
of the TCN plus the existing centred feature summaries. Halves are isolated in
audio interpolation, temporal context, rolling features and decoding.

## Learned audio

PANNs Cnn6 is frozen in evaluation mode. It has not been fine-tuned to this match,
and its scores are not calibrated referee-whistle probabilities. Spectators and
restart whistles can sound similar to whistles that stop play. The temporal model
must interpret sound alongside player and ball movement. Silence is not evidence
that play resumed, and possession D is not a whistle-presence target.

The original checkpoint supplies the exact 32 kHz STFT convolution filters and
mel filter bank. This avoids changes in library-default preprocessing. Source
and weight attribution/checksums are in third_party/PANNS_NOTICE.md. The code is
adapted from MIT-licensed source; unchanged pretrained weights are CC BY 4.0 and
are distributed separately from this public repository.

Audio is decoded with PyAV and resampled to 32 kHz mono. Resampled PTS values are
mapped onto the video timeline, rather than subtracting the audio stream's own
start time. Negative AAC priming samples are clipped and genuine gaps remain
zero-filled with recorded coverage. Two-second windows are sampled every 0.5
seconds, padded outside graded halves, and encoded once. Coarse embeddings are
interpolated within each half. They do not supply 40 ms onset precision; the
existing frame-level tone signal supplies finer timing information.

The 512-dimensional embeddings are reduced to 32 dimensions by PCA fitted only
on available training-frame embeddings. Scaling and imputation statistics also
use training frames only. Missing values have explicit indicators. The frozen
encoder may process validation audio at inference; no validation audio updates
the encoder, PCA, scaler or model.

## Reproduction

Install requirements-temporal.txt plus a compatible PyTorch >=2.4 build. On
Kaggle use the preinstalled CUDA build rather than replacing it with a CPU wheel.
The local checks use a separate CPU PyTorch 2.8.0 environment. Raw media, feature
caches, checkpoints and annotations remain private and outside GitHub.

```bash
python -m temporal_possession audio \
  --video /private/tactical/BAR-ATH_tactical.mp4 \
  --checkpoint /private/Cnn6_mAP=0.343.pth \
  --features /private/features.npz --out /private/audio.npz

python -m temporal_possession compare \
  --features /private/features.npz --audio /private/audio.npz \
  --train-labels /private/train.csv --validation-labels /private/validation.csv \
  --audit /private/annotation_audit.json --out /private/tcn_comparison \
  --compare /private/run02/learned.csv /private/possession_v6.csv

python -m temporal_possession predict \
  --features /private/features.npz --audio /private/audio.npz \
  --model /private/tcn_comparison/learned_audio --out /private/regenerated.csv
```

Prediction loads the saved preprocessing, all three checkpoints and decoder
settings, without reading labels. Caches and checkpoints are fingerprinted.
The experiment plan is saved before fitting and cannot be changed by a resume.
The notebook regenerates each output and checks byte identity.

## Verification and evaluation

Checks cover numerical agreement with an independent NumPy FFT, synthetic media
with delayed AAC audio, halftime isolation, train-only PCA/scaling, complete audio
feature removal in ablations, whole-half versus chunked inference, unique training
coverage, seeded training and save/reload/predict without labels. The original
possession, decoder and annotation-exporter tests remain part of the suite.

Report the official spellF1 and macro F1, D-class F1, and dead-ball onset matching
at 12- and 25-frame tolerances. One bounded D interval is insufficient to establish
a reliable whistle detector's event precision/recall. No whistle-event F1 is
reported because there is no verified acoustic-event reference. A model with the
highest score on this draft window is an experiment candidate, not a selected
final competition submission.

## Frozen contrastive audio comparison

Experiment 28 in [F1_TARGET.md](F1_TARGET.md) fixes a separate audio replacement
using [LAION CLAP](https://huggingface.co/laion/clap-htsat-unfused/tree/8fa0f1c6d0433df6e97c127f64b2a1d6c0dcda8a)
at revision `8fa0f1c6d0433df6e97c127f64b2a1d6c0dcda8a` (Apache-2.0).
The encoder stays frozen. Two-second clips use the same centers as PANNs, audio
PTS aligned to video PTS, 48 kHz decoding and zeros outside each graded half.
The checkpoint's repeat-padding supplies its expected input length.

The cache contains 512-dimensional normalized audio embeddings and two relative
scores from a fixed six-prompt comparison: referee whistle, mouth whistling,
crowd, commentary, ball kick and stadium ambience. These are not calibrated
whistle probabilities or new acoustic labels. Prompt strings, model-file hashes,
parameter count, timing and source-cache fingerprint are recorded in provenance.
There is no audio fine-tuning or prompt selection on match labels.

The graph model retains its original architecture and training settings. New
PCA-32 and scaling are fitted separately within each training fold, followed by
three seeds per fold and final training (18 fits). Original tracking and heuristic
whistle inputs remain available. Two controls and two fixed new policies are
selected by pooled training-window spellF1 before only the winner is scored on
the reused draft validation. The complete 136-test snapshot passed on Kaggle
before extraction; results belong in [EXPERIMENT_RESULTS.md](EXPERIMENT_RESULTS.md).

The private notebook is reproduced from [possession_clap.ipynb](notebooks/possession_clap.ipynb).
Feature extraction accepts no label files:

```bash
python -m temporal_possession.clap \
  --video /private/BAR-ATH_tactical.mp4 \
  --reference-audio /private/panns_audio_cache.npz \
  --out /private/clap_audio.npz --device cuda
```
