# Frozen video features for possession

The next controlled experiment adds visual observations to the graph + TCN
candidate. Audio features, supplied labels, training-window folds and the original
decoder family remain unchanged. Memory and span supervision remain disabled.

The first stage is a compatibility/memory pilot with the pinned
OpenGVLab/InternVideo2-Stage2_6B checkpoint. Its vision-only pooled-feature path
loads 5,938,852,096 frozen parameters, omitting unused text/distillation outputs.
FP16 parameters are split across the two free Kaggle T4s. PyTorch SDPA replaces
the optional FlashAttention implementation. A small architecture test verifies
agreement with the released feature computation before real-weight inference.

The pilot samples fixed unlabelled moments in both halves and checks finite,
unit-normalized, exactly repeatable features and GPU memory. It produces no
possession score and reads no reference labels. No model fine-tuning occurs.

Raw media and caches stay in private Kaggle storage. Model weights are downloaded
to temporary storage and are not included in the repository or output artifacts.

## Frozen feature ablation

The successful four-clip GPU pilot measured about 0.57 seconds per warmed-up
forward and roughly 6.1 GB peak allocation per T4 with a split encoder.

Visual sampling uses a fixed 50-frame grid (0.5 Hz) within each graded half and
four frames at offsets -18/-6/+6/+18, clamped at halftime. Views are the full image
and a 256-pixel region around the nearest tracked player of each team within 15 m
of the current ball or a ball observation at most two seconds old. The homography
maps normalized pitch coordinates to image pixels. Candidates are selected using
observations only; crop slots are followed across the four frames. Invalid or
stale geometry has an explicit availability mask. No new annotations are made.

The first diagnostic extraction covers all existing labelled windows plus 60 s
of preceding observation context and the TCN/interpolation margin. Outside those
fixed extraction intervals visual features are unavailable. This diagnostic cache
does not establish full-match vision performance. Expand to full-match extraction
only if the fixed comparison improves training-window OOF spellF1 and does not
regress on the reused main validation. Any expanded-cache results require fresh
OOF decoder selection and main scoring with unchanged neural checkpoints.

Wide features receive train-only 16-component PCA. Candidate-team features use
one shared 16-component PCA fitted jointly across both teams' training samples.
Availability-gated, zero-initialized linear projections add those features to the
existing global/team streams before GELU. Audio normalization, graph encoder,
TCN width/dilations, CE/boundary losses and epoch budgets remain unchanged.
No memory or span objective is enabled. A disabled-vision training test reproduces
the original graph model's parameter values exactly, including dropout RNG.

The experiment reuses the saved original graph controls and their exact fitted
audio/graph preprocessing. Only the visual-fusion arm is newly trained: five
whole-window seed-0 folds and three main seeds, each 60 epochs. Decoder cost 8
provides the fixed comparison; a separate 1/2/4/8 grid is selected using training
OOF only before main scoring. All settings and unsuccessful results are retained.
The same draft references remain byte-identical; OOF selection scores and reused
nine-possession validation are exploratory rather than an independent benchmark.

## Completed diagnostic results

The frozen-video diagnostic does not meet the predeclared expansion criterion. Keep the existing graph + TCN candidate; full-match vision extraction is not justified by this result.

| Candidate | Decoder cost | Validation spellF1 | Frame macro F1 | D F1 | Predicted / 9 |
|---|---:|---:|---:|---:|---:|
| Existing graph + TCN | 8 | 0.7200 | 0.6792 | 0.5438 | 6 |
| Graph + frozen video, fixed decoder | 8 | 0.7200 | 0.6626 | 0.4736 | 6 |
| Graph + frozen video, training-OOF decoder | 8 | 0.7200 | 0.6626 | 0.4736 | 6 |

| Decoder switch cost | Vision training OOF spellF1 | Predicted / 33 |
|---:|---:|---:|
| 1 | 0.3652 | 36 |
| 2 | 0.3765 | 35 |
| 4 | 0.3877 | 32 |
| 8 | 0.4207 | 25 |
