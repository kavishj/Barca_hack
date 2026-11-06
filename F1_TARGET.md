# Target: official spellF1 at least 0.80

The working candidate scores 0.7200 spellF1 and 0.6792 frame macro F1 on the
unchanged nine-possession draft validation. Its five training-window holdouts
score 0.5123 spellF1. The 5.94B frozen-video addition kept validation spellF1 at
0.7200, lowered frame macro F1 to 0.6626 and lowered training OOF spellF1 to
0.4207, so the graph + TCN remains the candidate.

## Evaluation rules

- Keep the original references and official scorer unchanged. Use no private
  ground truth, hand-corrected output, or new manual labels.
- Fit every scaler, PCA, duration statistic, class prior and loss weight using
  the corresponding training fold only. Hold out whole existing windows and
  preserve the existing 60-second purge.
- Choose among the predeclared variants using training-window OOF scores. Score
  only that selected new variant on the existing validation after selection.
  Archive all OOF results, unsuccessful variants and decisions.
- The same validation and training windows have already been inspected in prior
  experiments. Selection scores have selection bias. Reaching an observed 0.80
  on them will not establish 0.80 on the private full match or an unseen fixture.
  Do not present it as an independent benchmark or promise no generalization gap.
- Keep complete H/A/D coverage on graded frames and X outside graded halves.
  Preserve the supplied pre-kickoff D constraint. No abstention-based score gaming.
- Commit tested code, protocols and aggregate results regularly. Keep match
  assets, derived feature caches, weights, labels and predictions out of Git.

## Experiment 10, fixed before fitting

The training-only error audit found 18 of 33 spells unmatched at tIoU 0.5.
Their median duration was 6.38 s, compared with 20.44 s for matched spells.
Some misses have high true-team emission probability but are merged by decoding;
others have low emission probability. Both representation and objective matter.

With a fixed cost of 8 for each switch and probabilities clipped at 1e-6, even
perfect emissions cannot insert a 0.76-second H spell between A spells: its
maximum emission gain is about 10.5 while its two switches cost at least 16.
The official reference includes brief possessions and imposes no minimum length.

Predeclare a 2 x 2 comparison:

| Tracking input | State loss |
|---|---|
| Existing graph | Existing natural frame-frequency CE |
| Existing graph | Capped inverse-square-root live-spell weighting |
| Graph plus automatic contact/reception history | Existing CE |
| Graph plus automatic contact/reception history | Capped inverse-square-root live-spell weighting |

Reuse the existing graph/natural fits as controls. The three new variants keep
the original graph + TCN width, depth, audio, normalization, boundary BCE,
optimizer, seeds 0/1/2 and 60 epochs. Each uses five seed-0 whole-window folds
and three final seeds. No memory, span objective or large video encoder is added.

Contact and reception events are observations computed by the existing physical
signal code. Their recent team memberships and recencies enter graph nodes through
a zero-initialized shared linear projection. Base initialization and dropout RNG
are preserved. No reference labels enter event extraction.

For the weighted loss, only complete known H/A spells receive
`clip(sqrt(median_complete_live_length / length), 0.25, 4)` weights. Censored
window-edge spells retain weight 1. Live weights are normalized to preserve the
total live-frame mass; D frames retain weight 1. All statistics fit training only.

Evaluate each variant with two fixed decoders, without a parameter grid:

1. The previous switch cost 8 decoder and its original weak duration preference.
2. A boundary-conditioned positive switch cost
   `8 * training_boundary_band_prior / (q(t) + training_boundary_band_prior)`.
   The prior is fitted from known training boundary targets. This replaces the
   old constant-plus-boundary penalty. No duration preference is applied. Candidate
   boundary peaks must exceed twice the training prior (clamped to [0.01, 0.35]).
   There is no hard minimum duration and no reward for adding transitions.

Select the highest pooled training-window OOF official spellF1; ties prefer the
simpler existing architecture and then the fixed decoder. Record the choice
before main validation scoring. Report per-window variability, frame macro F1,
predicted/true spell counts and boundary metrics beside ranking spellF1.

### Experiment 10 outcome

All 24 new fits completed. None beat the existing control's 0.5123 OOF spellF1.
Fixed-decoder OOF scores were 0.4746 (balanced loss), 0.4949 (contact history),
and 0.4982 (both); adaptive decoding reduced each score. The training-only
selection retained the original model and reproduced its prediction file exactly.
Main validation therefore remains 0.7200. No other new arm was scored on it.

## Experiment 11, fixed before fitting

Test a different inductive bias: regularized histogram-boosted binary trees for
dead versus live, home versus away conditional on live play, and boundaries.
Combine the unchanged rich physical/fine-audio observations with the original
fold-fitted 32 audio PCA components. Add observation shifts at -10, -3, -1, +1,
+3 and +10 seconds for ball motion, activity, distance margin, team motion,
contact/reception history, visibility and whistle observations. Shifts stay
inside each half; missing observations stay missing. Exclude absolute frame/time
and the supplied known-dead flag from fitting. No new labels or pseudo-labels.

Augment each training frame by swapping teams and corresponding labels. Average
both perspectives at inference so H/A equivariance and dead-state invariance
hold exactly. Train on a fixed 5 Hz frame grid, preserving natural class counts.
Use 250 iterations, learning rate 0.05, at most 7 leaves, minimum 50 samples per
leaf, L2 10, seed 0, and no early stopping or internal validation split. These
settings are fixed for every model and fold, not selected using scores.

Use the same five whole-window folds and 60-second purge. Compare the trees
alone and their equal arithmetic posterior blend with the original graph model.
For each, test switch costs 1, 2, 4 and 8 with the original segment decoder;
reuse fold-training duration fits. Include graph/fixed-8 as the first-listed
control. Choose highest pooled OOF spellF1, breaking ties by listed order, save
the decision, then evaluate only that selection on the unchanged main validation.
Archive every OOF result, including failures. This is exploratory model selection
on reused windows and cannot demonstrate independent full-match performance.

### Experiment 11 outcome

All six boosted fits completed. The equal tree/graph blend with switch cost 2
improved training OOF spellF1 from 0.5123 to 0.5754, but its selected main score
fell to 0.6133. Tree-only OOF scores ranged from 0.4308 to 0.4692. The new blend
is the current training-OOF leader; the original graph retains the highest main
score (0.7200). These disagreeing rankings show the instability of this small,
reused validation set. Do not describe either as proven full-match improvement.

## Experiment 12, fixed before fitting

Test a longer temporal receptive field without increasing model capacity. Keep
the original graph encoder, shared team head, loss, optimizer, 60 epochs and seeds.
Run its temporal convolutions at the graph's existing 5 Hz observation cadence,
then interpolate state and boundary logits onto 25 Hz for the original losses.
The six dilation blocks now see about 25 seconds on each side instead of 5 seconds.
Use the original fold-fitted preprocessing and original 15 graph features; no
contact-history projection, memory or span losses. Keep the 60-second purge and
extend both training context sides to cover the new receptive field.

Fit five seed-0 whole-window folds and three final seeds, using the same window
supervision. Compare costs 1, 2, 4 and 8 with the unchanged segment decoder and
training-only durations. Include graph/fixed-8 and experiment-11 blend/fixed-2
as controls. Select maximum pooled OOF spellF1, control-first ties, and record
the decision before scoring only that selection on the original main validation.
No new parameter grid, labels, output edits or validation-based early stopping.

## Experiment 13, fixed before combining scores

Test complementary existing model errors without training any new parameters.
Use the original learned-audio TCN's expanded-data fold scores (experiment 5/6),
the original relational graph scores, and experiment-11 tree scores. All must
share the same references, five held-out windows and training-only preprocessing.
Compare an equal arithmetic graph/TCN mixture and an equal graph/tree/TCN mixture,
each with the existing costs 1, 2, 4 and 8. Use the original graph-training duration
fits and average boundary posteriors with the same weights as states. No learned
ensemble weights or main-validation calibration. Include graph/8 and graph/tree/2
as first-listed controls. Select pooled OOF spellF1 before one main evaluation.
Archive all results and retain the original prediction files as immutable controls.

### Experiment 13 outcome

The best new two-model mixture scored 0.4787 OOF spellF1; the best three-way
mixture scored 0.5586. Neither beat graph/tree at 0.5754, which was retained.
Its validation file reproduced experiment 11, so there is no new main improvement.

## Experiment 14, fixed before fitting

The supplied automatic Smart Tagging XML contains separate tactical-phase types
and 26 throw-in tags, while the existing features sum phase teams and omit throw-in
restarts. Preserve the original rich observations and add each automatic event
code/type separately for each team: active interval count, signed seconds to the
nearest interval start, and signed seconds to its end, capped to a 30-second
neighbourhood. Construct these inside each graded half, without reference labels.
Do not copy automatic tag intervals into output labels or add hard restart rules.

Run exactly experiment 11's symmetric factorized boosted model and its predeclared
tree-only/equal-graph-blend cost comparison, changing only the rich feature cache.
Reuse the same folds, purge, audio PCA and fixed tree settings. Select on training
OOF, then score only the selection on unchanged main validation. Report against
the existing 0.5754 OOF leader and 0.7200 main-score leader separately.

## Experiment 15, fixed before fitting

Test an explicit physical ownership expert instead of adding neural capacity.
Fit one L2 logistic owner classifier (C=0.1, no intercept, fixed 5 Hz training
grid, no class weights) on known live training frames only. Inputs are signed
team-relative proximity, density, motion/spread, tag balance, and exponentially
decayed previous/next automatic contact and reception observations (0.5/2/8 s).
Future event context remains within each half and at most 30 seconds. Replace
missing observations with zero evidence; fit RMS scales using that fold's live
training frames only. Signed inputs and the zero intercept impose team symmetry.
No absolute position, video time, previous predictions or reference-derived events.

Keep the graph's dead posterior and boundary scores unchanged. Mix only its
conditional H/A owner posterior with the physical expert, using fixed physical
weights 0.5, 0.75 or 1.0. Compare the existing costs 1, 2, 4 and 8, plus unchanged
graph/8 and graph/tree/2 controls. Same five windows and 60-second purge. Select
pooled training OOF spellF1 before scoring only the winner on unchanged main
validation. Log all 14 choices and acknowledge selection on reused windows.

## Experiment 16, fixed before fitting

Test label-preserving geometric augmentation to reduce memorization of pitch
regions. Train the original 25 Hz graph + TCN with the original fixed configuration,
five seed-0 folds and three final seeds. For each training window, independently
sample reflections across the pitch's x/y midlines. Reflect graph positions,
velocities, ball-relative vectors and relative velocities consistently. For a
y-reflection, also reflect raw ball-y and ball-vy observations through the saved
fold normalizer, preserving missingness. X-reflection reverses the attack direction
too, so the existing attack-relative scalar features remain unchanged. Distances,
team identity, supervision and labels are invariant. No augmentation at inference.

Keep a separate augmentation RNG so the original window shuffle sequence and model
initialization are preserved. No dropout, loss, epoch or architecture changes.
Compare costs 1, 2, 4 and 8 against unchanged graph/8 and graph/tree/2 controls;
choose pooled OOF spellF1 before one main evaluation. No validation-derived regions,
event timings, transforms or label changes.

## Experiment 17, fixed before external pretraining

Use the publicly accessible Metrica Sports Sample Games 1 and 2, pinned to
`metrica-sports/sample-data@e706dd506b360d69d9d123d5b8026e7294b13996`.
These are separate anonymized fixtures with 145,006 and 141,156 tracking frames,
not another possession answer key for the target match. The rules explicitly
allow public external data. Acknowledge Metrica Sports and retain the upstream
README notice; do not redistribute their raw assets in this repository.

Pretrain only the existing two-layer, width-32 spatial graph encoder and a shared
linear H/A owner head on closed successful PASS intervals from those event CSVs.
Exclude offside passes, malformed intervals and conflicting opposite-team labels;
use half-open intervals and the source's published synchronized 25 Hz frame IDs.
Do not infer dead labels or forward-fill ownership between events. Sample the
existing graph cadence (5 Hz). Preserve missing ball/player observations, pack
teams consistently into padded 47-token graphs, and identify goalkeeper slots by
their mean distance from midfield, independently of possession labels.

Pretraining uses seed 0, 20 epochs, batch 128, AdamW LR 0.001, weight decay 0.001,
cosine schedule, natural CE, no early stopping or target-match labels. Augment
each graph with random x/y reflections, 10% independent player dropout and 20%
ball dropout, consistently masking coordinates and ball-relative observations.
No external hyperparameter search or target-validation checkpoint selection.

Freeze this pretrained spatial encoder during the original target-match 60-epoch
graph+TCN training; keep it in eval mode. All other initialization, preprocessing,
losses, optimizer settings, five seed-0 folds, three final seeds and 60-second
purge remain unchanged. No target-match augmentation is added in this experiment.
Compare costs 1, 2, 4 and 8 against graph/8 and graph/tree/2 controls. Select pooled
training OOF spellF1 before one unchanged main evaluation. External pretraining
performance cannot be substituted for the requested target-match F1.

## Experiment 18, fixed before typed-cost decoding

The training audit identified very short live possessions, while generic adaptive
switching over-fragmented play. Separate the cost of a direct H/A turnover from
the cost of entering/leaving D. This changes no state emissions or neural weights.
Use a symmetric off-diagonal transition matrix: H↔A cost in {1,2,4}; H/A↔D cost
in {4,8}. Keep the original boundary penalty, candidate generation and weak
training-fitted duration preferences. No hard minimum duration or transition reward.

Apply the six matrices to two fixed score sources: the original graph, and the
graph with physical-owner weight 0.75 (the best new expert weight on experiment-15
OOF). Keep graph dead and boundary posteriors unchanged for the physical source.
Include unchanged graph/8 and graph/tree/2 as controls. Select the best of these
14 predeclared choices using pooled training-window OOF only, save the decision,
then score only the selected variant on unchanged main validation. Use the same
fold models, purge and references. Do not inspect main scores for other matrices.

## Experiment 19, fixed before additional fold fitting

Correct a model-selection mismatch: previous OOF predictions use one seed per
fold, whereas main predictions average three seeds. Decoder calibration should
use the same ensemble size as deployment. For the original graph architecture,
reuse every saved fold seed-0 checkpoint and fit seeds 1 and 2 in each of the five
folds (10 new fits), with unchanged preprocessing, configuration, masks and epochs.
Do not refit the existing three-seed global model or change any of its weights.

Average each fold's state and corrected boundary posteriors over seeds 0/1/2.
Compare this matched three-seed graph ensemble and its equal mixture with the
existing fold-specific trees, at costs 1, 2, 4 and 8. Choose the highest pooled
OOF spellF1 among these eight matched-ensemble candidates, cost/order ties as
listed. Archive the old seed-0 OOF controls as diagnostics, not as alternative
estimates of the three-seed model. Record the selection before one main scoring.
The global graph/trees score arrays and reference hashes must stay unchanged.

This repairs the ensemble-size mismatch; it does not turn reused windows into an
untouched benchmark. Report per-fold scores and the difference between old and
matched OOF estimates beside the selected main score. No new state model grid,
validation-based cost choice, early stopping, labels or output correction.
