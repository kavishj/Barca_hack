# Target: official spellF1 at least 0.80

The original graph candidate retains the highest observed main score: 0.7200
spellF1 and 0.6792 frame macro F1 on the unchanged nine-possession draft validation.
Its historical single-seed training OOF spellF1 is 0.5123. The 0.80 ranking target
has not been reached.

Experiment 20's direct external owner expert improves matched three-seed training
OOF spellF1 from 0.5133 to 0.6067, with 0.8120 frame macro F1. Its selected main
spellF1 is 0.6824 and frame macro F1 is 0.6485. It remains an experimental candidate;
the original prediction file is preserved. Fresh full-match inference from saved
weights, without target reference labels, reproduces its CSV byte-identically.
See EXPERIMENT_RESULTS.md for all verified outcomes and evaluation limitations.

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

## Experiment 20, fixed before fitting the external probe

Test the externally pretrained encoder as a direct ownership expert, bypassing
the target-trained temporal owner head. Reuse experiment 17's frozen encoder;
do not repeat or select a pretraining checkpoint. Encode its 11,904 external
closed-pass samples and fit a zero-intercept L2 logistic classifier (C=0.1,
max_iter=2000, tol=1e-8, natural frequency) on the 32-dimensional home-minus-away
pooled embedding. No target-match labels enter encoder/probe fitting. A swapped
team order negates the embedding difference and complements the owner posterior.

Apply the probe to the original target graph at 5 Hz, interpolating within halves.
Keep the graph's dead posterior. Mix its conditional H/A posterior with the
external expert at weights 0.5, 0.75 or 1.0 only when the raw ball is observed;
retain the original posterior at missing-ball frames. Add an observation-derived
turnover boundary score: absolute change in consecutive external owner posteriors
when both graph samples observe the ball, centred between those samples with the
existing +/-3-frame target-band radius. Mask it at missing-ball frames and take
the maximum with the original graph boundary posterior. No transition rewards.

Use experiment 19's matched three-seed fold graph posteriors and the unchanged
three-seed global graph; keep the existing tree caches. Compare those three
weights with costs 1, 2, 4 and 8, plus graph/8 and the OOF-selected experiment-19
variant as controls (deduplicate if they coincide). That control identity comes
only from experiment 19's decision file, without reading its main score. Keep
training-only durations; select pooled OOF spellF1 before one main scoring.
External pretraining/probe training scores do not count toward the requested
target-match F1. Archive every variant. This ensemble-size clarification is
recorded before probe fitting or access to experiment 19's results.

## Experiment 21, fixed before typed-adaptive decoding

Experiment 20 matches 24/33 OOF possessions at tIoU 0.3/0.4 but only 10/33 at
0.7. Its fixed cost 8 can erase subsecond turnovers even with perfect emissions.
Keep its selected external-owner weight 1, all matched three-seed scores, original
training durations and decoder candidate graph unchanged. Compare four policies:

1. Experiment-20 fixed-8 control, unchanged.
2. Typed fixed-8: H/A transitions use max(graph boundary, external turnover),
   while entering/leaving D uses the graph boundary only.
3. Typed adaptive: use policy 2, replacing only the H/A switch fee by
   `8 * training_turnover_band_prior / (external_turnover + prior)`.
4. Confidence-gated typed adaptive: attenuate external turnover variation by
   the geometric mean of adjacent graph confidence scores and require both
   observations to have observed ball position and valid ball velocity. Gate
   scores at missing-ball frames. Entering/leaving D still uses policy 2.

Fit the turnover-band prior on corresponding fold-training labels only: radius-3
bands around known H/A changes divided by usable known live frames, with Jeffreys
half-count smoothing. Exclude unknown edges. Keep the original boundary log
penalty, weak duration preferences and positive transition fees; add no hard
minimum duration, transition reward, state reweighting or parameter grid.
Use the same candidate boundaries for all policies. Select pooled OOF spellF1,
control-first ties, save the decision, then score only that policy on unchanged
main validation. Save the global training prior for label-free inference.

## Experiment 22, fixed before visibility-likelihood calibration

Known training frames have missing ball observations in 66.7% of D versus 14.7%
of live play. Test whether the D posterior relies too much on this marginal
observation cue. Fit `P(visible | D)` and `P(visible | live)` with Jeffreys
half-count smoothing on each corresponding training fold only. For observed
visibility v, subtract `gamma * log(P(v|D)/P(v|live))` from the graph's dead/live
log odds. Preserve its conditional H/A ratio, then apply the unchanged external
owner mixture. This is a nuisance-feature ablation, not a guaranteed causal or
unbiased posterior correction. Use gamma 0, 0.5 or 1; no fitted temperature/grid.

Cross those three fixed values with the fixed-8 decoder and experiment 21's
OOF-selected policy (deduplicate if it is fixed). Refit any turnover prior only
on the corresponding training fold. No neural/encoder/probe refitting, labels,
manual rules, duration changes or main-based parameter choices. Prefer lower
gamma and then the fixed decoder on exact OOF ties. Select pooled training OOF
spellF1 before one unchanged main evaluation; retain all failed variants.

## Experiment 23, fixed before temporal external pretraining

The instantaneous external owner can respond to momentary touches. Learn control
over time using more of the same public event data, without additional target
annotations. Use Sample Games 1/2 for fitting. Reserve Sample Game 3 for a single
external transfer check: its metadata is DEMO_1002, dated 2020-05-08, with different
half durations and a two-goal event sequence distinct from the target's supplied
four-goal automatic tags. It is not another answer key for the target fixture.

Derive weak external control sequences from the published event semantics:
PASS/RECOVERY/SET PIECE/SHOT starts establish the recorded team's live control;
BALL OUT ends, immediate FAULT RECEIVED, offside/end-half and goal ends establish
D. Ignore CHALLENGE/CARD as ownership changes. A BALL LOST end clears its losing
team only if it still owns the ball; do not erase an already recorded recovery
by the opposite team. Same-frame conflicting controls become X, and stop events
take priority. Start each half in D until a recorded restart; mask unresolved
live tails. Use the existing closed-pass labels to fill unknown pass spans and
mask contradictions. Never propagate labels across halves. D/X frames provide
context but no owner supervision. These are weak external labels, not target GT.

Freeze the experiment-17 graph encoder and experiment-20 linear probe. Extract
team embeddings, signed point logits, ball visibility and graph confidence at
5 Hz. Fit a shared width-32, four-block TCN with dilations 1/2/4/8 (about +/-6 s),
dropout 0.15 and a zero-initialized residual owner head over the point logits.
Add an invariant H/A turnover head with radius-1 coarse boundary targets and
0.25 weighted BCE; cap positive weight at 20 and correct its prior at inference.
Use seed 0, 60 epochs, 256-step cores, batch 8, AdamW LR/weight-decay 0.001 and
cosine schedule. Fit normalization on Games 1/2 only. No early stopping or grid.

Precompute consistent ball-masked embeddings. Each epoch, mask random contiguous
1–20-second blocks covering approximately 20% of the external timeline,
independently of labels. Preserve team symmetry and half resets. This trains the
head to handle missing observations without changing supervision.

After the fixed final epoch, compare conditional live-owner macro F1 and typed
turnover F1 within 0.5 seconds on untouched external Game 3 against the original
instantaneous probe. Neither metric may decrease, and at least one must improve,
before target OOF expansion. These conditional external scores do not satisfy
the requested target-match F1. No Game-3 checkpoint or hyperparameter selection.
If this gate passes, preregister the target comparison before reading new target
scores; retain the current fixed predictions and all external negative results.

Implementation details fixed before fitting: evaluate typed turnovers using both
source and destination (`typed='pair'`), one-to-one matching within 12 frames at
25 Hz. Interpolate conditional H/A probabilities to the original frame clock and
take the larger probability, with H on exact ties; no label-derived D gating or
boundary threshold in this external check. Restrict metric runs to known live
reference spans, while predictions themselves never see the reference. Fit shared
normalization moments on unmasked Games 1/2 observations, pool both teams, clip
normalized inputs to +/-8, and retain unscaled frozen point logits for the
residual head. Fixed renewed ball-mask blocks have uniform lengths 5..100 coarse
steps and geometric unmasked gaps averaging 210 steps. Loss masks respect the
assigned known-live run, excluding D/X and half edges from boundary supervision.

## Experiment 24, dead-ball transfer hypothesis fixed before reading Game-3 metrics

The remaining target OOF confusion includes 1,657 live frames predicted D and
1,673 dead frames predicted live in experiment 20. Test whether weak external
stoppage supervision helps, independently of experiment 23's owner-only gate.
Declare this second external hypothesis before either new Game-3 result is read.
Game 3 therefore checks two predeclared hypotheses; report that multiplicity and
do not treat its results as an untouched final test after these comparisons.

Use exactly the same pinned full-control caches and frozen spatial/probe features.
Train only on Games 1/2, with known H/A/D supervision and X censored. Keep the
shared temporal backbone, point-logit owner residual, input normalization,
missing-ball augmentation, seed, 60 fixed epochs, batches, cores, optimization
and half resets from experiment 23. Add an invariant dead/live log-odds head over
the mean and absolute difference of the two temporal team representations. Its
weights start at zero, with the training dead/live prior as bias. Combine it with
conditional H/A logits by factorization, preserving owner team symmetry. Train
three-state CE plus 0.25 boundary BCE with radius-1 targets for any known state
change, positive weight capped at 20; X and half edges never create transitions.

Use a matched non-temporal external baseline: frozen instantaneous H/A probe and
a C=0.1, unweighted logistic dead/live classifier fitted on Games 1/2 only. Its
invariant inputs are the mean and absolute difference of the 32-dimensional team
embeddings, absolute point-logit difference, ball visibility and graph confidence
(67 dimensions). Fit standardization on known training frames, clip to +/-8,
use an intercept, max_iter=2000, tol=1e-8, seed 0. Neither baseline nor temporal
model sees target observations or labels while fitting. Natural class mass is
preserved; no label balancing or manually derived missing-ball rule.

After both fixed fits, interpolate three-state probabilities to the original
Game-3 frame clock and use raw H/A/D argmax. Compare macro frame F1 over H/A/D
and the official spellF1 on its known weak external spans. Require neither to
decrease and at least one to improve before this branch's target OOF expansion.
Report these as weak external metrics, never target-match achievement. No
checkpoint, threshold or hyperparameter selection on Game 3. Preregister any
subsequent target blend comparison before reading new target-window scores.

## Experiment 25, fixed after experiment 23's external gate and before target scores

The fixed owner-only external model improved conditional live-owner macro F1
from 0.916737 to 0.945819 and typed turnover F1 from 0.254902 to 0.399317.
This passes experiment 23's predeclared external gate. Freeze its final weights,
normalization and encoder/probe; do not fit them on any target annotations.

Compare four fixed target policies: the experiment-20 fixed decoder, the
experiment-21 typed-gated decoder, and two new owner-only replacements. The new
variants replace the conditional H/A owner with the external temporal output,
either only when the raw target ball is observed (`temporal_observed`) or across
the graded timeline (`temporal_all`). Leave graph P(D) unchanged, with original
graph ownership retained wherever the variant is unavailable. Keep the same
switch cost 8, durations and candidate/boundary scores as experiment 20: maximum
of graph boundary and original instantaneous observed-ball owner variation. The
new auxiliary temporal boundary head is not used in this first owner ablation.
No blend grid, threshold tuning or decoder refit. Keep supplied kickoff D facts.

Use matched three-seed graph OOF scores, the same five whole-window folds and
60-second purge. Verify the two controls against saved CSVs. Select pooled OOF
spellF1 with controls first on ties; write the decision before main scoring.
Evaluate only that selected policy on unchanged, reused main validation. Record
all variants and failures, complete graded coverage and regeneration checks.
The public external check and reused target windows do not establish independent
full-match generalization; the target remains official target-match spellF1 0.80.

## Experiment 26, fixed after experiment 24's gate and before target H/A/D scores

The three-state external temporal model passed its fixed Game-3 check: weak
external macro frame F1 0.941231 -> 0.944240 and spellF1 0.525860 -> 0.570385.
Freeze the final 60-epoch checkpoint, source-only normalization and encoder/probe.
The owner-only target ablation (25) did not improve OOF and retained the typed
control; do not retune that ablation on main validation.

Compare the same two controls (experiment-20 fixed and experiment-21 typed-gated)
with three fixed full-probability blends. Start with the experiment-20 target
H/A/D posterior. Average it with the frozen external three-state temporal
posterior at external weights 0.25, 0.5 and 1.0 across the graded timeline.
Interpolate external three-state probabilities within each half and preserve
normalization. No target calibration, manual missing-ball override or label-based
gating. Keep switch cost 8, the original fold duration priors and maximum of
graph/instantaneous-owner boundaries; do not use the new boundary head yet.
Known kickoff D facts remain mandatory. No state or decoder fitting.

Use the matched three-seed graph folds, unchanged references, five whole-window
holdouts and 60-second purge. Controls must reproduce earlier predictions. Select
pooled OOF spellF1 with controls first on ties and write the decision before
scoring only the selected policy on reused main validation. Keep all failures;
an external score or frame F1 above 0.80 does not meet the target-match spellF1
goal, and repeated window selection does not establish independent generalization.

## Experiment 27, frozen external representations with fold-fitted linear adaptation

Experiment 26's unadapted full-probability blends scored 0.5400/0.5263/0.4918 OOF
spellF1, below the fixed controls. Test target-domain calibration of the frozen
experiment-24 representation rather than further external-only probability grids.
No new manual labels or external fixture tuning. Freeze all neural weights and
source normalization. Extract the two 32-dimensional temporal team representations
at 5 Hz, respecting the original half resets and +/-6-second receptive field.

Fit two target linear probes at each whole-window fold: an intercept-free H/A
probe on the 32-dimensional team difference plus instantaneous point-logit
difference (33 signed inputs), and an intercept-bearing dead/live probe on the
temporal mean/absolute difference, absolute point-logit difference, ball visibility
and graph confidence (67 invariant inputs). Use C=0.1, max_iter=2000, tol=1e-8,
seed 0, natural frame mass and no balancing for both logistic classifiers.
H/A uses known live training samples and symmetric zero-centered RMS scaling;
dead/live uses all known training samples and training-only mean/std scaling.
Clip normalized inputs to +/-8. Sample existing labels on the native coarse
clock; unknown frames remain excluded. Fit all scalers/coefficients only after
excluding the held-out window and the existing 60-second purge. Never fit on
main validation. This adds 101 linear coefficients/intercepts, not neural tuning.

Compare controls `fixed` and `typed_gated`, followed by three new policies:
`adapted_dead` averages graph and adapted P(D) equally while retaining the fixed
external instantaneous owner; `adapted_full` averages the complete fixed and
adapted H/A/D posteriors equally; `adapted_only` uses the adapted posterior alone.
Use switch cost 8, unchanged graph/instantaneous-owner boundaries and fold duration
priors. No new boundary head, blend grid or post-hoc threshold. Fit one global
pair of probes on known training windows only after OOF selection. Write the OOF
decision before scoring only the selected policy on reused main validation.
Keep the same references, full graded coverage, kickoff D facts and all negative
results. Repeated OOF selection remains exploratory, not a guarantee against
overfitting or an independent full-match score.

## Experiment 28, fixed frozen contrastive audio replacement

The source-only tracking models and target linear adaptation did not improve
OOF spellF1. Test an independent audio representation: frozen
`laion/clap-htsat-unfused`, revision `8fa0f1c6d0433df6e97c127f64b2a1d6c0dcda8a`
(Apache-2.0, LAION CLAP). Use its 512-dimensional normalized audio embeddings and
two fixed prompt-relative scores. No match audio fine-tuning, whistle annotations,
prompt tuning or acoustically derived reference labels.

Decode original audio PTS relative to video PTS at 48 kHz, preserving AAC priming,
gaps and half boundaries. Use the same two-second clips, 0.5-second centers and
endpoint coverage as the old PANNs cache. Use the checkpoint's repeat-padding and
unfused feature extraction. Fixed prompts, in order: `The sound of a referee
blowing a whistle during a soccer match.`, `The sound of a person whistling with
their mouth.`, `The sound of a crowd cheering in a soccer stadium.`, `The sound of
a commentator speaking during a soccer match.`, `The sound of a soccer ball being
kicked.`, `The ambient sound of a soccer stadium without a whistle.` Use the
checkpoint's learned audio/text logit scale and softmax across these six prompts;
keep the first two as relative scores, not calibrated referee-whistle probabilities.

Replace only the old learned-audio embedding/score cache in the original graph
TCN pipeline. Preserve tracking, heuristic whistle features, architecture, loss,
60 epochs and seeds 0/1/2. Fit new PCA-32 and all normalization only on each fold's
remaining known training frames after the 60-second purge. Use matched three-seed
predictions for all five whole-window holdouts and the final model; 18 new fits.
Retain control OOF predictions and unchanged reference hashes.

Compare `fixed` and `typed_gated` controls, then the new graph/audio posterior at
cost 8 (`clap_graph`) and that same posterior with the frozen experiment-20
instantaneous owner on observed-ball frames (`clap_external`). Both new variants
use the maximum of their new graph boundary and the original observed-ball
external owner variation. No blend or cost grid. Keep original fold duration
priors and kickoff D constraints. Write the pooled-OOF spellF1 selection before
scoring only that selected policy on reused main validation. Archive failures,
verify weights/input hashes and regenerate the selected CSV. A high frame/audio
score is not target spellF1 achievement, and these repeatedly reused windows
cannot establish independent full-match performance.

## Experiment 29, fixed PANNs/CLAP fusion before new ensemble scores

The fixed CLAP replacement improved matched OOF spellF1 to 0.6400 (frame macro
F1 0.843823) but scored 0.5500 on reused main validation. Do not promote it over
the preserved candidates or interpret its frame score as the target. Test whether
separating dead/live evidence from ownership and averaging two audio encoders
reduces variance. No new model, scaler, weight, threshold or decoder fitting.

Compare `fixed`, `typed_gated` and experiment-28 `clap_external` as controls.
Add three fixed policies: `dead_clap` replaces P(D) with the CLAP graph P(D) but
preserves the experiment-20 conditional owner everywhere; `dead_mean` averages
PANNs and CLAP P(D) equally while preserving that same owner; `state_mean` averages
the complete experiment-20 and experiment-28 posteriors equally. Both existing
posteriors use the same frozen instantaneous external owner on observed-ball
frames. Missing-ball ownership is precisely the difference between dead-only and
full-state fusion; no reference labels determine availability or weighting.

Use the new CLAP graph boundary for `dead_clap`; use the equal mean of the two
graph boundary scores for the two equal-mean variants. In all new policies take
the maximum with the original observed-ball external owner variation. Switch
cost remains 8, duration priors remain unchanged, and kickoff D facts apply.
Matched three-seed folds, five whole-window holdouts, the 60-second purge and
reference hashes remain unchanged. Verify all three controls, select pooled OOF
spellF1 with controls first on ties, save the decision and score only the winner
on reused main. Keep all failures and complete graded coverage. This repeated
development selection still cannot prove independent full-match generalization.

## Experiment 30, fixed shorter training against the frozen best OOF control

Experiment 29 did not improve spellF1: dead-only replacement tied 0.6400 and
equal-mean variants scored 0.6133/0.6200, leaving the CLAP control selected at
0.5500 main. The 60-epoch fits reached training losses around 0.04–0.08, so test a
smaller fixed training budget before further expanding models. Do not select an
epoch from the reused main window or add an early-stopping sweep.

Train the exact experiment-28 graph architecture with the same fixed CLAP cache,
loss, initialization seeds 0/1/2, dropout, weight decay, batches and labels, but
use 20 epochs and a cosine schedule with T_max=20. Reuse each corresponding
experiment-28 fold's preprocessing verbatim, verifying its training mask and all
scaler hashes, so the audio representation and normalization cannot change.
Keep duration priors fixed. No neural layer freezing, new augmentation or other
hyperparameter change. Fit all five matched three-seed folds and the three-seed
global model (18 fits), without intermediate-checkpoint selection.

Compare the unchanged experiment-29 OOF-selected policy as `control` with
`early_graph` and `early_external`. The new policies follow the experiment-28
graph-only and observed-ball external-owner definitions, respectively, at cost 8
and maximum of their own boundary with the original external owner variation.
Retain supplied kickoff D facts. Check the control CSVs, save the pooled-OOF
spellF1 decision before main scoring, and score only the winner on reused main
validation. If control wins, copy its verified output without a new alternative
main score. Preserve references, purge, graded coverage and all failed results.
This is a fixed regularization comparison, not independent validation of 0.80.
