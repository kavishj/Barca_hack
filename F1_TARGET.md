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
