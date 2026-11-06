# Graph, possession memory and span-supervision experiment

The pretrained PANNs audio encoder and its cached features stay unchanged. This
study implements the architectural improvements before evaluating replacement
pretrained backbones. All four variants use the same supplied training labels,
shared preprocessing and fixed original decoder with switch cost 8. No new human
labels or visual encoder are introduced.

## Controlled variants

1. symmetric_tcn: shared home/away streams, existing aggregate features and audio.
2. graph_tcn: add a player–ball graph encoder to those same temporal streams.
3. graph_memory: add confidence-gated causal possession memory.
4. graph_memory_span: add per-possession span endpoint and overlap supervision.

All modules exist in each model, so registered parameter counts are equal;
disabled branches have no effect and receive no gradient. Effective trained paths
differ by design. The matched symmetric control distinguishes graph effects from
changes in team sharing and whole-window optimisation. The previous calibrated
audio TCN remains an external comparison, with its predictions untouched.

## Spatial representation and symmetry

Raw tracking is sampled every five frames (5 Hz) independently within each half.
Forty-six player slots and one ball token encode position, velocity, relative
ball motion, goalkeeper status and observation masks. The ball token remains
present when untracked, with unknown position explicitly distinguished from zero.
Slot jumps over 15 m/s are rejected for player velocities; ball jumps over 60 m/s
are excluded from velocity features. Geometric coordinates remain observations.

Two 32-channel graph-attention layers use observed distance and same-team edge
biases. Team identifiers enter only equality relations and shared team pooling.
Home and away streams share all parameters. Swapping teams must swap H/A outputs
and preserve D/boundary outputs; permuting player slots must preserve predictions.
The missing-data masks follow those same permutations.

Existing spatial features are split into team-invariant and paired/signed fields.
Paired fields use one scaler fitted jointly over both teams' training rows. Audio
PCA and every scaler fit known training frames only. Two owner substates per team
are aggregated into H/A logits; they are latent and do not establish independently
verified controlled/in-transit action labels. The dead/boundary heads consume
symmetric functions of the two streams.

## Memory, timing and supervision

The diagonal gated state-space recurrence carries possession evidence across
occlusions. Low ball-control observation confidence slows updates. The recurrence
is causal in its feature sequence and resets at every half/training window. The
whole pipeline remains offline: existing features and the TCN use future context.
Deterministic triangular matrix scans in blocks of 32 avoid CUDA scan backward
limitations and numerical underflow. Graph embeddings and memory are interpolated
onto exact video frames, and the existing six-block TCN predicts at 25 fps.

Training uses whole known annotation windows with up to 60 seconds of observation
warmup and the TCN's future observation context. Only known frames in that window
contribute to state/boundary loss. State loss preserves natural frame weighting
across windows. The memory receives no ground-truth inputs or teacher forcing.

The auxiliary span head predicts positive left/right distances from each frame.
Unknown annotation-window edges mask the corresponding endpoint loss. IoU loss
uses only intervals whose two ends are known. Per-interval weights prevent long
possessions from dominating span supervision. The span head is initially an
auxiliary training objective; the current decoder remains the inference rule.

Each main variant averages seeds 0, 1 and 2 after 60 fixed epochs, without early
stopping or neural parameter search. Five diagnostic folds hold out one existing training
window, refit preprocessing/durations/models and compare all four variants with
seed 0. The original nine-possession draft validation is excluded/purged throughout.
In addition to the fixed-cost-8 comparison, a predeclared decoder-only grid of
1/2/4/8 is selected separately per arm using pooled training-window OOF spellF1.
Ties choose the lowest cost. The decisions are saved before main validation
scoring. Those OOF selection scores have selection bias. All raw and calibrated
outputs remain separate. All fitting finishes before main validation scores are opened. References remain
byte-identical, including the interval already flagged for boundary review.

The previous 0.6667 spellF1 candidate remains the reference point. Every new result
must be reported, including regressions. These reused draft labels and 33 training
possessions do not support a promised 0.98 or a full-match/private-test claim.

```bash
python -m relational_possession prepare --data-dir /private/data --out /private/graph.npz
python -m relational_possession compare --features /private/features.npz \
  --audio /private/audio_cache.npz --graph /private/graph.npz \
  --train-labels /private/train.csv --validation-labels /private/validation.csv \
  --audit /private/annotation_audit.json --out /private/relational --cross-validate --calibrate
python -m relational_possession predict --features /private/features.npz \
  --audio /private/audio_cache.npz --graph /private/graph.npz \
  --model /private/relational/global/graph_memory_span --out /private/predictions.csv --calibrated
```

Inference checks input/preprocessing/checkpoint hashes and reads no reference
labels. The provided pre-kickoff dead-ball interval is applied automatically.
All raw data, graph caches, checkpoints and labels remain private.

## Completed results

The highest newly observed validation spellF1 is 0.7200 from Player–ball graph + TCN, versus 0.6667 for the previous candidate. This reused nine-possession comparison is exploratory; the previous candidate is preserved.

| Model | Decoder cost | Validation spellF1 | Frame macro F1 | D F1 | Predicted / 9 possessions |
|---|---:|---:|---:|---:|---:|
| Previous calibrated audio TCN | 8 | 0.6667 | 0.6217 | 0.4584 | 6 |
| Shared-team TCN control — fixed decoder | 8 | 0.6667 | 0.6256 | 0.4574 | 6 |
| Shared-team TCN control — training-OOF decoder | 8 | 0.6667 | 0.6256 | 0.4574 | 6 |
| Player–ball graph + TCN — fixed decoder | 8 | 0.7200 | 0.6792 | 0.5438 | 6 |
| Player–ball graph + TCN — training-OOF decoder | 8 | 0.7200 | 0.6792 | 0.5438 | 6 |
| Graph + possession memory — fixed decoder | 8 | 0.6400 | 0.6149 | 0.4000 | 6 |
| Graph + possession memory — training-OOF decoder | 8 | 0.6400 | 0.6149 | 0.4000 | 6 |
| Graph + memory + span supervision — fixed decoder | 8 | 0.6667 | 0.6284 | 0.4058 | 6 |
| Graph + memory + span supervision — training-OOF decoder | 2 | 0.6667 | 0.6176 | 0.3836 | 6 |

| Model | Cost 1 OOF spellF1 | Cost 2 | Cost 4 | Cost 8 | Selected cost |
|---|---:|---:|---:|---:|---:|
| Shared-team TCN control | 0.4588 | 0.4529 | 0.4571 | 0.4852 | 8 |
| Player–ball graph + TCN | 0.4581 | 0.4581 | 0.4814 | 0.5123 | 8 |
| Graph + possession memory | 0.4317 | 0.4459 | 0.4610 | 0.4690 | 8 |
| Graph + memory + span supervision | 0.4444 | 0.4600 | 0.4542 | 0.4542 | 2 |
