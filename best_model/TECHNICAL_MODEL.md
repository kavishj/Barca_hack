# Packaged model and measured results

The packaged candidate is the original `graph_tcn` arm, three seeds 0/1/2 and
fixed decoder switch cost 8. Its saved output has 0.7200 spell F1 and
0.679217402885496 frame macro F1 on the reused 4,078-frame, nine-spell validation
window. Its matched three-seed training-window OOF spell F1 is 0.4881356. The
historical one-seed OOF estimate was 0.5122807; these should not be conflated.
The best later matched OOF result is CLAP/external ownership at 0.6400, but that
model scored 0.5500 on validation and is not packaged as the best observed
validation candidate. No hidden/full-match F1 is available.

## Architecture

```
tracking -> 47-node player/ball graph, 15 features/node at 5 Hz
         -> two graph-attention layers, width 32, four heads -> team embeddings

video audio -> frozen PANNs Cnn6 -> 512-D embeddings + two AudioSet scores
           -> saved training-only PCA-32

86 tracking/physical/heuristic input features + audio + team graph embeddings
    -> shared home/away temporal CNN at 25 fps
    -> H/A/D state probabilities + transition-boundary probability
    -> duration-aware dynamic-programming decoder -> frame,label CSV
```

Graph distance and team-equality edge biases respect team swaps and player
permutations. Home and away share weights. The TCN has width 32, six residual
blocks with dilations 1/2/4/8/16/32, and approximately +/-5 seconds of feature
context. This is offline inference: future observations can be used. Invariant
mean/absolute-difference features feed the dead-ball and boundary heads.

The graph model has 74,999 registered parameters per seed, including unused
memory/span branches. Those branches are disabled in this selected arm. The
frozen PANNs implementation has 5,920,911 parameters including its checkpoint
spectral frontend. The three seed probabilities are averaged before decoding.
This model does not use CLAP, AST, an LLM, the external ownership probe or new
automatic-phase overrides.

## Training and evaluation

- Known training supervision: 21,787 frames, five windows, 33 live spells.
- Validation: 4,078 known frames in one separate window, nine live spells.
- Five whole-window diagnostic folds, with a 60-second purge and no random
  adjacent-frame split. Matched later checks use all three seeds per fold.
- Audio PCA, scaling and duration priors use training frames/segments only.
- State cross-entropy plus 0.25 weighted boundary BCE; positive weight capped
  at 20 and corrected at inference. No span loss in this arm.
- AdamW, learning rate 0.001, weight decay 0.001, dropout 0.15, cosine schedule,
  60 epochs, gradient clipping 1.0. Observation context is confined to each half.
- The metric is official one-to-one possession spell F1 averaged over tIoU
  thresholds 0.3/0.4/0.5/0.6/0.7. Frame macro F1 is secondary.

Source/scaler/checkpoint hashes and reference immutability have been checked.
The inference runner reads original observations or their exact frozen feature
caches, saved weights and preprocessing. It does not read annotations or a
stored prediction file. Its expected CSV hash is used only after inference as a
reproduction check; mismatched predictions are not corrected.

All five training windows and the small validation window have been reused in
development. Selection bias and limited validation coverage remain; the measured
0.7200 is not a guarantee of independent generalization or an achieved 0.80 score.
