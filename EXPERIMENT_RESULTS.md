# Verified experiment ledger

Ranking metric: official one-to-one possession spellF1 averaged over tIoU 0.3–0.7. Frame macro F1 is a secondary metric.

All experiments keep the same 21,787 known training frames, five whole-window holdouts with a 60-second purge, and 4,078-frame, nine-possession draft validation. No new manual labels, reference edits, prediction corrections, private full-match ground truth or abstention were used. Match assets and derived private artifacts remain outside Git.

| Study | Best new OOF variant | New OOF spellF1 | OOF-selected variant | Selected main spellF1 |
|---|---|---:|---|---:|
| 9: Frozen 5.94B visual encoder | video / cost 8 | 0.4207 | video / cost 8 | 0.7200 |
| 10: Contact/history and spell weighting | event_balanced__fixed | 0.4982 | graph_natural__fixed | 0.7200 |
| 11: Symmetric boosted trees | blend__2 | 0.5754 | blend__2 | 0.6133 |
| 12: Five-Hz temporal graph | coarse__1 | 0.4267 | blend__2 | 0.6133 |
| 13: Equal complementary ensembles | three_way__1 | 0.5586 | graph_tree__2 | 0.6133 |
| 14: Separate automatic tag types | blend__4 | 0.5091 | graph__8 | 0.7200 |
| 15: Physical ownership expert | 0.75__4 | 0.5571 | blend__2 | 0.6133 |
| 16: Pitch-reflection augmentation | reflected__4 | 0.5079 | blend__2 | 0.6133 |
| 17: Frozen external-pass spatial encoder | external__4 | 0.4968 | blend__2 | 0.6133 |
| 18: Separate turnover/stoppage costs | physical__2_4 | 0.5571 | blend__2_2 | 0.6133 |
| 19: Matched three-seed OOF calibration | blend__1 | 0.5133 | blend__1 | 0.6133 |
| 20: Direct external ownership expert | 1.0__8 | 0.6067 | 1.0__8 | 0.6824 |
| 21: Typed adaptive turnover fees | typed_gated | 0.6154 | typed_gated | 0.4000 |
| 22: Visibility likelihood ablation | 0.5__typed_gated | 0.6031 | 0.0__typed_gated | 0.4000 |
| 25: Frozen external temporal owner | temporal_observed | 0.6034 | typed_gated | 0.4000 |
| 26: Frozen external full H/A/D blends | external_0.25 | 0.5400 | typed_gated | 0.4000 |
| 27: Fold-fitted linear temporal adaptation | adapted_dead | 0.5180 | typed_gated | 0.4000 |
| 28: Frozen CLAP audio replacement | clap_external | 0.6400 | clap_external | 0.5500 |
| 29: Fixed PANNs/CLAP state fusion | dead_clap | 0.6400 | clap_external | 0.5500 |
| 30: Fixed 20-epoch CLAP training | early_external | 0.6033 | control (clap_external) | 0.5500 |
| 31: Factorized tabular dead-ball expert | trees_audio_mean | 0.6164 | control (clap_external) | 0.5500 |

The original graph control has **0.5123 OOF / 0.7200 main spellF1**, with 0.7907 / 0.6792 frame macro F1. The historical graph/tree blend reached **0.5754** using one-seed fold estimates and **0.6133** on main. Matching the fold ensemble to the three deployment seeds lowered the best OOF estimate to **0.5133**, with unchanged main spellF1 **0.6133**. The one-seed blend's frame macro F1 was **0.8198**, which is not the ranking metric. These rankings disagree; neither establishes independent full-match improvement. The direct external expert then improved matched OOF spellF1 to **0.6067** and frame macro F1 to **0.8120**, with **0.6824** main spellF1. Its full-match CSV regenerates byte-identically from weights and observations, with no target reference labels. Typed adaptive turnover fees increased matched OOF spellF1 slightly to **0.6154**, but the selected candidate fell to **0.4000** on main. Removing marginal visibility evidence from dead/live odds did not improve OOF: the selected correction strength was zero, reproducing experiment 21 exactly. Neither change replaces the preserved candidates. The original 0.7200 main-score candidate remains preserved. The 0.80 ranking target has not been reached.

Each new study records its protocol before fitting, saves its OOF decision before main scoring, and scores only that selected variant on main validation. All listed results were independently rescored using the unchanged official evaluator. Saved selected outputs regenerate byte-identically. New neural checkpoints, tree-model hashes, source/input hashes and fold preprocessing were verified; graded H/A/D coverage and supplied pre-kickoff D constraints hold.

The same training and validation windows were examined in earlier development. OOF scores select models/decoders and carry selection bias; the main validation is also reused. Do not treat these numbers as an untouched test or a guarantee against overfitting. No private full-match F1 is available.

Private reproducibility artifacts are under `/workspace/kaggle-run/run09` through `run31`, including per-fold predictions, pooled scores, unsuccessful variants, checkpoints, input hashes and independent verification reports. Experimental plans and code are committed on `codex/possession-f1-080`; there has been no GitHub push, environment publication or competition submission.

Current candidate reproduction and tested inference commands are in [CURRENT_MODELS.md](CURRENT_MODELS.md). Full regression at the external owner snapshot: 122 tests, 121 passed and one CUDA-only skip locally; all 122 passed on Kaggle GPU. The later three-state snapshot passed all 126 tests on GPU; the subsequent four target-ablation checks also pass locally.

Experiments 23/24 are external-source studies, not entries in the target-score
table. Both fit only public Games 1/2 and check two hypotheses declared before
either Game-3 result was read. Owner-only transfer improved conditional live-owner
macro F1 0.916737 -> 0.945819 and typed turnover F1 0.254902 -> 0.399317. Three-state
transfer improved weak external spellF1 0.525860 -> 0.570385 and macro frame F1
0.941231 -> 0.944240. Both fixed 60-epoch models regenerated on CPU; source hashes,
training-only normalization and the matched static dead/live baseline were checked.
These shared external development checks are not an independent final benchmark.

The owner-only target ablation did not improve: observed-ball replacement scored
0.603390 OOF spellF1 and full-timeline replacement 0.586667. Both controls reproduced
previous fold predictions. The selected typed-gated control retained 0.4000 main
spellF1 and reproduced its saved CSV byte-identically. No owner-only temporal
variant was promoted and no additional main candidate was inspected.

The three-state external transfer also failed to improve target OOF spellF1:
external weights 0.25/0.5/1.0 scored 0.5400/0.5263/0.4918. The first blend's
frame macro F1 was 0.8158; that does not meet the spellF1 target. Controls and
selected CSV reproduced previous results exactly. Frozen external probabilities
regenerated from source weights and observations. The next fixed study tests
fold-fitted linear adaptation of frozen temporal representations rather than
more unadapted source-probability weights.

Fold-fitted linear adaptation also failed to improve OOF spellF1: dead-only blend
0.5180, full blend 0.4351 and adapted-only 0.4857. All five fold masks excluded the
held window and its 60-second purge; every scaler/coefficient and adapted-score
array refitted exactly. No neural weights were fitted on target labels. The
selected control and full-match CSV reproduced the previous result. The full
132-test snapshot passed locally with one CUDA-only skip. A separately fixed
contrastive-audio experiment is now testing CLAP without audio labels or prompt
tuning. The preserved 0.7200 candidate and the unmet 0.80 target are unchanged.

Before experiment 28's audio extraction, the complete 136-test snapshot passed
on Kaggle GPU. Locally 135 passed with one CUDA-only skip. The pinned CLAP model
loaded successfully and extraction started; this is runtime validation, not a
new possession-score result.

The fixed CLAP run completed all 18 three-seed fits. `clap_graph` scored 0.528814
OOF spellF1 (0.823282 frame macro F1), and `clap_external` improved OOF spellF1 to
0.640000 (0.843823 frame macro F1). The selected model reached only 0.550000 main
spellF1 and 0.630875 frame macro F1, with seven predicted versus nine GT live spells.
It does not replace the preserved 0.7200 main candidate. All six training masks and
normalizers, all 18 checkpoint hashes, fixed model/prompts, audio centers and unit
embeddings were verified. The selected CSV regenerated byte-identically from the
new neural weights and observation caches on CPU.

CLAP has 153,492,890 frozen parameters and encoded 12,712 two-second windows. No
whistle labels or audio fine-tuning were used. The higher reused-window OOF result
did not establish transfer to the reused main window; neither result demonstrates
independent full-match F1. Experiment 29 fixes probability fusion before scoring.

The fixed audio fusion did not improve spellF1. Dead-only CLAP replacement tied
0.6400, while equal-mean dead and full-state fusion scored 0.613333 and 0.620000.
The selected CLAP control retained 0.5500 main spellF1 and reproduced its CSV
byte-identically. All 30 fold and six pooled scores, three controls, source input
hashes and fold-only priors were independently verified. Two new algebra and
team-symmetry checks passed. A fixed 20-epoch comparison now tests reduced training
budget using the exact same fold preprocessing, rather than another encoder.

The fixed external ownership expert and its variation boundary were additionally
recomputed directly from its encoder/probe weights, with no target reference
files. Both matched their saved arrays exactly. Repeating the selected CLAP
reproduction with those regenerated inputs again produced the exact CSV. The
complete 140-test snapshot passed locally (one CUDA-only skip) and on Kaggle GPU
before the fixed shorter-training experiment.

Shorter training did not improve OOF: early graph 0.513208 and early external
0.603279, retaining the 0.6400 control and its 0.5500 main score. All 18 checkpoints
ran exactly 20 epochs; all six fold masks and preprocessing files matched the
60-epoch source byte for byte. Architecture/configuration differed only in epochs
and its corresponding cosine schedule. First-epoch losses matched the original
runs exactly, supporting an isolated training-budget comparison. The control
CSV was copied exactly; no new alternative main prediction was scored.

The factorized tabular dead-ball expert did not improve OOF: tree-only D scored
0.501695, tree/graph D mean 0.537705 and tree/PANNs/CLAP D mean 0.616393. Ownership
was kept independent from the tabular owner; a probability-saturation guard
preserved the frozen expert's owner even when the old P(D) rounded to one.
All 20 fold and four pooled scores, source hashes and the byte-identical control
were independently checked. The 0.6400 OOF control and 0.7200 preserved highest
observed main candidate remain unchanged.

The complete 146-test snapshot passed locally with one CUDA-only skip before
the fixed AST experiment. Schema 2 adds explicit 768-dimensional audio embeddings;
legacy 512-dimensional PANNs/CLAP caches keep schema 1. Native-rate timing and
fold-only PCA tests pass for the new encoder. This is setup validation, not a new
possession-score result.
