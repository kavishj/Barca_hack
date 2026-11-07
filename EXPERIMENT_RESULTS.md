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

The original graph control has **0.5123 OOF / 0.7200 main spellF1**, with 0.7907 / 0.6792 frame macro F1. The historical graph/tree blend reached **0.5754** using one-seed fold estimates and **0.6133** on main. Matching the fold ensemble to the three deployment seeds lowered the best OOF estimate to **0.5133**, with unchanged main spellF1 **0.6133**. The one-seed blend's frame macro F1 was **0.8198**, which is not the ranking metric. These rankings disagree; neither establishes independent full-match improvement. The direct external expert then improved matched OOF spellF1 to **0.6067** and frame macro F1 to **0.8120**, with **0.6824** main spellF1. Its full-match CSV regenerates byte-identically from weights and observations, with no target reference labels. Typed adaptive turnover fees increased matched OOF spellF1 slightly to **0.6154**, but the selected candidate fell to **0.4000** on main. Removing marginal visibility evidence from dead/live odds did not improve OOF: the selected correction strength was zero, reproducing experiment 21 exactly. Neither change replaces the preserved candidates. The original 0.7200 main-score candidate remains preserved. The 0.80 ranking target has not been reached.

Each new study records its protocol before fitting, saves its OOF decision before main scoring, and scores only that selected variant on main validation. All listed results were independently rescored using the unchanged official evaluator. Saved selected outputs regenerate byte-identically. New neural checkpoints, tree-model hashes, source/input hashes and fold preprocessing were verified; graded H/A/D coverage and supplied pre-kickoff D constraints hold.

The same training and validation windows were examined in earlier development. OOF scores select models/decoders and carry selection bias; the main validation is also reused. Do not treat these numbers as an untouched test or a guarantee against overfitting. No private full-match F1 is available.

Private reproducibility artifacts are under `/workspace/kaggle-run/run09` through `run26`, including per-fold predictions, pooled scores, unsuccessful variants, checkpoints, input hashes and independent verification reports. Experimental plans and code are committed on `codex/possession-f1-080`; there has been no GitHub push, environment publication or competition submission.

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
