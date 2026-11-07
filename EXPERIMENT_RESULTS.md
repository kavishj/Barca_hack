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

The original graph control has **0.5123 OOF / 0.7200 main spellF1**, with 0.7907 / 0.6792 frame macro F1. The historical graph/tree blend reached **0.5754** using one-seed fold estimates and **0.6133** on main. Matching the fold ensemble to the three deployment seeds lowered the best OOF estimate to **0.5133**, with unchanged main spellF1 **0.6133**. The one-seed blend's frame macro F1 was **0.8198**, which is not the ranking metric. These rankings disagree; neither establishes independent full-match improvement. The direct external expert then improved matched OOF spellF1 to **0.6067** and frame macro F1 to **0.8120**, with **0.6824** main spellF1. Its full-match CSV regenerates byte-identically from weights and observations, with no target reference labels. The original 0.7200 main-score candidate remains preserved. The 0.80 ranking target has not been reached.

Each new study records its protocol before fitting, saves its OOF decision before main scoring, and scores only that selected variant on main validation. All listed results were independently rescored using the unchanged official evaluator. Saved selected outputs regenerate byte-identically. New neural checkpoints, tree-model hashes, source/input hashes and fold preprocessing were verified; graded H/A/D coverage and supplied pre-kickoff D constraints hold.

The same training and validation windows were examined in earlier development. OOF scores select models/decoders and carry selection bias; the main validation is also reused. Do not treat these numbers as an untouched test or a guarantee against overfitting. No private full-match F1 is available.

Private reproducibility artifacts are under `/workspace/kaggle-run/run09` through `run20`, including per-fold predictions, pooled scores, unsuccessful variants, checkpoints, input hashes and independent verification reports. Experimental plans and code are committed on `codex/possession-f1-080`; there has been no GitHub push, environment publication or competition submission.

Current candidate reproduction and tested inference commands are in [CURRENT_MODELS.md](CURRENT_MODELS.md). Full regression: 105 tests, 104 passed and one CUDA-only skip locally. The 101-test preceding snapshot also passed on Kaggle GPU.
