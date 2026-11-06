# Training-window decoder regularisation

The fixed expanded-data study showed 77 predicted versus 33 true pooled
training-window holdout possessions for the original learned-audio TCN, and
74 versus 33 for its short rich structured counterpart. This follow-up tests
one parameter: switch cost 1, 2, 4 or 8. All neural weights, scalers, PCA and
duration distributions are reused unchanged from that study.

Each saved fold model predicts only with its already fitted preprocessing.
Its cost-1 predictions must reproduce the original held-out frames exactly.
The full half containing that fold is decoded at each cost; the other half
cannot affect the result. Inference preserves original batching for numerical
reproducibility. Only the held-out window contributes to the selection metric.

For each family, choose the highest pooled OOF official spellF1; ties choose
the lowest cost. Save selected_decoder.json before opening the main validation
labels. Only the selected cost is applied to that family's existing three-seed
global score arrays. Copied model metadata records the selected cost; checkpoint
files remain byte-identical. Regeneration reads no reference labels.

The pooled CV scores are used for parameter selection and are therefore not an
unbiased performance estimate after selection. The main nine-possession draft
validation window was already examined in earlier work. Report all costs and
both selected models without claiming an independent or full-match 0.98 result.
No label is edited, no short possession is removed by a hard duration threshold,
and supplied pre-kickoff facts remain enforced automatically.

```bash
python -m structured_possession.calibrate --source-study /private/run05/study \
  --features /private/features.npz --rich-features /private/rich_features.npz \
  --audio /private/audio_cache.npz --train-labels /private/train.csv \
  --validation-labels /private/validation.csv --out /private/calibration
```

## Completed results

| Switch cost | Original learned-audio OOF spellF1 | Predicted / 33 | Structured-rich OOF spellF1 | Predicted / 33 |
|---:|---:|---:|---:|---:|
| 1.0 | 0.2109 | 77 | 0.2430 | 74 |
| 2.0 | 0.2314 | 69 | 0.2928 | 64 |
| 4.0 | 0.2818 | 55 | 0.3356 | 54 |
| 8.0 | 0.3836 | 40 | 0.3167 | 39 |

| Candidate | Validation spellF1 | Macro F1 | Predicted / 9 |
|---|---:|---:|---:|
| Previous best, with supplied kickoff constraint | 0.6182 | 0.7239 | 13 |
| Calibrated baseline, switch cost 8 | 0.6667 | 0.6217 | 6 |
| Calibrated structured, switch cost 4 | 0.5176 | 0.5946 | 8 |

The highest newly observed validation score is 0.6667 from calibrated baseline; the old candidate scored 0.6182. This is an exploratory candidate comparison.
