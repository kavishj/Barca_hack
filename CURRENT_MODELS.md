# Current models and reproduction

The official 0.80 spellF1 target is **not reached**. The original graph + audio
TCN retains the highest observed draft-validation spellF1, **0.7200**. Its private
prediction file and weights remain unchanged.

The direct external ownership expert reached matched training-window OOF
spellF1 **0.6067**, with frame macro F1 **0.8120**. Its single OOF-selected main
score is **0.6824 spellF1 / 0.6485 frame macro F1**. This is a separate experimental
candidate. Its encoder/probe were fitted on public Metrica Sample Games 1/2;
the original graph supplies dead-ball evidence and ownership at missing-ball
frames. Credit and source pin: [EXTERNAL_PRETRAINING.md](EXTERNAL_PRETRAINING.md).

Later adaptive turnover fees increased OOF spellF1 to **0.6154** but fell to
**0.4000** on main. A visibility-likelihood ablation selected zero correction and
reproduced that result. Both are archived experiments; the two candidates above
remain preserved. External temporal-control pretraining is a separate study with
a public third-match transfer gate before any target-window expansion.

All five target training windows and the nine-possession main validation have
already been reused. Model/decoder selection uses OOF, then only the selected
variant is scored on main. These are exploratory estimates; neither proves
0.80 on an unseen fixture or the private full match. The matched-ensemble study
reduced a one-seed OOF estimate of 0.5754 to 0.5133, illustrating selection
instability. A fresh, untouched evaluation window is needed before asserting
generalization. No reference was edited and no new manual labels were added.

## Predict without target reference labels

Install `requirements-temporal.txt` and PyTorch >=2.4; preserve Kaggle's installed
CUDA PyTorch. The tested external inference used CPU PyTorch 2.8.0, with an encoder
pretrained on Kaggle using PyTorch 2.11.0+cu130. Keep match assets and observation
caches private. The saved bundle contains:

```
deployment.json
graph_tcn/model.json, seed_0/1/2.pt and seed_0/1/2.json
preprocessing/audio_normalizer.npz, symmetric.npz
encoder.pt
probe.npz
```

Use the existing private observation caches with their recorded hashes:

```bash
python -m external_possession.predict \
  --model /private/direct_model \
  --features /private/features.npz \
  --audio /private/audio_cache.npz \
  --graph /private/graph.npz \
  --out /private/direct_predictions.csv --device cpu
```

This recomputes the original three neural seeds, external graph embeddings,
owner probabilities and segmentation from saved weights. It accepts no target
label files and uses no stored frame predictions. Its complete 199,319-row CSV
reproduced the experiment byte for byte:
`3cb5686891835b1bf62beddfbe086ce0277dca3db5191a2e04dd5a5441e1582e`.

The original graph candidate can be regenerated with the same caches:

```bash
python -m relational_possession predict \
  --model /private/direct_model/graph_tcn \
  --features /private/features.npz --audio /private/audio_cache.npz \
  --graph /private/graph.npz --out /private/original_graph.csv --device cpu
```

Its retained CSV SHA-256 is
`c5dd08253f385e37cbdcb0f32006fde09ea65ada622a8f9548524cab6d633857`.
All metrics use the unchanged official scorer. Independent checks verify complete
graded H/A/D coverage, X outside the halves and zero violations of the supplied
pre-kickoff dead-ball fact.

Source, preprocessing, model and reference hashes and all fold/pooled scores were
checked independently. The regression suite ran 122 tests: 121 passed and one
CUDA-only check was skipped locally; the preceding 101-test source snapshot also
passed on Kaggle's GPU. Full-match inference from weights is an additional real
integration check. See [EXPERIMENT_RESULTS.md](EXPERIMENT_RESULTS.md).
