# Best observed validation candidate: graph + PANNs + TCN

## GitHub download

Download [releases/barca_best_code_weights.zip](../releases/barca_best_code_weights.zip)
using GitHub's **Download raw file** button, or follow the bundle link on
[release best-graph-panns-072-v1](https://github.com/kavishj/Barca_hack/releases/tag/best-graph-panns-072-v1).
It contains the companion weights, manifest and pinned inference source needed
by this runner. The source-only `best_model/` directory in a clone is not a
complete installation. The [main README](../README.md) explains the design
decisions, results, exact commands and research training pipeline.

The prediction CSV is also tracked at
[submissions/Barca_hack_possession.csv](../submissions/Barca_hack_possession.csv)
and linked from the release. Licensed observation caches are not uploaded.

This is the preserved candidate with **0.7200 validation spell F1** and
**0.679217 frame macro F1**. Its validation window contains 4,078 known frames
and nine live-possession spells. The window has been reused during development;
this is not an independent full-match score, and the 0.80 target is not reached.
The CLAP experiment has better training-window OOF spell F1 (0.6400), but its
selected validation score is 0.5500, so it is not the model packaged here.

## Files

- `predict_best.py`: inference from the saved weights; no reference labels read.
- `model/graph_tcn/`: all three seed weights and architecture/decoder metadata.
- `model/preprocessing/`: the saved training-only PCA and normalizers.
- `model/panns/`: the unchanged public Cnn6 audio checkpoint.
- `manifest.json`: input/model hashes, video clock and expected prediction hash.
- `data_cache/`: exact label-free observation caches, in the ready bundle only.
- `TECHNICAL_MODEL.md`: architecture, training and evaluation details.

The separate `Barca_hack_possession.csv` is the submission-format prediction file.
The ready bundle also contains a copy under `predictions/`.
No competition form has been submitted. Its SHA-256 is:

```
c5dd08253f385e37cbdcb0f32006fde09ea65ada622a8f9548524cab6d633857
```

## Install

Use Python 3.12 or 3.13. In an isolated CPU environment:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip install -r requirements-cpu.txt
```

CPU reproduction was verified with PyTorch 2.8.0. On Kaggle, preserve its installed
CUDA PyTorch instead of installing `requirements-cpu.txt`; install the other
requirements and select `--device cuda`. No Transformers model download is
required: this bundle contains the PANNs checkpoint and trained graph weights.

## Ready bundle: exact cached observations

Run from the extracted `barca_best_model` directory:

```bash
python predict_best.py --out submission_generated.csv --device cpu
```

This computes all three neural models and the decoder. It does not load a stored
prediction CSV. It checks the resulting file against the preserved candidate's
hash and writes `submission_generated.csv.verification.json`.

## Code-and-weights bundle: original supplied assets

Point to a folder containing the five tactical assets, recursively:

```bash
python predict_best.py --data-dir /path/to/data \
  --work-dir /path/to/writable/work \
  --out submission_generated.csv --device cuda
```

Required files: `BAR-ATH_tactical.mp4`, `BAR-ATH_tactical_atd.xml`,
`BAR-ATH_tactical_atd.txt`, `BAR-ATH_tactical_homography.csv`, and
`BAR-ATH_tactical_smart_tagging.xml`. Their content hashes must match the original
bundle. Folder nesting and file modification times need not match. The runner
rebuilds tracking/heuristic features, the player-ball graph and frozen PANNs audio
features before inference. It accepts no training or validation label paths.

The raw-data route needs writable space for intermediate audio/tracking caches;
GPU inference is recommended. Cached-observation inference can run on CPU.

## Validate

```bash
python evaluation/validate_submission.py submission_generated.csv
```

The CSV has header `frame,label`, frame IDs 0..199318 and labels H/A/D/X.
All graded frames receive H, A or D; X is used outside the graded halves.
The provided pre-kickoff dead interval is applied by the original decoder.

The ready bundle contains derived match observation caches and should remain
private. Original licensed video/tracking and human annotation files are not
included in either archive. Repository source licensing and the public audio
checkpoint attribution are in `LICENSE` and `third_party/PANNS_NOTICE.md`.
