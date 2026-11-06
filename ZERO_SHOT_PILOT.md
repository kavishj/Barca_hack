# Frozen large-model zero-shot pilot

The pilot uses Qwen2.5-Omni-7B with no fine-tuning or labelled prompt examples.
Official checkpoint: https://huggingface.co/Qwen/Qwen2.5-Omni-7B
Pinned revision: ae9e1690543ffd5c0221dc27f79834d0294cba00

The complete checkpoint has 10,732,225,440 parameters. The audio/video/text
understanding model has 8,931,813,888 nominal parameters with speech generation
disabled. Four-bit NF4 quantisation reduces storage, not the nominal architecture.
Audio and vision encoders remain FP16. All weights remain frozen.

Its actual LICENSE is Apache 2.0, retained in third_party/QWEN_OMNI_LICENSE.txt.
The smaller Omni-3B and VL-3B checkpoints use a research-only license; the present
experiment uses the 7B checkpoint. Public weights are downloaded in Kaggle /tmp,
not committed to Git or sent any private media for hosted inference.

## Fixed comparison

Select 24 uniform stratum midpoints inside the same 24958–29035 draft validation
interval, without class-based or model-disagreement selection. Each query receives
a five-second private clip, ten video frames, 16 kHz audio, and seven physical
tracking observations at -2, -1, 0, +1 and +2 seconds from the target instant.
The fixed prompt defines H/A/D/X and relevant possession rules. There are no
labelled demonstrations, reference answers or predictions from other models.
Inference and scoring are separate commands. Invalid answers become X; abstentions
count as misses. Optional whistle descriptions do not determine parsing of a
valid possession answer and have no verified acoustic-event reference.

Compare sparse-frame accuracy, macro F1, D F1 and coverage with the previous TCN
and CPU models at exactly the same 24 moments. No spellF1 is reported for this
sparse pilot, and its classification score is not a full-window result. Only two
sampled moments may fall inside the bounded D interval; uncertainty is substantial.

## Implementation

foundation_possession/pilot.py provides infer and score commands. The default
infer command requires --features, --video and --out; score requires --run,
--truth and optional --compare prediction CSVs. No reference file is read during
inference. Audio/video decoding uses a shared timestamp origin and local PyAV;
the processor uses 2 fps video timing without resampling the already sampled frames.

The loader downloads the four source shards containing original thinker tensors,
omits the speech-only fifth shard, and writes an index selecting unchanged thinker
keys. No tensor values are trained or edited. The source/revision and omitted
modules are recorded. Generation is greedy and text-only. The full nominal model
architecture was counted with empty weights, and the multimodal processor passed
a local synthetic check before any large model download.

Use Kaggle's CUDA PyTorch and matching torchvision. Install requirements-learning.txt
normally in an isolated target, then requirements-foundation.txt with --no-deps,
preserving the CUDA build. Supporting regex, tqdm, requests, Pillow and psutil
come from the Kaggle image. Model files stay outside notebook saved outputs.

All clips, observations and results stay private. These are exploratory results
on a small draft-labelled window, with no final-test or verified whistle-event F1.

## Original pilot and fixed follow-up

The original interface completed all 24 queries with valid JSON, but achieved
6/24 accuracy (25%), macro F1 0.1379 and D F1 0. It returned Athletic 20 times,
dead twice and unknown twice, with no Barcelona predictions. Its reasons usually
discussed whether play was live without identifying a club. On the same moments,
the learned-audio TCN achieved 18/24 accuracy (75%), macro F1 0.6607 and D F1 0.5.
The original outputs are retained, including all errors and abstentions.

The original tracking field names were abbreviated. Missing-ball booleans could
also appear false despite being unobserved. One fixed follow-up uses full team
names, separate play-state and controlling-club outputs, explicit tracking field
definitions, and null for every unobserved ball-derived feature. Its second mode
uses the same semantic output but audio/video alone. Both modes are frozen before
their run and query the same original 24 moments. The model is loaded once.
Use infer --interface dual to run both modes into separate output folders.

This revision follows inspection of validation errors; it remains zero-shot in
the sense of no training or labelled demonstrations, but it is an exploratory
prompt comparison. All three interfaces must be reported. It cannot establish
generalisation on an independent holdout. No further prompt search is planned
within this pilot.

## Completed comparison

| Method | Accuracy | Macro F1 | D F1 | Non-abstention coverage |
|---|---:|---:|---:|---:|
| Omni: original prompt, audio/video + tracking | 6/24 (25.0%) | 0.1379 | 0.0000 | 91.7% |
| Omni: explicit club/state output + tracking | 13/24 (54.2%) | 0.3273 | 0.0000 | 87.5% |
| Omni: explicit club/state output, audio/video alone | 13/24 (54.2%) | 0.2342 | 0.0000 | 100.0% |
| Existing TCN + frozen audio encoder | 18/24 (75.0%) | 0.6607 | 0.5000 | 100.0% |
| Existing tracking TCN | 17/24 (70.8%) | 0.5761 | 0.4000 | 100.0% |
| Existing CPU learned model | 18/24 (75.0%) | 0.6380 | 0.4000 | 100.0% |
| Teammate v6 | 15/24 (62.5%) | 0.4322 | 0.4000 | 100.0% |

All seven methods were independently rescored against the same 24 draft-labelled
moments. These are exploratory sparse classification scores, not spellF1 or a
final-test result. All 39 checks passed locally and in Kaggle. Original and
follow-up results, sources, raw replies and runtime flags are retained privately.
