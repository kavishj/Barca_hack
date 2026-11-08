# Handoff verification

The preserved candidate's CSV contains 199,319 data rows, frame IDs 0..199318,
and passes the unchanged official submission-format validator.

Three real inference checks reproduced its exact bytes:

1. CPU PyTorch 2.8.0 using the ready bundle's saved observation caches.
2. CPU inference from the extracted code-and-weights archive, using those caches.
3. A private Kaggle T4 / PyTorch 2.11.0+cu130 run starting from the five original
   tactical assets. That run received no observation caches, human annotation
   files or stored prediction CSV. It rebuilt the features, PANNs audio and graph,
   ran all three saved neural seeds and the original decoder, and passed the hash.

Prediction SHA-256:

```
c5dd08253f385e37cbdcb0f32006fde09ea65ada622a8f9548524cab6d633857
```

The private raw-reproduction notebook is:
https://www.kaggle.com/code/kavish03/barca-best-reproduction-072-v1

Both archives pass ZIP integrity checks. Their inference source, model weights,
preprocessing and original-input content hashes are recorded in `manifest.json`.
`REPRODUCTION_VERIFICATION.json` contains the CPU and raw-CUDA checks.

This verifies reproducibility and file format. The 0.7200 spell F1 remains a
measurement on the reused small validation window, not a private full-match
score or a claim that the 0.80 target has been reached.
