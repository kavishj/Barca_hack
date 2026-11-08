# Frozen inference bundle

[barca_best_code_weights.zip](barca_best_code_weights.zip) contains the verified
inference source, three graph/TCN checkpoints, trained preprocessing, public
PANNs Cnn6 checkpoint, licences and reproduction evidence. It is tracked directly
in Git, so a normal clone includes the complete bundle. On GitHub use the ZIP's
**Download raw file** button.

The archive has 46 files and is 22,181,972 bytes. SHA-256:

```text
f19d9472ede53c435a7f0b02d0b1a7021d3ac9da79edd75e2bb821b144e20749
```

It contains no original match assets, frame-level observation caches, additional
human annotations or stored prediction CSV. Match observations are rebuilt from
the separately licensed bundle at inference time.

See the [main README](../README.md#run-the-saved-model) for installation and
execution, and [release checksums](../best_model/release_checksums.json) for
the archive and separate [prediction CSV](../submissions/Barca_hack_possession.csv).
