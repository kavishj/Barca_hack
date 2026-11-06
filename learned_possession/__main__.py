"""Prepare, train, evaluate and predict: python -m learned_possession --help."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

import joblib
import numpy as np
import pandas as pd

from .features import FeatureCache
from .model import fit, predict
from .segment import DecoderParams

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    h = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def metrics_module():
    sys.path.insert(0, str(ROOT / "evaluation"))
    import possession_metrics
    return possession_metrics


def read_labels(path, n):
    pm = metrics_module()
    table = pm.read_label_csv(path)
    reason = pm.validate_submission(table, n)
    if reason:
        raise ValueError(f"{path}: {reason}")
    return pm.label_codes(table)


def write_labels(path, labels):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(dict(frame=np.arange(len(labels)), label=labels)).to_csv(
        path, index=False, lineterminator="\n")
    read_labels(path, len(labels))


def parameters(args):
    return DecoderParams(**{k: getattr(args, k) for k in DecoderParams.__dataclass_fields__})


def report(labels, truth, bootstrap):
    pm = metrics_module()
    r = pm.evaluate(truth, pm.label_codes(labels), bootstrap_n=bootstrap)
    return {k: r[k] for k in ("macro_f1", "spell_f1_mean", "spell_bootstrap_low",
                              "spell_bootstrap_high", "n_graded", "n_gt_spells",
                              "n_pred_spells", "mean_spell_duration_s", "per_class")}


def prepare(args):
    if args.data_dir:
        os.environ["MTAH_DATA_DIR"] = str(Path(args.data_dir).resolve())
    # Imports occur after setting MTAH_DATA_DIR. Also support callers that already
    # imported games (e.g. notebooks). Kaggle input mounts are read-only;
    # redirect the existing parser/audio caches without editing baseline sources.
    import games
    if args.data_dir:
        games.DATA_DIR = os.environ["MTAH_DATA_DIR"]
    from games import asset_paths, require
    require(["atd_meta", "atd_raw", "homography", "smart_tagging", "video"])
    from possession import data, audio, signals
    from .features import build
    paths = asset_paths()
    sources = {k: dict(name=Path(v).name, size=Path(v).stat().st_size,
                      mtime_ns=Path(v).stat().st_mtime_ns) for k, v in paths.items()}
    signature = hashlib.sha256(json.dumps(sources, sort_keys=True).encode()).hexdigest()[:16]
    work = Path(args.out).resolve().parent / "raw_cache" / signature
    data.CACHE_DIR = audio.CACHE_DIR = str(work)
    match = data.load_match()
    cache = build(match, signals.compute(match), dict(sources=sources))
    cache.save(args.out)
    print(f"Cached {len(cache.x)} frames × {len(cache.names)} features at {args.out}")


def train(args):
    cache = FeatureCache.load(args.features)
    train_codes = read_labels(args.train_labels, len(cache.x))
    validation = read_labels(args.validation_labels, len(cache.x))
    bundle = fit(cache, train_codes, validation, args.purge_seconds,
                 args.iterations, args.seed, args.threads)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    bundle["metadata"].update(feature_cache_sha256=digest(args.features),
                               training_labels_sha256=digest(args.train_labels),
                               validation_labels_sha256=digest(args.validation_labels),
                               evaluation_kind=("public_smoke_not_independent"
                                                if args.command == "public-smoke"
                                                else "supplied_disjoint_windows"))
    params = parameters(args)
    bundle["decoder_params"] = vars(params)
    joblib.dump(bundle, out / "model.joblib")
    labels = predict(bundle, cache, params, args.threads)
    write_labels(out / "learned.csv", labels)
    # Evaluate exactly the validation mask passed through the training guard.
    validation = np.where(cache.graded_mask(), validation, 3)
    scores = {"learned": report(labels, validation, args.bootstrap)}
    for baseline in args.compare:
        codes = read_labels(baseline, len(cache.x))
        scores[str(baseline)] = report(np.array(["H", "A", "D", "X"])[codes],
                                       validation, args.bootstrap)
    payload = dict(training=bundle["metadata"], decoder=vars(params), scores=scores)
    (out / "report.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    import sklearn
    import scipy
    versions = dict(python=sys.version, numpy=np.__version__, pandas=pd.__version__,
                    sklearn=sklearn.__version__, scipy=scipy.__version__, joblib=joblib.__version__)
    (out / "versions.json").write_text(json.dumps(versions, indent=2) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))
    print(f"Saved model, predictions, report and versions under {out}")


def infer(args):
    cache = FeatureCache.load(args.features)
    bundle = joblib.load(args.model)
    params = DecoderParams(**bundle["decoder_params"])
    labels = predict(bundle, cache, params, args.threads)
    write_labels(args.out, labels)
    print(f"Wrote {len(labels)} rows to {args.out}; SHA-256 {digest(args.out)}")


def public_smoke(args):
    """Exercise real data using a fixed public-window split, never a blind benchmark."""
    cache = FeatureCache.load(args.features)
    public = ROOT / "evaluation" / "gt" / "BAR-ATH_possession_gt_public.csv"
    codes = read_labels(public, len(cache.x))
    runs = metrics_module().valid_runs(codes)
    if len(runs) != 1 or runs[0][1] - runs[0][0] != 7500 or cache.fps != 25:
        raise ValueError("public smoke expects the supplied 7,500-frame, 25 fps window")
    first, stop = runs[0]
    split = first + 4500
    training, validation = np.full(len(codes), "X"), np.full(len(codes), "X")
    letters = np.array(["H", "A", "D", "X"])[codes]
    training[first:split] = letters[first:split]
    validation[split:stop] = letters[split:stop]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    args.train_labels, args.validation_labels = out / "smoke_train.csv", out / "smoke_validation.csv"
    write_labels(args.train_labels, training)
    write_labels(args.validation_labels, validation)
    print("PUBLIC SMOKE ONLY: first 3 min train, last 2 min evaluate, with temporal purging. "
          "These public labels were already used in v6 development. This is not evidence "
          "of generalisation and must not be used for hyperparameter selection.", flush=True)
    train(args)


def decoder_arguments(parser):
    for name, default in vars(DecoderParams()).items():
        parser.add_argument("--" + name.replace("_", "-"), type=type(default), default=default)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("prepare", help="cache label-free observations from the tactical assets")
    p.add_argument("--data-dir", help="directory containing tactical/; otherwise use MTAH_DATA_DIR")
    p.add_argument("--out", required=True)
    p.set_defaults(run=prepare)
    for command in ("train", "public-smoke"):
        p = sub.add_parser(command, help=("fit separate labelled windows" if command == "train"
                                         else "fixed public split: plumbing check, not model selection"))
        p.add_argument("--features", required=True)
        if command == "train":
            p.add_argument("--train-labels", required=True)
            p.add_argument("--validation-labels", required=True)
        p.add_argument("--out", required=True)
        p.add_argument("--compare", nargs="*", default=[])
        p.add_argument("--purge-seconds", type=float, default=10)
        p.add_argument("--iterations", type=int, default=100)
        p.add_argument("--seed", type=int, default=0)
        p.add_argument("--threads", type=int, default=4)
        p.add_argument("--bootstrap", type=int, default=1000)
        decoder_arguments(p)
        p.set_defaults(run=train if command == "train" else public_smoke)
    p = sub.add_parser("predict", help="regenerate CSV from a trained model and feature cache")
    p.add_argument("--features", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--threads", type=int, default=4)
    p.set_defaults(run=infer)
    args = parser.parse_args()
    try:
        if getattr(args, "bootstrap", 0) < 0:
            raise ValueError("bootstrap draws must be nonnegative")
        args.run(args)
    except (ValueError, OSError, ImportError) as error:
        parser.exit(2, f"{error}\n")


if __name__ == "__main__":
    main()
