"""Fixed three-arm temporal/audio comparison with evaluation-only validation."""

from dataclasses import asdict
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

from learned_possession.features import FeatureCache
from learned_possession.model import boundary_targets, fit_durations, training_mask
from learned_possession.segment import DecoderParams, decode
from learned_possession.__main__ import read_labels, write_labels
from .audio import AudioCache, sha
from .network import (ARMS, NetworkConfig, Normalizer, TemporalNet, deterministic,
                      infer_seed, train_seed)

ROOT = Path(__file__).resolve().parents[1]
LABELS = np.array(["H", "A", "D", "X"])


def save_json(path, payload):
    Path(path).write_text(json.dumps(payload, indent=2, sort_keys=True,
                                    default=lambda value: value.tolist()) + "\n")


def metrics(truth, labels, bootstrap=1000):
    sys.path.insert(0, str(ROOT / "evaluation"))
    import possession_metrics as pm
    codes = pm.label_codes(labels)
    result = pm.evaluate(truth, codes, bootstrap_n=bootstrap)
    runs = pm.valid_runs(truth)
    gt = [t for t in pm.transitions(truth, runs) if t[2] == pm.D]
    pred = [t for t in pm.transitions(codes, runs) if t[2] == pm.D]
    result["dead_onsets"] = {tol: pm.transition_scores(gt, pred, tol) for tol in (12, 25)}
    result["whistle_event_f1"] = None
    result["whistle_event_note"] = "No verified acoustic-event reference; possession D is not whistle presence"
    return result


def decode_scores(probability, boundary, cache, durations, params):
    labels = np.full(len(cache.x), "X", dtype="<U1")
    for _, first, last in cache.halves:
        labels[first:last + 1] = LABELS[decode(probability[first:last + 1],
                                             boundary[first:last + 1], cache.fps,
                                             params, durations)]
    return labels


def ensemble(arm_dir, cache, audio, device):
    arm_dir = Path(arm_dir)
    info = json.loads((arm_dir / "model.json").read_text())
    if info["schema"] != 1 or info["arm"] not in ARMS:
        raise ValueError("Unknown temporal model schema")
    normalizer = Normalizer.load(arm_dir.parent / "normalizer.npz")
    x = normalizer.transform(cache, audio, info["arm"])
    config = NetworkConfig(**info["network"])
    probability = np.zeros((len(cache.x), 3), dtype=np.float64)
    boundary = np.zeros(len(cache.x), dtype=np.float64)
    for seed in info["seeds"]:
        deterministic(seed)
        model = TemporalNet(x.shape[1], config).to(device)
        checkpoint = arm_dir / f"seed_{seed}.pt"
        fit = json.loads((arm_dir / f"seed_{seed}.json").read_text())
        if sha(checkpoint) != fit["checkpoint_sha256"]:
            raise ValueError("Temporal checkpoint checksum mismatch")
        model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
        p, q = infer_seed(model, x, cache.halves, fit["boundary_positive_weight"], device)
        probability += p
        boundary += q
        del model
    count = len(info["seeds"])
    return probability.astype(np.float32) / count, boundary.astype(np.float32) / count, info


def compare(args):
    if args.device == "cuda" and not torch.cuda.is_available():
        raise ValueError("GPU experiment requires a CUDA device; no silent CPU fallback")
    config = NetworkConfig(epochs=args.epochs)
    seeds = list(args.seeds)
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("Use a fixed list of distinct seeds")
    deterministic(0)
    cache, audio = FeatureCache.load(args.features), AudioCache.load(args.audio)
    train = read_labels(args.train_labels, len(cache.x))
    validation = read_labels(args.validation_labels, len(cache.x))
    selected, valid = training_mask(cache, train, validation, round(cache.fps * 10))
    codes = np.where(selected, train, 3)
    boundary_y, boundary_selected = boundary_targets(codes, round(cache.fps * 0.12))
    durations, duration_counts = fit_durations(codes, cache.fps, cache.halves)
    params = DecoderParams()
    audit = json.loads(Path(args.audit).read_text())
    if (sha(args.train_labels) != audit["train_labels_sha256"]
            or sha(args.validation_labels) != audit["validation_labels_sha256"]):
        raise ValueError("Annotation audit hashes differ from supplied label files")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    plan = dict(schema=1, evaluation_kind="provisional_draft_annotation_holdout",
                network=asdict(config), decoder=asdict(params), seeds=seeds, arms=ARMS,
                epochs_fixed_before_evaluation=True, early_stopping=False, parameter_search=False,
                feature_cache_sha256=sha(args.features), audio_cache_sha256=sha(args.audio),
                train_labels_sha256=sha(args.train_labels), validation_labels_sha256=sha(args.validation_labels),
                annotation_audit=audit, training_frames=int(selected.sum()), validation_frames=int(valid.sum()),
                training_counts={str(LABELS[c]): int(np.sum(codes == c)) for c in range(3)},
                purge_seconds=10, effective_observation_radius_seconds=config.radius / cache.fps + 2.5,
                duration_examples=duration_counts, durations=durations, audio_finetuning=False,
                acoustic_labels_created=False, pca_components=32, pca_fitted_on="available training frames only",
                normalisation_fitted_on="training frames only", equal_input_dimensions=True,
                final_test_evaluated=False, ensemble_policy="mean probabilities of all predeclared seeds",
                heldout_policy="Validation labels used for masking/purging and final scoring, never fitting or stopping")
    plan = json.loads(json.dumps(plan))
    plan_path = out / "experiment.json"
    if plan_path.exists() and json.loads(plan_path.read_text()) != plan:
        raise ValueError("Output directory has a different frozen experiment plan")
    save_json(plan_path, plan)
    print("Frozen experiment plan saved; no validation scoring until all arms finish.", flush=True)
    normal_path = out / "normalizer.npz"
    if normal_path.exists():
        normalizer = Normalizer.load(normal_path)
    else:
        normalizer = Normalizer.fit(cache, audio, selected, components=32)
        normalizer.save(normal_path)
    started = time.monotonic()
    outputs = {}
    for arm in ARMS:
        arm_dir = out / arm
        arm_dir.mkdir(exist_ok=True)
        x = normalizer.transform(cache, audio, arm)
        info = dict(schema=1, arm=arm, seeds=seeds, network=asdict(config),
                    input_channels=x.shape[1], durations=durations, decoder=asdict(params),
                    feature_cache_sha256=plan["feature_cache_sha256"],
                    audio_cache_sha256=plan["audio_cache_sha256"],
                    annotation_status=audit["annotation_status"],
                    evaluation_kind=plan["evaluation_kind"], normalizer_sha256=sha(normal_path))
        save_json(arm_dir / "model.json", info)
        print(f"Training arm {arm}; all arms have {x.shape[1]} input channels", flush=True)
        for seed in seeds:
            checkpoint = arm_dir / f"seed_{seed}.pt"
            fit_path = arm_dir / f"seed_{seed}.json"
            if checkpoint.exists() and fit_path.exists():
                if sha(checkpoint) != json.loads(fit_path.read_text())["checkpoint_sha256"]:
                    raise ValueError("Saved checkpoint differs from its fit record")
                print(f"Reusing completed {arm} seed {seed}", flush=True)
                continue
            model, fit = train_seed(x, codes, selected, boundary_y, boundary_selected,
                                    cache.halves, config, seed, args.device)
            torch.save(model.state_dict(), checkpoint)
            fit["checkpoint_sha256"] = sha(checkpoint)
            save_json(fit_path, fit)
            del model
        del x
        p, q, _ = ensemble(arm_dir, cache, audio, args.device)
        labels = decode_scores(p, q, cache, durations, params)
        write_labels(arm_dir / "predictions.csv", labels)
        np.savez_compressed(arm_dir / "ensemble_scores.npz", probability=p, boundary=q)
        outputs[arm] = labels
    # This is the first point where validation classes influence any calculation.
    truth = np.where(cache.graded_mask(), validation, 3)
    scores = {arm: metrics(truth, labels, args.bootstrap) for arm, labels in outputs.items()}
    for baseline in args.compare:
        codes = read_labels(baseline, len(cache.x))
        scores[str(baseline)] = metrics(truth, LABELS[codes], args.bootstrap)
    report = dict(experiment=plan, elapsed_seconds=time.monotonic() - started, scores=scores,
                  prediction_sha256={arm: sha(out / arm / "predictions.csv") for arm in ARMS})
    save_json(out / "report.json", report)
    import sklearn
    save_json(out / "versions.json", dict(python=sys.version, numpy=np.__version__, sklearn=sklearn.__version__,
                                         torch=torch.__version__, cuda=torch.version.cuda,
                                         device=torch.cuda.get_device_name(0) if args.device == "cuda" else "cpu"))
    for name, score in scores.items():
        print(name, "spellF1", round(score["spell_f1_mean"], 4),
              "macroF1", round(score["macro_f1"], 4),
              "D_F1", round(score["per_class"]["D"]["f1"], 4), flush=True)


def predict(args):
    deterministic(0)
    cache, audio = FeatureCache.load(args.features), AudioCache.load(args.audio)
    info = json.loads((Path(args.model) / "model.json").read_text())
    if (sha(args.features) != info["feature_cache_sha256"]
            or sha(args.audio) != info["audio_cache_sha256"]
            or sha(Path(args.model).parent / "normalizer.npz") != info["normalizer_sha256"]):
        raise ValueError("Prediction inputs or preprocessing differ from the frozen model")
    p, q, _ = ensemble(args.model, cache, audio, args.device)
    labels = decode_scores(p, q, cache, info["durations"], DecoderParams(**info["decoder"]))
    write_labels(args.out, labels)
    print("Regenerated prediction SHA-256", sha(args.out), flush=True)
