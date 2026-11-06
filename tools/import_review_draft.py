"""Import explicitly accepted draft annotations for a provisional experiment.

The strict reviewed-label exporter is unchanged. This separate importer only
uses intervals bounded by two supplied marks; it never extends an open tail,
fills an unmarked prefix, changes a label or promotes a draft to reviewed.
"""

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.annotation_pack import (FPS, N_FRAMES, PUBLIC, integer,
                                   plan_digest, validate_plan)

LABELS = np.array(["H", "A", "D", "X"])


def metric_module():
    sys.path.insert(0, str(ROOT / "evaluation"))
    import possession_metrics
    return possession_metrics


def bounded_codes(window, entry):
    points = entry.get("boundaries")
    if not isinstance(points, list):
        raise ValueError("Expected a boundary list")
    previous = window["start_frame"] - 1
    for point in points:
        if not isinstance(point, list) or len(point) != 2:
            raise ValueError("Expected [frame, label] pairs")
        frame, label = point
        if (not integer(frame) or not previous < frame <= window["stop_frame"]
                or frame < window["start_frame"] or label not in ("H", "A", "D", "X")):
            raise ValueError("Invalid, duplicate, unordered or out-of-window boundary")
        previous = frame
    codes = np.full(N_FRAMES, 3, dtype=np.int8)
    for (a, label), (b, _) in zip(points, points[1:]):
        codes[a:b] = np.flatnonzero(LABELS == label)[0]
    known = np.flatnonzero(codes != 3)
    summary = dict(status=entry.get("status"), supplied_marks=len(points),
                   redundant_marks=sum(a[1] == b[1] for a, b in zip(points, points[1:])),
                   known_frames=len(known), known_seconds=len(known) / FPS,
                   first_known=int(known[0]) if len(known) else None,
                   stop_known=int(known[-1]) + 1 if len(known) else None,
                   class_counts={str(l): int(np.sum(codes == i)) for i, l in enumerate(LABELS[:3])},
                   unknown_core_frames=int(np.sum(codes[window["start_frame"]:window["stop_frame"]] == 3)),
                   open_last_mark=points[-1] if points and points[-1][1] != "X" else None)
    return codes, summary


def prepare_inputs(plan, session, train_ids, validation_ids, public=None, allow_drafts=False):
    validate_plan(plan)
    if (session.get("schema") != 1 or session.get("n_frames") != N_FRAMES
            or session.get("fps") != FPS or session.get("plan_sha256") != plan_digest(plan)):
        raise ValueError("Review timing or frozen plan fingerprint mismatch")
    if session.get("provenance") != "independent_video_review_no_model_predictions":
        raise ValueError("Expected independent video annotation provenance")
    windows = {w["id"]: w for w in plan["windows"]}
    if set(session.get("windows", {})) != set(windows):
        raise ValueError("Review windows differ from the frozen plan")
    if (not train_ids or not validation_ids or len(set(train_ids)) != len(train_ids)
            or len(set(validation_ids)) != len(validation_ids)
            or set(train_ids) & set(validation_ids)):
        raise ValueError("Choose distinct, nonempty training and validation window lists")
    train, validation = np.full(N_FRAMES, 3, dtype=np.int8), np.full(N_FRAMES, 3, dtype=np.int8)
    audit = dict(schema=1, evaluation_kind="provisional_draft_annotation_holdout",
                 annotation_status="Original review status preserved; not certified ground truth",
                 plan_sha256=plan_digest(plan), labeller=session.get("labeller"), windows=[],
                 calibration_used_as_new_training_labels=False, public_training_included=public is not None,
                 cutoff_policy="Only between supplied marks; prefixes and open last segments remain X",
                 final_test="No final-test score; allocated validation windows are no longer a final holdout")
    if public is not None:
        if np.shape(public) != (N_FRAMES,) or not np.isin(public, range(4)).all():
            raise ValueError("Invalid public label array")
        if np.any(public[:PUBLIC[0]] != 3) or np.any(public[PUBLIC[1]:] != 3):
            raise ValueError("Public training labels must stay inside the published five minutes")
        train[:] = public
    for split, ids, target in (("train", train_ids, train), ("validation", validation_ids, validation)):
        for ident in ids:
            if ident not in windows:
                raise ValueError("Unknown window ID")
            entry = session["windows"][ident]
            if entry.get("status") != "reviewed" and not allow_drafts:
                raise ValueError("Draft annotations need explicit --allow-drafts for a provisional run")
            if entry.get("status") not in ("draft", "reviewed"):
                raise ValueError("Unknown review status")
            codes, summary = bounded_codes(windows[ident], entry)
            if not summary["known_frames"]:
                raise ValueError(f"{ident}: no bounded, known annotation intervals")
            selected = codes != 3
            if np.any(target[selected] != 3):
                raise ValueError("Labels overlap")
            target[selected] = codes[selected]
            audit["windows"].append(dict(id=ident, original_split=windows[ident]["split"],
                                        allocated_split=split, **summary))
    if np.any((train != 3) & (validation != 3)):
        raise ValueError("Training and validation overlap")
    if set(train[train != 3].tolist()) != {0, 1, 2}:
        raise ValueError("Training needs H, A and D")
    # Calibration is compared only with the already-public reference, never used
    # as an additional independently labelled window.
    if public is not None and session.get("calibration_review", {}).get("boundaries"):
        calibration = dict(start_frame=12000, stop_frame=15750)
        codes, summary = bounded_codes(calibration, session["calibration_review"])
        truth = np.where(codes != 3, public, 3)
        pm = metric_module()
        scores = pm.evaluate(truth, codes, bootstrap_n=0)
        audit["calibration"] = dict(**summary,
            scores={k: scores[k] for k in ("macro_f1", "spell_f1_mean", "n_graded", "n_gt_spells", "n_pred_spells")},
            missed_public_dead_frames=int(np.sum((truth == 2) & (codes != 2))))
    audit["training_counts"] = {str(l): int(np.sum(train == i)) for i, l in enumerate(LABELS[:3])}
    audit["validation_counts"] = {str(l): int(np.sum(validation == i)) for i, l in enumerate(LABELS[:3])}
    return train, validation, audit


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--review", required=True, type=Path)
    parser.add_argument("--train-window", nargs="+", required=True)
    parser.add_argument("--validation-window", nargs="+", required=True)
    parser.add_argument("--include-public-training", action="store_true")
    parser.add_argument("--allow-drafts", action="store_true")
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    public_path = ROOT / "evaluation/gt/BAR-ATH_possession_gt_public.csv"
    pm = metric_module()
    public = pm.label_codes(pm.read_label_csv(public_path)) if args.include_public_training else None
    train, validation, audit = prepare_inputs(
        json.loads(args.plan.read_text()), json.loads(args.review.read_text()),
        args.train_window, args.validation_window, public, args.allow_drafts)
    args.out.mkdir(parents=True, exist_ok=True)
    audit.update(source_review_sha256=sha(args.review), source_plan_sha256=sha(args.plan),
                 public_labels_sha256=sha(public_path) if public is not None else None)
    for name, codes in (("train", train), ("validation", validation)):
        path = args.out / f"{name}.csv"
        pd.DataFrame(dict(frame=np.arange(N_FRAMES), label=LABELS[codes])).to_csv(path, index=False)
        reason = pm.validate_submission(pm.read_label_csv(path), N_FRAMES)
        if reason:
            raise ValueError(reason)
        audit[name + "_labels_sha256"] = sha(path)
    (args.out / "annotation_audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
