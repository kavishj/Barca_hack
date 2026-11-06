"""Build a private, offline video-labelling pack and export reviewed split labels.

No possession model is imported. Generated plans, annotations and media belong in
data/labels/ or another private directory, never in the public source tree.
"""

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FPS, N_FRAMES = 25, 199319
HALVES = ((11185, 90652), (110808, 190198))
PUBLIC = (11185, 18685)
SPLITS = ("train", "validation", "test")
# Seeded temporal-stratum selection made before inspecting model predictions or
# video content. These frozen starts make the proposed experiment reproducible.
FROZEN_WINDOWS = (
    ("test", 1, 24935), ("validation", 1, 36748),
    ("train", 1, 55960), ("train", 1, 70260), ("train", 1, 83218),
    ("train", 2, 122726), ("test", 2, 133202), ("train", 2, 152232),
    ("validation", 2, 169253), ("train", 2, 182935),
)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def plan_digest(plan):
    return hashlib.sha256(json.dumps(plan, sort_keys=True).encode()).hexdigest()


def default_plan():
    windows = []
    for i, (split, half, start) in enumerate(FROZEN_WINDOWS):
        windows.append(dict(id=f"{split}_h{half}_{i % 5 + 1:02}", split=split,
                            half=half, start_frame=start, stop_frame=start + 4500,
                            context_start=start - 250, context_stop=start + 4750,
                            selection="seeded_uniform_within_time_stratum"))
    return dict(schema=1, seed=20261104, fps=FPS, n_frames=N_FRAMES,
                selection_inputs="timeline only; no predictions or labels used",
                public_window_excluded=list(PUBLIC), windows=windows)


def integer(value):
    return isinstance(value, int) and not isinstance(value, bool)


def validate_plan(plan):
    if plan.get("schema") != 1 or plan.get("fps") != FPS or plan.get("n_frames") != N_FRAMES:
        raise ValueError("Plan must describe this match's 199319 frames at 25 fps")
    windows = plan.get("windows", [])
    if not windows or {w.get("split") for w in windows} != set(SPLITS):
        raise ValueError("Plan must contain train, validation and test windows")
    seen = set()
    spans = []
    for w in windows:
        if not isinstance(w.get("id"), str) or not w["id"] or w["id"] in seen:
            raise ValueError("Window IDs must be nonempty and unique")
        seen.add(w["id"])
        if w.get("half") not in (1, 2):
            raise ValueError("Invalid half")
        fields = ("start_frame", "stop_frame", "context_start", "context_stop")
        if not all(integer(w.get(k)) for k in fields):
            raise ValueError("Window endpoints must be integer video frames")
        a, b, ca, cb = (w[k] for k in fields)
        ha, hb = HALVES[w["half"] - 1]
        if not ha <= ca <= a < b <= cb <= hb:
            raise ValueError("Window and context must stay inside one graded half")
        if max(ca, PUBLIC[0]) < min(cb, PUBLIC[1]):
            raise ValueError("New windows cannot overlap the public label window")
        spans.append((ca, cb))
    for (_, b), (c, _) in zip(sorted(spans), sorted(spans)[1:]):
        if b + 250 > c:
            raise ValueError("Context windows need at least a 10-second separation")
    return plan


def empty_session(plan):
    return dict(schema=1, plan_sha256=plan_digest(plan), n_frames=N_FRAMES, fps=FPS,
                labeller="", calibration_confirmed=False,
                calibration_review=dict(boundaries=[], status="draft", notes="",
                                        reviewer="", reviewed_at=None),
                provenance="independent_video_review_no_model_predictions",
                windows={w["id"]: dict(boundaries=[], status="draft", notes="",
                                      reviewer="", reviewed_at=None)
                         for w in plan["windows"]})


def validate_boundaries(window, entry):
    points = entry.get("boundaries")
    if not isinstance(points, list):
        raise ValueError("Each window needs a boundaries list")
    a, b = window["start_frame"], window["stop_frame"]
    previous = a - 1
    for point in points:
        if not isinstance(point, list) or len(point) != 2:
            raise ValueError("Boundaries must be [frame, label] pairs")
        frame, label = point
        if not integer(frame) or not previous < frame <= b or frame < a or label not in ("H", "A", "D", "X"):
            raise ValueError("Boundaries must increase, stay inside the window and use H/A/D/X")
        previous = frame
    if not points or points[0][0] != a or points[-1] != [b, "X"]:
        raise ValueError(f"{window['id']}: mark the start and close the window with X at {b}")
    if not any(label != "X" for _, label in points[:-1]):
        raise ValueError(f"{window['id']}: window has no known labels")
    return points


def export_reviewed(plan, session, out):
    """Reject drafts; X preserves uncertainty and all unreviewed/outside frames."""
    validate_plan(plan)
    if session.get("plan_sha256") != plan_digest(plan):
        raise ValueError("Annotation file belongs to a different plan")
    if session.get("schema") != 1 or session.get("n_frames") != N_FRAMES or session.get("fps") != FPS:
        raise ValueError("Annotation schema or video timing mismatch")
    if not str(session.get("labeller", "")).strip() or session.get("calibration_confirmed") is not True:
        raise ValueError("Record a labeller and confirm public-window calibration first")
    if session.get("provenance") != "independent_video_review_no_model_predictions":
        raise ValueError("This export requires independent video review")
    expected = {w["id"] for w in plan["windows"]}
    if set(session.get("windows", {})) != expected:
        raise ValueError("Annotation windows must match the plan exactly")
    arrays = {split: ["X"] * N_FRAMES for split in SPLITS}
    summaries = []
    # Validate everything before writing any CSV, so a partly reviewed session
    # cannot silently become a validation set.
    for w in plan["windows"]:
        entry = session["windows"][w["id"]]
        if entry.get("status") != "reviewed" or not str(entry.get("reviewer", "")).strip() or not entry.get("reviewed_at"):
            raise ValueError(f"{w['id']}: draft or missing review record; no export written")
        points = validate_boundaries(w, entry)
        counts = {label: 0 for label in "HADX"}
        for (a, label), (b, _) in zip(points, points[1:]):
            arrays[w["split"]][a:b] = [label] * (b - a)
            counts[label] += b - a
        summaries.append(dict(id=w["id"], split=w["split"], counts=counts,
                              coverage=1 - counts["X"] / (w["stop_frame"] - w["start_frame"]),
                              reviewer=entry["reviewer"], notes=entry.get("notes", "")))
    if not all(label in arrays["train"] for label in "HAD"):
        raise ValueError("Training labels must contain H, A and D")
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    hashes = {}
    for split, labels in arrays.items():
        path = out / f"{split}.csv"
        with path.open("w", newline="") as handle:
            writer = csv.writer(handle, lineterminator="\n")
            writer.writerow(("frame", "label"))
            writer.writerows(enumerate(labels))
        hashes[split] = digest(path)
    report = dict(plan_sha256=plan_digest(plan), labeller=session["labeller"],
                  created_at=datetime.now(timezone.utc).isoformat(), windows=summaries,
                  label_sha256=hashes, test_policy="Do not use test.csv for fitting or model selection",
                  boundary_policy="Fixed windows censor edge spells; leave ambiguous spans X")
    (out / "review_report.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def build(plan, out):
    validate_plan(plan)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    session = empty_session(plan)
    template = Path(__file__).with_name("annotation_review.html").read_text()
    calibration = dict(id="calibration_public", split="calibration", half=1,
                       start_frame=12000, stop_frame=15750,
                       context_start=11750, context_stop=16000)
    public_path = ROOT / "evaluation" / "gt" / "BAR-ATH_possession_gt_public.csv"
    with public_path.open(newline="") as handle:
        public = {int(row["frame"]): row["label"] for row in csv.DictReader(handle)}
    truth, last = [], None
    for f in range(calibration["start_frame"], calibration["stop_frame"]):
        label = public[f]
        if label != last:
            truth.append([f, label])
            last = label
    truth.append([calibration["stop_frame"], "X"])
    payload = json.dumps(dict(plan=plan, session=session, calibration=calibration,
                              calibration_truth=truth)).replace("<", "\\u003c")
    html = template.replace("/*__PACK_DATA__*/null", payload)
    (out / "review.html").write_text(html)
    (out / "plan.json").write_text(json.dumps(plan, indent=2) + "\n")
    # Do not overwrite a person's saved work on repeated pack builds.
    initial = out / "blank-review.json"
    initial.write_text(json.dumps(session, indent=2) + "\n")
    (out / "START_HERE.txt").write_text(
        "Open review.html in desktop Chrome or Edge. Choose the original "
        "BAR-ATH_tactical.mp4 from your computer; nothing is uploaded.\n"
        "Read the instructions, use the public-calibration window first, then "
        "review the 10 assigned windows. H = Barcelona, A = Athletic, D = dead, "
        "X = uncertain. Sound helps find fouls.\n"
        "Download a JSON backup after each window. Browser autosave is convenient "
        "but is not a portable backup.\n"
        "Watch each full window, refine every change with frame steps, and mark "
        "reviewed only after this second pass. Close with X at the stop frame.\n"
        "Keep all video, screenshots and labels private. The test windows are "
        "labelled blind and their scores are reserved for the final comparison.\n"
        "Return the downloaded review JSON to generate training/validation/test "
        "CSVs. No new ground-truth labels are included in this pack.\n")
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("build")
    p.add_argument("--plan", type=Path, help="existing frozen plan; otherwise use the default")
    p.add_argument("--out", type=Path, required=True)
    p = sub.add_parser("export")
    p.add_argument("--plan", type=Path, required=True)
    p.add_argument("--review", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text()) if args.plan else default_plan()
    if args.command == "build":
        print(build(plan, args.out))
    else:
        report = export_reviewed(plan, json.loads(args.review.read_text()), args.out)
        print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
