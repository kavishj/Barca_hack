"""Checks that private annotation exports cannot leak across windows or splits."""

import copy
import csv
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from tools.annotation_pack import (N_FRAMES, build, default_plan, empty_session,
                                   export_reviewed, validate_plan)


class AnnotationPackTests(unittest.TestCase):
    def reviewed_fixture(self):
        # Artificial labels solely for exporter tests; not annotations of footage.
        plan = default_plan()
        session = empty_session(plan)
        session.update(labeller="test reviewer", calibration_confirmed=True)
        for w in plan["windows"]:
            a, b = w["start_frame"], w["stop_frame"]
            session["windows"][w["id"]].update(
                boundaries=[[a, "D"], [a + 25, "H"], [a + 50, "X"],
                            [a + 75, "A"], [b, "X"]], status="reviewed",
                reviewer="test reviewer", reviewed_at="2026-11-04T00:00:00+00:00")
        return plan, session

    def test_plan_has_both_halves_and_separate_context(self):
        plan = validate_plan(default_plan())
        seconds = {}
        for split in ("train", "validation", "test"):
            windows = [w for w in plan["windows"] if w["split"] == split]
            self.assertEqual({w["half"] for w in windows}, {1, 2})
            seconds[split] = sum(w["stop_frame"] - w["start_frame"] for w in windows) / 25
        self.assertEqual(seconds, dict(train=1080, validation=360, test=360))
        bad = copy.deepcopy(plan)
        bad["windows"][1].update(start_frame=25000, stop_frame=29500,
                                 context_start=24750, context_stop=29750)
        with self.assertRaisesRegex(ValueError, "separation"):
            validate_plan(bad)
        bad = copy.deepcopy(plan)
        bad["windows"][0].update(start_frame=12000, stop_frame=16500,
                                 context_start=11750, context_stop=16750)
        with self.assertRaisesRegex(ValueError, "public"):
            validate_plan(bad)

    def test_drafts_cannot_become_ground_truth(self):
        plan, session = self.reviewed_fixture()
        for edit in (lambda s: s["windows"][plan["windows"][0]["id"]].update(status="draft"),
                     lambda s: s.update(calibration_confirmed=False),
                     lambda s: s.update(plan_sha256="different"),
                     lambda s: s.update(provenance="model_prediction")):
            with self.subTest(edit=edit), tempfile.TemporaryDirectory() as directory:
                bad = copy.deepcopy(session)
                edit(bad)
                with self.assertRaises(ValueError):
                    export_reviewed(plan, bad, Path(directory) / "labels")
                self.assertFalse((Path(directory) / "labels").exists())

    def test_boundary_mistakes_do_not_write_partial_export(self):
        plan, session = self.reviewed_fixture()
        w = plan["windows"][0]
        a, b = w["start_frame"], w["stop_frame"]
        for points in ([[a, "H"]], [[a + 1, "H"], [b, "X"]],
                       [[a, "H"], [a, "A"], [b, "X"]],
                       [[a, "H"], [b, "A"]], [[a, "X"], [b, "X"]],
                       [[a, "H"], [b + 1, "X"]], [[a + .5, "H"], [b, "X"]],
                       [[a, None], [b, "X"]], [[a, "HD"], [b, "X"]]):
            with self.subTest(points=points), tempfile.TemporaryDirectory() as directory:
                bad = copy.deepcopy(session)
                bad["windows"][w["id"]]["boundaries"] = points
                with self.assertRaises(ValueError):
                    export_reviewed(plan, bad, Path(directory) / "labels")
                self.assertFalse((Path(directory) / "labels").exists())

    def test_full_length_exports_preserve_uncertainty_and_split_masks(self):
        plan, session = self.reviewed_fixture()
        with tempfile.TemporaryDirectory() as directory:
            report = export_reviewed(plan, session, directory)
            tables = {}
            for split in ("train", "validation", "test"):
                with (Path(directory) / f"{split}.csv").open() as handle:
                    rows = list(csv.DictReader(handle))
                self.assertEqual(len(rows), N_FRAMES)
                self.assertEqual([int(r["frame"]) for r in rows], list(range(N_FRAMES)))
                tables[split] = [r["label"] for r in rows]
            for frame in range(N_FRAMES):
                known = [split for split in tables if tables[split][frame] != "X"]
                self.assertLessEqual(len(known), 1)
                windows = [w for w in plan["windows"]
                           if w["start_frame"] <= frame < w["stop_frame"]]
                if known:
                    self.assertEqual(known, [windows[0]["split"]])
            for w in plan["windows"]:
                a, b, split = w["start_frame"], w["stop_frame"], w["split"]
                self.assertEqual(tables[split][a + 49:a + 76], ["H"] + ["X"] * 25 + ["A"])
                self.assertEqual(tables[split][a - 1], "X")
                self.assertEqual(tables[split][b], "X")
            self.assertTrue(all(abs(w["coverage"] - (1 - 25 / 4500)) < 1e-12
                                for w in report["windows"]))

    def test_build_starts_empty_and_embeds_only_public_calibration_labels(self):
        with tempfile.TemporaryDirectory() as directory:
            # Synthetic renderer fixture only: never require target labels on an
            # external-only training machine, and never write to the real key.
            root = Path(directory)
            public = root / "evaluation/gt/BAR-ATH_possession_gt_public.csv"
            public.parent.mkdir(parents=True)
            with public.open("w", newline="") as handle:
                writer = csv.writer(handle)
                writer.writerow(["frame", "label"])
                writer.writerows((f, "H" if f < 13000 else "D")
                                 for f in range(12000, 15750))
            path = root / "pack"
            with mock.patch("tools.annotation_pack.ROOT", root):
                build(default_plan(), path)
            session = json.loads((path / "blank-review.json").read_text())
            self.assertTrue(all(not e["boundaries"] and e["status"] == "draft"
                                for e in session["windows"].values()))
            self.assertIn('"calibration_truth":', (path / "review.html").read_text())
            self.assertIn('[[12000, "H"], [13000, "D"], [15750, "X"]]',
                          (path / "review.html").read_text())
            self.assertNotIn("/*__PACK_DATA__*/null", (path / "review.html").read_text())


if __name__ == "__main__":
    unittest.main()
