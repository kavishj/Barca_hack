"""Draft import must not guess missing labels or contaminate a holdout."""

import copy
import unittest
import numpy as np

from tools.annotation_pack import N_FRAMES, PUBLIC, default_plan, empty_session
from tools.import_review_draft import bounded_codes, prepare_inputs


class DraftImportTests(unittest.TestCase):
    def fixture(self):
        p = default_plan()
        s = empty_session(p)
        w = p["windows"][-1]
        a = w["start_frame"]
        s["windows"][w["id"]]["boundaries"] = [[a + 10, "H"], [a + 50, "A"],
                                              [a + 70, "D"], [a + 90, "D"],
                                              [a + 100, "H"]]
        v = p["windows"][0]
        s["windows"][v["id"]]["boundaries"] = [[v["start_frame"] + 20, "H"],
                                              [v["stop_frame"] - 10, "X"]]
        return p, s, w, v

    def test_only_bounded_intervals_are_imported_and_repeats_do_not_split(self):
        _, s, w, _ = self.fixture()
        codes, audit = bounded_codes(w, s["windows"][w["id"]])
        a = w["start_frame"]
        self.assertTrue(np.all(codes[:a + 10] == 3))
        np.testing.assert_array_equal(codes[a + 10:a + 100], [0] * 40 + [1] * 20 + [2] * 30)
        self.assertTrue(np.all(codes[a + 100:] == 3))
        self.assertEqual(audit["status"], "draft")
        self.assertEqual(audit["redundant_marks"], 1)
        self.assertEqual(audit["known_frames"], 90)

    def test_drafts_require_explicit_acceptance_and_reallocation_is_recorded(self):
        p, s, w, v = self.fixture()
        with self.assertRaisesRegex(ValueError, "allow-drafts"):
            prepare_inputs(p, s, [w["id"]], [v["id"]])
        train, validation, audit = prepare_inputs(p, s, [w["id"]], [v["id"]], allow_drafts=True)
        self.assertFalse(np.any((train != 3) & (validation != 3)))
        record = audit["windows"][1]
        self.assertEqual((record["original_split"], record["allocated_split"]), ("test", "validation"))
        self.assertEqual(s["windows"][v["id"]]["status"], "draft")
        self.assertEqual(audit["evaluation_kind"], "provisional_draft_annotation_holdout")

    def test_invalid_marks_and_cross_split_reuse_are_rejected(self):
        p, s, w, v = self.fixture()
        with self.assertRaisesRegex(ValueError, "distinct"):
            prepare_inputs(p, s, [w["id"]], [w["id"]], allow_drafts=True)
        cases = [
            [[w["start_frame"], "H"], [w["start_frame"], "D"]],
            [[w["start_frame"] - 1, "H"]],
            [[w["stop_frame"] + 1, "H"]],
            [[w["start_frame"], "unknown"]],
        ]
        for points in cases:
            with self.subTest(points=points), self.assertRaises(ValueError):
                bounded_codes(w, dict(boundaries=points))
        bad = copy.deepcopy(s)
        bad["plan_sha256"] = "wrong"
        with self.assertRaisesRegex(ValueError, "fingerprint"):
            prepare_inputs(p, bad, [w["id"]], [v["id"]], allow_drafts=True)

    def test_public_reference_is_not_replaced_with_calibration_drafts(self):
        p, s, w, v = self.fixture()
        public = np.full(N_FRAMES, 3, dtype=np.int8)
        public[PUBLIC[0]:PUBLIC[1]] = 0
        s["calibration_review"]["boundaries"] = [[12000, "A"], [12050, "D"], [12100, "X"]]
        train, _, audit = prepare_inputs(p, s, [w["id"]], [v["id"]], public, True)
        np.testing.assert_array_equal(train[PUBLIC[0]:PUBLIC[1]], public[PUBLIC[0]:PUBLIC[1]])
        self.assertFalse(audit["calibration_used_as_new_training_labels"])
        public[1] = 0
        with self.assertRaisesRegex(ValueError, "published five"):
            prepare_inputs(p, s, [w["id"]], [v["id"]], public, True)


if __name__ == "__main__":
    unittest.main()
