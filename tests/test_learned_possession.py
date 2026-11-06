"""Correctness and leakage checks; fixtures are synthetic, never match results."""

import itertools
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd

from learned_possession.features import FeatureCache, build
from learned_possession.model import boundary_targets, fit_durations, training_mask
from learned_possession.segment import DecoderParams, decode
from possession.data import Match

ROOT = Path(__file__).resolve().parents[1]


def cache(x, fps=25, halves=None):
    return FeatureCache(np.asarray(x, dtype=np.float32),
                        tuple("feature_" + str(i) for i in range(np.shape(x)[1])), fps,
                        halves or ((1, 0, len(x) - 1),), {"fixture": "synthetic"})


class DecoderTests(unittest.TestCase):
    def test_soft_duration_objective_matches_exhaustive_search(self):
        rng = np.random.default_rng(7)
        probability = rng.dirichlet(np.ones(3), size=7)
        boundary = rng.uniform(0.05, 0.95, size=7)
        durations = [(np.log(2), 0.8), (np.log(3), 1.0), (np.log(4), 0.9)]
        p = DecoderParams(grid_frames=1, duration_weight=0.7)

        def objective(path):
            total = sum(np.log(probability[t, c]) for t, c in enumerate(path))
            edges = np.r_[0, np.flatnonzero(np.diff(path)) + 1, len(path)]
            for t in edges[1:-1]:
                total += p.boundary_weight * np.log(boundary[t]) - p.switch_cost
            for a, b in zip(edges[:-1], edges[1:]):
                if a and b < len(path):
                    mu, sigma = durations[path[a]]
                    total -= p.duration_weight * 0.5 * ((np.log(b - a) - mu) / sigma) ** 2
            return total

        expected = max(objective(path) for path in itertools.product(range(3), repeat=7))
        result = decode(probability, boundary, 1, p, durations)
        self.assertAlmostEqual(objective(result), expected, places=10)

    def test_dense_decoder_matches_exhaustive_state_sequences(self):
        rng = np.random.default_rng(2026)
        p = DecoderParams(grid_frames=1, switch_cost=0.7,
                          boundary_weight=0.4, duration_weight=0)
        for _ in range(12):
            probability = rng.dirichlet(np.ones(3), size=6)
            boundary = rng.uniform(0.05, 0.95, size=6)
            score = np.log(np.clip(probability, 1e-6, 1)) / 2
            bonus = np.log(boundary) * p.boundary_weight

            def objective(path):
                total = sum(score[t, c] for t, c in enumerate(path))
                return total + sum(bonus[t] - p.switch_cost for t in range(1, len(path))
                                   if path[t] != path[t - 1])

            expected = max(objective(path) for path in itertools.product(range(3), repeat=6))
            actual = decode(probability, boundary, 2, p)
            self.assertAlmostEqual(objective(actual), expected, places=10)

    def test_short_possession_and_quick_stoppage_can_survive(self):
        truth = np.array([0] * 25 + [1] * 10 + [0] * 25 + [2] * 10 + [0] * 25)
        probability = np.full((len(truth), 3), 0.0005)
        probability[np.arange(len(truth)), truth] = 0.999
        boundary = np.full(len(truth), 0.02)
        boundary[np.flatnonzero(np.diff(truth)) + 1] = 0.99
        result = decode(probability, boundary, 25, DecoderParams(duration_weight=0))
        np.testing.assert_array_equal(result, truth)

    def test_no_state_is_carried_across_halves(self):
        for state in (0, 1, 2):
            probability = np.full((10, 3), 0.01)
            probability[:, state] = 0.98
            path = decode(probability, np.full(10, 0.01), 25)
            np.testing.assert_array_equal(path, np.full(10, state))

    def test_invalid_probabilities_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "invalid state"):
            decode(np.array([[0.5, 0.5, np.nan]]), np.array([0.5]), 25)


class LeakageTests(unittest.TestCase):
    def test_overlapping_labels_are_rejected(self):
        c = cache(np.zeros((100, 1)))
        train = np.tile([0, 1, 2, 3], 25)
        with self.assertRaisesRegex(ValueError, "overlap"):
            training_mask(c, train, train, 0)

    def test_purge_and_half_mask_are_respected(self):
        c = cache(np.zeros((100, 1)), halves=((1, 10, 89),))
        train = np.full(100, 3)
        train[10:20], train[20:30], train[30:50] = 0, 1, 2
        train[:10] = 0
        validation = np.full(100, 3)
        validation[50:60] = 1
        selected, valid = training_mask(c, train, validation, 10)
        self.assertFalse(selected[:10].any())
        self.assertFalse(selected[40:].any())
        self.assertEqual(valid.sum(), 10)

    def test_unknown_window_edges_do_not_create_boundary_targets(self):
        codes = np.array([3] * 5 + [0] * 20 + [1] * 20 + [3] * 5)
        target, selected = boundary_targets(codes, 2)
        self.assertFalse(target[5])
        self.assertFalse(selected[5])
        np.testing.assert_array_equal(np.flatnonzero(target), np.arange(23, 28))

    def test_duration_fit_discards_censored_segments(self):
        codes = np.array([3] * 5 + [2] * 10 + [0] * 20 + [2] * 10 + [0] * 40 + [3] * 5)
        distributions, counts = fit_durations(codes, 25, [(1, 0, len(codes) - 1)])
        self.assertEqual(counts, [1, 0, 1])
        self.assertEqual(distributions, [None, None, None])


class FeatureTests(unittest.TestCase):
    def test_prepare_reads_raw_inputs_and_writes_caches_outside_input(self):
        import games
        from learned_possession.__main__ import prepare
        from possession import data, audio
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset, output = root / "input", root / "output" / "features.npz"
            tactical = dataset / "tactical"
            tactical.mkdir(parents=True)
            stem = tactical / "BAR-ATH_tactical"
            teams = '<Team id="h"/><Team id="a"/>'
            players = ''.join(f'<Player teamId="{team}"/>' for team in ['h'] * 24 + ['a'] * 22)
            Path(str(stem) + "_atd.xml").write_text('<Root>' + teams + players + '</Root>')
            group = ';'.join(['0.3,0.4,2'] * 24 + ['0.7,0.6,2'] * 22)
            Path(str(stem) + "_atd.txt").write_text(''.join(
                f'{i + 1}:{group}:{"NaN,NaN,NaN" if 160 <= i < 200 else "0.31,0.4,NaN"}\n'
                for i in range(400)))
            h = pd.DataFrame(np.tile(np.eye(3).ravel(), (400, 1)),
                             columns=['h00', 'h01', 'h02', 'h10', 'h11', 'h12', 'h20', 'h21', 'h22'])
            h.insert(0, 'frame', np.arange(400))
            h.to_csv(str(stem) + "_homography.csv", index=False)
            Path(str(stem) + "_smart_tagging.xml").write_text(
                '<root><instance><start>1</start><end>2</end><code>PROGRESSION</code>'
                '<label><group>Team</group><text>Barcelona</text></label></instance></root>')
            Path(str(stem) + ".mp4").write_bytes(b'audio is stubbed in this parser integration test')
            config = dict(games.GAMES['BAR-ATH'], offsets={1: 0, 2: 0},
                          halves={1: (20, 179), 2: (220, 379)})
            # Actual ATD/XML/homography/tag parsing is exercised; decoding the real
            # stadium audio still requires the licensed mp4 and PyAV.
            with patch.dict(games.GAMES, {'BAR-ATH': config}), \
                 patch.object(games, 'DATA_DIR', games.DATA_DIR), \
                 patch.object(data, 'CACHE_DIR', data.CACHE_DIR), \
                 patch.object(audio, 'CACHE_DIR', audio.CACHE_DIR), \
                 patch.dict('os.environ', {}, clear=False), \
                 patch('possession.signals.whistle_track', lambda n: np.zeros(n)):
                prepare(SimpleNamespace(data_dir=str(dataset), out=str(output)))
            result = FeatureCache.load(output)
            self.assertEqual(len(result.x), 400)
            self.assertEqual(len(result.provenance['sources']), 5)
            self.assertFalse((dataset / 'cache').exists())
            self.assertTrue(list((output.parent / 'raw_cache').rglob('*.npz')))

    def test_missing_ball_is_nan_and_windows_never_cross_halftime(self):
        n = 40
        players = np.zeros((n, 46, 2), dtype=np.float32)
        players[:, :, 0], players[:, :, 1] = 30, 20
        ball = np.full((n, 2), np.nan, dtype=np.float32)
        team = np.r_[np.zeros(24), np.ones(22)].astype(np.int8)
        match = Match(players, np.ones((n, 46)), team, ball,
                      np.repeat(np.eye(3)[None], n, axis=0), (105.3, 68), 25,
                      [(1, 0, 14), (2, 25, 39)], True)
        sig = SimpleNamespace(ball_visible=np.zeros(n, dtype=bool),
                              ball_speed=np.full(n, np.nan), activity=np.r_[np.ones(20), np.full(20, 9)],
                              d_home=np.full(n, np.nan), d_away=np.full(n, np.nan),
                              camera_x=np.zeros(n), blackout_len=np.full(n, n))
        for name in ("whistle", "tag_restart", "tag_phase", "ball_still", "ball_oob",
                     "ball_placed", "ball_placed_touchline", "pre_kickoff"):
            setattr(sig, name, np.zeros(n))
        result = build(match, sig)
        feature = lambda name: result.x[:, result.names.index(name)]
        self.assertTrue(np.isnan(feature("home_nearest_1")).all())
        self.assertTrue(np.isnan(feature("home_within_2m")).all())
        self.assertTrue((feature("activity_mean_5s")[:15] == 1).all())
        self.assertTrue((feature("activity_mean_5s")[25:] == 9).all())
        self.assertTrue(np.isnan(feature("activity_mean_5s")[15:25]).all())
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "cache.npz"
            result.save(path)
            loaded = FeatureCache.load(path)
            np.testing.assert_array_equal(loaded.x, result.x)
            self.assertEqual(loaded.names, result.names)


class CommandLineTests(unittest.TestCase):
    def test_train_evaluate_reload_and_predict_roundtrip(self):
        rng = np.random.default_rng(19)
        n = 1400
        y = (np.arange(n) // 50 % 3).astype(np.int8)
        x = np.eye(3)[y] + rng.normal(0, 0.03, size=(n, 3))
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            features = directory / "features.npz"
            cache(x, fps=10, halves=((1, 20, 679), (2, 720, 1379))).save(features)
            train, validation = np.full(n, "X"), np.full(n, "X")
            train[20:650] = np.array(["H", "A", "D"])[y[20:650]]
            validation[850:1370] = np.array(["H", "A", "D"])[y[850:1370]]
            for name, labels in (("train", train), ("validation", validation)):
                pd.DataFrame(dict(frame=np.arange(n), label=labels)).to_csv(
                    directory / (name + ".csv"), index=False)
            out = directory / "run"
            commands = [
                ["train", "--features", str(features), "--train-labels", str(directory / "train.csv"),
                 "--validation-labels", str(directory / "validation.csv"), "--out", str(out),
                 "--iterations", "10", "--bootstrap", "10", "--threads", "1"],
                ["predict", "--features", str(features), "--model", str(out / "model.joblib"),
                 "--out", str(directory / "regenerated.csv"), "--threads", "1"],
            ]
            for command in commands:
                result = subprocess.run([sys.executable, "-m", "learned_possession", *command],
                                        cwd=ROOT, text=True, capture_output=True)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertEqual((out / "learned.csv").read_bytes(),
                             (directory / "regenerated.csv").read_bytes())
            report = json.loads((out / "report.json").read_text())
            self.assertEqual(report["scores"]["learned"]["n_graded"], 520)
            labels = pd.read_csv(out / "learned.csv").label.values
            self.assertTrue((labels[680:720] == "X").all())
            self.assertFalse((labels[20:680] == "X").any())


if __name__ == "__main__":
    unittest.main()
