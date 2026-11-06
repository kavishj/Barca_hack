"""Independent numerical, alignment, leakage and checkpoint-roundtrip checks."""

from argparse import Namespace
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

HAS_TORCH = importlib.util.find_spec("torch") is not None
if HAS_TORCH:
    import torch
    from learned_possession.features import FeatureCache
    from learned_possession.model import boundary_targets
    from learned_possession.__main__ import write_labels
    from temporal_possession.audio import (AudioCache, CheckpointSpectrogram, covered_fraction,
                                            decode_audio, interpolate, place_samples)
    from temporal_possession.network import (ARMS, NetworkConfig, Normalizer, TemporalNet,
                                              context, deterministic, epoch_tiles, infer_seed, train_seed)


def fixture(n=300):
    rng = np.random.default_rng(7)
    halves = ((1, 0, n // 2 - 11), (2, n // 2 + 10, n - 1))
    x = rng.normal(size=(n, 2)).astype(np.float32)
    x[20:30, 0] = np.nan
    cache = FeatureCache(x, ("activity", "whistle"), 25, halves, {"fixture": "synthetic"})
    centers, ids = [], []
    for h, first, last in halves:
        c = np.unique(np.r_[np.arange(first, last, 10), last])
        centers.extend(c)
        ids.extend([h] * len(c))
    k = len(centers)
    audio = AudioCache(np.array(centers), np.array(ids), rng.normal(size=(k, 512)).astype(np.float32),
                       rng.uniform(size=(k, 2)).astype(np.float32), np.ones(k, np.float32),
                       halves, 25, {"fixture": "synthetic"})
    return cache, audio


@unittest.skipUnless(HAS_TORCH, "Install PyTorch for temporal-model checks")
class AudioTests(unittest.TestCase):
    def test_checkpoint_stft_matches_independent_numpy_fft(self):
        spec = CheckpointSpectrogram()
        k, j = np.arange(513)[:, None], np.arange(1024)[None, :]
        window = np.hanning(1025)[:-1]
        real = (np.cos(2 * np.pi * k * j / 1024) * window).astype(np.float32)
        imag = (-np.sin(2 * np.pi * k * j / 1024) * window).astype(np.float32)
        with torch.no_grad():
            spec.stft.conv_real.weight.copy_(torch.from_numpy(real[:, None]))
            spec.stft.conv_imag.weight.copy_(torch.from_numpy(imag[:, None]))
        wave = np.random.default_rng(0).normal(size=6400).astype(np.float32)
        padded = np.pad(wave, (512, 512), mode="reflect")
        expected = np.array([np.abs(np.fft.rfft(padded[a:a + 1024] * window)) ** 2
                             for a in range(0, len(padded) - 1023, 320)])
        with torch.inference_mode():
            actual = spec(torch.from_numpy(wave[None])).numpy()[0, 0]
        np.testing.assert_allclose(actual, expected, rtol=0.001, atol=0.01)

    def test_audio_interpolation_never_crosses_halftime(self):
        _, audio = fixture()
        values = np.where((audio.half_ids == 1)[:, None], 1, 100).astype(np.float32)
        out = interpolate(audio, values, np.array([139, 140, 150, 159, 160]))
        self.assertEqual(out[0, 0], 1)
        self.assertTrue(np.isnan(out[1:4]).all())
        self.assertEqual(out[4, 0], 100)

    def test_timestamp_placement_clips_priming_preserves_gaps_and_rejects_reordering(self):
        out = np.zeros(30, np.float32)
        spans = []
        place_samples(out, np.arange(10, dtype=np.float32), -4, spans)
        np.testing.assert_array_equal(out[:6], np.arange(4, 10))
        place_samples(out, np.ones(5), 10, spans)
        self.assertEqual(spans, [[0, 6], [10, 15]])
        self.assertEqual(covered_fraction(0, 20, spans), 11 / 20)
        self.assertTrue(np.all(out[6:10] == 0))
        with self.assertRaisesRegex(ValueError, "backwards"):
            place_samples(out, np.ones(5), 2, spans)

    @unittest.skipUnless(shutil.which("ffmpeg") and importlib.util.find_spec("av"),
                         "PyAV and ffmpeg needed for synthetic media alignment")
    def test_delayed_audio_is_not_shifted_to_video_zero(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "delayed.mp4"
            subprocess.run(["ffmpeg", "-v", "error", "-f", "lavfi", "-i",
                            "color=c=black:s=32x32:r=25:d=2", "-f", "lavfi", "-i",
                            "sine=frequency=3800:sample_rate=48000:duration=1",
                            "-filter_complex", "[1:a]asetpts=PTS+0.2/TB[a]",
                            "-map", "0:v", "-map", "[a]", "-c:v", "libx264", "-c:a", "aac",
                            "-y", str(path)], check=True)
            wave, spans = decode_audio(path, Path(folder) / "wave.f32", 50, 25)
            energy = np.sqrt(np.mean(np.array(wave).reshape(-1, 320) ** 2, axis=1))
            onset = np.flatnonzero(energy > 0.01)[0] * 0.01
            self.assertTrue(0.15 <= onset <= 0.23, onset)
            self.assertLess(float(energy[:12].max()), 0.001)
            self.assertGreater(spans[0][0], 0)
            del wave


@unittest.skipUnless(HAS_TORCH, "Install PyTorch for temporal-model checks")
class PreprocessingTests(unittest.TestCase):
    def test_only_training_frames_fit_pca_and_scaling(self):
        cache, audio = fixture()
        selected = np.zeros(300, bool)
        selected[10:110] = True
        first = Normalizer.fit(cache, audio, selected, components=4)
        changed, new_audio = copy.deepcopy(cache), copy.deepcopy(audio)
        changed.x[160:] += 50000
        new_audio.embeddings[new_audio.half_ids == 2] += 10000
        new_audio.scores[new_audio.half_ids == 2] = 1
        second = Normalizer.fit(changed, new_audio, selected, components=4)
        for a, b in ((first.mean, second.mean), (first.scale, second.scale),
                     (first.pca_mean, second.pca_mean), (first.pca_components, second.pca_components)):
            np.testing.assert_array_equal(a, b)

    def test_ablation_removes_audio_values_and_missingness_with_equal_capacity(self):
        cache, audio = fixture()
        selected = np.zeros(300, bool)
        selected[10:110] = True
        norm = Normalizer.fit(cache, audio, selected, components=4)
        original = {arm: norm.transform(cache, audio, arm) for arm in ARMS}
        changed, new_audio = copy.deepcopy(cache), copy.deepcopy(audio)
        changed.x[:, 1] = np.nan
        new_audio.embeddings += 100
        new_audio.scores[:] = 0
        np.testing.assert_array_equal(original["tracking"], norm.transform(changed, new_audio, "tracking"))
        np.testing.assert_array_equal(original["heuristic_audio"], norm.transform(cache, new_audio, "heuristic_audio"))
        self.assertFalse(np.allclose(original["learned_audio"], norm.transform(cache, new_audio, "learned_audio")))
        self.assertEqual(len({x.shape for x in original.values()}), 1)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "norm.npz"
            norm.save(path)
            np.testing.assert_array_equal(original["learned_audio"],
                                          Normalizer.load(path).transform(cache, audio, "learned_audio"))


@unittest.skipUnless(HAS_TORCH, "Install PyTorch for temporal-model checks")
class TemporalTests(unittest.TestCase):
    def test_chunking_matches_whole_half_and_other_half_cannot_influence_scores(self):
        deterministic(4)
        config = NetworkConfig(width=8, dilations=(1, 2), dropout=0)
        model = TemporalNet(4, config).eval()
        x = np.random.default_rng(2).normal(size=(270, 4)).astype(np.float32)
        halves = ((1, 10, 129), (2, 150, 249))
        full, qfull = infer_seed(model, x, halves, 3, "cpu", core=120, batch_size=1)
        chunk, qchunk = infer_seed(model, x, halves, 3, "cpu", core=16, batch_size=3)
        np.testing.assert_allclose(chunk, full, atol=1e-6)
        np.testing.assert_allclose(qchunk, qfull, atol=1e-6)
        x[150:] += 10000
        changed, _ = infer_seed(model, x, halves, 3, "cpu", core=16, batch_size=3)
        np.testing.assert_array_equal(chunk[10:130], changed[10:130])

    def test_every_known_training_frame_is_supervised_once_despite_unknown_gaps(self):
        selected = np.zeros(150, bool)
        selected[20:60] = True
        selected[70:100] = True
        count = np.zeros(150, int)
        for start, _, _, first, stop in epoch_tiles(selected, ((1, 0, 149),), 64, np.random.default_rng(0)):
            a, b = max(start, first), min(start + 64, stop)
            count[a:b] += 1
        np.testing.assert_array_equal(count, selected.astype(int))

    def test_training_is_seeded_and_weights_reload_without_labels(self):
        from temporal_possession.network import train_seed
        x = np.random.default_rng(1).normal(size=(240, 4)).astype(np.float32)
        codes = np.repeat(np.array([0, 1, 2, 0, 1, 2], np.int8), 40)
        selected = np.ones(240, bool)
        by, bm = boundary_targets(codes, 3)
        config = NetworkConfig(width=4, dilations=(1,), dropout=0.1, epochs=2, core_frames=32, batch_size=4)
        m1, meta = train_seed(x, codes, selected, by, bm, ((1, 0, 239),), config, 5, "cpu")
        m2, _ = train_seed(x, codes, selected, by, bm, ((1, 0, 239),), config, 5, "cpu")
        for key in m1.state_dict():
            torch.testing.assert_close(m1.state_dict()[key], m2.state_dict()[key], rtol=0, atol=0)
        p1, _ = infer_seed(m1, x, ((1, 0, 239),), meta["boundary_positive_weight"], "cpu")
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "weights.pt"
            torch.save(m1.state_dict(), path)
            restored = TemporalNet(4, config)
            restored.load_state_dict(torch.load(path, weights_only=True))
            p2, _ = infer_seed(restored, x, ((1, 0, 239),), meta["boundary_positive_weight"], "cpu")
            np.testing.assert_array_equal(p1, p2)

    def test_experiment_and_prediction_roundtrip(self):
        from temporal_possession.experiment import compare, predict
        cache, audio = fixture(2000)
        train, validation = np.full(2000, "X"), np.full(2000, "X")
        train[100:250], train[250:400], train[400:600] = "H", "A", "D"
        validation[1400:1500], validation[1500:1600], validation[1600:1700] = "H", "A", "D"
        sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            features, sound = root / "features.npz", root / "audio.npz"
            cache.save(features)
            audio.save(sound)
            tr, va = root / "train.csv", root / "validation.csv"
            write_labels(tr, train)
            write_labels(va, validation)
            audit = root / "audit.json"
            audit.write_text(json.dumps(dict(train_labels_sha256=sha(tr), validation_labels_sha256=sha(va),
                                              annotation_status="synthetic fixture")))
            run = root / "run"
            compare(Namespace(device="cpu", epochs=1, seeds=[0], features=str(features), audio=str(sound),
                              train_labels=str(tr), validation_labels=str(va), audit=str(audit),
                              out=str(run), bootstrap=0, compare=[]))
            for arm in ARMS:
                output = root / f"{arm}.csv"
                with patch("temporal_possession.experiment.read_labels", side_effect=AssertionError("labels read during prediction")):
                    predict(Namespace(features=str(features), audio=str(sound), device="cpu",
                                      model=str(run / arm), out=str(output)))
                self.assertEqual(output.read_bytes(), (run / arm / "predictions.csv").read_bytes())
            info = json.loads((run / "report.json").read_text())
            self.assertEqual(info["experiment"]["training_frames"], 500)
            self.assertEqual(info["experiment"]["validation_frames"], 300)


if __name__ == "__main__":
    unittest.main()
