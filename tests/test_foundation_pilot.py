"""Zero-shot protocol checks without downloading or executing a large model."""

import json
from pathlib import Path
import shutil
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from foundation_possession.pilot import (OBSERVATIONS, TASK, download_understanding_model,
                                         media, parse_response, prompt, select_frames,
                                         semantic_prompt, parse_semantic_response)
from learned_possession.features import FeatureCache


class FoundationPilotTests(unittest.TestCase):
    def test_semantic_interface_maps_clubs_and_dead_state_without_letter_ambiguity(self):
        self.assertEqual(parse_semantic_response('{"play_state":"live","team_in_control":"Barcelona"}')["label"], "H")
        self.assertEqual(parse_semantic_response('{"play_state":"live","team_in_control":"Athletic"}')["label"], "A")
        self.assertEqual(parse_semantic_response('{"play_state":"dead","team_in_control":null}')["label"], "D")
        self.assertEqual(parse_semantic_response('{"play_state":"live","team_in_control":null}')["label"], "X")
        self.assertEqual(parse_semantic_response('{"play_state":"uncertain","team_in_control":"Barcelona"}')["label"], "X")

    def test_semantic_tracking_does_not_claim_missing_ball_is_inside_or_moving(self):
        x = np.zeros((200, len(OBSERVATIONS)), np.float32)
        x[:, list(OBSERVATIONS).index("activity")] = 1
        cache = FeatureCache(x, OBSERVATIONS, 25, ((1, 0, 199),), {})
        text = semantic_prompt(cache, 100)
        rows = json.loads(text.split("\n")[-1])
        self.assertFalse(rows[2]["ball_tracked"])
        self.assertIsNone(rows[2]["ball_outside_pitch"])
        self.assertIsNone(rows[2]["nearest_Barcelona_player_distance_to_ball_m"])
        self.assertEqual(rows[2]["mean_outfield_player_speed_m_s"], 1)
        self.assertNotIn("tracking observations", semantic_prompt(cache, 100, False).lower())

    def test_hf_loader_can_ignore_unindexed_speech_tensors(self):
        import importlib.util
        if importlib.util.find_spec("transformers") is None:
            self.skipTest("Transformers needed for a tiny synthetic sharded-checkpoint check")
        import torch
        from safetensors.torch import save_file
        from transformers import BertConfig, BertModel
        config = BertConfig(vocab_size=16, hidden_size=8, num_hidden_layers=1,
                            num_attention_heads=2, intermediate_size=16)
        original = BertModel(config)
        state = {k: v.clone().contiguous() for k, v in original.state_dict().items()}
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            config.save_pretrained(root)
            shard = dict(state)
            shard["talker.unused"] = torch.ones(2)
            save_file(shard, root / "part.safetensors")
            index = dict(metadata=dict(source_revision="synthetic"),
                         weight_map={k: "part.safetensors" for k in state})
            (root / "model.safetensors.index.json").write_text(json.dumps(index))
            restored = BertModel.from_pretrained(root, local_files_only=True)
            for key in state:
                torch.testing.assert_close(restored.state_dict()[key], state[key], rtol=0, atol=0)

    def test_temporal_sampling_is_fixed_uniform_and_inside_known_window(self):
        frames = select_frames(24958, 29036, 24)
        self.assertEqual(len(set(frames)), 24)
        self.assertEqual(frames, sorted(frames))
        self.assertEqual(frames[0], 25042)
        self.assertEqual(frames[-1], 28951)
        self.assertGreaterEqual(frames[0] - 62, 24958)
        self.assertLess(frames[-1] + 62, 29036)
        self.assertLessEqual(max(np.diff(frames)) - min(np.diff(frames)), 1)
        with self.assertRaises(ValueError):
            select_frames(10, 5)

    def test_strict_state_parser_preserves_abstentions_and_optional_whistle_format(self):
        self.assertEqual(parse_response('prefix {"label":"H","whistle_in_clip":"no","reason":"pass"} suffix')["label"], "H")
        self.assertEqual(parse_response('{"label":"a"}')["label"], "A")
        self.assertTrue(parse_response('{"label":"X","whistle_in_clip":"uncertain"}')["parse_valid"])
        self.assertEqual(parse_response('{"label":"Barcelona"}')["label"], "X")
        self.assertFalse(parse_response("I cannot answer")["parse_valid"])
        self.assertEqual(parse_response('{"label":"D","whistle_in_clip":true}')["label"], "D")

    def test_prompt_uses_only_whitelisted_observations_and_no_reference_fields(self):
        names = tuple(OBSERVATIONS) + ("whistle", "reference_label", "model_prediction")
        values = np.ones((500, len(names)), np.float32)
        values[:, -3:] = 999
        values[200, 0] = np.nan
        cache = FeatureCache(values, names, 25, ((1, 0, 499),), {"reference_label": "secret"})
        text = prompt(cache, 200)
        self.assertNotIn("reference_label", text)
        self.assertNotIn("model_prediction", text)
        self.assertNotIn("999", text)
        self.assertNotIn("secret", text)
        self.assertIn("null", text)
        self.assertTrue(text.startswith(TASK))
        with self.assertRaises(ValueError):
            prompt(cache, 5)

    def test_understanding_index_omits_speech_weights_without_editing_tensor_files(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            snapshot = root / "snapshot"
            snapshot.mkdir()
            (snapshot / "config.json").write_text(json.dumps(dict(enable_audio_output=True)))
            (snapshot / "LICENSE").write_text("test license")
            index = dict(metadata=dict(total_size=12), weight_map={
                "thinker.layer.weight": "shard1.safetensors",
                "thinker.other.weight": "shard2.safetensors",
                "talker.weight": "shard2.safetensors",
                "token2wav.weight": "shard3.safetensors"})
            (snapshot / "model.safetensors.index.json").write_text(json.dumps(index))
            for i in range(1, 4):
                (snapshot / f"shard{i}.safetensors").write_bytes(bytes([i] * 4))
            calls = []

            def fake_download(*args, **kwargs):
                calls.append(kwargs)
                return str(snapshot)

            with patch.dict("sys.modules", {"huggingface_hub": SimpleNamespace(snapshot_download=fake_download)}):
                out = download_understanding_model(root / "cache")
            selected = json.loads((out / "model.safetensors.index.json").read_text())
            self.assertEqual(set(selected["weight_map"]), {"thinker.layer.weight", "thinker.other.weight"})
            self.assertEqual(calls[1]["allow_patterns"], ["shard1.safetensors", "shard2.safetensors"])
            self.assertFalse(json.loads((out / "config.json").read_text())["enable_audio_output"])
            self.assertEqual((out / "shard2.safetensors").read_bytes(), bytes([2] * 4))
            self.assertFalse((out / "shard3.safetensors").exists())

    @unittest.skipUnless(shutil.which("ffmpeg"), "Synthetic media requires ffmpeg")
    def test_media_uses_exact_center_and_aligned_audio(self):
        import importlib.util
        if importlib.util.find_spec("av") is None or importlib.util.find_spec("torch") is None:
            self.skipTest("PyAV and PyTorch needed for media utilities")
        import subprocess
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "synthetic.mp4"
            subprocess.run(["ffmpeg", "-v", "error", "-f", "lavfi", "-i",
                            "color=c=black:s=64x36:r=25:d=6", "-f", "lavfi", "-i",
                            "sine=frequency=3800:sample_rate=48000:duration=6",
                            "-c:v", "libx264", "-c:a", "aac", "-y", str(path)], check=True)
            images, wave, info = media(path, 75, clip_frames=50, sample_frames=4, size=(64, 36))
            self.assertEqual(images.shape, (4, 36, 64, 3))
            self.assertEqual(len(wave), 32000)
            self.assertEqual(info["first_frame"], 50)
            self.assertEqual(info["requested_video_seconds"], 1.0)
            self.assertIn(75, info["sampled_frames"])
            self.assertGreater(info["audio_coverage"], 0.99)
            self.assertGreater(float(np.std(wave)), 0.01)


if __name__ == "__main__":
    unittest.main()
