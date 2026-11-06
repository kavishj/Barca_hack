"""Frozen PANNs Cnn6 features, with timestamp-preserving audio extraction.

Architecture adapted from Qiuqiang Kong's MIT-licensed PANNs implementation.
See third_party/PANNS_NOTICE.md. Checkpoint filters supply the exact STFT and
mel bank, so inference does not depend on a changing librosa implementation.
"""

from dataclasses import dataclass
from fractions import Fraction
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

SAMPLE_RATE = 32000
CHECKPOINT_SHA256 = "8a17541944f2623a2a0164e072ac2f915dcedca99968083cc44bfbc7eb682275"
WHISTLE_INDICES = (402, 40)  # AudioSet Whistle; Whistling (not referee-specific).


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


class CheckpointSpectrogram(nn.Module):
    def __init__(self):
        super().__init__()
        self.stft = nn.Module()
        self.stft.conv_real = nn.Conv1d(1, 513, 1024, stride=320, bias=False)
        self.stft.conv_imag = nn.Conv1d(1, 513, 1024, stride=320, bias=False)

    def forward(self, wave):
        x = F.pad(wave[:, None], (512, 512), mode="reflect")
        real, imag = self.stft.conv_real(x), self.stft.conv_imag(x)
        return (real.square() + imag.square()).transpose(1, 2)[:, None]


class CheckpointLogmel(nn.Module):
    def __init__(self):
        super().__init__()
        self.melW = nn.Parameter(torch.empty(513, 64), requires_grad=False)

    def forward(self, power):
        return 10 * torch.log10(torch.clamp(power @ self.melW, min=1e-10))


class ConvBlock5x5(nn.Module):
    def __init__(self, source, target):
        super().__init__()
        self.conv1 = nn.Conv2d(source, target, 5, padding=2, bias=False)
        self.bn1 = nn.BatchNorm2d(target)

    def forward(self, x):
        return F.avg_pool2d(F.relu(self.bn1(self.conv1(x))), (2, 2))


class FrozenCnn6(nn.Module):
    def __init__(self):
        super().__init__()
        self.spectrogram_extractor = CheckpointSpectrogram()
        self.logmel_extractor = CheckpointLogmel()
        self.bn0 = nn.BatchNorm2d(64)
        self.conv_block1 = ConvBlock5x5(1, 64)
        self.conv_block2 = ConvBlock5x5(64, 128)
        self.conv_block3 = ConvBlock5x5(128, 256)
        self.conv_block4 = ConvBlock5x5(256, 512)
        self.fc1 = nn.Linear(512, 512)
        self.fc_audioset = nn.Linear(512, 527)

    @classmethod
    def from_checkpoint(cls, path, device="cpu"):
        if sha(path) != CHECKPOINT_SHA256:
            raise ValueError("PANNs checkpoint checksum differs from the pinned public artifact")
        model = cls()
        model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True)["model"], strict=True)
        model.eval().requires_grad_(False)
        return model.to(device)

    def forward(self, wave):
        if self.training:
            raise ValueError("This PANNs encoder is frozen; use eval mode")
        x = self.logmel_extractor(self.spectrogram_extractor(wave))
        x = self.bn0(x.transpose(1, 3)).transpose(1, 3)
        for block in (self.conv_block1, self.conv_block2, self.conv_block3, self.conv_block4):
            x = block(x)
        x = x.mean(dim=3)
        x = x.amax(dim=2) + x.mean(dim=2)
        embedding = F.relu(self.fc1(x))
        scores = torch.sigmoid(self.fc_audioset(embedding))[:, WHISTLE_INDICES]
        return embedding, scores


@dataclass
class AudioCache:
    centers: np.ndarray
    half_ids: np.ndarray
    embeddings: np.ndarray
    scores: np.ndarray
    availability: np.ndarray
    halves: tuple
    fps: int
    provenance: dict

    def __post_init__(self):
        n = len(self.centers)
        if (self.embeddings.shape != (n, 512) or self.scores.shape != (n, 2)
                or self.half_ids.shape != (n,) or self.availability.shape != (n,)
                or not np.isfinite(self.embeddings).all() or not np.isfinite(self.scores).all()
                or not np.isfinite(self.availability).all() or not np.isfinite(self.centers).all()
                or np.any((self.availability < 0) | (self.availability > 1))
                or np.any((self.scores < 0) | (self.scores > 1))):
            raise ValueError("Invalid frozen audio cache")
        self.halves = tuple(tuple(int(v) for v in h) for h in self.halves)
        covered = np.zeros(n, dtype=bool)
        for half, first, last in self.halves:
            selected = self.half_ids == half
            c = self.centers[selected]
            if (not len(c) or np.any(np.diff(c) <= 0)
                    or c[0] != first or c[-1] != last):
                raise ValueError("Audio centers must increase and cover each half's endpoints")
            covered |= selected
        if not covered.all() or self.fps <= 0:
            raise ValueError("Audio cache half or fps mismatch")

    def save(self, path):
        meta = dict(schema=1, halves=self.halves, fps=self.fps, provenance=self.provenance)
        np.savez_compressed(path, centers=self.centers, half_ids=self.half_ids,
                            embeddings=self.embeddings, scores=self.scores,
                            availability=self.availability, metadata=np.array(json.dumps(meta)))

    @classmethod
    def load(cls, path):
        with np.load(path, allow_pickle=False) as a:
            m = json.loads(str(a["metadata"]))
            if m["schema"] != 1:
                raise ValueError("Unknown audio cache schema")
            return cls(a["centers"], a["half_ids"], a["embeddings"], a["scores"],
                       a["availability"], m["halves"], m["fps"], m["provenance"])


def interpolate(cache, values, query):
    """Interpolate independently per half; no halftime extrapolation."""
    query = np.asarray(query)
    values = np.asarray(values)
    if values.ndim != 2 or len(values) != len(cache.centers):
        raise ValueError("Audio values disagree with centers")
    result = np.full((len(query), values.shape[1]), np.nan, dtype=np.float32)
    for half, first, last in cache.halves:
        keep = cache.half_ids == half
        target = (query >= first) & (query <= last)
        for column in range(values.shape[1]):
            result[target, column] = np.interp(query[target], cache.centers[keep], values[keep, column])
    return result


def place_samples(destination, samples, start, spans):
    """Write timestamped samples and record their union, clipping AAC priming."""
    a, b = max(0, start), min(len(destination), start + len(samples))
    if a >= b:
        return
    if spans and a < spans[-1][1] - 1:
        raise ValueError("Audio timestamps moved backwards")
    destination[a:b] = samples[a - start:b - start]
    if spans and a <= spans[-1][1]:
        spans[-1][1] = max(spans[-1][1], b)
    else:
        spans.append([a, b])


def decode_audio(video_path, out, n_frames, fps):
    import av
    n = int(Fraction(n_frames, fps) * SAMPLE_RATE)
    wave = np.memmap(out, mode="w+", dtype=np.float32, shape=(n,))
    wave[:] = 0
    spans = []
    with av.open(str(video_path)) as container:
        video = container.streams.video[0]
        audio = container.streams.audio[0]
        if video.frames != n_frames or video.average_rate != fps:
            raise ValueError("Video frame count or fps differs from the feature cache")
        origin = (video.start_time or 0) * video.time_base
        resampler = av.AudioResampler(format="fltp", layout="mono", rate=SAMPLE_RATE)

        def write(frame):
            if frame.pts is None or frame.time_base is None:
                raise ValueError("Resampled audio lacks a timestamp")
            start = round((frame.pts * frame.time_base - origin) * SAMPLE_RATE)
            place_samples(wave, frame.to_ndarray().reshape(-1), start, spans)

        for frame in container.decode(audio):
            for resampled in resampler.resample(frame):
                write(resampled)
        for resampled in resampler.resample(None):
            write(resampled)
    wave.flush()
    return wave, spans


def covered_fraction(first, stop, spans):
    if stop <= first:
        return 0.0
    return sum(max(0, min(stop, b) - max(first, a)) for a, b in spans) / (stop - first)


def extract(video_path, checkpoint, out, halves, n_frames, fps=25, device="cuda",
            batch_size=32, wave_path="/tmp/barca_audio_32000.f32"):
    if fps != 25 or batch_size < 1:
        raise ValueError("Expected 25 fps and a positive batch size")
    model = FrozenCnn6.from_checkpoint(checkpoint, device)
    print("Decoding timestamped 32 kHz audio; preserving gaps and AAC priming alignment", flush=True)
    wave, spans = decode_audio(video_path, wave_path, n_frames, fps)
    print(f"Audio decoded into {len(spans)} contiguous span(s); starting frozen Cnn6", flush=True)
    centers, ids, clips, availability, embeddings, scores = [], [], [], [], [], []

    def flush():
        if clips:
            with torch.inference_mode():
                e, s = model(torch.from_numpy(np.stack(clips)).to(device))
            embeddings.append(e.cpu().numpy())
            scores.append(s.cpu().numpy())
            clips.clear()

    for half, first, last in halves:
        left, right = round(first / fps * SAMPLE_RATE), round((last + 1) / fps * SAMPLE_RATE)
        locations = np.unique(np.r_[np.arange(first, last, fps * 0.5), last])
        for location in locations:
            center = round((location + 0.5) / fps * SAMPLE_RATE)
            start, stop = center - SAMPLE_RATE, center + SAMPLE_RATE
            a, b = max(start, left), min(stop, right)
            clip = np.zeros(SAMPLE_RATE * 2, dtype=np.float32)
            clip[a - start:b - start] = wave[a:b]
            centers.append(location)
            ids.append(half)
            availability.append(covered_fraction(a, b, spans))
            clips.append(clip)
            if len(clips) >= batch_size:
                flush()
                if len(centers) % 2048 == 0:
                    print(f"Encoded {len(centers)} audio windows", flush=True)
        flush()
        print(f"Encoded audio half {half}: {len(centers)} total centers", flush=True)
    metadata = dict(checkpoint_sha256=sha(checkpoint), checkpoint_license="CC-BY-4.0",
                    source="PANNs Cnn6, AudioSet, frozen eval; no match audio fine-tuning",
                    source_commit="d2f4b8c18eab44737fcc0de1248ae21eb43f6aa4",
                    sample_rate=SAMPLE_RATE, clip_seconds=2, hop_seconds=0.5,
                    timestamp_policy="Resampled audio PTS relative to video PTS; AAC priming clipped",
                    padded_context="Zero outside each graded half", audio_spans=spans,
                    video_name=Path(video_path).name, video_bytes=Path(video_path).stat().st_size,
                    whistle_classes=["Whistle", "Whistling"], whistle_class_indices=list(WHISTLE_INDICES),
                    temporal_resolution="Coarse two-second embeddings; existing tone feature supplies finer timing")
    cache = AudioCache(np.asarray(centers), np.asarray(ids, dtype=np.int8),
                       np.concatenate(embeddings), np.concatenate(scores),
                       np.asarray(availability, dtype=np.float32), halves, fps, metadata)
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    cache.save(out)
    del wave
    Path(wave_path).unlink()
    return cache
