"""Small offline TCN, train-only preprocessing and strictly gated audio arms."""

from dataclasses import dataclass
import os
import random

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
from sklearn.decomposition import PCA
from threadpoolctl import threadpool_limits
import torch
from torch import nn
from torch.nn import functional as F

from .audio import interpolate

ARMS = ("tracking", "heuristic_audio", "learned_audio")


def deterministic(seed, threads=4):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(threads)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


@dataclass(frozen=True)
class NetworkConfig:
    width: int = 32
    dilations: tuple = (1, 2, 4, 8, 16, 32)
    dropout: float = 0.15
    core_frames: int = 256
    batch_size: int = 8
    epochs: int = 60
    learning_rate: float = 0.001
    weight_decay: float = 0.001

    def __post_init__(self):
        if (self.width < 1 or not self.dilations or any(d < 1 for d in self.dilations)
                or not 0 <= self.dropout < 1 or min(self.core_frames, self.batch_size, self.epochs) < 1
                or self.learning_rate <= 0 or self.weight_decay < 0):
            raise ValueError("Invalid temporal network settings")

    @property
    def radius(self):
        return 2 * sum(self.dilations)


class ResidualBlock(nn.Module):
    def __init__(self, width, dilation, dropout):
        super().__init__()
        self.conv1 = nn.Conv1d(width, width, 3, padding=dilation, dilation=dilation)
        self.conv2 = nn.Conv1d(width, width, 3, padding=dilation, dilation=dilation)
        # Only channels at a timestamp are normalised; never the whole time axis.
        self.norm1 = nn.LayerNorm(width)
        self.norm2 = nn.LayerNorm(width)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        z = self.conv1(x)
        z = self.dropout(F.gelu(self.norm1(z.transpose(1, 2)).transpose(1, 2)))
        z = self.conv2(z)
        z = self.dropout(F.gelu(self.norm2(z.transpose(1, 2)).transpose(1, 2)))
        return x + z


class TemporalNet(nn.Module):
    def __init__(self, input_channels, config):
        super().__init__()
        self.config = config
        self.input = nn.Conv1d(input_channels, config.width, 1)
        self.blocks = nn.Sequential(*(ResidualBlock(config.width, d, config.dropout)
                                      for d in config.dilations))
        self.state = nn.Conv1d(config.width, 3, 1)
        self.boundary = nn.Conv1d(config.width, 1, 1)

    def forward(self, x):
        h = self.blocks(F.gelu(self.input(x)))
        return self.state(h), self.boundary(h)[:, 0]


@dataclass
class Normalizer:
    base_names: tuple
    names: tuple
    mean: np.ndarray
    scale: np.ndarray
    pca_mean: np.ndarray
    pca_components: np.ndarray

    @classmethod
    def fit(cls, cache, audio, selected, components=32):
        if cache.halves != audio.halves or cache.fps != audio.fps:
            raise ValueError("Audio and tracking timelines disagree")
        selected = np.asarray(selected, dtype=bool)
        if selected.shape != (len(cache.x),) or not selected.any():
            raise ValueError("Preprocessing requires a nonempty training mask")
        frames = np.flatnonzero(selected)
        embeddings = interpolate(audio, audio.embeddings, frames)
        available = interpolate(audio, audio.availability[:, None], frames)[:, 0]
        valid = available >= 0.95
        if np.sum(valid) < components or not 1 <= components <= 512:
            raise ValueError("Too little available training audio for the fixed PCA size")
        with threadpool_limits(limits=4):
            pca = PCA(n_components=components, svd_solver="randomized", random_state=0).fit(embeddings[valid])
        score_names=tuple(audio.provenance.get('score_names',('panns_whistle','panns_whistling')))
        if len(score_names)!=2 or len(set(score_names))!=2:raise ValueError('Expected two named audio scores')
        names = tuple(cache.names) + tuple(f"audio_pc_{i:02}" for i in range(components)) + score_names + ('audio_available',)
        provisional = cls(tuple(cache.names), names, None, None,
                          pca.mean_.astype(np.float32), pca.components_.astype(np.float32))
        raw = provisional.raw(cache, audio)
        seen = np.isfinite(raw[selected])
        total = np.where(seen, raw[selected], 0).sum(axis=0, dtype=np.float64)
        mean = np.divide(total, seen.sum(axis=0), out=np.zeros(raw.shape[1]), where=seen.sum(axis=0) > 0)
        square = np.where(seen, (raw[selected] - mean) ** 2, 0).sum(axis=0, dtype=np.float64)
        variance = np.divide(square, seen.sum(axis=0), out=np.ones(raw.shape[1]), where=seen.sum(axis=0) > 0)
        scale = np.sqrt(variance)
        scale[scale < 1e-5] = 1
        provisional.mean, provisional.scale = mean.astype(np.float32), scale.astype(np.float32)
        return provisional

    def raw(self, cache, audio):
        if tuple(cache.names) != tuple(self.base_names) or cache.halves != audio.halves or cache.fps != audio.fps:
            raise ValueError("Saved preprocessing schema differs from cache")
        with threadpool_limits(limits=4):
            projected = (audio.embeddings - self.pca_mean) @ self.pca_components.T
        coarse = np.column_stack((projected, audio.scores, audio.availability))
        extra = interpolate(audio, coarse, np.arange(len(cache.x)))
        bad = ~np.isfinite(extra[:, -1]) | (extra[:, -1] < 0.95)
        extra[bad, :-1] = np.nan
        extra[:, -1] = np.nan_to_num(extra[:, -1])
        return np.column_stack((cache.x, extra)).astype(np.float32)

    def transform(self, cache, audio, arm):
        if arm not in ARMS:
            raise ValueError("Unknown audio ablation")
        raw = self.raw(cache, audio)
        finite = np.isfinite(raw)
        z = np.where(finite, (raw - self.mean) / self.scale, 0)
        x = np.column_stack((np.clip(z, -8, 8), ~finite)).astype(np.float32)
        gate = np.ones(len(self.names), dtype=bool)
        if arm != "learned_audio":
            gate[len(self.base_names):] = False
        if arm == "tracking":
            gate[:len(self.base_names)] = [not n.startswith("whistle") for n in self.base_names]
        # Mask values AND missingness, preventing an indirect audio leak.
        x[:, ~np.tile(gate, 2)] = 0
        return x

    def save(self, path):
        np.savez_compressed(path, base_names=np.array(self.base_names), names=np.array(self.names),
                            mean=self.mean, scale=self.scale, pca_mean=self.pca_mean,
                            pca_components=self.pca_components)

    @classmethod
    def load(cls, path):
        with np.load(path, allow_pickle=False) as a:
            return cls(tuple(a["base_names"].tolist()), tuple(a["names"].tolist()), a["mean"],
                       a["scale"], a["pca_mean"], a["pca_components"])


def context(x, start, core, radius, first, stop):
    """Fixed-size context, zero outside this half, never across halftime."""
    left, right = start - radius, start + core + radius
    out = np.zeros((right - left, x.shape[1]), dtype=np.float32)
    a, b = max(first, left), min(stop, right)
    if b > a:
        out[a - left:b - left] = x[a:b]
    return out.T


def known_runs(mask):
    edges = np.diff(np.r_[False, mask, False].astype(np.int8))
    return list(zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)))


def epoch_tiles(selected, halves, core, rng):
    tiles = []
    for _, first, last in halves:
        for a, b in known_runs(selected[first:last + 1]):
            a, b = a + first, b + first
            offset = int(rng.integers(core))
            tiles.extend((start, first, last + 1, a, b) for start in range(a - offset, b, core))
    rng.shuffle(tiles)
    return tiles


def train_seed(x, codes, selected, boundary_y, boundary_selected, halves, config, seed, device):
    deterministic(seed)
    rng = np.random.default_rng(seed)
    model = TemporalNet(x.shape[1], config).to(device)
    positive = float(boundary_y[boundary_selected].sum())
    negative = float(boundary_selected.sum()) - positive
    if min(positive, negative) <= 0:
        raise ValueError("Boundary training requires positive and negative examples")
    pos_weight = min(20.0, negative / positive)
    # Correct this weighted-BCE prior shift before decoding at inference.
    nn.init.constant_(model.boundary.bias, np.log(pos_weight * positive / negative))
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate,
                                  weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.epochs)
    log = []
    for epoch in range(config.epochs):
        model.train()
        tiles = epoch_tiles(selected, halves, config.core_frames, rng)
        losses = []
        for start in range(0, len(tiles), config.batch_size):
            batch = tiles[start:start + config.batch_size]
            data, targets, masks, by, bm = [], [], [], [], []
            for frame, first, stop, run_first, run_stop in batch:
                data.append(context(x, frame, config.core_frames, config.radius, first, stop))
                indices = np.arange(frame, frame + config.core_frames)
                inside = (indices >= run_first) & (indices < run_stop)
                clipped = np.clip(indices, 0, len(codes) - 1)
                targets.append(codes[clipped])
                masks.append(inside & selected[clipped])
                by.append(boundary_y[clipped])
                bm.append(inside & boundary_selected[clipped])
            data = torch.from_numpy(np.stack(data)).to(device)
            target = torch.from_numpy(np.stack(targets).astype(np.int64)).to(device)
            mask = torch.from_numpy(np.stack(masks)).to(device)
            btarget = torch.from_numpy(np.stack(by).astype(np.float32)).to(device)
            bmask = torch.from_numpy(np.stack(bm)).to(device)
            logits, bscore = model(data)
            region = slice(config.radius, config.radius + config.core_frames)
            logits, bscore = logits[:, :, region], bscore[:, region]
            loss = F.cross_entropy(logits.transpose(1, 2)[mask], target[mask])
            if bmask.any():
                loss = loss + 0.25 * F.binary_cross_entropy_with_logits(
                    bscore[bmask], btarget[bmask], pos_weight=torch.tensor(pos_weight, device=device))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.detach()))
        scheduler.step()
        average = float(np.mean(losses))
        if not np.isfinite(average):
            raise ValueError("Non-finite temporal training loss")
        log.append(average)
        if epoch == 0 or (epoch + 1) % 10 == 0:
            print(f"seed {seed} epoch {epoch + 1}/{config.epochs} train loss {average:.5f}", flush=True)
    model.eval()
    return model, dict(seed=seed, training_loss=log, boundary_positive_weight=pos_weight,
                      boundary_prior_correction="subtract log(positive_weight)",
                      parameters=sum(p.numel() for p in model.parameters()))


def infer_seed(model, x, halves, positive_weight, device, core=2048, batch_size=8):
    model.eval()
    probability = np.full((len(x), 3), 1 / 3, dtype=np.float32)
    boundary = np.full(len(x), 0.5, dtype=np.float32)
    radius = model.config.radius
    tiles = [(start, first, last + 1) for _, first, last in halves
             for start in range(first, last + 1, core)]
    with torch.inference_mode():
        for offset in range(0, len(tiles), batch_size):
            batch = tiles[offset:offset + batch_size]
            data = np.stack([context(x, start, core, radius, first, stop)
                             for start, first, stop in batch])
            logits, blogits = model(torch.from_numpy(data).to(device))
            p = logits[:, :, radius:radius + core].softmax(dim=1).transpose(1, 2).cpu().numpy()
            q = (blogits[:, radius:radius + core] - np.log(positive_weight)).sigmoid().cpu().numpy()
            for i, (start, _, stop) in enumerate(batch):
                count = min(core, stop - start)
                probability[start:start + count] = p[i, :count]
                boundary[start:start + count] = q[i, :count]
    return probability, boundary
