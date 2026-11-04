"""The tactical video's stadium audio as a per-video-frame spectrogram.

48 kHz audio and 25 fps video means one 1920-sample window per video frame, so row i of the
spectrogram is video frame i exactly. Decoded with PyAV (pip install av) and cached under
data/cache/ (gitignored with the rest of data/).
"""

import os

import numpy as np

from .data import CACHE_DIR, GAME
from games import require

# Params
SAMPLE_RATE = 48000
HOP = 1920                 # samples per video frame at 25 fps
N_FFT = 2048
MAX_HZ = 12000             # keep 0..12 kHz; whistles and crowd both live well below this
CHUNK_FRAMES = 25 * 60     # process a minute of video at a time


def _spectra(mono, n_frames):
    """(n_frames, bins) log-magnitude spectra of consecutive HOP-sample windows (zero-padded)."""
    need = n_frames * HOP
    if len(mono) < need:
        mono = np.pad(mono, (0, need - len(mono)))
    frames = mono[:need].reshape(n_frames, HOP)
    window = np.hanning(HOP).astype(np.float32)
    spec = np.abs(np.fft.rfft(frames * window, n=N_FFT, axis=1))
    keep = int(MAX_HZ / (SAMPLE_RATE / N_FFT)) + 1
    return np.log1p(spec[:, :keep]).astype(np.float32)


WHISTLE_BAND = (3600.0, 4100.0)   # the referee's whistle: a steady 3.80-3.91 kHz tone, measured on
                                  # the public window's stoppages and the first-half kick-off
BACKGROUND_FRAMES = 501            # crowd background: per-bin rolling median over ~20 s
WHISTLE_MIN_FRAMES = 6             # a whistle holds its pitch for 0.25 s or more; chants wander


def whistle_strength(spec):
    """(T,) per-frame tonal strength in the whistle band: the peak of the background-whitened
    spectrum over the band minus the band's median (a narrow line, not broadband noise), kept only
    where the peak frequency holds steady (within one bin) for WHISTLE_MIN_FRAMES in a row."""
    import pandas as pd
    hz = bin_hz()
    lo, hi = np.searchsorted(hz, WHISTLE_BAND[0]), np.searchsorted(hz, WHISTLE_BAND[1])
    band = spec[:, lo - 4:hi + 4].astype(np.float64)
    background = pd.DataFrame(band).rolling(BACKGROUND_FRAMES, center=True, min_periods=25).median().values
    white = band - background
    inner = white[:, 4:-4]
    peak_bin = np.argmax(inner, axis=1)
    contrast = inner.max(axis=1) - np.median(white, axis=1)
    # steady[t]: frame t's peak sits within one bin of frame t-1's. A run of k steady frames
    # means k+1 frames on the same pitch.
    steady = np.r_[False, np.abs(np.diff(peak_bin)) <= 1]
    edges = np.diff(np.r_[0, steady.astype(np.int8), 0])
    held = np.zeros(len(steady), dtype=bool)
    for start, stop in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)):
        if stop - start + 1 >= WHISTLE_MIN_FRAMES:
            held[start - 1:stop] = True
    return np.where(held, np.nan_to_num(contrast), 0.0)


def bin_hz():
    keep = int(MAX_HZ / (SAMPLE_RATE / N_FFT)) + 1
    return np.arange(keep) * SAMPLE_RATE / N_FFT


def load_spectrogram(game=GAME, n_video_frames=None):
    """(T, bins) float32 log-magnitude spectrogram, one row per video frame."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    cache = os.path.join(CACHE_DIR, "{}_tactical_audio_spec.npy".format(game))
    if os.path.isfile(cache):
        return np.load(cache)

    import av
    path = require(["video"], game)[0]
    container = av.open(path)
    stream = container.streams.audio[0]
    if stream.rate != SAMPLE_RATE:
        raise ValueError("expected {} Hz audio, got {}".format(SAMPLE_RATE, stream.rate))
    origin = stream.start_time or 0

    pieces, out, buffered, first = [], [], 0, None
    for frame in container.decode(stream):
        samples = frame.to_ndarray().mean(axis=0).astype(np.float32)   # mono
        if first is None:
            first = frame.pts - origin
            if first < 0:          # AAC priming samples sit before t = 0
                samples = samples[-first:]
            elif first > 0:
                samples = np.concatenate((np.zeros(first, np.float32), samples))
        pieces.append(samples)
        buffered += len(samples)
        if buffered >= CHUNK_FRAMES * HOP:
            block = np.concatenate(pieces)
            n = len(block) // HOP
            out.append(_spectra(block[:n * HOP], n))
            pieces, buffered = [block[n * HOP:]], len(block) - n * HOP
    block = np.concatenate(pieces) if pieces else np.zeros(0, np.float32)
    n = int(np.ceil(len(block) / HOP))
    if n:
        out.append(_spectra(block, n))
    spec = np.concatenate(out)
    if n_video_frames is not None:
        spec = spec[:n_video_frames] if len(spec) >= n_video_frames else np.pad(
            spec, ((0, n_video_frames - len(spec)), (0, 0)))
    np.save(cache, spec)
    return spec
