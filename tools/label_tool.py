"""Frame viewer for hand-labelling ball possession on the tactical video.

    python tools/label_tool.py --labeller alice               # resumes where alice stopped
    python tools/label_tool.py --labeller alice --frame 40000
    python tools/label_tool.py --labeller alice --export      # -> data/labels/alice.csv
    python tools/label_tool.py --labeller alice --export --window 30000 34499
    python tools/label_tool.py --snapshot 11185               # frame + overlay as a PNG, no window

H / A / D start a segment of that label at the current frame and X starts an unlabelled stretch,
which is how a labelled window is closed. Labels are stored as segment starts in
data/labels/<labeller>.json, saved on every edit; data/ is gitignored, so hand labels never reach
the public repo. --export writes a full-length frame,label CSV with X outside the labelled spans,
which evaluation/run_evaluation.py --gt and tools/spells.py --gt accept. Keys and labelling
conventions: tools/LABELLING.md.

The video is decoded with PyAV (pip install av), because the installed opencv is built against
numpy 1 and the pipeline is pinned to numpy 2. Every packet's pts is exactly frame * 1000 (time base
1/25000, no B-frames, a keyframe every 50 frames; checked on all 199,319 frames), and every decoded
frame's pts is checked against the frame asked for, so the picture shown is always frame N.
"""

import argparse
import bisect
import json
import os
import re
import sys
import time
from collections import OrderedDict

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "evaluation"))
import possession_metrics as pm  # noqa: E402
from games import DATA_DIR, GAME, GAMES, half_video_blocks, video_path  # noqa: E402

# Params
N_FRAMES = GAMES[GAME]["n_video_frames"]
FPS = GAMES[GAME]["fps"]
LABEL_DIR = os.path.join(DATA_DIR, "labels")
START_FRAME = half_video_blocks(GAME)[0][1]   # first graded frame, the first-half kick-off
VIDEO_SIZE = (1920, 1080)                      # the homography maps the pitch to these pixels
DISPLAY_WIDTH = 1280                           # shrunk to fit the screen if needed
FORWARD_DECODE = 50    # frames; a jump further ahead seeks instead (at most one GOP to decode)
CACHE_MB = 300         # display-size frames kept for instant stepping back
UNDO_DEPTH = 500
SPEEDS = (0.25, 0.5, 1, 2, 4)
STRIP_SPANS = (10, 30, 120, 600)   # seconds either side of the cursor in the timeline strip
STRIP_H, STRIP_GUTTER = 14, 64
TRAIL = 10             # ball positions drawn behind the current one
COLOURS = {"H": "#1f77b4", "A": "#d62728", "D": "#9a9a9a", "X": "#303030"}
RING = {0: (90, 160, 255), 1: (255, 70, 70)}   # overlay ring per team: Barcelona, Athletic
BALL, LINES = (255, 230, 0), (120, 255, 255)
BG, FG = "#1b1b1b", "#e8e8e8"

HELP = """\
Left / Right          -1 / +1 frame          Shift: 5   Ctrl: 25 (1 s)
Down / Up             -250 / +250 frames (10 s)
Space                 play / pause           - / +   playback speed
G                     go to a frame (type it, Enter)
[ / ]                 previous / next segment start
H  A  D               a segment of H / A / D starts at this frame
X                     unlabelled from this frame on (closes a labelled window)
Backspace / Delete    remove the start of the segment you are in
Ctrl+Z  (or U)        undo
O / P                 tracking overlay / pitch lines
T                     timeline zoom          Ctrl+E  export CSV
? / F1                this help              Q       quit (labels are saved on every edit)"""


# -- video -------------------------------------------------------------------------------------

class VideoReader:
    """Frame-accurate random access. Seeks land on the keyframe at or before N and decode forward.

    Frames decoded on the way to N are kept as raw YUV (converting to RGB is the costly step), and
    frames shown are kept at display size in an LRU cache, so stepping back costs one seek per GOP.
    """

    def __init__(self, path, size, cache_mb=CACHE_MB):
        import av
        self.container = av.open(path)
        self.stream = self.container.streams.video[0]
        self.stream.thread_type = "AUTO"
        tick = 1 / (self.stream.time_base * self.stream.average_rate)
        if tick.denominator != 1 or self.stream.frames != N_FRAMES:
            raise ValueError("{}: expected {} frames on an integer pts grid, got {} frames, "
                             "{} ticks per frame".format(path, N_FRAMES, self.stream.frames, tick))
        self.tick = int(tick)
        self.origin = self.stream.start_time or 0
        self.cache_mb = cache_mb
        self.resize(size)

    def resize(self, size):
        self.size = size
        self.cache = OrderedDict()
        self.max_cached = max(2 * FORWARD_DECODE,
                              self.cache_mb * 2 ** 20 // (size[0] * size[1] * 3))
        self.walk = {}       # raw frames decoded since the last seek
        self.decoder = None
        self.last = None     # index of the last frame the decoder produced

    def index_of(self, frame):
        offset = frame.pts - self.origin
        if offset % self.tick:
            raise RuntimeError("pts {} is off the frame grid".format(frame.pts))
        return offset // self.tick

    def get(self, n):
        """Frame n as an RGB PIL image at display size."""
        if not 0 <= n < N_FRAMES:
            raise IndexError(n)
        if n in self.cache:
            self.cache.move_to_end(n)
            return self.cache[n]
        raw = self.walk[n] if n in self.walk else self._decode_to(n)
        image = raw.to_image(width=self.size[0], height=self.size[1], interpolation="BILINEAR")
        self.cache[n] = image
        if len(self.cache) > self.max_cached:
            self.cache.popitem(last=False)
        return image

    def _decode_to(self, n):
        if self.last is None or not self.last < n <= self.last + FORWARD_DECODE:
            self.container.seek(self.origin + n * self.tick, stream=self.stream,
                                backward=True, any_frame=False)
            self.decoder = self.container.decode(self.stream)
            self.walk = {}
        for frame in self.decoder:
            i = self.index_of(frame)
            self.last = i
            self.walk[i] = frame
            if len(self.walk) > FORWARD_DECODE:
                del self.walk[min(self.walk)]
            if i == n:
                return frame
            if i > n:
                raise RuntimeError("decoder went past frame {} to {}".format(n, i))
        self.last = None
        raise RuntimeError("video ended before frame {}".format(n))


# -- tracking overlay --------------------------------------------------------------------------

def pitch_lines(length, width):
    """Polylines of the pitch markings, in normalised pitch coordinates."""
    def box(depth, span):
        y0, y1 = (width - span) / 2, (width + span) / 2
        return [[(0, y0), (depth, y0), (depth, y1), (0, y1)],
                [(length, y0), (length - depth, y0), (length - depth, y1), (length, y1)]]
    angle = np.linspace(0, 2 * np.pi, 49)
    lines = [[(0, 0), (length, 0), (length, width), (0, width), (0, 0)],
             [(length / 2, 0), (length / 2, width)],
             list(zip(length / 2 + 9.15 * np.cos(angle), width / 2 + 9.15 * np.sin(angle)))]
    lines += box(16.5, 40.32) + box(5.5, 18.32)
    scale = np.array([length, width])
    return [np.asarray(line, dtype=np.float64) / scale for line in lines]


class Overlay:
    """The tracking drawn onto the picture: players ringed by team, the ball and its trail, and
    optionally the pitch markings (which show at a glance whether the homography fits)."""

    def __init__(self, match):
        self.match = match
        self.scale = np.array(match.pitch, dtype=np.float64)
        self.lines = pitch_lines(*match.pitch)
        # Per-frame teams: a goalkeeper slot is coloured by the goal it guards, not its metadata
        # team, which the tracker gets wrong (see possession.data.GK_SLOTS).
        self.teams = match.team_by_frame()

    def project(self, n, xy, size):
        """(k, 2) normalised pitch -> (k, 2) pixels at display size; NaN in, NaN out."""
        h = self.match.homography[n]
        p = np.c_[xy, np.ones(len(xy))] @ h.T
        with np.errstate(invalid="ignore", divide="ignore"):
            px = p[:, :2] / p[:, 2:]
        px[p[:, 2] <= 0] = np.nan
        return px * (size[0] / VIDEO_SIZE[0], size[1] / VIDEO_SIZE[1])

    def draw(self, image, n, tracks=True, lines=False):
        from PIL import ImageDraw
        draw = ImageDraw.Draw(image)
        r = max(4, image.size[0] // 160)
        if lines:
            for line in self.lines:
                px = self.project(n, line, image.size)
                if np.isfinite(px).all():
                    draw.line([tuple(p) for p in px], fill=LINES, width=1)
        if not tracks:
            return
        players = self.project(n, self.match.players[n] / self.scale, image.size)
        for (x, y), team in zip(players, self.teams[n]):
            if np.isfinite(x) and np.isfinite(y):
                draw.ellipse([x - r, y - r * 0.6, x + r, y + r * 0.6], outline=RING[int(team)],
                             width=2)
        first = max(0, n - TRAIL)
        ball = self.project(n, self.match.ball[first:n + 1] / self.scale, image.size)
        for x, y in ball[:-1]:
            if np.isfinite(x) and np.isfinite(y):
                draw.ellipse([x - 1.5, y - 1.5, x + 1.5, y + 1.5], fill=BALL)
        x, y = ball[-1]
        if np.isfinite(x) and np.isfinite(y):
            draw.ellipse([x - r * 0.6, y - r * 0.6, x + r * 0.6, y + r * 0.6], fill=BALL,
                         outline=(0, 0, 0), width=1)


# -- labels ------------------------------------------------------------------------------------

def minimal(starts):
    """Drop starts that repeat the label already in effect, so starts are exactly the changes."""
    out, current = [], "X"
    for frame, label in sorted(starts):
        if label != current:
            out.append((frame, label))
            current = label
    return out


class Labels:
    """One labeller's segments as starts: each (frame, label) holds until the next start, and frames
    before the first start are X. Saved to JSON on every edit."""

    def __init__(self, labeller, directory=LABEL_DIR):
        self.labeller = labeller
        self.path = os.path.join(directory, labeller + ".json")
        self.starts, self.cursor, self.undo_stack = [], None, []
        if os.path.isfile(self.path):
            with open(self.path) as handle:
                stored = json.load(handle)
            starts = [(int(f), str(lab).upper()) for f, lab in stored["boundaries"]]
            bad = [s for s in starts if s[1] not in pm.LABELS or not 0 <= s[0] < N_FRAMES]
            if len({f for f, _ in starts}) < len(starts):
                bad.append("a frame that starts two segments")
            if bad or stored.get("n_frames", N_FRAMES) != N_FRAMES:
                raise ValueError("{}: bad boundaries {}".format(self.path, bad[:3]))
            self.starts = minimal(starts)
            self.cursor = stored.get("cursor")
        self._refresh()

    def _refresh(self):
        self.frames = [f for f, _ in self.starts]
        self.codes = np.full(N_FRAMES, pm.X, dtype=np.int8)
        for (a, label), b in zip(self.starts, self.frames[1:] + [N_FRAMES]):
            self.codes[a:b] = pm.LABELS.index(label)

    def segment_at(self, n):
        """(start, stop, label) of the segment holding frame n; start None before the first
        start, stop None for the last (still open) segment."""
        i = bisect.bisect_right(self.frames, n) - 1
        start, label = self.starts[i] if i >= 0 else (None, "X")
        stop = self.frames[i + 1] if i + 1 < len(self.frames) else None
        return start, stop, label

    def mark(self, n, label):
        before = self.segment_at(n)[2]
        replaced = n in self.frames
        if not self._commit([s for s in self.starts if s[0] != n] + [(n, label)], n):
            return "no change: {} is already in effect at {}".format(label, n)
        if replaced:
            return "the segment starting at {} is now {}".format(n, label)
        return "{} starts at {} (was {})".format(label, n, before)

    def remove(self, n):
        start, _, label = self.segment_at(n)
        if start is None:
            return "no segment start at or before {}".format(n)
        self._commit([s for s in self.starts if s[0] != start], start)
        return "removed the {} start at {}; {} now holds there".format(
            label, start, self.segment_at(start)[2])

    def undo(self):
        if not self.undo_stack:
            return None
        self.starts, frame = self.undo_stack.pop()
        self._refresh()
        self.save()
        return frame

    def _commit(self, starts, frame):
        starts = minimal(starts)
        if starts == self.starts:
            return False
        self.undo_stack = (self.undo_stack + [(self.starts, frame)])[-UNDO_DEPTH:]
        self.starts = starts
        self._refresh()
        self.save()
        return True

    def save(self, cursor=None):
        if cursor is not None:
            self.cursor = int(cursor)
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        head = json.dumps({
            "labeller": self.labeller,
            "video": os.path.basename(video_path()),
            "n_frames": N_FRAMES,
            "fps": FPS,
            "format": "each [frame, label] starts a segment that runs to the next start; "
                      "X = unlabelled, so a labelled window ends with an X start",
            "cursor": self.cursor,
        }, indent=2)
        rows = ",\n".join('    [{}, "{}"]'.format(f, lab) for f, lab in self.starts)
        text = head[:-2] + ',\n  "boundaries": [\n' + rows + "\n  ]\n}\n"
        tmp = self.path + ".tmp"
        with open(tmp, "w", newline="\n") as handle:
            handle.write(text)
        os.replace(tmp, self.path)   # an interrupted save never leaves a half-written file


def export_codes(labels, window=None):
    """(per-frame codes, notes). Each segment is confined to the half it falls in, and the last
    segment is dropped while it is still open, so labels never leak past what was watched.
    window=(first, last) keeps only those frames, e.g. the stretch two labellers both did."""
    codes = np.full(N_FRAMES, pm.X, dtype=np.int8)
    halves = [(first, last + 1) for _, first, last in half_video_blocks(GAME)]
    notes = []
    for (a, label), b in zip(labels.starts, labels.frames[1:] + [None]):
        if label == "X":
            continue
        if b is None:
            notes.append("the {} segment from {} is still open, so it is not exported: press X on "
                         "the first frame after your window".format(label, a))
            continue
        pieces = [(max(a, f), min(b, l)) for f, l in halves if max(a, f) < min(b, l)]
        if not pieces:
            notes.append("the {} segment {}-{} lies outside both halves, not exported".format(
                label, a, b - 1))
            continue
        if pieces[0] != (a, b):
            notes.append("the {} segment {}-{} was cut to the graded frames {}-{}".format(
                label, a, b - 1, pieces[0][0], pieces[0][1] - 1))
        codes[pieces[0][0]:pieces[0][1]] = pm.LABELS.index(label)
    if window is not None:
        codes[:window[0]] = pm.X
        codes[window[1] + 1:] = pm.X
    return codes, notes


def merge_codes(names, window=None):
    """Several labellers' exports as one; where their windows overlap, the first name wins."""
    codes, notes = np.full(N_FRAMES, pm.X, dtype=np.int8), []
    for i, name in enumerate(names):
        more, extra = export_codes(Labels(name), window)
        both = (codes != pm.X) & (more != pm.X)
        if both.any():
            notes.append("{} overlaps {} on {} frames, {:.1%} with the same label; the earlier "
                         "labels are kept".format(name, ", ".join(names[:i]), int(both.sum()),
                                                  np.mean(codes[both] == more[both])))
        codes = np.where(codes == pm.X, more, codes)
        notes += [(name + ": " if len(names) > 1 else "") + note for note in extra]
    return codes, notes


def export(codes, notes, path):
    """Write the full-length CSV, read it back the way the scorer does, and summarise it."""
    letters = np.array(pm.LABELS)[codes]
    with open(path, "w", newline="\n") as handle:
        handle.write("frame,label\n")
        handle.writelines("{},{}\n".format(i, label) for i, label in enumerate(letters))

    table = pm.read_label_csv(path)
    reason = pm.validate_submission(table, N_FRAMES)
    if reason or not np.array_equal(pm.label_codes(table), codes):
        raise RuntimeError("{} does not read back as written: {}".format(path, reason))
    runs = pm.valid_runs(codes)
    spells = pm.possession_spells(codes, runs)
    lines = ["wrote {}: {} labelled frames ({:.1f} min) in {} window(s)".format(
        path, int((codes != pm.X).sum()), (codes != pm.X).sum() / FPS / 60, len(runs))]
    for a, b in runs:
        window = codes[a:b]
        lines.append("  {:>6}-{:<6} {:5.1f} min   H {:>5}  A {:>5}  D {:>5} frames".format(
            a, b - 1, (b - a) / FPS / 60, *[int((window == k).sum()) for k in (pm.H, pm.A, pm.D)]))
    lines.append("  {} possessions ({} H, {} A), {} dead-ball stretches".format(
        len(spells), sum(k == pm.H for *_, k in spells), sum(k == pm.A for *_, k in spells),
        sum(k == pm.D for *_, k in pm.segments(codes, runs))))
    lines += ["  note: " + note for note in notes]
    return "\n".join(lines)


# -- window ------------------------------------------------------------------------------------

def clock(n):
    """'video mm:ss.ff', plus the time into the half when n is graded."""
    text = "video {:02d}:{:02d}.{:02d}".format(n // FPS // 60, n // FPS % 60, n % FPS)
    for half, first, last in half_video_blocks(GAME):
        if first <= n <= last:
            k = n - first
            text += "   half {} +{:02d}:{:02d}.{:02d}".format(
                half, k // FPS // 60, k // FPS % 60, k % FPS)
    return text


class App:
    def __init__(self, labels, reader, overlay, compare, start, width):
        import tkinter as tk
        from tkinter import font
        from PIL import ImageTk
        self.labels, self.reader, self.overlay, self.compare = labels, reader, overlay, compare
        self.n = min(max(start, 0), N_FRAMES - 1)
        self.playing, self.speed, self.play_id = False, 1, 0
        self.show_tracks, self.show_lines, self.span = overlay is not None, False, STRIP_SPANS[0]

        root = self.root = tk.Tk()
        root.title("possession labels: {}".format(labels.labeller))
        root.configure(bg=BG)
        mono = font.nametofont("TkFixedFont").copy()
        big, self.small = mono.copy(), mono.copy()
        big.configure(size=16, weight="bold")
        self.small.configure(size=8)

        self.video = tk.Label(root, bg="black", bd=0)
        self.video.pack()
        self.strip = tk.Canvas(root, bg=BG, highlightthickness=0,
                               height=(1 + len(compare)) * (STRIP_H + 3) + 16)
        self.strip.pack(fill="x")
        row = tk.Frame(root, bg=BG)
        row.pack(fill="x", padx=6, pady=(4, 0))
        self.frame_text = tk.Label(row, font=big, bg=BG, fg=FG)
        self.frame_text.pack(side="left")
        self.badge = tk.Label(row, font=big, width=3, fg="white")
        self.badge.pack(side="left", padx=10)
        self.info = tk.Label(row, font=mono, bg=BG, fg=FG, justify="left", anchor="w")
        self.info.pack(side="left", fill="x")
        row = tk.Frame(root, bg=BG)
        row.pack(fill="x", padx=6, pady=4)
        tk.Label(row, text="go to frame (G):", font=mono, bg=BG, fg=FG).pack(side="left")
        self.entry = tk.Entry(row, width=9, font=mono)
        self.entry.pack(side="left", padx=(4, 12))
        self.entry.bind("<Return>", self.jump)
        self.entry.bind("<Escape>", lambda e: root.focus_set())
        self.message = tk.Label(row, font=mono, bg=BG, fg="#ffd27f", anchor="w")
        self.message.pack(side="left", fill="x")
        self.help = tk.Label(root, text=HELP, font=mono, bg="#2a2a2a", fg=FG, justify="left",
                             padx=10, pady=8)

        # Fit the picture to the screen now that the rest of the window has a height.
        root.update_idletasks()
        spare_h = root.winfo_screenheight() - root.winfo_reqheight() - 120
        w = int(min(width, root.winfo_screenwidth() - 40, spare_h * 16 / 9)) // 2 * 2
        self.size = (w, w * 9 // 16)
        reader.resize(self.size)
        self.photo = ImageTk.PhotoImage("RGB", self.size)
        self.video.configure(image=self.photo, width=self.size[0], height=self.size[1])
        self.strip.configure(width=w)

        root.bind("<KeyPress>", self.on_key)
        root.protocol("WM_DELETE_WINDOW", self.quit)
        self.say("press ? for keys")
        self.render()
        root.focus_force()

    # navigation and drawing

    def go(self, n):
        self.n = int(min(max(n, 0), N_FRAMES - 1))
        self.render()

    def render(self):
        image = self.reader.get(self.n)
        if self.overlay is not None and (self.show_tracks or self.show_lines):
            image = image.copy()
            self.overlay.draw(image, self.n, self.show_tracks, self.show_lines)
        self.photo.paste(image)
        self.draw_strip()
        start, stop, label = self.labels.segment_at(self.n)
        self.frame_text.configure(text="frame {:>6}".format(self.n))
        self.badge.configure(text=label, bg=COLOURS[label] if label != "X" else "#555555")
        if start is None:
            segment = "unlabelled (no segment starts before this frame)"
        elif stop is None:
            segment = "{} since {} -- open: press X where your window ends".format(label, start)
        else:
            segment = "{} {}-{} ({:.2f} s)".format(label, start, stop - 1, (stop - start) / FPS)
        state = "playing x{}".format(self.speed) if self.playing else "paused"
        self.info.configure(text="{}   {}\nsegment {}".format(clock(self.n), state, segment))

    def draw_strip(self):
        c, w = self.strip, self.size[0]
        c.delete("all")
        bar = w - STRIP_GUTTER
        span = self.span * FPS
        frames = self.n + ((np.arange(bar) + 0.5) / bar * 2 - 1) * span
        frames = np.floor(frames).astype(np.int64)
        inside = (frames >= 0) & (frames < N_FRAMES)
        rows = [("you", self.labels.codes)] + self.compare
        for r, (name, codes) in enumerate(rows):
            y = r * (STRIP_H + 3) + 2
            sampled = np.full(bar, pm.X, dtype=np.int8)
            sampled[inside] = codes[frames[inside]]
            edges = np.concatenate(([0], np.flatnonzero(np.diff(sampled)) + 1, [bar]))
            for a, b in zip(edges[:-1], edges[1:]):
                c.create_rectangle(STRIP_GUTTER + a, y, STRIP_GUTTER + b, y + STRIP_H, width=0,
                                   fill=COLOURS[pm.LABELS[sampled[a]]])
            c.create_text(4, y + STRIP_H / 2, text=name[:9], anchor="w", fill=FG, font=self.small)
        lo, hi = self.n - span, self.n + span
        for f in self.labels.frames[bisect.bisect_left(self.labels.frames, lo):
                                    bisect.bisect_right(self.labels.frames, hi)]:
            x = STRIP_GUTTER + (f - lo) / (2 * span) * bar
            c.create_line(x, 0, x, STRIP_H + 4, fill="white")
        mid = STRIP_GUTTER + bar / 2
        bottom = len(rows) * (STRIP_H + 3)
        c.create_line(mid, 0, mid, bottom + 2, fill="#ffd27f", width=2)
        for x, text, anchor in ((STRIP_GUTTER, "-{} s".format(self.span), "nw"),
                                (mid, "frame {}".format(self.n), "n"),
                                (w - 2, "+{} s   (T: zoom)".format(self.span), "ne")):
            c.create_text(x, bottom + 2, text=text, anchor=anchor, fill="#aaaaaa", font=self.small)

    def say(self, text):
        self.message.configure(text=text)

    # input

    def on_key(self, event):
        if event.widget is self.entry:
            return
        key, shift, ctrl = event.keysym, bool(event.state & 0x1), bool(event.state & 0x4)
        low = key.lower()
        if key in ("Left", "Right"):
            step = 25 if ctrl else 5 if shift else 1
            self.go(self.n + (step if key == "Right" else -step))
        elif key in ("Up", "Down"):
            self.go(self.n + (250 if key == "Up" else -250))
        elif ctrl and low == "z" or low == "u":
            frame = self.labels.undo()
            self.say("undone" if frame is not None else "nothing to undo")
            self.go(self.n if frame is None else frame)
        elif ctrl and low == "e":
            self.write_csv()
        elif low in ("h", "a", "d", "x") and not ctrl:
            self.labels.cursor = self.n
            self.say(self.labels.mark(self.n, low.upper()))
            self.render()
        elif key in ("BackSpace", "Delete"):
            self.say(self.labels.remove(self.n))
            self.render()
        elif key in ("bracketleft", "bracketright"):
            frames = self.labels.frames
            i = bisect.bisect_left(frames, self.n) - 1 if key == "bracketleft" \
                else bisect.bisect_right(frames, self.n)
            if 0 <= i < len(frames):
                self.go(frames[i])
            else:
                self.say("no segment start {} this frame".format(
                    "before" if key == "bracketleft" else "after"))
        elif key == "space":
            self.playing = not self.playing
            if self.playing:
                self.play_id += 1
                self.play(self.play_id)
            self.render()
        elif key in ("minus", "plus", "equal", "KP_Subtract", "KP_Add"):
            i = SPEEDS.index(self.speed) + (-1 if key in ("minus", "KP_Subtract") else 1)
            self.speed = SPEEDS[min(max(i, 0), len(SPEEDS) - 1)]
            self.render()
        elif low == "g":
            self.entry.delete(0, "end")
            self.entry.focus_set()
        elif low == "o" and self.overlay is not None:
            self.show_tracks = not self.show_tracks
            self.render()
        elif low == "p" and self.overlay is not None:
            self.show_lines = not self.show_lines
            self.render()
        elif low == "t":
            self.span = STRIP_SPANS[(STRIP_SPANS.index(self.span) + 1) % len(STRIP_SPANS)]
            self.render()
        elif key in ("question", "F1"):
            if self.help.winfo_ismapped():
                self.help.place_forget()
            else:
                self.help.place(in_=self.video, x=12, y=12)
        elif key == "Escape":
            self.playing = False
            self.help.place_forget()
            self.render()
        elif low == "q":
            self.quit()

    def jump(self, _event=None):
        text = self.entry.get().strip().replace(",", "").replace("_", "")
        self.root.focus_set()
        if not re.fullmatch(r"[+-]?\d+", text):
            self.say("not a frame number: {!r}".format(text))
            return
        self.go(self.n + int(text) if text[0] in "+-" else int(text))

    def play(self, play_id):
        if not self.playing or play_id != self.play_id:
            return   # paused, or a newer Space press started its own loop
        t0 = time.perf_counter()
        step = max(1, int(round(self.speed)))
        if self.n + step >= N_FRAMES:
            self.playing = False
        self.go(self.n + step)
        wait = 1000.0 * step / (FPS * self.speed) - 1000.0 * (time.perf_counter() - t0)
        self.root.after(max(1, int(wait)), self.play, play_id)

    def write_csv(self):
        path = os.path.splitext(self.labels.path)[0] + ".csv"
        summary = export(*export_codes(self.labels), path)
        print(summary)
        notes = "  (notes in the console)" if "note:" in summary else ""
        self.say(summary.splitlines()[0] + notes)

    def quit(self):
        self.playing = False
        self.labels.save(self.n)
        self.root.destroy()


# -- entry point -------------------------------------------------------------------------------

def load_overlay():
    from possession.data import load_match
    print("loading tracking for the overlay (--no-overlay to skip) ...")
    return Overlay(load_match())


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--labeller", help="your name; labels go to data/labels/<labeller>.json")
    parser.add_argument("--frame", type=int, help="start here (default: where you stopped)")
    parser.add_argument("--export", nargs="?", const="", metavar="CSV",
                        help="write the full-length frame,label CSV (default "
                             "data/labels/<labeller>.csv) and exit")
    parser.add_argument("--window", type=int, nargs=2, metavar=("FIRST", "LAST"),
                        help="with --export: keep only frames FIRST..LAST (default file name "
                             "data/labels/<labeller>_<FIRST>-<LAST>.csv)")
    parser.add_argument("--merge", nargs="+", metavar="LABELLER",
                        help="with --export: one CSV from several labellers (default "
                             "data/labels/merged.csv); where windows overlap the first named wins")
    parser.add_argument("--compare", nargs="*", default=[], metavar="CSV",
                        help="frame,label CSVs to show as extra timeline strips (public GT, a "
                             "submission, another labeller's export)")
    parser.add_argument("--no-overlay", action="store_true", help="don't load the tracking")
    parser.add_argument("--width", type=int, default=DISPLAY_WIDTH, help="picture width in pixels")
    parser.add_argument("--snapshot", type=int, metavar="FRAME",
                        help="save one full-size frame with the overlay as a PNG and exit")
    parser.add_argument("--out", help="where --snapshot writes (default data/labels/frame_<N>.png)")
    parser.add_argument("--video", default=video_path(), help=argparse.SUPPRESS)
    args = parser.parse_args()

    if args.snapshot is not None:
        reader = VideoReader(args.video, VIDEO_SIZE)
        image = reader.get(args.snapshot).copy()
        if not args.no_overlay:
            load_overlay().draw(image, args.snapshot, tracks=True, lines=True)
        out = args.out or os.path.join(LABEL_DIR, "frame_{}.png".format(args.snapshot))
        os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
        image.save(out)
        print("wrote", out)
        return

    names = args.merge if args.export is not None and args.merge else [args.labeller]
    if not all(name and re.fullmatch(r"[A-Za-z0-9_-]+", name) for name in names):
        parser.error("--labeller NAME is required (letters, digits, _ and - only)")
    if args.export is not None:
        missing = [n for n in names if not os.path.isfile(os.path.join(LABEL_DIR, n + ".json"))]
        if missing:
            parser.error("no labels for {} in {}".format(", ".join(missing), LABEL_DIR))
        stem = os.path.join(LABEL_DIR, "merged" if args.merge else names[0])
        if args.window:
            stem += "_{}-{}".format(*args.window)
        print(export(*merge_codes(names, args.window), args.export or stem + ".csv"))
        return
    labels = Labels(args.labeller)

    if sys.platform == "win32":
        try:   # draw at the screen's real resolution instead of letting Windows upscale a blur
            import ctypes
            ctypes.windll.shcore.SetProcessDpiAwareness(1)
        except (AttributeError, OSError):
            pass
    compare = [(os.path.splitext(os.path.basename(p))[0].replace("BAR-ATH_possession_", ""),
                pm.label_codes(pm.read_label_csv(p))) for p in args.compare]
    for name, codes in compare:
        if len(codes) != N_FRAMES:
            parser.error("{} has {} rows, expected {}".format(name, len(codes), N_FRAMES))
    overlay = None if args.no_overlay else load_overlay()
    reader = VideoReader(args.video, (args.width, args.width * 9 // 16))
    start = args.frame if args.frame is not None else labels.cursor or START_FRAME
    app = App(labels, reader, overlay, compare, start, args.width)
    app.root.mainloop()


if __name__ == "__main__":
    main()
