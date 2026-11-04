"""Render a 2x2 montage of video frames, with the tracking overlay and each model's label, to
adjudicate what really happened at a moment the models disagree on.

    python tools/review.py --frames 17370 17380 17390 17400 --out review.png
    python tools/review.py --around 17380 --span 50 --sub evaluation/results/possession_v1.csv

Each tile is captioned with its frame, the match clock, the public GT (where it exists) and the
label every --sub file gives that frame.
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools"))
from label_tool import Overlay, VideoReader, clock  # noqa: E402
from games import video_path  # noqa: E402
from possession.data import load_match, load_public_gt  # noqa: E402

# Params
TILE = (960, 540)


def montage(frames, reader, overlay, captions):
    from PIL import Image, ImageDraw, ImageFont
    try:
        font = ImageFont.truetype("arial.ttf", 22)
    except OSError:
        font = ImageFont.load_default()
    canvas = Image.new("RGB", (TILE[0] * 2, TILE[1] * 2))
    for k, (n, caption) in enumerate(zip(frames, captions)):
        tile = reader.get(n).copy()
        overlay.draw(tile, n, tracks=True, lines=False)
        draw = ImageDraw.Draw(tile)
        draw.rectangle([0, 0, TILE[0], 30], fill=(0, 0, 0))
        draw.text((8, 3), caption, fill=(255, 255, 255), font=font)
        canvas.paste(tile, ((k % 2) * TILE[0], (k // 2) * TILE[1]))
    return canvas


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--frames", type=int, nargs=4)
    parser.add_argument("--around", type=int)
    parser.add_argument("--span", type=int, default=50, help="frames either side for --around")
    parser.add_argument("--sub", nargs="*", default=[])
    parser.add_argument("--out", required=True)
    parser.add_argument("--blind", action="store_true",
                        help="caption frame numbers only: no GT, no model labels")
    args = parser.parse_args()

    if args.frames:
        frames = args.frames
    else:
        frames = np.linspace(args.around - args.span, args.around + args.span, 4).round().astype(int)
    gt = load_public_gt()
    subs = [(os.path.splitext(os.path.basename(p))[0][:12],
             pd.read_csv(p, dtype={"label": "object"})["label"].values) for p in args.sub]
    captions = []
    for n in frames:
        parts = ["{} {}".format(n, clock(n))]
        if not args.blind:
            if gt[n] != "X":
                parts.append("GT " + gt[n])
            parts += ["{} {}".format(name, labels[n]) for name, labels in subs]
        captions.append("   ".join(parts))

    reader = VideoReader(video_path(), TILE)
    overlay = Overlay(load_match())
    montage(frames, reader, overlay, captions).save(args.out)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
