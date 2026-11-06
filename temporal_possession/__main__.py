"""python -m temporal_possession audio|compare|predict."""

import argparse
from .audio import extract
from .network import deterministic
from .experiment import compare, predict
from learned_possession.features import FeatureCache


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    a = sub.add_parser("audio")
    a.add_argument("--video", required=True)
    a.add_argument("--checkpoint", required=True)
    a.add_argument("--features", required=True)
    a.add_argument("--out", required=True)
    a.add_argument("--device", default="cuda", choices=("cuda", "cpu"))
    a.add_argument("--batch-size", type=int, default=32)
    c = sub.add_parser("compare")
    for name in ("features", "audio", "train-labels", "validation-labels", "audit", "out"):
        c.add_argument("--" + name, required=True)
    c.add_argument("--device", default="cuda", choices=("cuda", "cpu"))
    c.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    c.add_argument("--epochs", type=int, default=60)
    c.add_argument("--bootstrap", type=int, default=1000)
    c.add_argument("--compare", nargs="*", default=[])
    p = sub.add_parser("predict")
    for name in ("features", "audio", "model", "out"):
        p.add_argument("--" + name, required=True)
    p.add_argument("--device", default="cuda", choices=("cuda", "cpu"))
    args = parser.parse_args()
    if args.command == "audio":
        deterministic(0)
        cache = FeatureCache.load(args.features)
        extract(args.video, args.checkpoint, args.out, cache.halves, len(cache.x),
                cache.fps, args.device, args.batch_size)
    elif args.command == "compare":
        compare(args)
    else:
        predict(args)


if __name__ == "__main__":
    main()
