"""Собрать MP4 из кадров браузерных действий; это не непрерывная запись экрана."""

import argparse
from pathlib import Path

import imageio.v2 as imageio
from PIL import Image, ImageOps

parser = argparse.ArgumentParser(description="Собрать видео из кадров сеанса")
parser.add_argument("run_dir", type=Path)
parser.add_argument("--seconds-per-frame", type=float, default=1.5)
args = parser.parse_args()
frames = sorted(args.run_dir.glob("frame-*.png"))
if not frames:
    parser.error("В каталоге нет frame-*.png")
if args.seconds_per_frame <= 0:
    parser.error("--seconds-per-frame должен быть > 0")
output = args.run_dir / "replay.mp4"
with imageio.get_writer(output, fps=10, codec="libx264", macro_block_size=16) as writer:
    for path in frames:
        with Image.open(path) as source:
            frame = ImageOps.pad(source.convert("RGB"), (1280, 720), color="#111827")
            import numpy as np

            for _ in range(max(1, round(args.seconds_per_frame * 10))):
                writer.append_data(np.asarray(frame))
print(output.resolve())
