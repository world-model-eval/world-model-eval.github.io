#!/usr/bin/env python3
"""make_gif_grid.py

Create a single animated GIF laid out as a grid containing all individual GIFs
found within the `rollouts/` sub-directory of the current working directory.

Usage
-----
Simply run the script from the repository root (where the `rollouts/` folder
lives):

    python make_gif_grid.py            # produces grid.gif in the repo root

Optional arguments:
    --output <path>    Path to the GIF that will be written. Defaults to
                       "grid.gif" in the current working directory.
    --fps <value>      Frames per second of the resulting animation. Default: 10
                       (⇒ frame duration of 100 ms).

Dependencies
------------
This script relies on Pillow and imageio.  Install them via pip if they are not
present:

    pip install pillow imageio
"""
from __future__ import annotations

import argparse
import glob
import math
import os
from pathlib import Path
from typing import List

import imageio.v2 as imageio  # type: ignore
from PIL import Image  # type: ignore

ROLL_OUT_DIR = Path(__file__).parent / "rollouts"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a grid GIF from individual GIFs in rollouts directory.")
    parser.add_argument("--output", "-o", default="grid.gif", help="Output GIF path. Default: grid.gif")
    parser.add_argument("--fps", type=float, default=10.0, help="Frames per second of output GIF. Default: 10")
    return parser.parse_args()


def discover_gifs(directory: Path) -> List[Path]:
    """Return a sorted list of .gif files inside *directory*."""
    return sorted(Path(p) for p in glob.glob(str(directory / "*.gif")))


def compute_grid_dims(n: int) -> tuple[int, int]:
    """Compute (rows, cols) for *n* grid cells, trying to keep it square-ish."""
    if n == 0:
        raise ValueError("No GIFs found in rollouts/ directory.")
    cols = math.ceil(math.sqrt(n))
    rows = math.ceil(n / cols)
    return rows, cols


def get_common_cell_size(gif_paths: List[Path]) -> tuple[int, int]:
    """Return the (width, height) all GIF frames will be resized to.

    Uses the *smallest* width and height across the first frames of all GIFs so
    that no GIF gets upscaled (which leads to blurring).
    """
    widths, heights = [], []
    for p in gif_paths:
        with Image.open(p) as im:
            widths.append(im.width)
            heights.append(im.height)
    return min(widths), min(heights)


def load_gif_frames(path: Path, target_size: tuple[int, int]) -> List[Image.Image]:
    """Read *path* and return a list of PIL frames, each resized to *target_size*."""
    frames: List[Image.Image] = []
    reader = imageio.get_reader(path)
    for frame in reader:
        im = Image.fromarray(frame)
        if im.size != target_size:
            im = im.resize(target_size, Image.NEAREST)
        frames.append(im.convert("RGBA"))
    reader.close()
    return frames


def main() -> None:
    args = parse_args()

    gif_paths = discover_gifs(ROLL_OUT_DIR)
    if not gif_paths:
        print("[make_gif_grid] No .gif files found in", ROLL_OUT_DIR)
        return

    rows, cols = compute_grid_dims(len(gif_paths))
    cell_w, cell_h = get_common_cell_size(gif_paths)

    print(f"[make_gif_grid] Building {rows}x{cols} grid – cell size {cell_w}x{cell_h}px – {len(gif_paths)} GIFs total.")

    # Load frames for each GIF and remember the longest sequence length
    all_gif_frames: List[List[Image.Image]] = []
    max_frames = 0
    for p in gif_paths:
        frames = load_gif_frames(p, (cell_w, cell_h))
        all_gif_frames.append(frames)
        max_frames = max(max_frames, len(frames))

    # Ensure every GIF has at least one frame
    assert max_frames > 0, "All GIFs appear to be empty!"

    # Prepare the blank canvas for each output frame
    grid_w, grid_h = cell_w * cols, cell_h * rows
    frame_duration = 1.0 / args.fps  # seconds per frame

    out_frames: List[Image.Image] = []
    for t in range(max_frames):
        canvas = Image.new("RGBA", (grid_w, grid_h), (255, 255, 255, 0))
        for idx, gif_frames in enumerate(all_gif_frames):
            r, c = divmod(idx, cols)
            frame_idx = min(t, len(gif_frames) - 1)  # hold last frame once GIF ends
            canvas.paste(gif_frames[frame_idx], (c * cell_w, r * cell_h))
        out_frames.append(canvas)

    # Convert RGBA → P palette mode to reduce file size on save
    first_frame, *extra_frames = (f.convert("P", palette=Image.ADAPTIVE) for f in out_frames)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    first_frame.save(
        output_path,
        save_all=True,
        append_images=extra_frames,
        duration=int(frame_duration * 1000),  # Pillow expects milliseconds
        loop=0,
        optimize=False,
    )

    print(f"[make_gif_grid] Saved animated grid GIF → {output_path.relative_to(Path.cwd())}")


if __name__ == "__main__":
    main() 