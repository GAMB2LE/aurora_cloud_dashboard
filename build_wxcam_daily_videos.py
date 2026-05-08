#!/usr/bin/env python3
"""Build daily wxcam mp4 products from hourly raw clips."""

from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

from wxcam_catalog import WXCAM_IMAGE_TYPES

RAW_DEFAULT = Path("/project/aurora/raw/wxcam")
OUTPUT_DEFAULT = Path("/data/aurora/products/wxcam/daily_videos")
DAY_DIR_REGEX = re.compile(r"^\d{8}$")


def _iter_day_dirs(stream_root: Path):
    if not stream_root.exists():
        return
    for path in sorted(stream_root.iterdir()):
        if path.is_dir() and DAY_DIR_REGEX.match(path.name):
            yield path


def _list_clips(day_dir: Path, pattern: str) -> list[Path]:
    return sorted(path for path in day_dir.glob(pattern) if path.is_file())


def _needs_refresh(clips: list[Path], output_path: Path) -> bool:
    if not clips:
        return False
    if not output_path.exists():
        return True
    output_mtime_ns = output_path.stat().st_mtime_ns
    return any(path.stat().st_mtime_ns > output_mtime_ns for path in clips)


def _build_output(clips: list[Path], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=output_path.parent) as tmpdir_name:
        tmpdir = Path(tmpdir_name)
        list_path = tmpdir / "clips.txt"
        tmp_output = tmpdir / output_path.name
        lines = []
        for path in clips:
            escaped = str(path).replace("'", r"'\''")
            lines.append(f"file '{escaped}'\n")
        list_path.write_text(
            "".join(lines),
            encoding="utf-8",
        )
        subprocess.run(
            [
                shutil.which("ffmpeg") or "ffmpeg",
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-fflags",
                "+genpts",
                "-f",
                "concat",
                "-safe",
                "0",
                "-i",
                str(list_path),
                "-an",
                "-c:v",
                "libx264",
                "-preset",
                "veryfast",
                "-crf",
                "18",
                "-pix_fmt",
                "yuv420p",
                "-movflags",
                "+faststart",
                str(tmp_output),
            ],
            check=True,
        )
        tmp_output.replace(output_path)


def build_daily_videos(raw_root: Path, output_root: Path) -> None:
    built = 0
    skipped = 0
    for image_type, spec in WXCAM_IMAGE_TYPES.items():
        stream_root = raw_root / spec["stream"]
        video_glob = spec["video_glob"]
        type_output_root = output_root / image_type
        for day_dir in _iter_day_dirs(stream_root):
            clips = _list_clips(day_dir, video_glob)
            if not clips:
                continue
            output_path = type_output_root / f"{day_dir.name}.mp4"
            if not _needs_refresh(clips, output_path):
                skipped += 1
                continue
            _build_output(clips, output_path)
            built += 1
            print(f"Built wxcam daily video: {image_type} {day_dir.name} -> {output_path}")
    print(f"wxcam daily videos complete: built={built} skipped={skipped} output_root={output_root}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build daily wxcam mp4 products from hourly raw clips.")
    parser.add_argument("--raw-root", type=Path, default=RAW_DEFAULT)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_DEFAULT)
    args = parser.parse_args()
    build_daily_videos(args.raw_root, args.output_root)


if __name__ == "__main__":
    main()
