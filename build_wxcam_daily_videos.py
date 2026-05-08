#!/usr/bin/env python3
"""Build daily wxcam mp4 products and hourly thumbnails from raw clips."""

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
THUMBNAIL_DEFAULT = Path("/data/aurora/products/wxcam/hourly_thumbnails")
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


def _thumbnail_path(thumbnail_root: Path, image_type: str, day_token: str, clip_path: Path) -> Path:
    return thumbnail_root / image_type / day_token / f"{clip_path.stem}.jpg"


def _probe_duration_seconds(path: Path) -> float:
    output = subprocess.check_output(
        [
            shutil.which("ffprobe") or "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(path),
        ],
        text=True,
    ).strip()
    return float(output) if output else 0.0


def _render_thumbnail(clip_path: Path, tmp_output: Path, seek_seconds: float) -> None:
    subprocess.run(
        [
            shutil.which("ffmpeg") or "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-ss",
            f"{seek_seconds:.3f}",
            "-i",
            str(clip_path),
            "-frames:v",
            "1",
            "-q:v",
            "2",
            "-vf",
            "scale=360:-2",
            str(tmp_output),
        ],
        check=True,
    )


def _build_thumbnail(clip_path: Path, output_path: Path) -> bool:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    duration_seconds = _probe_duration_seconds(clip_path)
    seek_seconds = max(min(duration_seconds * 0.5, max(duration_seconds - 0.01, 0.0)), 0.0)
    with tempfile.TemporaryDirectory(dir=output_path.parent) as tmpdir_name:
        tmpdir = Path(tmpdir_name)
        tmp_output = tmpdir / output_path.name
        _render_thumbnail(clip_path, tmp_output, seek_seconds)
        if not tmp_output.exists():
            _render_thumbnail(clip_path, tmp_output, 0.0)
        if not tmp_output.exists():
            return False
        tmp_output.replace(output_path)
    return True


def build_daily_videos(raw_root: Path, output_root: Path, thumbnail_root: Path) -> None:
    built = 0
    skipped = 0
    thumbs_built = 0
    thumbs_skipped = 0
    thumbs_failed = 0
    for image_type, spec in WXCAM_IMAGE_TYPES.items():
        stream_root = raw_root / spec["stream"]
        video_glob = spec["video_glob"]
        type_output_root = output_root / image_type
        for day_dir in _iter_day_dirs(stream_root):
            clips = _list_clips(day_dir, video_glob)
            if not clips:
                continue
            for clip_path in clips:
                thumb_path = _thumbnail_path(thumbnail_root, image_type, day_dir.name, clip_path)
                if _needs_refresh([clip_path], thumb_path):
                    try:
                        built_thumb = _build_thumbnail(clip_path, thumb_path)
                    except Exception as exc:
                        print(f"Skipping wxcam thumbnail {clip_path}: {exc}")
                        thumbs_failed += 1
                        continue
                    if built_thumb:
                        thumbs_built += 1
                    else:
                        print(f"Skipping wxcam thumbnail {clip_path}: ffmpeg produced no frame")
                        thumbs_failed += 1
                else:
                    thumbs_skipped += 1
            output_path = type_output_root / f"{day_dir.name}.mp4"
            if not _needs_refresh(clips, output_path):
                skipped += 1
                continue
            _build_output(clips, output_path)
            built += 1
            print(f"Built wxcam daily video: {image_type} {day_dir.name} -> {output_path}")
    print(
        "wxcam daily products complete: "
        f"videos_built={built} videos_skipped={skipped} "
        f"thumbnails_built={thumbs_built} thumbnails_skipped={thumbs_skipped} thumbnails_failed={thumbs_failed} "
        f"video_root={output_root} thumbnail_root={thumbnail_root}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Build daily wxcam mp4 products and hourly thumbnails from raw clips.")
    parser.add_argument("--raw-root", type=Path, default=RAW_DEFAULT)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_DEFAULT)
    parser.add_argument("--thumbnail-root", type=Path, default=THUMBNAIL_DEFAULT)
    args = parser.parse_args()
    build_daily_videos(args.raw_root, args.output_root, args.thumbnail_root)


if __name__ == "__main__":
    main()
