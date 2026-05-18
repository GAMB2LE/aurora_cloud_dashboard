#!/usr/bin/env python3
"""Build daily wxcam mp4 products, rolling latest videos, and hourly thumbnails."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

from PIL import Image

from extra_housekeeping import (
    extra_housekeeping_daily_png,
    extra_housekeeping_latest_png,
    plot_wxcam_housekeeping_day,
    plot_wxcam_housekeeping_latest,
)
from wxcam_catalog import WXCAM_IMAGE_TYPES, parse_timestamp

RAW_DEFAULT = Path("/project/aurora/raw/wxcam")
OUTPUT_DEFAULT = Path("/data/aurora/products/wxcam/daily_videos")
THUMBNAIL_DEFAULT = Path("/data/aurora/products/wxcam/hourly_thumbnails")
QUICKLOOK_DEFAULT = Path("/data/aurora/products/quicklooks/wxcam")
CATALOG_DEFAULT = Path("/data/aurora/products/wxcam/wxcam_catalog.sqlite")
DAY_DIR_REGEX = re.compile(r"^\d{8}$")
SETTLED_VIDEO_GRACE_SECONDS = 10 * 60


def _iter_day_dirs(stream_root: Path, reverse: bool = False):
    if not stream_root.exists():
        return
    for path in sorted(stream_root.iterdir(), reverse=reverse):
        if path.is_dir() and DAY_DIR_REGEX.match(path.name):
            yield path


def _list_files(day_dir: Path, pattern: str, recursive: bool = False) -> list[Path]:
    iterator = day_dir.rglob(pattern) if recursive else day_dir.glob(pattern)
    return sorted(path for path in iterator if path.is_file())


def _settled_video_clips(clips: list[Path], *, now_ts: float, grace_seconds: int = SETTLED_VIDEO_GRACE_SECONDS) -> list[Path]:
    # During long raw backfills, the newest hourly clip may still be growing on disk.
    # Hold those back for the next timer pass instead of failing the whole product job.
    cutoff = now_ts - grace_seconds
    settled: list[Path] = []
    for path in clips:
        try:
            if path.stat().st_mtime <= cutoff:
                settled.append(path)
        except FileNotFoundError:
            continue
    return settled


def _prime_latest_clips(day_dirs: list[Path], video_glob: str, *, now_ts: float, max_clips: int = 24) -> list[Path]:
    seeded: list[Path] = []
    for day_dir in day_dirs:
        seeded.extend(_settled_video_clips(_list_files(day_dir, video_glob), now_ts=now_ts))
        seeded.sort()
        if len(seeded) >= max_clips:
            break
    return seeded[-max_clips:]


def _needs_refresh(clips: list[Path], output_path: Path) -> bool:
    if not clips:
        return False
    if not output_path.exists():
        return True
    if output_path.suffix.lower() == ".mp4":
        signature_path = _video_signature_path(output_path)
        if not signature_path.exists():
            return True
        try:
            return signature_path.read_text(encoding="utf-8") != _clip_signature(clips)
        except OSError:
            return True
    output_mtime_ns = output_path.stat().st_mtime_ns
    return any(path.stat().st_mtime_ns > output_mtime_ns for path in clips)


def _clip_signature(clips: list[Path]) -> str:
    lines = []
    for path in clips:
        stat_result = path.stat()
        lines.append(f"{path.resolve()}\t{stat_result.st_size}\t{stat_result.st_mtime_ns}")
    return "\n".join(lines) + "\n"


def _video_signature_path(output_path: Path) -> Path:
    return output_path.with_suffix(output_path.suffix + ".inputs")


def _build_output(clips: list[Path], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    signature = _clip_signature(clips)
    with tempfile.TemporaryDirectory(dir=output_path.parent) as tmpdir_name:
        tmpdir = Path(tmpdir_name)
        list_path = tmpdir / "clips.txt"
        tmp_output = tmpdir / output_path.name
        tmp_signature = tmpdir / f"{output_path.name}.inputs"
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
        tmp_signature.write_text(signature, encoding="utf-8")
        tmp_output.replace(output_path)
        tmp_signature.replace(_video_signature_path(output_path))


def _thumbnail_path(thumbnail_root: Path, image_type: str, day_token: str, source_path: Path) -> Path:
    return thumbnail_root / image_type / day_token / f"{source_path.stem}.jpg"


def _representative_hourly_images(day_dir: Path, pattern: str) -> dict[int, Path]:
    chosen: dict[int, Path] = {}
    scores: dict[int, tuple[int, int, str]] = {}
    for image_path in _list_files(day_dir, pattern, recursive=True):
        timestamp = parse_timestamp(image_path)
        if timestamp is None:
            continue
        hour = timestamp.hour
        seconds_after_hour = timestamp.minute * 60 + timestamp.second
        score = (
            abs(seconds_after_hour - 30 * 60),
            seconds_after_hour,
            image_path.name,
        )
        if hour not in chosen or score < scores[hour]:
            chosen[hour] = image_path
            scores[hour] = score
    return chosen


def _build_image_thumbnail(image_path: Path, output_path: Path) -> bool:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    resample = getattr(Image, "Resampling", Image).LANCZOS
    with tempfile.TemporaryDirectory(dir=output_path.parent) as tmpdir_name:
        tmpdir = Path(tmpdir_name)
        tmp_output = tmpdir / output_path.name
        with Image.open(image_path) as image:
            preview = image.convert("RGB")
            preview.thumbnail((540, 540), resample)
            preview.save(tmp_output, format="JPEG", quality=88, optimize=True)
        if not tmp_output.exists():
            return False
        tmp_output.replace(output_path)
    return True


def build_daily_videos(
    raw_root: Path,
    output_root: Path,
    thumbnail_root: Path,
    quicklook_root: Path,
    catalog_path: Path,
) -> None:
    built = 0
    latest_built = 0
    skipped = 0
    latest_skipped = 0
    failed = 0
    latest_failed = 0
    thumbs_built = 0
    thumbs_skipped = 0
    thumbs_failed = 0
    day_tokens_seen: set[str] = set()
    now_ts = datetime.now(timezone.utc).timestamp()
    today_token = datetime.now(timezone.utc).strftime("%Y%m%d")
    today_iso = f"{today_token[:4]}-{today_token[4:6]}-{today_token[6:8]}"
    if catalog_path.exists():
        latest_hk = extra_housekeeping_latest_png(quicklook_root, "wxcam")
        if latest_hk is not None:
            plot_wxcam_housekeeping_latest(catalog_path, "HK_WXcam - Latest 24 hours", latest_hk)
        today_hk = extra_housekeeping_daily_png(quicklook_root, "wxcam", today_token)
        if today_hk is not None:
            plot_wxcam_housekeeping_day(catalog_path, today_iso, f"HK_WXcam - {today_iso}", today_hk)
    for image_type, spec in WXCAM_IMAGE_TYPES.items():
        stream_root = raw_root / spec["stream"]
        image_glob = spec["image_glob"]
        video_glob = spec["video_glob"]
        type_output_root = output_root / image_type
        all_clips: list[Path] = []
        latest_path = type_output_root / "latest.mp4"
        latest_finalized = False
        primed_days: set[str] = set()
        day_dirs = list(_iter_day_dirs(stream_root, reverse=True))

        primed_latest_clips = _prime_latest_clips(day_dirs, video_glob, now_ts=now_ts)
        if primed_latest_clips:
            if _needs_refresh(primed_latest_clips, latest_path):
                try:
                    _build_output(primed_latest_clips, latest_path)
                except subprocess.CalledProcessError as exc:
                    print(f"Skipping wxcam rolling latest video for {image_type}: {exc}")
                    latest_failed += 1
                else:
                    latest_built += 1
                    latest_finalized = True
                    print(f"Built wxcam rolling latest video: {image_type} -> {latest_path}")
            else:
                latest_skipped += 1
                latest_finalized = True

        if day_dirs:
            newest_day = day_dirs[0]
            newest_day_clips = _settled_video_clips(_list_files(newest_day, video_glob), now_ts=now_ts)
            newest_output = type_output_root / f"{newest_day.name}.mp4"
            if newest_day_clips:
                if _needs_refresh(newest_day_clips, newest_output):
                    try:
                        _build_output(newest_day_clips, newest_output)
                    except subprocess.CalledProcessError as exc:
                        print(f"Skipping wxcam daily video build for {image_type} {newest_day.name}: {exc}")
                        failed += 1
                    else:
                        built += 1
                        primed_days.add(newest_day.name)
                        print(f"Built wxcam daily video: {image_type} {newest_day.name} -> {newest_output}")
                else:
                    primed_days.add(newest_day.name)

        for day_dir in day_dirs:
            day_tokens_seen.add(day_dir.name)
            hourly_images = _representative_hourly_images(day_dir, image_glob)
            for image_path in hourly_images.values():
                thumb_path = _thumbnail_path(thumbnail_root, image_type, day_dir.name, image_path)
                if _needs_refresh([image_path], thumb_path):
                    try:
                        built_thumb = _build_image_thumbnail(image_path, thumb_path)
                    except Exception as exc:
                        print(f"Skipping wxcam thumbnail {image_path}: {exc}")
                        thumbs_failed += 1
                        continue
                    if built_thumb:
                        thumbs_built += 1
                    else:
                        print(f"Skipping wxcam thumbnail {image_path}: no thumbnail produced")
                        thumbs_failed += 1
                else:
                    thumbs_skipped += 1
            clips = _list_files(day_dir, video_glob)
            settled_clips = _settled_video_clips(clips, now_ts=now_ts)
            if settled_clips:
                all_clips.extend(settled_clips)
                all_clips.sort()
                if not latest_finalized and len(all_clips) >= 24:
                    latest_clips = all_clips[-24:]
                    if _needs_refresh(latest_clips, latest_path):
                        try:
                            _build_output(latest_clips, latest_path)
                        except subprocess.CalledProcessError as exc:
                            print(f"Skipping wxcam rolling latest video for {image_type}: {exc}")
                            latest_failed += 1
                        else:
                            latest_built += 1
                            latest_finalized = True
                            print(f"Built wxcam rolling latest video: {image_type} -> {latest_path}")
                    else:
                        latest_skipped += 1
                        latest_finalized = True
            if not settled_clips:
                continue
            output_path = type_output_root / f"{day_dir.name}.mp4"
            if day_dir.name in primed_days and not _needs_refresh(settled_clips, output_path):
                continue
            if not _needs_refresh(settled_clips, output_path):
                skipped += 1
                continue
            try:
                _build_output(settled_clips, output_path)
            except subprocess.CalledProcessError as exc:
                print(f"Skipping wxcam daily video build for {image_type} {day_dir.name}: {exc}")
                failed += 1
                continue
            built += 1
            print(f"Built wxcam daily video: {image_type} {day_dir.name} -> {output_path}")
        if all_clips and not latest_finalized:
            latest_clips = all_clips[-24:]
            if _needs_refresh(latest_clips, latest_path):
                try:
                    _build_output(latest_clips, latest_path)
                except subprocess.CalledProcessError as exc:
                    print(f"Skipping wxcam rolling latest video for {image_type}: {exc}")
                    latest_failed += 1
                    continue
                latest_built += 1
                print(f"Built wxcam rolling latest video: {image_type} -> {latest_path}")
            else:
                latest_skipped += 1
    if catalog_path.exists():
        for day_token in sorted(day_tokens_seen):
            hk_out = extra_housekeeping_daily_png(quicklook_root, "wxcam", day_token)
            if hk_out is None:
                continue
            if hk_out.exists() and day_token != today_token:
                continue
            plot_wxcam_housekeeping_day(catalog_path, f"{day_token[:4]}-{day_token[4:6]}-{day_token[6:8]}", f"HK_WXcam - {day_token[:4]}-{day_token[4:6]}-{day_token[6:8]}", hk_out)
        latest_hk = extra_housekeeping_latest_png(quicklook_root, "wxcam")
        if latest_hk is not None:
            plot_wxcam_housekeeping_latest(catalog_path, "HK_WXcam - Latest 24 hours", latest_hk)
    print(
        "wxcam daily products complete: "
        f"videos_built={built} videos_skipped={skipped} videos_failed={failed} "
        f"latest_built={latest_built} latest_skipped={latest_skipped} latest_failed={latest_failed} "
        f"thumbnails_built={thumbs_built} thumbnails_skipped={thumbs_skipped} thumbnails_failed={thumbs_failed} "
        f"video_root={output_root} thumbnail_root={thumbnail_root} quicklook_root={quicklook_root}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Build daily wxcam mp4 products and hourly thumbnails from raw clips.")
    parser.add_argument("--raw-root", type=Path, default=RAW_DEFAULT)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_DEFAULT)
    parser.add_argument("--thumbnail-root", type=Path, default=THUMBNAIL_DEFAULT)
    parser.add_argument("--quicklook-root", type=Path, default=QUICKLOOK_DEFAULT)
    parser.add_argument("--catalog-path", type=Path, default=CATALOG_DEFAULT)
    args = parser.parse_args()
    build_daily_videos(args.raw_root, args.output_root, args.thumbnail_root, args.quicklook_root, args.catalog_path)


if __name__ == "__main__":
    main()
