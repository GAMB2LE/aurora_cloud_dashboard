from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from display_artifact_manifest import build_manifest, load_manifest, write_manifest


class DisplayArtifactManifestTests(unittest.TestCase):
    def test_manifest_contains_only_supported_derived_files(self) -> None:
        with TemporaryDirectory() as raw_directory:
            root = Path(raw_directory)
            (root / "nested").mkdir()
            (root / "latest.json").write_text("{}", encoding="utf-8")
            (root / "nested" / "plot.png").write_bytes(b"png")
            (root / "source.zarr").mkdir()
            (root / "source.zarr" / "chunk.0").write_bytes(b"raw")
            payload = build_manifest({"display": root})

        files = payload["groups"]["display"]["files"]
        self.assertEqual([item["path"] for item in files], ["latest.json", "nested/plot.png"])
        self.assertEqual(payload["artifactCount"], 2)
        self.assertTrue(payload["contentRevision"])

    def test_manifest_is_atomically_readable_after_write(self) -> None:
        with TemporaryDirectory() as raw_directory:
            root = Path(raw_directory)
            source = root / "source"
            source.mkdir()
            (source / "plot.png").write_bytes(b"png")
            destination = root / "latest.json"
            write_manifest(destination, build_manifest({"display": source}))
            loaded = load_manifest(destination)

        self.assertTrue(loaded["available"])
        self.assertEqual(loaded["artifactCount"], 1)


if __name__ == "__main__":
    unittest.main()
