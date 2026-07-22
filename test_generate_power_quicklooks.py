from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np
import pandas as pd
import xarray as xr

import generate_power_quicklooks as quicklooks


class _Figure:
    def __init__(self, section: str) -> None:
        self.section = section

    def write_json(self, path: Path) -> None:
        Path(path).write_text(json.dumps({"section": self.section}), encoding="utf-8")


def test_latest_prewarms_publish_all_sections_and_live_window_metadata() -> None:
    times = pd.date_range(pd.Timestamp.now(tz="UTC").tz_localize(None) - pd.Timedelta(hours=2), periods=5, freq="30min")
    dataset = xr.Dataset({"BatterySOC": (("time",), np.linspace(80, 82, len(times)))}, coords={"time": times})
    with TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        paths = {
            "PREWARM_DIR": root,
            "PREWARM_JSON": root / "power_latest_interactive.json",
            "PREWARM_CURRENT_JSON": root / "power_current_latest_interactive.json",
            "PREWARM_FORECAST_JSON": root / "power_forecast_latest_interactive.json",
            "PREWARM_METADATA_JSON": root / "power_prewarms_metadata.json",
        }
        with (
            patch.multiple(quicklooks, **paths),
            patch.object(quicklooks, "combine_summary_datasets", side_effect=lambda _instrument, *items: next(item for item in items if item is not None)),
            patch.object(quicklooks, "build_summary_plotly", side_effect=lambda _ds, _instrument, **kwargs: _Figure(str(kwargs.get("panel_groups")))),
        ):
            written = quicklooks.generate_latest_prewarms(dataset, None, None)

        assert set(written) == {"all", "current", "forecast"}
        assert all(path.exists() for path in written.values())
        metadata = json.loads((root / "power_prewarms_metadata.json").read_text(encoding="utf-8"))
        assert metadata["display_end_utc"] >= metadata["observed_latest_utc"]
        assert set(metadata["sections"]) == {"all", "current", "forecast"}
