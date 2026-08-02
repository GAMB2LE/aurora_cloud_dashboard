from pathlib import Path


def test_dashboard_rebuild_does_not_control_retired_science_runner() -> None:
    source = (Path(__file__).parent / "rebuild_dashboard_zarrs_from.py").read_text()

    assert "aurora-les-operational-run" not in source
