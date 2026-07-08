from datetime import datetime, timezone

from uas_mqtt import load_uas_mqtt_log, parse_uas_mqtt_line


def test_parse_tier_change_line():
    record = parse_uas_mqtt_line("2026-07-08 13:40:22: Tier change 4 3", line_number=7)

    assert record is not None
    assert record.timestamp == datetime(2026, 7, 8, 13, 40, 22, tzinfo=timezone.utc)
    assert record.event_type == "tier_change"
    assert record.reported_tier == 4
    assert record.effective_tier == 3
    assert record.line_number == 7


def test_parse_sample_line():
    record = parse_uas_mqtt_line("2026-07-08 13:44:42: 4 4")

    assert record is not None
    assert record.event_type == "sample"
    assert record.reported_tier == 4
    assert record.effective_tier == 4


def test_parse_malformed_line_returns_none():
    assert parse_uas_mqtt_line("not a uas mqtt line") is None


def test_load_log_tracks_empty_and_malformed_lines(tmp_path):
    log = tmp_path / "menapia_mqtt.log"
    log.write_text(
        "\n".join(
            [
                "2026-07-08 13:40:22: Tier change 4 4",
                "",
                "bad line",
                "2026-07-08 13:40:32: 4 4",
            ]
        )
    )

    result = load_uas_mqtt_log(log)

    assert not result.missing
    assert result.error is None
    assert len(result.records) == 2
    assert len(result.malformed_lines) == 1
    assert result.malformed_lines[0].startswith("3: bad line")


def test_load_missing_log_marks_missing(tmp_path):
    result = load_uas_mqtt_log(tmp_path / "missing.log")

    assert result.missing
    assert result.records == ()
