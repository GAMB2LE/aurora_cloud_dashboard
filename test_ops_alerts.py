from send_ops_alerts import evaluate_alerts


def _ids(snapshot):
    return {alert.id for alert in evaluate_alerts(snapshot)}


def test_low_internal_temperature_alerts_separately_from_high_temp():
    assert "power:internal_temp_low" in _ids({"aps_internal_temp_c": 4.9})
    assert "power:internal_temp" in _ids({"aps_internal_temp_c": 45.0})


def test_normal_internal_temperature_does_not_alert():
    ids = _ids({"aps_internal_temp_c": 10.0})

    assert "power:internal_temp_low" not in ids
    assert "power:internal_temp" not in ids


def test_dewpoint_alert_requires_internal_humidity_available():
    assert "power:internal_dewpoint" not in _ids(
        {
            "aps_internal_humidity_available_state": 0,
            "aps_internal_dewpoint_margin_c": -1.0,
        }
    )


def test_dewpoint_alerts_when_margin_is_at_or_below_zero():
    ids = _ids(
        {
            "aps_internal_humidity_available_state": 1,
            "aps_internal_dewpoint_margin_c": 0.0,
            "aps_internal_humidity_pct": 100.0,
            "aps_internal_dewpoint_c": 5.0,
            "aps_internal_dewpoint_temp_c": 5.0,
        }
    )

    assert "power:internal_dewpoint" in ids
