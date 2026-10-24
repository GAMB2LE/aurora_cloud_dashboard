"""Keep automatic app fallback separate from public development experiments."""

import os
from unittest.mock import patch

import mobile_catalog


def test_development_does_not_advertise_operational_fallback_by_default():
    with patch.dict(os.environ, {"AURORA_SITE_ENV": "development"}, clear=True):
        assert mobile_catalog.deployment_descriptor()["operationalMirror"] is False


def test_operational_mirror_is_explicit_and_keeps_browser_on_production():
    with patch.dict(os.environ, {
        "AURORA_SITE_ENV": "development",
        "AURORA_DOMAIN": "data-ocean.gamb2le.co.uk",
        "AURORA_OPERATIONAL_MIRROR": "true",
        "AURORA_POWER_CANDIDATE_API_ENABLED": "true",
    }, clear=True):
        deployment = mobile_catalog.deployment_descriptor()
        assert deployment["operationalMirror"] is True
        assert deployment["dataRole"] == "live-mirror"
        assert deployment["dashboardURL"] == "https://data.gamb2le.co.uk/app"
        assert not mobile_catalog.power_candidate_evaluation_enabled()


def test_continuity_and_production_cannot_claim_operational_mirror():
    for settings in [
        {"AURORA_SITE_ENV": "production"},
        {"AURORA_SITE_ENV": "development", "AURORA_DATA_MODE": "direct-field-continuity"},
    ]:
        with patch.dict(os.environ, {"AURORA_OPERATIONAL_MIRROR": "true", **settings}, clear=True):
            assert mobile_catalog.deployment_descriptor()["operationalMirror"] is False
