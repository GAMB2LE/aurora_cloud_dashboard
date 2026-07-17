"""Evaluate whether the Earthkit decoder has enough shadow evidence for review.

This command never changes the configured decoder.  It creates a transparent,
machine-readable promotion gate for an operator to review after the observation
window has elapsed.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _parse_timestamp(value: object) -> datetime | None:
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None


def _read_history(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    records: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            records.append(payload)
    return records


def evaluate(
    records: list[dict[str, Any]],
    *,
    minimum_days: float,
    minimum_samples: int,
    maximum_ssrd_difference: float,
) -> dict[str, Any]:
    timestamps = [timestamp for item in records if (timestamp := _parse_timestamp(item.get("opened_at_utc")))]
    compared = [item for item in records if item.get("shadow_status") == "compared"]
    failed = [item for item in records if item.get("shadow_status") != "compared"]
    mismatch = [item for item in compared if not item.get("valid_times_match")]
    max_difference = max(
        (float(item["ssrd_max_abs_difference_j_m2"]) for item in compared if item.get("ssrd_max_abs_difference_j_m2") is not None),
        default=None,
    )
    observed_hours = (
        (max(timestamps) - min(timestamps)).total_seconds() / 3600.0 if len(timestamps) >= 2 else 0.0
    )
    reasons: list[str] = []
    if observed_hours < minimum_days * 24.0:
        reasons.append("minimum_observation_window_not_reached")
    if len(compared) < minimum_samples:
        reasons.append("minimum_successful_comparisons_not_reached")
    if failed:
        reasons.append("shadow_decoder_failures_present")
    if mismatch:
        reasons.append("valid_time_mismatch_present")
    if max_difference is None or max_difference > maximum_ssrd_difference:
        reasons.append("ssrd_difference_exceeds_tolerance")
    return {
        "generated_at_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "decision": "eligible_for_operator_review" if not reasons else "not_eligible",
        "automatic_provider_switch": False,
        "observation_count": len(records),
        "successful_comparison_count": len(compared),
        "failure_count": len(failed),
        "valid_time_mismatch_count": len(mismatch),
        "observed_hours": round(observed_hours, 3),
        "maximum_ssrd_difference_j_m2": max_difference,
        "requirements": {
            "minimum_observation_days": minimum_days,
            "minimum_successful_comparisons": minimum_samples,
            "maximum_ssrd_difference_j_m2": maximum_ssrd_difference,
        },
        "reasons": reasons,
    }


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--history", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--minimum-days", type=float, default=7.0)
    parser.add_argument("--minimum-samples", type=int, default=50)
    parser.add_argument("--maximum-ssrd-difference", type=float, default=0.001)
    args = parser.parse_args()
    result = evaluate(
        _read_history(args.history),
        minimum_days=args.minimum_days,
        minimum_samples=args.minimum_samples,
        maximum_ssrd_difference=args.maximum_ssrd_difference,
    )
    _write_json(args.output, result)
    print(f"Wrote {args.output}: {result['decision']}")


if __name__ == "__main__":
    main()
