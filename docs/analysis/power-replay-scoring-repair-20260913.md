# Mature replay truth repair, 13 September 2026

The first immutable replay issue generated successfully but its evidence reader
stopped at issue time. Its only paired SOC point was the authoring anchor.

Scoring now reads matured observations through the smaller of current UTC and
the forecast endpoint. Training power and PDU state remain cut off at issue time.
The evaluation cutoff is recorded separately from the forecast observation cutoff.
Zero-lead anchors are retained for provenance but excluded from deterministic
and ensemble skill. Regression coverage adds observations after immutable issue
creation, checks multiple scored positive leads and preserves baseline digests.

Deploy only to an implementation-keyed isolated replay tree. Existing evidence,
the frozen baseline, public products and instrument controls remain unchanged.
Missing baseline ensemble, MPPT-active history and verified panel geometry remain
explicit evidence gates; this code does not fabricate or waive them.

## Verified live result

Deployed code: `cecffb8ba5ef46bb48e681af11de695e15d9794b`.
Implementation digest: `3633138d278681fce1ee1934beb306c0417b0c5faee2c4c8495a0577da94e571`.
The first recovered issue completed at 11:05:21 UTC: processed 1, failed 0,
pending 52; peak memory 878.4 MiB. Each lane has 32 scored positive-lead
points within 96 hours: 2, 6, 8 and 16 across the four lead buckets, versus
the previous single anchor point. This is only one independent cycle/day,
so these results are diagnostic, not promotion evidence.

Hybrid SOC MAEs for this one issue are 0.676, 3.816, 11.890 and 8.849 points;
baseline values are 0.676, 5.078, 11.203 and 9.387. The mixed result must not
be represented as a general improvement. All six ECMWF provider tests pass
on the deployed runtime. The disposable local environment lacks Earthkit,
explaining its two provider-test failures; 283 power tests pass locally.

The temporary idle-window CPU quota was restored to 25%; the 1.5 GiB cap
was retained throughout. Both original timers were restored, and the wider
evaluator was restarted only after the replay completed. The existing hourly
monitor was updated to the new release references and scientific blockers.
