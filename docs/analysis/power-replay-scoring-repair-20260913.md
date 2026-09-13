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
