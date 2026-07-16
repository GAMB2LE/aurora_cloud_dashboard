# APS load forecast model

## Decision

The operational load model is `kit_mode_persistence_v4`. Its observed target is
the APS energy balance:

`load = SolarWatts_East + SolarWatts_South + SolarWatts_West - BatteryWatts`

APS `BatteryWatts` is positive during charging and negative during discharge.
This balance therefore measures total system consumption, including the 48 V
DC load. `ACOutputWatts + DCInverterWatts` is not used as the primary target:
in `DC-Only` it reports about `9 W` of inverter idle while the battery balance
shows about `220-226 W` of real consumption.

For each uninterrupted operating mode, the preferred baseline is the median of
`-BatteryWatts` during the latest 48 hours when summed measured solar is at most
`10 W`. At least four 15-minute samples are required. With no solar input this
is a direct battery-side measurement of total station consumption. If those
dark samples are unavailable, the model uses the recent full power-balance
median instead.

The model reduces load and AC output to 15-minute medians. Fresh measured PDU
power identifies a named AC mode before the slower AC-output median has fully
changed state. Outlet power at or above `5 W` is active; relay state is used
only when finite outlet watts are unavailable. Outlet 5 maps to `CL61`, so the
current Ceilometer state is `DC-Only + CL61`. Sustained AC output below `25 W`
with no powered PDU signature is named `DC-Only`. Known mappings also include
UAS, Radar, and HATPRO, and multiple active items produce a combined name such
as `DC-Only + Radar + HATPRO`.

Recognition and durable learning are separate. The forecast can switch to a
powered PDU mode immediately, using its previously learned level or the clean
DC baseline plus current PDU watts. A new training observation is retained only
after the corresponding AC/DC state has been stable for at least 30 minutes and
has at least two aggregated samples. One independent observation per hour is
kept for each named mode, up to seven days. The median of those observations is
the learned mode level used through the forecast horizon. A known AC mode can
also be recognised by its learned power level when PDU data are temporarily
stale.

The `DC-Only` registry is independently checked against zero-solar,
battery-discharge samples with AC output below `25 W`. Transition-level
outliers are removed before the baseline is reused, preventing a newly started
AC load from contaminating the minimum-power profile.

The model deliberately does not predict when operators will switch equipment.
The current mode persists until the measured AC/PDU state changes. Fixed
`100-600 W` SOC scenarios remain the way to test hypothetical future loads.

## Evidence

The diagnostic run on 2026-07-16 used
`/data/aurora/products/power/power.zarr` and 15-minute median samples. In the
current `DC-Only` run, AC output was `0 W`, inverter idle was about `8.9 W`, and
55 zero-solar 15-minute battery-discharge samples gave a `223 W` baseline. The
independently measured ASFS 48 V trace was about `157 W`; it is the main 48 V
branch, while the battery-side balance also includes the rest of the DC system
and conversion losses.

Later on 2026-07-16, the Ceilometer transition supplied the first named AC-mode
case. Fresh PDU outlet 5 data reported `223 W`, median AC output was about
`174 W`, and the APS power balance measured about `455 W` total station load.
The prior `DC-Only` baseline was independently recovered as `223.7 W` from 59
zero-solar 15-minute samples (`220.0-225.5 W` 10th-90th percentile). The mode
therefore resolves as `DC-Only + CL61`, with outlet 5 as its primary signature.

A rolling-origin comparison used issue times every six hours during the latest
seven-day verification interval, up to 21 days of history per issue, and lead
times from 3 through 24 hours. The target for every model was power-balance
total load. Errors are in watts across 184 forecast-observation pairs.

| Model | MAE | Bias | RMSE |
|---|---:|---:|---:|
| UTC-hour median | 330.28 | 126.74 | 353.16 |
| Raw last-value persistence | 135.35 | 14.72 | 279.15 |
| `kit_mode_persistence_v2` | 131.53 | 15.62 | 275.97 |
| `kit_mode_persistence_v3` | 131.28 | -33.53 | 232.73 |

This table evaluates the version-3 predecessor. On this limited backtest,
version 3 reduced RMSE by about 16% relative to
version 2 and slightly reduced MAE, but its negative bias shows that a single
`DC-Only` baseline cannot represent unlabelled higher-load operation. These are
historical results, not a guarantee of future skill. Named PDU kit modes will
separate those periods as they are observed; there are not yet enough repeated
kit transitions to estimate operator switching times.

## Learning and verification

Every 15-minute learning run re-anchors SOC to the latest actual observation,
recognises the current mode, refreshes its robust load level, and updates the
mode registry no more than once per independent hour. Archived load forecasts
carry `LoadModelVersion`. Load MAE, bias, and skill only use rows from version
`4`, preventing retired model errors from contaminating the improvement loop.
Version 4 therefore starts a fresh load-skill series rather than mixing its
errors with versions 0-3. Its transition safeguards need new archived cases
before they can be compared independently with the version-3 backtest above.

The 50-member SOC ensemble also re-anchors when the deterministic forecast's
SOC anchor, calibrated solar factor, learned load, model version, or named mode
changes, even when ECMWF is still on the same 00/12 UTC ensemble cycle. The
cycle's accumulated SSRD values at the AURORA grid point are cached as a small
site Zarr, so hourly re-anchoring does not redownload or reparse the global
GRIB. This keeps the probabilistic forecast aligned with mode transitions such
as `DC-Only` to `DC-Only + CL61`.

When additional AC kit is switched on, the next sustained PDU signature creates
or updates that named mode automatically. The dashboard forecast-load legend
shows the recognised mode so operators can check that classification directly.
