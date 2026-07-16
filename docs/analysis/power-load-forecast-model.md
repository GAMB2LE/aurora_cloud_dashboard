# APS load forecast model

## ECMWF data provider

The deterministic solar forecast has a provider boundary controlled by
`AURORA_ECMWF_PROVIDER` (or `--provider`):

- `legacy` uses `ecmwf-opendata` for retrieval and `cfgrib` for decoding.
- `earthkit` uses `earthkit-data` for retrieval and decoding, with an automatic
  legacy fallback that preserves a usable forecast.
- `shadow` publishes the legacy result and compares Earthkit against the same
  input file. The comparison is written to
  `/data/aurora/products/power/ecmwf_provider_shadow.json` by default.

Both providers are normalized to the same site-level xarray contract before
the calibrated solar, load, and SOC model runs. Forecast Zarr variables and
paths therefore remain unchanged. Provider, cycle, selected grid point,
timings, parity, and fallback details are stored as forecast metadata and
propagated into the display summary metadata.

The 50-member ensemble continues to use direct ecCodes nearest-point streaming.
This avoids allocating the complete global member-by-step grid; it should only
move to Earthkit after an equivalent bounded-memory benchmark passes.

## Decision

The planning load model is `hybrid_state_space_v5`. Its observed target is
the APS energy balance:

`load = SolarWatts_East + SolarWatts_South + SolarWatts_West - BatteryWatts`

APS `BatteryWatts` is positive during charging and negative during discharge.
This balance therefore measures total system consumption, including the 48 V
DC load. `ACOutputWatts + DCInverterWatts` is not used as the primary target:
in `DC-Only` it reports about `9 W` of inverter idle while the battery balance
shows about `220-226 W` of real consumption.

The discrete part is an HMM-like finite-state classifier. Fresh PDU outlet watts
provide direct evidence for CL61, Radar, HATPRO, UAS, and their combinations;
the APS AC output and learned total-load level provide secondary evidence.
Stale PDU data cannot assert that a kit remains on. A transition prior prevents
single noisy observations from making the classification oscillate, while the
posterior probability supplies a dashboard confidence value.

The continuous part is a robust Kalman learner over additive components: DC,
CL61, Radar, HATPRO, UAS, and Unknown AC. Every observation supplies a total
power-balance equation and fresh PDU outlet watts can additionally constrain an
individual kit component. Innovation clipping prevents a transition spike or a
bad sample from moving the learned level arbitrarily. The persisted mean and
full covariance become the load ensemble used by the SOC forecast.

Recognition and durable learning remain separate. The latest state can change
as soon as fresh PDU and APS data identify it, but saved component parameters
advance only for timestamps newer than the previous training cursor. This makes
the five-minute process incremental and idempotent. Zero-solar battery
discharge remains the strongest evidence for the DC component because it
measures the complete battery-side load.

Future operator choices are represented explicitly instead of guessed. Named
plans include current mode, DC-Only, DC + CL61 continuously on, an optimized
CL61 schedule, a custom CL61 start/duration, and any other kit combinations the
model has learned. The old `100-600 W` plot is retained only as a backwards-
compatible data contract and is no longer the operating interface.

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

## Learning, planning, and verification

The operating-state job runs every five minutes, re-anchors SOC to the latest
actual observation, and consumes only newly arrived samples for parameter
learning. A separate planning job retrieves the latest eligible ECMWF 00 or 12
UTC deterministic cycle twice daily and forecasts 240 hours. Its native cycle
is reused by the faster state job, so a kit transition changes the load and SOC
plans without redownloading ECMWF.

The optimizer considers CL61 on/off schedules over the first 96 hours, then
propagates each candidate with CL61 off through the complete 240-hour planning
forecast. It maximizes collection time only among candidates whose P10 SOC
stays at or above 40% for all 240 hours, with a minimum 12-hour CL61 run and at
most one start per UTC day. It is advisory only. A custom start/duration editor
runs the same ensemble calculation immediately and marks the plan safe,
marginal, or unsafe from its minimum P10 SOC.

Archived deterministic forecasts carry `LoadModelVersion`. Load MAE, bias, and
skill only use rows from a matching model version, preventing retired model
errors from contaminating the improvement loop. Version 5 therefore starts a
fresh verification series. SOC MAE is reported by lead bucket; solar and load
MAE/bias diagnose the two principal error sources. Skill is measured against
persistence, and the fixed-lead hindcast shows what the dashboard would have
forecast 6, 24, 48, and 72 hours before each observation.

The 50-member SOC ensemble also re-anchors when the deterministic forecast's
SOC anchor, calibrated solar factor, learned load, model version, or named mode
changes, even when ECMWF is still on the same 00/12 UTC ensemble cycle. The
cycle's accumulated SSRD values at the AURORA grid point are cached as a small
site Zarr, so hourly re-anchoring does not redownload or reparse the global
GRIB. This keeps the probabilistic forecast aligned with mode transitions such
as `DC-Only` to `DC-Only + CL61`.

When additional AC kit is switched on, its fresh PDU signature creates or
updates that named mode automatically. The dashboard shows the recognised mode,
confidence, component-aware plan curves, uncertainty, load axis, and the 40%
minimum operational reference so operators can inspect the classification and
risk directly.
