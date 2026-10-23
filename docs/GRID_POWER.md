# Grid power and measured DC charging

The APS MultiPlus-II is used for charging; the SureSine supplies AC loads.
This addition uses the existing Modbus logger, daily CSV, append-to-Zarr job,
forecast services and dashboard products. It adds no service, timer, broker,
database or dependency and does not issue device-control commands.

## Measurements

- `GridWatts`: signed AC input watts, Cerbo system unit 100 register 820 (L1).
  Keep it missing unless active input register 826 reports grid (1). Modbus
  returns zero for some unavailable values, so zero alone is not validity.
- `MultiPlusDCWatts`: DC voltage times signed DC current from a single paired
  read of VE.Bus unit 227 registers 26–27, scaled by 100 and 10 respectively.
  This is device instance 276, verified on the APS on 2026-10-23. Positive
  values supply the DC bus. Missing/failed reads and implausible 48 V bus
  voltages remain missing. The unit key `multiplus_monitor` is deliberately
  separate from the logger's dormant legacy charging-control key.

The mappings were checked against Victron's
[Modbus register list](https://github.com/victronenergy/dbus_modbustcp/blob/master/attributes.csv)
and [unit mapping](https://github.com/victronenergy/dbus_modbustcp/blob/master/unitid2di.csv),
and a read-only live check returned 53.96 V and 47.3 A (2552.308 W DC).

Both fields are optional in storage. Older records remain NaN. Daily CSV
headers remain fixed: the fields first appear in the next UTC daily file after
logger deployment. Do not rewrite the current or historical raw CSV files.

## Forecast meaning

Qualified station demand is:

    SolarEast + SolarSouth + SolarWest + max(MultiPlusDCWatts, 0) - BatteryWatts

All inputs must be present; negative inferred demand is invalid. Missing DC
power is not zero. Historical unknown-supply rows cannot train the corrected
load model. If no recent qualified load exists, the deterministic forecast is
not replaced and the operating product reports unavailable. Model versions
isolate previous load learning and SOC corrections.

The standard forecasts and operating recommendations assume no grid charging.
An optional `GridContinuesSOCP50` curve uses the same SOC anchor, weather,
load members and battery parameters as the current-state operating forecast.
It adds the preceding 15-minute median measured DC supply inside the existing
SOC integrator, retaining battery efficiency, power limits and SOC bounds.
It is shown for up to 96 hours and is a conditional scenario, not a prediction
of grid availability or charger taper.

Eligibility requires at least 90% paired AC/DC coverage, a valid final sample,
nonnegative input and charging values, and an observation age of at most 20
minutes. One-second coverage bins allow at most one second of nearest-sample
tolerance to accommodate the logger's approximately 1.1-second cadence.
A valid zero rate produces the same curve as the baseline. Missing or stale
telemetry omits the comparison. Its rate, contract, anchor and median curve
are retained in the existing operating recommendation archive.

No-grid SOC calibration rejects verification intervals with observed charging
or insufficient DC telemetry, including charging between otherwise valid SOC
endpoints. The two measured powers appear on the existing Battery Charging
panel. The conditional median appears on the existing 24 h and 96 h panels.

## Rollout and rollback

1. Preserve the current source and deploy the compatible cloud reader first.
2. Deploy the logger's two changed source files through its existing Git
   checkout and restart only `powermonitor.service`. Check fresh acquisition
   and the DC read. Expect the new CSV columns at the next UTC boundary.
3. Check raw CSV → Zarr values and missing-history semantics before activating
   the corrected forecast. Do not activate it before qualified DC samples exist.
4. Use the normal clean development release, smoke test, annotated production
   tag and exact-tag production release. Check both forecast cards and the
   existing battery panel. No broad infrastructure convergence is required.
5. Confirm the current-state comparison shares the SOC anchor and that missing
   DC data hides it. Verify the archived charging rate and curve.

For rollback, redeploy the preceding application and logger commits. Preserve
raw data and added Zarr fields; older code may ignore them. Keep the original
forecast timestamps when retaining a previous product.

Pre-implementation checkpoint refs in both repositories:
`codex/grid-integration-pre-implementation-20261023`.
Logger checkpoint: `ba519f2f1cad21aa7fca96a850d320bca6d56b0a`.
Dashboard checkpoint: `8f6b021b792cba9e076cf4c60044b52b94af9c3a`.

As inspected on 2026-10-23, deployment is pending: the development checkout
contains unrelated tracked edits and untracked enclosure/infrastructure files.
The standard release refuses a dirty checkout; this change does not overwrite
or package those files. Live reader, logger and forecast rollout has not begun.
