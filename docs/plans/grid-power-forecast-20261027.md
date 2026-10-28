# Grid power in the AURORA forecast

Plan prepared 27 October 2026 and revised 28 October 2026. No model, service, device or app changes have been applied as part of this plan.

## Agreed behaviour

Show two forecasts on the Power tab, on the primary and development dashboards and in the iOS and Android apps:

| Curve | Starting point | Future solar | Future grid contribution |
| --- | --- | --- | --- |
| **Grid Charging** | Latest valid measured battery SOC and its timestamp | Forecast production as if the previous solar array were connected | Qualified recent MultiPlus DC charging output, assumed to continue |
| **No Grid** | Exactly the same SOC and timestamp | Forecast production as if the previous solar array were connected | Zero grid charging |

Label the curves **“Grid Charging”** and **“No Grid”**, with a shared explanation: **“Both forecasts assume the solar panels are connected. The panels are currently removed.”** Grid Charging is the leading curve when fresh data supports its charging assumption. Show the measured charging rate and its observation time; do not imply that grid availability is predicted or guaranteed.

Retain the existing 24-hour and 96-hour views. Share the SOC anchor, forecast solar and weather, load assumptions, battery parameters and corresponding uncertainty samples. Only grid supply differs: Grid Charging adds the measured DC charging rate; No Grid adds zero grid input. With a zero grid rate the curves coincide; with nonnegative grid supply, Grid Charging must not fall below No Grid under the same integration assumptions. The current measured solar and grid displays remain observations, distinct from these future assumptions.

## 1. Establish the inputs and solar reference

- Trace stored GridWatts, MultiPlusDCWatts, SOC, solar and load inputs through the canonical product and the separate development continuity snapshot. Check both actual forecast input paths; a working dashboard card does not prove forecast input availability.
- Recheck the previously documented continuity-schema gap before changing it. Infrastructure PR97 remains open and records that gap; its October 26 evidence is not a fresh check of today's snapshot.
- Identify when solar was removed and select a verified healthy calibration from before removal, with the matching array configuration. Pin the same calibration and configuration identity for both hypothetical connected-solar scenarios.
- Exclude removed/disconnected periods from solar calibration and weather-to-PV bias learning. Do not let recent zero output suppress the hypothetical future array output. Do not silently switch to the checked-in physical PV candidate: its array configuration is provisional.
- If a verified healthy calibration/configuration cannot be established from records, make that a specific implementation question before publishing either connected-solar scenario. Do not invent array capacity or use a zeroed calibration.

## 2. Build one paired forecast calculation

- Start both curves at the latest valid measured SOC, using one timestamp and forecast issue identity. Re-anchor together during the existing refresh cycle; do not start either curve from an older forecast value.
- Use the same existing weather-driven solar model and pinned connected-array calibration for both curves, preserving normal day/night behaviour and future weather variation. Pair the same solar/weather samples across the two scenarios.
- Infer real station demand from actual observed solar plus positive measured MultiPlus DC supply minus net battery power. Never insert hypothetical future solar into historical observations or inferred load. Where disconnected hardware produces missing telemetry, distinguish a confirmed physical zero from an unknown measurement and retain raw gaps.
- Add the same forecast solar supply to both curves inside the existing battery integrator. For Grid Charging, also add measured MultiPlus DC supply once; for No Grid, add zero grid supply. Use MultiPlusDCWatts, not AC GridWatts, so AC conversion losses are not counted as energy delivered to the battery. Retain the battery model's efficiency, power limits and 0–100% SOC bounds.
- Reuse the existing 15-minute median DC-rate calculation as the first version of Grid Charging. It is a conditional constant-rate assumption, not a model of future outages, charger taper or maximum charger capacity. A valid zero rate makes Grid Charging coincide with No Grid because both still receive the same forecast solar. Zero must not be described as proof that grid power is off.
- A fresh valid final sample showing zero charging must immediately remove the previous positive charging assumption, even if most of the preceding 15 minutes were positive. Show the zero-rate comparison or an explicit transition/unavailable state; do not keep forecasting an old positive rate. Negative DC output is not positive charging.
- Keep uncertainty conditional on each scenario's stated solar/grid assumptions. Compute each curve and its uncertainty band from matched solar/weather, load and battery samples, adding the assumed DC grid rate only to Grid Charging; never wrap the grid median in no-grid bounds. Solar-weather uncertainty applies to both scenarios. Do not present confidence bands as confidence that the grid will stay available.

## 3. Close validity and learning gaps before activation

- Require valid, nonfuture SOC and paired AC/DC samples, at least 90% coverage over the preceding 15 minutes, and no more than 20 minutes of grid-observation age.
- Validate the latest raw sample before time binning; nearest-sample matching must not conceal a missing or invalid final reading. Reject duplicate/invalid timestamp ambiguity explicitly.
- Recheck grid-assumption validity when publishing and serving the forecast. Weather validity alone must not keep an old grid curve current. Expire the cached grid assumption locally in every client without new requests.
- If grid telemetry becomes unknown, show Grid Charging as unavailable with a reason; do not call it grid off. Retain the No Grid scenario only when its own SOC, load, solar-model and weather inputs qualify. Retained previous products must keep their real timestamps and stale status.
- Make deterministic and ensemble learning/skill checks use consistent eligibility rules that respect each scenario's charging and solar-availability assumptions. Do not train a no-grid model against SOC changes caused by grid charging. Both forecasts assume connected solar, so do not score either against removed-array operation as if its assumptions matched reality.

## 4. Share assumptions through the existing API

Extend the existing forecast payload additively with one scenario-assumption block containing:

- shared SOC anchor value and timestamp;
- exact scenario labels and trace IDs;
- shared solar status: hypothetical connected array, with the same calibration/configuration identity for both scenarios;
- grid status/reason, assumed DC watts, observation time, averaging window and expiry;
- forecast issue time and product freshness, separate from API request time.

Both native apps already support arbitrary traces in the existing SOC panels, and shared source already contains GridContinuesSOCP50. Reuse the existing same-solar comparison calculation, chart transport and battery integration, extending them with the explicit assumptions and validity checks above. Version the scenario contract and include trace identities so retained products are not silently relabelled and updated clients can distinguish the fully qualified paired forecasts from older payloads. No additional endpoint or background fetch is needed.

Update the shared chart labels, summaries, explanatory text and legends to **Grid Charging** and **No Grid**. Show the shared hypothetical solar assumption and input-age information consistently on web, iOS and Android. Android needs the richer forecast freshness/status fields already available to iOS; its request timestamp must not be presented as the model's update time.

These are planning scenarios. Keep both scenarios' hypothetical solar assumption out of device control and do not silently substitute either scenario into existing operating recommendations or autonomous-runtime claims.

## 5. Validate before release

Use offline fixtures and isolated replay from existing stored observations and weather products:

- Both curves share the exact measured starting SOC, time axis, solar/weather forcing and load/battery samples. No Grid's future grid contribution is identically zero.
- With a zero grid rate, the curves coincide for any shared solar forecast. With nonnegative grid supply, Grid Charging is never below No Grid within numerical tolerance, and both remain bounded.
- Known energy-balance examples prove that neither AC input nor hypothetical solar is counted twice.
- Removing or zeroing current observed solar does not erase either connected-array future forecast. Both retain the same hypothetical solar input, and observed history remains unchanged.
- Healthy historical solar periods reproduce plausible solar generation, including night-time zero output. Removed-array periods are excluded from calibration and inappropriate skill scoring.
- Missing, stale, future-dated, gapped, negative and invalid final samples behave correctly. A cached grid curve expires even if weather remains valid or the app is offline.
- Primary, development and continuity paths preserve the optional inputs and the same scenario metadata.
- Missing shared solar calibration/weather disables both connected-solar scenarios; missing grid-rate telemetry disables Grid Charging while No Grid remains available if its other inputs qualify. Shared SOC and load validity still apply to both.
- Both forecast horizons, all four presentation surfaces, old payload compatibility and unchanged request counts pass checks.

## 6. Release in stages

1. Reconcile the current forecast source pins and any parallel forecast work. The live APIs currently identify forecast revisions different from the presentation checkout; do not replace an entire scientific writer checkout with the dashboard revision.
2. Fix the input propagation and model/metadata gaps in isolated changes. Run the paired forecasts into separate validation outputs using stored inputs; leave published products intact.
3. Use the existing focused forecast-release approach to stage development first, preserving schedules, locks, output paths and writer ownership. Verify the actual API and browser results.
4. Promote the reviewed forecast/API changes to production. Release the matching iOS and Android presentation changes and verify real store/tester availability separately from successful builds. Android's existing upload-certificate blocker still needs resolution for a new native release.
5. On failure, restore the previous code selection and published forecast bundle without deleting or rewriting observations, learned-state history or archives.

## Equipment impact

All work uses existing cloud-stored telemetry, existing weather products and existing forecast/app refresh schedules. No new Victron polling, charger commands, charge-limit changes or physical solar reconnection are required.

## Existing implementation references

- `aurora-charger-indicator/power_grid.py`: qualified load and measured DC-rate helpers.
- `aurora-charger-indicator/power_operating_scenarios.py`: paired SOC integration and GridContinuesSOCP50.
- `aurora-charger-indicator/grouped_timeseries.py`: 24/96-hour traces, metadata merge and serving.
- `aurora-charger-indicator/mobile_catalog.py`: shared app payload.
- `aurora-charger-indicator/docs/GRID_POWER.md`: current measured-grid contract.
- Android `Models.kt` and `PowerScreen.kt`; iOS `AppConfiguration.swift` and `InteractiveDataView.swift`: assumption and freshness presentation.
- [Infrastructure PR97](https://github.com/GAMB2LE/aurora-cloud-infra/pull/97): existing focused forecast-release preparation and prior input-path gate.
