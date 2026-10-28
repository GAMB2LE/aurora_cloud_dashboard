# Review of “Integrating autonomous operation”

Review date: 28 October 2026. Host: Rocinante-5.local.
Document: https://docs.google.com/document/d/1RNK1-_voDKZ-L7ZU_Q3K_7FnMNIwUIEBe4U2Jch98Pc/edit

The initial evidence-led revision was 857 words. At the author's request, a subsequent style edit shortened it to **494 words**, using plainer language and a direct first-person plural voice. The factual distinctions are unchanged; implementation detail remains in this record. Counts exclude title, citation markers and drafting sources. The original Google Doc text is retained alongside it. This record separates source implementation, deployment evidence, experimental results and proposals; it is supporting material, not part of the manuscript section.

## Changes to the draft

- Retained the connection between energy availability and scientific priorities, and made the operator's ability to conserve charge for a chosen later observing window explicit.
- Explained finite operating configurations, component learning and automatic startup/heater/blower phases without implying the system predicts future operator commands.
- Identified ECMWF and empirical calibrated solar conversion as the deployed baseline. Physical three-array solar and interfaces to other weather sources are experimental.
- Clarified the reported battery SOC anchor and conditional ensemble spread. No independently validated accuracy or reliability claim is made.
- Added the distinction between implemented grid-supply accounting, incomplete operational integration, and the later proposed disconnected-solar/connected-solar comparison.
- Retained advisory instrument-priority scheduling and future scientific-event targeting. Acknowledged implemented experimental control machinery and existing separate instrument commands; automatic execution of forecast schedules is not established.
- Replaced the generic learning-improves-performance implication with an explicit independent-validation limit.

## Capability and evidence matrix

| Capability | Classification at review | Evidence and limit |
| --- | --- | --- |
| ECMWF deterministic and 50-member SSRD, empirical irradiance-to-electricity conversion and lead-dependent correction | Implemented and deployed baseline | Live forecast metadata says power-v10, calibrated_ssrd_scalar, ssrd_scalar_lead_mos_v1. |
| Finite operating-state classification, additive component learning, recurring startup/heater/fan phases | Implemented; v10 phase model deployed | Live metadata says finite_controlled_state_phases_v10. Explicit schedules change controllable states; automatic internal phases vary within states. |
| Battery integration with capacity, separate efficiencies, limits and SOC bounds | Implemented and deployed | Live API reports bounded bidirectional physics. Fitted parameters are effective model estimates, not independently established hardware ratings. |
| Ensemble uncertainty from weather, state/phase loads and battery parameters | Implemented and deployed, reliability not independently validated | Battery perturbations include heuristic bounded spreads; P10 is not a demonstrated 90% survival guarantee. |
| CL61-first four-day advisory schedule evaluated over ten days | Implemented; fresh development output; production planning unavailable at review | Development says advisory_only, actuation false and no_safe_schedule. Production retains stale planning input. |
| Operator-selected instrument/start/duration comparisons | Implemented advisory capability | Custom operating-plan calculation and UI exist. No automatic scientific-event detection/value ranking found. |
| Direct instrument commands and guarded CL61 prototype | Existing separate commands; experimental controller | Forecast diagnostic intent is observe_only and not executable; production CL61 status unavailable. |
| New simplified shared-request controller | Proposed | ac_instrument_control README at c94c919 explicitly says no controller built or deployed. |
| Physical three-array PV and additional residual-load v12 lanes | Implemented experimental code; isolated evaluation machinery deployed | Unpublished/not accepted. Surveyed array parameters and independent validation remain outstanding. |
| Historical SOC research package | Substantial local experimental implementation and development diagnostics | Entire aurora-soc-research directory is untracked in the parent workspace, not a separate Git repository. No completed acceptance campaign. |
| Charger-aware load balance and conditional old grid comparison | Implemented; isolated validation performed; operational writers not activated | Grid telemetry/display deployment is separate from forecast integration. |
| Grid Charging with zero solar versus No Grid with hypothetical connected solar | Proposed in 27 October plan | Distinct from older add-grid-to-solar code; requires verified pre-removal solar calibration. |

## Fresh deployment observations

Read-only public GETs and SSH metadata reads were made at **21:47–21:49 UTC on 28 October**. These are timestamped observations, not a continuing health claim.

- Both presentation manifests report dashboard revision `7399c08a60e37fa2b7b48b21369b1bfb6ac54c1d`. Forecast producer provenance differs: production `36770c2d6e38f6f9c37f83be6e8fb772902c71df`; development `e053de8133564234763fd5c9eef95c7ee25ea23d`. Both forecast products remain power-v10. A newer web display does not prove newer forecast physics.
- Both use the 28 October 12 UTC ECMWF cycle. Production deterministic/ensemble products were generated at 21:43:50/21:40:33 UTC. Development deterministic/planning/ensemble products were generated at 21:34:51/21:10:55/21:35:25 UTC.
- Development's active coherent generation was `20261028T213553Z-b7737bca`, updated 21:37:43 from observations through 21:27:19. It reports `controlAuthority=advisory_only`, `cl61ActuationEnabled=false`, `no_safe_schedule`, `fixed_baseline_below_reserve`. Its public panel says “No Feasible CL61-first Schedule”.
- Production's current SOC forecast is present, but operating scenarios were unavailable at 21:46:32. The planning solar horizon ends **16 October**, whereas **1 November** is required for the next 96 hours. The retained planning artifact was generated **6 October, 20:54:19 UTC**, revision `6114f54566472011cbd51ab403c9e6d8e241e8c3`. Production exposes SOC panels but no operating-plan panels. This defect was documented, not repaired during manuscript work.
- Production's CL61 automation endpoint returned `available=false` and `capability=false`. Development's fresh diagnostic returned `observe_only`, `control_eligible=false`, `capability=false`, and signature algorithm `none`. An action of “hold” or desired=false here is diagnostic, not an instruction to switch equipment.
- The v12 candidate's last complete status is dated 8 September and says `retain_unpublished_candidate` / `not_accepted`. Its 28 October 21:14:20 attempt deferred because model evaluation was active. GFS/GEFS/ICON/IFS manifests were not enrolled in that candidate.

Endpoints: [production manifest](https://data.gamb2le.co.uk/mobile/v1/manifest), [production forecast](https://data.gamb2le.co.uk/mobile/v1/power?window=96h&group=forecast), [development manifest](https://data-ocean.gamb2le.co.uk/mobile/v1/manifest), [development forecast](https://data-ocean.gamb2le.co.uk/mobile/v1/power?window=96h&group=forecast), and the corresponding `/mobile/v1/power/cl61-automation` endpoints.

Cloud evidence came from product `.zattrs`, active-generation metadata, candidate status/acceptance records and read-only service status. Production uses `/data/aurora/products/power`; development uses `/data/aurora/dev-products/power`. The active development generation takes precedence over stale mutable files in its parent directory.

## Local repositories, branches and existing changes

Paths below are relative to `/Users/rrniii/Documents/GAMB2LE`. Remote heads were checked with `git ls-remote`; the newer realtime-scripts main was fetched for comparison only. No existing checkout was switched, reset, staged, committed or cleaned.

| Repository / working copy | Inspected local state | Relevant difference |
| --- | --- | --- |
| aurora_cloud_dashboard | codex/power-forecast-integrity-20260912 at 0b901de; 18 modified tracked and 3 untracked forecast/grid files | Older primary checkout. Its dirty grid work largely overlaps newer main; dirty does not imply a new unpushed capability. |
| aurora-charger-indicator | clean main at 7399c08; live origin/main verified | Main already contains grid_v11 code, which differs from live v10 producer metadata. |
| .worktrees/aurora_cloud_dashboard-v12 | clean codex/power-v12-candidate at 178b203 | Candidate implementation already represented in main; no branch-only commits beyond main. |
| .worktrees/cl61-dashboard | codex/cl61-automation-machinery at 4839054; 25 staged files | Staged identity/evaluation/shadow work is largely older than main; no evidence of enabled control. |
| aurora-cloud-infra | local main 674dcbb; live main dddb6d7 | Existing reverse-tunnel changes preserved. Later grid records are in aurora-charger-release-infra at dddb6d7. |
| realtime-scripts | local main 84dbb52 with extensive staged and unstaged changes; live main 2054fb7 | Existing CL61 candidate modules match newer main. The latest remote commits change logger/ASFS/BLE/log records, not forecast control. |
| aurora-edge-infra | local 653a456 with extensive existing staged/untracked work | CL61 deployment role exists but defaults disable install/control/automatic stop; not proof of installed control. |
| ass-services | clean main 028acc6 | Separate instrument start/stop commands; retired SOC-check timer is not active forecast automation evidence. |
| ac_instrument_control | clean main c94c919, verified live remote | Latest design only; two local design branches are ancestors. |
| aurora-model-evaluation | codex/gfs-bounded-site-decoding-20260912 at 8c71f3d; two untracked helpers | Separate weather/model evaluation workflow. Local-only 85528d7/8e32966 add storage/headroom safeguards, not power-forecast forcing. |
| aurora-soc-research and SOC_FORECAST*.md | Untracked files under the parent GAMB2LE repository | Actual local experimental work invisible to a review of GitHub mains. |

All local branch tips in the six central forecast/control repositories were compared with known live remote-head ancestry. Relevant tips absent from live remote ancestry include dashboard `225ec5d` (coherent-bundle serving), `9069947` (grid checkpoint), `1baa8c8` (preservation checkpoint), `c70edec` (storage probe/retired timer), realtime `e278c3d` (grid checkpoint) and `ff682cc` (preserved edits), plus the model-evaluation safeguards above. Their subjects/diffs do not add autonomous scientific-event targeting. Several other historical branch tips concern archive, timestamps or presentation; absence of a branch or commit from remote ancestry does not prove unique functionality after squash/cherry-pick merging.

A malformed dashboard ref named `refs/remotes/origin/main 2` prevented one broad all-ref log command. Specific valid-ref and live-remote-hash comparisons succeeded; the ref was left untouched. No attempt was made to publish existing unpushed operational work.

## Source-level evidence

The following file/line references identify inspected source versions. Dashboard paths refer to main at 7399c08 unless an older working-copy path is explicitly named.

- `generate_power_soc_ensemble.py:325–358,701–783`: ECMWF member retrieval and joint load/battery spread. Capacity perturbations are 6% when calibrated or 10% otherwise; efficiencies vary by 3%. These are model assumptions.
- `generate_power_soc_forecast.py:3267–3341,3979–4028`: empirical solar calibration/lead correction, interval coverage and eligibility.
- `power_battery_model.py:13–28,117–219`: energy integration and bounded calibration. Default or fitted usable capacity is not an independently measured bank rating.
- `docs/analysis/power-load-forecast-model.md:42–126` and `power_load_dynamics.py:284–416,491–562`: controllable states, recurring internal phases, sparse-data fallbacks and UAS evidence.
- `power_operating_scenarios.py:1190–1285`: state classification and robust component updates.
- Inspected primary working-copy `power_operating_scenarios.py:2936–2967,3449–3459,3690–3698,4356–4400`: reserve CL61 first; add Radar/HATPRO from residual reserve; 96/240-hour horizons; P10 >=40%; advisory-only authority. This is not a simple strict rank ordering of every instrument.
- Primary working-copy `app.py:8580–8597,8640–8651` and `power_operating_scenarios.py:4510–4545`: specified instrument/start/duration scenarios.
- `docs/runtime/cl61-automation.md:17–41,117–134`: diagnostic intent, scheduling rules and absent signed operational intents.
- `realtime-scripts/cl61_automation/controller.py:14–28,144–208,255`; `aurora-edge-infra/roles/cl61_automation/defaults/main.yml:2–8,44`: experimental executable controller, disabled defaults.
- `ac_instrument_control/README.md:3–11,53–93`: proposed replacement and required future observation/commissioning. Its `docs/EXISTING_SYSTEM.md:3–16` records 26 October checks showing no installed CL61 shadow/controller units; that historical receipt is not a present instrument-health claim.
- `docs/analysis/power-solar-physical-candidate.md:3–28,59–80,118–130` and `docs/analysis/power-v12-hybrid-candidate.md:10–33,50–72,90–121`: provisional geometry, isolated lanes, source enrollment and evaluation gates.
- `aurora-cloud-infra/docs/DEVELOPMENT_EXPERIMENTS.md:19–39,81–101`: advisory runtime ownership, candidate isolation and non-actuating shadow output.

## Local research and grid records that changed the interpretation

`aurora-soc-research/HISTORICAL_ABLATION_DIAGNOSTIC_20260923.md` is a development diagnostic, not held-out acceptance. All 53 prior generation events admitted zero training samples for the **additional experimental load correction**. Its load-only lane therefore equals the already state-aware baseline; the combined lane equals physical-solar-only. This does not mean the baseline lacked finite-state learning.

The replay has 14 cycle groups over 78 hours of scored anchors, zero exact +48-hour rows, and no finite available-PV truth. It records a short delivered-solar error improvement, but its +96-hour SOC endpoints perform worse than persistence. These are grounds for development, not an achieved accuracy target or calibrated uncertainty. `NATIVE_SOC_ENDPOINT_STATUS_20260923.md` has 34 bracketed targets but none eligible for final scoring. `MATCHED_SOURCE_INTERVALS_20260923.md` and the latest `IMPLEMENTATION_STATUS.md` document further source/interval preparation, not completed acceptance. The research scheduler/controller manage computational tasks; they do not schedule scientific instruments.

`output/grid-power-release-20261025/RELEASE.md:3–34` and `isolated-acceptance-20261026.json` record verified grid telemetry and an isolated model check, but blocked forecast-writer activation because the continuity snapshots lack required optional fields. `output/grid-power-forecast-plan-20261027.md:3–42,54–84` is explicitly planning-only: Grid Charging has measured DC supply and zero solar; No Grid has zero grid and hypothetical connected-array solar. It requires healthy pre-removal calibration and does not forecast grid availability or charger taper.

These local records were read directly; names such as “controller”, “accepted” in an isolated check, or a passing software suite were not treated as operational deployment or independent scientific validation.

## Preservation and delivery

The manuscript and this review were prepared on a new documentation-only branch, `codex/bams-autonomous-operation-20261028`, based on verified dashboard main 7399c08, in a separate worktree. Only manuscript/supporting documentation is included. Existing working-tree status and staged/unstaged diff fingerprints were verified unchanged in six relevant repositories. No instrument commands, service changes, deployments, model training, data repair or Tailscale changes were performed.

The companion [deployment evidence](deployment-evidence-20261028.json) retains selected live metadata and SHA-256 fingerprints of eight local research/grid records. It contains no credentials or raw scientific time series. Code review and document checks were used for this prose-only change; no operational model or instrument test was run.
