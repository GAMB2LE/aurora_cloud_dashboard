# Literature supporting “Integrating autonomous operation”

Reviewed 28 October 2026 for the GAMB2LE BAMS manuscript. Companion files: [revised section](integrating-autonomous-operation.md), [local implementation and deployment audit](integrating-autonomous-operation-review-20261028.md), [BibTeX bibliography](integrating-autonomous-operation-literature.bib), and [verified metadata and access notes](integrating-autonomous-operation-literature.json).

The literature supports the section's central argument: use expected energy supply, learned demand and observing priorities to decide what can run, including saving energy for valuable observing periods. The individual ideas have substantial precedent. The useful emphasis for AURORA is their integration in a polar cloud observatory, together with an honest account of what currently works. This search does not establish a “first” or “unique” claim.

The closest field precedent is [Anderson et al. (2005)](https://doi.org/10.1175/JTECH1779.1): Antarctic sodar operators chose meteorologically interesting periods within the available battery budget. [HYDRANT](https://doi.org/10.5194/tc-9-285-2015) supplies a closer cloud-instrument comparison and documents winter power gaps. [CORAL](https://doi.org/10.5194/amt-14-1715-2021) demonstrates forecast-informed automatic lidar operation, but uses external electrical power. These are different kinds of autonomy and should not be treated as interchangeable.

For the specific idea of balancing an event against a longer mission, [Giglio et al. (2025)](https://doi.org/10.1029/2025GL117069) is particularly useful: forecast-informed ocean-float programming protected battery life and the routine Argo mission. It is an ocean analogue with human planning, not evidence that AURORA already identifies or ranks events automatically.

## What belongs in the short section

Six papers have been added to the draft: Anderson, Gorodetskaya, Sperati, Gneiting, Giglio and Diggle. The section remains 527 words of prose. The other references below support the rationale, methods discussion and future evaluation without adding technical detail to the manuscript.

| Question | Most useful literature | Manuscript implication |
|---|---|---|
| Has energy-limited atmospheric observing been demonstrated? | Anderson 2005; Gorodetskaya 2015; Clauer 2014 | Acknowledge field precedent; distinguish purposeful selection from power-related data loss. |
| Can weather forecasts inform energy budgets? | Sharma 2014; Sperati 2016 | Explain the weather → generation → reserve connection. Do not transfer another site's forecast skill. |
| How should stored energy and demand be treated? | Kansal 2007; Taneja 2008 | Include generation, demand, usable storage and losses. Neither supplies AURORA's battery capacity or reserve threshold. |
| Is learned operating behaviour a new general concept? | Hart 1992; Kolter and Jaakkola 2012 | State-based loads have precedents. Describe AURORA's actual measured-state and phase logic without claiming it uses those papers' algorithms. |
| What does a forecast range establish? | Gneiting 2023; Lauret 2019 | Check the range against observations and useful benchmarks. Do not equate a plotted spread with validated reliability. |
| Why conserve power for future science? | Padhy 2010; Zhang 2016; Giglio 2025 | There is a strong rationale, but gains in AURORA's scientific return have not yet been demonstrated. |
| What is “scientific value”? | Wang 2026; Krause 2008; Yoo 2020 | Improving a forecast, resolving an event and maintaining a representative record are different objectives. Operator-assigned instrument priority is the current implementation. |
| What separates advice from control? | Kaifler and Kaifler 2021; Chien 2005 | Planning, execution, local operating checks and confirmed useful observations require distinct evidence. |
| How should selective observations be interpreted? | Diggle 2010; Thompson 1990; Maury 2022 | Record when and why sampling changes. Selection needs accounting for; bias is not automatically unavoidable or uncorrectable. |

## Implications for the project evidence

**Implemented and deployed claims still come from the local audit.** The search does not change the finding that AURORA provides advisory instrument-priority schedules, while automatic forecast-driven switching and automatic scientific-event ranking remain development objectives. A comparison paper cannot establish a local deployment.

**Solar generation and battery reserve need separate checks.** The reviewed energy-budget studies justify accounting for supply, demand, storage and losses. They do not validate the station's effective battery capacity, initial charge, temperature response, charging limits or losses. A better irradiance forecast does not, by itself, validate a battery trajectory. The newer physical solar and load models should remain experimental until evaluated independently of their fitting data.

**Uncertainty must be tested at the decision horizon.** The solar literature evaluates both forecast accuracy and whether reported ranges agree with observations. For AURORA, that means checking generation, load and charge separately, then checking the complete proposed schedule. A lower-percentile battery curve should not be presented as an independently verified probability of staying above the reserve throughout the whole plan. This is our interpretation for the station's decision problem, not a result demonstrated for AURORA by the cited papers.

**Useful observations need an explicit objective.** The present priority order and minimum run lengths support sustained measurements. Choosing an anticipated cloud or precipitation event remains an operator decision. Future event selection should be evaluated against declared science questions, a feasible baseline schedule, missed events and loss of background coverage. More samples or more instrument-hours alone do not establish greater scientific value.

**Changes made to the prose are modest.** Added field context, a solar-forecasting citation, support for uncertainty checks, and the ocean-float analogy. Replaced the categorical “sampling bias introduced” wording with “account for how choosing when to observe affects the record.” This preserves the concern without asserting that every selective design produces the same bias. Technical algorithms, performance percentages and unverified novelty claims stay out of the section.

## Annotated primary literature

The 34 records below include 27 papers for which full text or substantive full-text sections were inspected, six assessed through primary abstracts, and one historical reference checked at bibliography/metadata level. Each entry states its access depth and transfer limits. Field data used in a simulation are not counted as a deployed controller. Complete author lists are in the BibTeX and JSON files.

### Field observatories and operational autonomy

**1. Philip S. Anderson, Russell S. Ladkin, Ian A. Renfrew (2005). [An Autonomous Doppler Sodar Wind Profiling System](https://doi.org/10.1175/JTECH1779.1).** Journal of Atmospheric and Oceanic Technology **22**(9), 1309–1325.

A two-year Antarctic deployment used solar/wind generation and batteries for an approximately 70 W active sodar. Available energy could not sustain continuous profiling. Halley personnel selected scientifically interesting meteorological situations and battery-supported observing intervals, usually 12–48 hours. This is a direct precedent for conserving energy for valuable atmospheric observations. Selection was operator-mediated using current conditions and battery voltage; it was not a demonstrated automatic forecast-and-battery optimizer.

Evidence: Full paper, especially pp. 1312–1315, power architecture and sounding criteria; field profiles and failures are reported. Metadata checked against the UEA author repository and journal information. [Primary source](https://ueaeprints.uea.ac.uk/id/eprint/26881/1/anderson_etal_doppler_sodar_system_JOATech_2005.pdf). Citation key: `Anderson2005`.

**2. Irina V. Gorodetskaya and coauthors (2015). [Cloud and precipitation properties from ground-based remote-sensing instruments in East Antarctica](https://doi.org/10.5194/tc-9-285-2015).** The Cryosphere **9**, 285–304.

HYDRANT combined ceilometer, precipitation radar, infrared measurements and surface meteorology at Princess Elisabeth Antarctica, a predominantly wind/solar-powered station remotely controlled outside summer. The study reports 14 months of cloud and 26 months of snowfall observations during 2010–2013; power failures interrupted winter cloud measurements. The ceilometer's typical total consumption was 310 W, versus 25 W for the unheated MRR. This is a close same-domain renewable-power comparator, but station infrastructure and remote control are distinct from a stand-alone autonomous scheduler.

Evidence: Full paper: station and operating context pp. 287–288; instrument power Tables 2 and 4; actual scientific results. No forecast-based allocation of energy among observing priorities is demonstrated in this paper. [Primary source](https://tc.copernicus.org/articles/9/285/2015/tc-9-285-2015.pdf). Citation key: `Gorodetskaya2015`.

**3. Bernd Kaifler, Natalie Kaifler (2021). [A Compact Rayleigh Autonomous Lidar (CORAL) for the middle atmosphere](https://doi.org/10.5194/amt-14-1715-2021).** Atmospheric Measurement Techniques **14**(2), 1715–1732.

CORAL automatically starts, stops and recovers using local weather, all-sky images, ECMWF forecasts and lidar returns. It obtained temperature profiles on 243 of 365 nights in southern Argentina during October 2019–October 2020. This establishes forecast-informed autonomous operation of a substantial atmospheric remote-sensing instrument. Its observing suitability logic is transferable, but electrical power is externally supplied; the result does not demonstrate renewable-energy forecasting, battery allocation, or scientific prioritization across multiple instruments.

Evidence: Full paper, abstract and software/observing examples; introduction explicitly retains electrical-power infrastructure. Do not misread the quoted 2.4 kW cooling capacity as electrical consumption. [Primary source](https://amt.copernicus.org/articles/14/1715/2021/). Citation key: `Kaifler2021`.

**4. C. R. Clauer and coauthors (2014). [An autonomous adaptive low-power instrument platform (AAL-PIP) for remote high-latitude geospace data collection](https://doi.org/10.5194/gi-3-211-2014).** Geoscientific Instrumentation, Methods and Data Systems **3**, 211–227.

Solar/battery Antarctic platforms retain low-power magnetic observations while scheduling the more demanding GPS receiver and reducing winter telemetry. The paper reports field engineering records and actual scheduled GPS observations; low battery charge leads to hibernation. It also describes instrument triggering from monitored magnetic conditions. That trigger capability is not separately validated by a documented autonomous-trigger event. The five-year unattended lifetime is a design target. Scientific-selection principles transfer, but these are low-power geospace instruments, not high-power cloud profilers.

Evidence: Full paper: pp. 212–213, operating policies; Fig. 4, January 2011–February 2014 engineering record; Figs. 15–16 and p. 225, field observations and winter limits. [Primary source](https://gi.copernicus.org/articles/3/211/2014/gi-3-211-2014.pdf). Citation key: `Clauer2014`.

**5. Matthew D. Shupe and coauthors (2013). [High and Dry: New Observations of Tropospheric and Cloud Properties above the Greenland Ice Sheet](https://doi.org/10.1175/BAMS-D-11-00249.1).** Bulletin of the American Meteorological Society **94**(2), 169–186.

ICECAPS at Summit combines active and passive remote sensors with radiosondes to characterize clouds and atmospheric structure over Greenland. Its first 20 months revealed year-round liquid cloud occurrence and seasonal moisture/cloud/snowfall variation. It is a strong science and instrument-integration precedent for linking atmospheric observations to ice-sheet mass and energy budgets. This article is not evidence for an off-grid, renewable-powered or autonomously science-prioritized observatory; Summit's supporting station infrastructure matters to the comparison.

Evidence: Publisher abstract and metadata verified through search indexing; direct publisher opening returned HTTP 403. Author NOAA bibliography confirms complete author order, year, pages and DOI. Detailed engineering/full-text assertions were not extracted from this source. [Primary source](https://psl.noaa.gov/people/matthew.shupe/icecaps/icecaps.html). Citation key: `Shupe2013`.

**6. Alfonso Ferrone, Alexis Berne (2023). [Radar and ground-level measurements of clouds and precipitation collected during the POPE 2020 campaign at Princess Elisabeth Antarctica](https://doi.org/10.5194/essd-15-1115-2023).** Earth System Science Data **15**, 1115–1132.

The 2019–2020 campaign deployed a 94 GHz cloud radar near the station and three lower-power precipitation radars at remote sites without connection to its grid. Instrument placement reflected power and access requirements. A scanning X-band radar was usually started manually before precipitation events. This is a useful same-domain comparison of instrument-dependent operating costs and purposeful event sampling. The article does not establish the remote radars' generation technology or demonstrate autonomous energy-aware event selection.

Evidence: Full text, Sections 2.1–2.3, deployment periods and power/access constraints; public measurement dataset. Do not describe every radar as remotely powered or renewable merely because the campaign used Princess Elisabeth. [Primary source](https://essd.copernicus.org/articles/15/1115/2023/). Citation key: `Ferrone2023`.

**7. Michael C. B. Ashley and coauthors (2012). [PLATO-R: a new concept for Antarctic science](https://doi.org/10.1117/12.925514).** Proceedings of SPIE **8444**, 84441R.

PLATO-R was deployed to Ridge A in January 2012 to provide heat, communications and up to 1 kW for astronomical experiments. Its engineering lineage combines fuel generators with summer solar generation, preserving fuel for winter. The paper reports earlier PLATO operation for 1235 uninterrupted days with annual servicing. This is a substantial autonomous polar-power comparator, but neither renewable-only nor maintenance-free; those long-duration results refer to earlier PLATO, not the newly deployed PLATO-R.

Evidence: Full peer-reviewed conference paper, introduction and system description. Useful engineering context across disciplines; it does not establish meteorological event selection or energy-forecast optimization. [Primary source](https://researchers.mq.edu.au/en/publications/plato-r-a-new-concept-for-antarctic-science/). Citation key: `Ashley2012`.

### Energy budgets and observing priorities

**8. Aman Kansal and coauthors (2007). [Power management in energy harvesting sensor networks](https://doi.org/10.1145/1274858.1274870).** ACM Transactions on Embedded Computing Systems **6**(4), 32.

Full text; mathematical models and evaluation with energy data measured by the Heliomote prototype. Source/consumption models, storage losses and adaptive duty cycles establish the energy-neutral principle. The algorithm holds energy to accommodate learning errors and lower-than-expected harvesting. This supports AURORA's general budgeting/reserve rationale. Prediction uses past harvesting profiles, not numerical weather forecasts; neither indefinite polar operation nor a particular AURORA reserve is demonstrated.

Evidence: Full text; published article. Location: sections 2–5, especially printed pp. 14–15 and conclusion p. 35. [Primary source](https://www.microsoft.com/en-us/research/wp-content/uploads/2007/12/kansal_TECS07.pdf). Citation key: `Kansal2007`.

**9. Jay Taneja, Jaein Jeong, David Culler (2008). [Design, Modeling, and Capacity Planning for Micro-solar Power Sensor Networks](https://doi.org/10.1109/IPSN.2008.67).** 2008 International Conference on Information Processing in Sensor Networks (ipsn 2008), 407-418.

Full text; HydroWatch forest watershed deployments and an urban comparison, with measured energy flows. Models connect panels, regulation, storage and application demand; the field results expose the importance of shading and seasonal supply. This directly supports site-specific generation/storage/load accounting. It is low-power micro-solar sensing and capacity planning, not meteorological-forecast scheduling or demonstrated prioritization of scientific events. Its numerical power/storage budgets cannot be transferred to AURORA.

Evidence: Full text; institutional copy. Location: deployment description and empirical analysis, PDF pp. 2–3 and 8–11. [Primary source](https://www.cs.virginia.edu/~bjc8c/class/cs6501-f17/taneja08hydrowatch.pdf). Citation key: `Taneja2008`.

**10. Clemens Moser and coauthors (2007). [Real-time scheduling for energy harvesting sensor nodes](https://doi.org/10.1007/s11241-007-9027-0).** Real-Time Systems **37**(3), 233-260.

Publisher abstract only for this exact journal article. Formal analysis and simulations combine task deadlines, harvesting behavior and finite storage; conventional earliest-deadline scheduling need not suffice. It supports the rationale for considering future energy and timing jointly when deciding whether to run tasks. Its optimality applies to its mathematical assumptions. This is not environmental field verification, a science-value objective, or validation of AURORA's scheduling policy.

Evidence: Publisher abstract only. Location: publisher abstract. Do not confuse the earlier conference paper with this journal article. [Primary source](https://link.springer.com/article/10.1007/s11241-007-9027-0). Citation key: `Moser2007`.

**11. Navin Sharma and coauthors (2014). [Leveraging weather forecasts in renewable energy systems](https://doi.org/10.1016/j.suscom.2014.07.005).** Sustainable Computing: Informatics and Systems **4**(3), 160-171.

Full text; measured weather/energy inputs and trace-driven case studies. Meteorological forecasts inform hour-to-day solar/wind energy estimates. The ViSE instrument testbed includes radar, but scheduling comparisons replay traces with scaled generation and assumed demand/storage. This strongly supports using forecasts to admit sustained observing tasks. It does not prove field-deployed forecast control throughout that evaluation, nor transfer prediction skill to AURORA.

Evidence: Full text; published article on author institutional website. Location: forecasting/model sections; ViSE evaluation on PDF pp. 8–9, including generation scaling and workload assumptions. [Primary source](https://lass.cs.umass.edu/papers/pdf/suscom14-weather.pdf). Citation key: `Sharma2014`.

**12. Paritosh Padhy and coauthors (2010). [A utility-based adaptive sensing and multihop communication protocol for wireless sensor networks](https://doi.org/10.1145/1754414.1754423).** ACM Transactions on Sensor Networks **6**(3), 27.

Full text; adaptive sensing/routing evaluated in simulation using data and energy costs from deployed GlacsWeb glacier probes. Information gain and energy opportunity cost guide acquisition/communication. This supports explaining scientific usefulness as an allocation objective. The evaluated protocol is not a field demonstration of autonomous event targeting; it concerns battery-powered sensing, not forecast renewable supply. The authors acknowledge that reduced sampling can miss short events.

Evidence: Full text; published article in institutional repository. Location: sensing section; section 6.1, printed pp. 27:24–27:26, explicitly identifies simulations and field-derived data. [Primary source](https://eprints.soton.ac.uk/267848/2/a27-padhy.pdf). Citation key: `Padhy2010`.

**13. Jianhui Zhang, Zhi Li, Shaojie Tang (2016). [Value of Information Aware Opportunistic Duty Cycling in Solar Harvesting Sensor Networks](https://doi.org/10.1109/TII.2015.2508745).** IEEE Transactions on Industrial Informatics **12**(1), 348-360.

Full author preprint; final journal bibliographic metadata independently verified. Theory and simulations, including SolarMote traces, allocate energy among harvesting/storage, sensing and communications according to data value. It supports the general choice between spending energy now and preserving it for useful future work. The data-value proxy does not equal a meteorological science objective, and no AURORA or field-control validation follows.

Evidence: Full author preprint; journal metadata independently verified. Location: algorithm and experimental sections, preprint pp. 10–12. Use 2016 journal year and all three authors; arXiv landing metadata differs. [Primary source](https://arxiv.org/pdf/1511.08331). Citation key: `Zhang2016`.

**14. Bruno Srbinovski and coauthors (2016). [An Energy Aware Adaptive Sampling Algorithm for Energy Harvesting WSN with Energy Hungry Sensors](https://doi.org/10.3390/s16040448).** Sensors **16**(4), 448.

Full publisher text retrieved through indexed primary-source results; direct endpoints were intermittently rate-limited/challenged. Measured wind-sensor and beehive gas-sensor data parameterize MATLAB evaluation. Sampling changes with battery level and signal dynamics, including acquisition/warm-up costs. It supports accounting for operating modes and the information lost when conserving energy. The adaptive algorithm's reported performance is simulated using field data, not a deployed AURORA analogue.

Evidence: Full publisher text via indexed primary result; direct access intermittently rate limited. Location: experimental evaluation and conclusion; aliasing is explicitly acknowledged. [Primary source](https://www.mdpi.com/1424-8220/16/4/448). Citation key: `Srbinovski2016`.

**15. Jianwei Hao and coauthors (2023). [DynaES: Dynamic Energy Scheduling for Energy Harvesting Environmental Sensors](https://doi.org/10.1109/IPCCC59175.2023.10253869).** 2023 IEEE International Performance, Computing, and Communications Conference (IPCCC), 365-374.

Full text; a trace-based simulator uses NSRDB/PVDAQ records and measured computing/sensor power costs. Predicted generation, sensor priorities and adjustable sensing intervals jointly determine energy use. This is a recent precedent for connecting budgets with domain-defined priorities. Predictions come from local histories, not online weather forecasts. Evaluation is simulation, not autonomous environmental deployment, and no numerical improvement or reliability guarantee transfers to AURORA.

Evidence: Full text; author PDF. Location: introduction pp. 1–2 and evaluation pp. 6–8. [Primary source](https://inkeekim.github.io/papers/2023-ipccc-dynaes.pdf). Citation key: `Hao2023`.

**16. Antonio Caruso and coauthors (2024). [A dynamic programming schedule trading off quality and stability in task allocation for energy-neutral Internet of Things devices harvesting solar energy](https://doi.org/10.52953/GQZA5788).** ITU Journal on Future and Evolving Technologies **5**(1), 13-25.

Full text; dynamic programming balances task quality and schedule stability under energy/battery constraints. The scheduler runs on an Arduino DUE, with simulated solar supply and prescribed tasks. It supports discussing continuity as a scheduling objective alongside utility. Hardware execution is demonstrated, but environmental field deployment and AURORA-specific minimum observing durations are not. Its stability metric is an analogy to scientific continuity, not a direct measurement of scientific value.

Evidence: Full text; publisher PDF. Location: evaluation on PDF pp. 7–9. [Primary source](https://www.itu.int/dms_pub/itu-s/opb/jnl/S-JNL-VOL5.ISSUE1-2024-A02-PDF-E.pdf). Citation key: `Caruso2024`.

### Weather, solar generation and uncertainty

**17. Simone Sperati, Stefano Alessandrini, Luca Delle Monache (2016). [An application of the ECMWF Ensemble Prediction System for short-term solar power forecasting](https://doi.org/10.1016/j.solener.2016.04.016).** Solar Energy **133**, 437-450.

Uses ECMWF ensembles with statistical correction to forecast solar power. It is a direct precedent for turning weather ensembles into generation estimates and correcting them against local measurements. The reported skill belongs to those plants and horizons; it does not validate a polar array or AURORA’s battery forecast.

Evidence: Publisher full-text sections and UCAR institutional record. [Primary source](https://www.sciencedirect.com/science/article/pii/S0038092X1630041X). Citation key: `Sperati2016`.

**18. Tilmann Gneiting, Sebastian Lerch, Benedikt Schulz (2023). [Probabilistic solar forecasting: Benchmarks, post-processing, verification](https://doi.org/10.1016/j.solener.2022.12.054).** Solar Energy **252**, 72-80.

Compares probabilistic forecasts, post-processing and reference methods. This is the clearest citation for checking whether forecast ranges capture observations and outperform a meaningful baseline. A displayed ensemble spread alone is insufficient. Its irradiance evaluations do not establish the probability that a battery stays above a reserve throughout a multi-day schedule.

Evidence: Publisher full-text sections and KIT institutional record. [Primary source](https://www.sciencedirect.com/science/article/pii/S0038092X22009343). Citation key: `Gneiting2023`.

**19. Guillermo Terrén-Serrano, Ranjit Deshmukh, Manel Martínez-Ramón (2026). [Probabilistic day-ahead forecasting of system-level renewable energy and electricity demand](https://doi.org/10.1038/s41467-026-69015-w).** Nature Communications **17**(1), 3307.

Studies joint probabilistic forecasts of electricity demand, solar and wind generation, including implications for reserve allocation. It supports considering uncertainty in both supply and demand. This is a grid-scale analogue with different data, decisions and constraints; it cannot supply a validated reserve policy or error distribution for AURORA.

Evidence: Publisher full-text results, discussion and methods; Crossref metadata. [Primary source](https://www.nature.com/articles/s41467-026-69015-w). Citation key: `TerrenSerrano2026`.

**20. Leticia de Oliveira Santos, Tarek AlSkaif, Paulo C.M. Carvalho (2026). [Hybrid photovoltaic power forecasting by combining physical and machine learning models](https://doi.org/10.1016/j.solener.2026.114951).** Solar Energy **317**, 114951.

Compares physical, statistical and hybrid models rather than assuming extra physical detail improves forecasts. Useful support for treating AURORA’s more detailed solar model as a candidate requiring independent evaluation. Results are specific to the two sites and tested models; they neither justify adopting the candidate nor demonstrate performance in Greenland.

Evidence: Publisher full-text sections; Crossref metadata. [Primary source](https://www.sciencedirect.com/science/article/pii/S0038092X26006407). Citation key: `deOliveiraSantos2026`.

**21. Philippe Lauret, Mathieu David, Pierre Pinson (2019). [Verification of solar irradiance probabilistic forecasts](https://doi.org/10.1016/j.solener.2019.10.041).** Solar Energy **194**, 254-271.

Explains complementary checks of forecast reliability, spread and overall performance, illustrated with two sites. Useful for designing the independent evaluation promised in the section. These are checks of irradiance forecasts, not evidence that AURORA’s current battery ranges are calibrated or that a schedule will remain feasible.

Evidence: Publisher full-text sections; author and DTU bibliographic records; PDF retrieval unsuccessful. [Primary source](https://www.sciencedirect.com/science/article/pii/S0038092X19310382). Citation key: `Lauret2019`.

**22. Philippe Lauret and coauthors (2024). [The added value of combining solar irradiance data and forecasts: A probabilistic benchmarking exercise](https://doi.org/10.1016/j.renene.2024.121574).** Renewable Energy **237**, 121574.

Benchmarks probabilistic methods and finds value in combining ground, satellite and numerical-weather information. Useful recent support for testing inputs and simple baselines before adding model complexity. Its short forecasting horizons differ from AURORA’s multi-day planning, and it does not support describing unused weather-model slots as an operational blend.

Evidence: Institutional abstract and published metadata. [Primary source](https://research-portal.uu.nl/en/publications/the-added-value-of-combining-solar-irradiance-data-and-forecasts-/). Citation key: `Lauret2024`.

### Learning operating demand

**23. G.W. Hart (1992). [Nonintrusive appliance load monitoring](https://doi.org/10.1109/5.192069).** Proceedings of the IEEE **80**(12), 1870-1891.

Provides historical context for representing equipment through operating states. Retained as a discovery and background reference, not used to attribute a particular algorithm to AURORA. AURORA has measured outlet information and its own state and phase logic; household load disaggregation is a related problem, not evidence of the same implementation.

Evidence: Author bibliography and Crossref metadata only; original full paper not inspected. [Primary source](https://www.georgehart.com/research/nalmrefs.html). Citation key: `Hart1992`.

**24. J. Zico Kolter, Tommi Jaakkola (2012). [Approximate Inference in Additive Factorial HMMs with Application to Energy Disaggregation](https://proceedings.mlr.press/v22/zico12.html).** Proceedings of the Fifteenth International Conference on Artificial Intelligence and Statistics **22**, 1472-1482.

Shows how multiple appliance states can explain an aggregate electricity signal. Useful conceptual support for learning recurring operating behaviour. It does not establish AURORA’s model structure, learned heater cycles or forecast accuracy. Keep the manuscript description tied to inspected project code and avoid calling it a hidden Markov model unless that is actually implemented.

Evidence: Official PMLR full-text PDF and bibliography. [Primary source](https://proceedings.mlr.press/v22/zico12.html). Citation key: `Kolter2012`.

### Adaptive observing and interpreting the record

**25. R. H. Langland and coauthors (1999). [The North Pacific Experiment (NORPEX-98): Targeted Observations for Improved North American Weather Forecasts](https://doi.org/10.1175/1520-0477(1999)080<1363:TNPENT>2.0.CO;2).** Bulletin of the American Meteorological Society **80**(7), 1363-1384.

NORPEX deployed targeted dropsondes and enhanced satellite winds to improve winter-storm forecasts. Initial impact experiments reported average benefits, but individual cases could degrade and the incremental contribution depended on other assimilated data. Human-run aircraft campaign; a precedent for scientific targeting rather than station autonomy, temporal power scheduling or demonstrated optimal observation value.

Evidence: publisher abstract only. [Primary source](https://doi.org/10.1175/1520-0477(1999)080<1363:tnpent>2.0.co;2). Citation key: `langland1999`.

**26. Jia Wang and coauthors (2026). [Impacts of atmospheric river reconnaissance dropsondes on ECMWF Integrated Forecasting System precipitation forecasts](https://doi.org/10.1002/qj.70019).** Quarterly Journal of the Royal Meteorological Society **152**(774), e70019.

Observing-system experiments for two AR Recon seasons found benefits from targeted dropsondes, varying by region, lead time and assimilation configuration. Mission tracks considered meteorological structure and forecast sensitivity; some forecasts degraded. Supports evaluating whether a targeting strategy benefits its declared objective. It does not measure autonomous station energy management or guarantee that more targeted observations always help.

Evidence: publisher full text. [Primary source](https://rmets.onlinelibrary.wiley.com/doi/10.1002/qj.70019). Citation key: `wang2026`.

**27. Christophe Reymann and coauthors (2018). [Adaptive sampling of cumulus clouds with UAVs](https://doi.org/10.1007/s10514-017-9625-1).** Autonomous Robots **42**(2), 491-512.

A proposed aircraft-fleet planner uses continually updated atmospheric maps to seek informative trajectories while exploiting airflow for energy-efficient flight. Evaluated in realistic cloud/aircraft simulations, not an operational autonomous cloud campaign. Useful conceptual link between science and energy objectives; do not transfer its performance to a fixed polar observatory.

Evidence: publisher abstract only. [Primary source](https://link.springer.com/article/10.1007/s10514-017-9625-1). Citation key: `reymann2018`.

**28. Donata Giglio and coauthors (2025). [Adaptive Sampling of the Upper Ocean by Autonomous Floats During Atmospheric River Precipitation](https://doi.org/10.1029/2025GL117069).** Geophysical Research Letters **52**(23), e2025GL117069.

Forecast-informed float programming intensified sampling near predicted rain, then returned floats to their primary Argo mission. Battery preservation was an explicit design priority. The experiment detected short-lived upper-ocean responses. Ocean analogue, with daily forecast analysis and surface communication/programming; not evidence of fully autonomous onboard event selection. Even intensified profiles missed shorter-lived structures. This is the strongest precedent for balancing episodic science against sustained monitoring, not a quantified AURORA saving.

Evidence: publisher full text, especially section2.2.1. [Primary source](https://agupubs.onlinelibrary.wiley.com/doi/10.1029/2025GL117069). Citation key: `giglio2025`.

**29. Steve Chien and coauthors (2005). [Using Autonomy Flight Software to Improve Science Return on Earth Observing One](https://doi.org/10.2514/1.12923).** Journal of Aerospace Computing, Information, and Communication **2**(4), 196-216.

EO-1 integrated onboard event analysis, resource-constrained replanning, command execution and independent fault protection as distinct components; science detection could request follow-up observations. Explicit spacecraft analogue. It illustrates that an observation recommendation is not equivalent to an operational closed loop. It does not establish AURORA control capability, safety or scientific return.

Evidence: published full text hosted by NASA/JPL. [Primary source](https://ai.jpl.nasa.gov/public/documents/papers/chien-JACIC2005-UsingAutonomy.pdf). Citation key: `chien2005`.

**30. Andreas Krause, Ajit Singh, Carlos Guestrin (2008). [Near-Optimal Sensor Placements in Gaussian Processes: Theory, Efficient Algorithms and Empirical Studies](https://jmlr.org/papers/v9/krause08a.html).** Journal of Machine Learning Research **9**(8), 235-284.

Sensor selection can target expected reduction in uncertainty about unobserved locations, rather than simply select the most uncertain location. The paper develops approximation algorithms and tests them on two datasets. Spatial design under specified probabilistic assumptions; neither a general definition of scientific value nor a ready-made event scheduler. Do not claim AURORA already uses this criterion.

Evidence: publisher full text. [Primary source](https://jmlr.org/papers/volume9/krause08a/krause08a.pdf). Citation key: `krause2008`.

**31. Peter J. Diggle, Raquel Menezes, Ting-li Su (2010). [Geostatistical Inference Under Preferential Sampling](https://doi.org/10.1111/j.1467-9876.2009.00701.x).** Journal of the Royal Statistical Society Series C: Applied Statistics **59**(2), 191-232.

When sample selection depends on the underlying phenomenon, ignoring that dependence can mislead inference. Simulations and environmental biomonitoring demonstrate the issue and a joint modelling approach. Primarily spatial statistics; application to time-selective AURORA operation is a design implication, not a measured bias estimate. It does not show that every adaptive design is biased or prescribe a background-observation frequency.

Evidence: published full text plus publisher metadata. [Primary source](https://www.stat.ubc.ca/~jim/Diggle2010.pdf). Citation key: `diggle2010`.

**32. Steven K. Thompson (1990). [Adaptive Cluster Sampling](https://doi.org/10.1080/01621459.1990.10474975).** Journal of the American Statistical Association **85**(412), 1050-1059.

Conventional estimators can be biased under observation-dependent expansion of a sample; the paper constructs design-unbiased estimators for its adaptive cluster designs. Supports wording about accounting for sampling choices, rather than saying selective operation necessarily introduces uncorrectable bias. Its estimators do not automatically apply to a battery-limited time series; the relevant inclusion/selection mechanism must first be defined.

Evidence: publisher abstract only. [Primary source](https://doi.org/10.1080/01621459.1990.10474975). Citation key: `thompson1990`.

**33. Eun-Hye Yoo, Andrew Zammit-Mangion, Michael G. Chipeta (2020). [Adaptive spatial sampling design for environmental field prediction using low-cost sensing technologies](https://doi.org/10.1016/j.atmosenv.2019.117091).** Atmospheric Environment **221**, 117091.

The design combines prediction uncertainty with a risk criterion; simulation experiments show that adaptive deployment benefits depend on sensor quantity and quality. Air-pollution spatial design and simulation, not an operational power-aware station. Useful for explaining why “scientific value” requires an explicit objective, while uncertainty alone is insufficient.

Evidence: author-deposited full manuscript plus publisher metadata. [Primary source](https://arxiv.org/pdf/2303.02050). Citation key: `yoo2020`.

**34. Nicolas Maury and coauthors (2022). [Use of large-eddy simulations to design an adaptive sampling strategy to assess cumulus cloud heterogeneities by remotely piloted aircraft](https://doi.org/10.5194/amt-15-335-2022).** Atmospheric Measurement Techniques **15**(2), 335-352.

Simulated adaptive aircraft trajectories and spatial mapping were assessed against known cloud structure. Reconstruction depended on the number of aircraft, mapping method and the cloud's evolution; rapidly changing phases remained harder to capture. Simulation evidence for designing and evaluating an observing strategy, not real-world autonomous cloud flight. Supports testing whether a chosen sampling pattern represents the process of interest, rather than equating collected data volume with scientific completeness.

Evidence: publisher full text. [Primary source](https://amt.copernicus.org/articles/15/335/2022/). Citation key: `maury2022`.

## Closely related project material

A [NOAA seminar by Michael Gallagher, 10 September 2024](https://psl.noaa.gov/seminars/Abstracts/2024/752.html), reports a renewable-powered autonomous ICECAPS-MELT prototype at Camp Raven with cloud, precipitation and surface-energy instruments. This is important project lineage, particularly because it concerns the same scientific community. It is first-party project evidence, not a peer-reviewed validation of forecast-based scientific scheduling.

The [NOAA author publication list](https://psl.noaa.gov/people/matthew.shupe/pubs.html), updated 9 October 2026 and checked during this search, places *How Greenland melts under cloud and clear skies* under **Under Review**, with status **submitted**. No published DOI was verified. It has therefore not been entered as a published 2026 article or merged with the seminar into an invented reference. An author-held manuscript can be considered separately using its actual status.

## Search coverage, verification and limitations

This was a focused deep search, completed 28 October 2026, rather than a systematic review or an exhaustive citation census. Four search strands were pursued in parallel, with exact-title and DOI checks and backward citation following:

- **Field systems:** autonomous Antarctic atmospheric observatories; solar/wind-powered sodar and cloud radar; renewable power limits; polar adaptive instruments; autonomous lidar and ECMWF forecasts; ICECAPS, ICECAPS-MELT, HYDRANT, POPE, CORAL, AAL-PIP and PLATO.
- **Energy and priorities:** energy-harvesting sensor networks; energy-neutral scheduling; battery reserves; weather-aware energy budgets; forecast-assisted task admission; value of information; utility-based glacier sensing; continuity and priority scheduling. Searches included recent 2024–2026 work.
- **Forecasts and learned demand:** ECMWF ensemble solar forecasting; probabilistic solar benchmarking and verification; physical/statistical photovoltaic models; joint generation and demand uncertainty; finite-state appliance loads and additive state models. Specific recent checks included Lauret 2024, Terrén-Serrano 2026 and de Oliveira Santos 2026.
- **Scientific targeting:** adaptive atmospheric and cloud sampling; targeted observations and forecast impact; atmospheric-river reconnaissance; autonomous event sampling; scientific planning versus execution; preferential sampling and adaptive sample design. Recent checks included Giglio 2025 and Wang 2026.

Substantive claims rely on papers, publisher records, and author or institutional copies. Crossref and primary records were used to verify titles, authors, years and identifiers; temporary rate limits were resolved with primary publication records. Conference papers are labelled in the bibliography. Final-volume years are retained for Wang (2026; first online 2025) and Reymann (2018; first online 2017). The publisher's later upload date is not used as Thompson's publication year. PMLR and JMLR papers have canonical URLs rather than invented DOIs. Giglio's sixth author is retained exactly as the publisher/DOI record renders the name; a submission bibliography should use the publisher's current author record.

Several publisher endpoints were inaccessible or intermittently failed. Abstract-only entries are marked individually; Hart's original article was not read. No claims were filled in from secondary summaries. Zotero was reachable, but five local library searches timed out, so this review does not claim to have reconciled the selection against the user's existing collection. No references were added to or changed in Zotero.

Screening excluded papers solely about very short image-based solar nowcasting, long-term climate-driven solar-resource changes, or generic grid optimisation where the connection to station decisions was weak. Recent simulation papers were not selected merely because they were newer. The 2025 solar/supercapacitor field study by [Gurung and coauthors](https://doi.org/10.1155/dsn/4253030), [AWARE](https://doi.org/10.1175/BAMS-D-18-0278.1), and [CRUX (2026)](https://amt.copernicus.org/articles/19/5683/2026/) were screened but are less direct comparators for the short section. In particular, a short deployment or a proposed renewable design does not establish year-round autonomous operation. The BAMS targeted-observing review by [Majumdar (2016)](https://doi.org/10.1175/BAMS-D-14-00259.1) is useful further orientation, but is not counted among the primary studies.

No instruments, operational services, deployments or scheduling policies were changed for this literature review.
