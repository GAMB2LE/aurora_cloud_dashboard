# Integrating autonomous operation

For AURORA, managing power is part of deciding what to observe and when. Running every instrument continuously can use the available energy before an important observing period. Waiting until the battery is nearly empty leaves few options. We therefore combine weather forecasts, measurements of power use and observing priorities to assess what the station can support over the coming days.

Measurements from Iceland show why this needs to reflect how the equipment actually behaves. Demand changes as instruments start, heaters cycle and aircraft recharge, while computers, communications and surface measurements continue to draw power. We treat each combination of operating instruments as a distinct state and use statistical learning to update its expected demand as new measurements arrive. Within each state, the model learns recurring changes, such as the ceilometer's heater and blower cycle. This separates the decision to switch an instrument on from the automatic changes in its power use once it is running. [1][2]

We use ECMWF forecasts of incoming solar radiation, adjusted against measured generation, to estimate the electricity the solar array could provide. Combining this with the latest reported battery charge, expected demand, usable capacity and charging losses gives a forecast of the energy reserve. We calculate a range of possible outcomes to account for uncertainty in the weather and system behaviour, and update the forecasts as new measurements arrive. We are still evaluating how well that range captures the errors seen in practice. [2][7]

The current planning tool gives priority to sustained ceilometer observations, adding cloud radar and microwave radiometer measurements when the remaining energy allows. It plans four days ahead and checks the reserve over ten days, while avoiding short runs and repeated starts. Operators can also compare particular instrument combinations, start times and observing durations. If the available energy cannot support a schedule within the chosen reserve limits, the system reports this. [5]

These comparisons allow us to consider conserving power now for more valuable observations later. In Greenland, for example, an anticipated period of clouds or precipitation could justify running lower-priority instruments less beforehand so that several instruments can observe the event together. For now, operators make this choice. Automatically identifying and ranking such periods is a further development objective. We also need sufficient background observations to interpret each event and understand the sampling bias introduced by choosing when to observe. [5][7]

The forecasting system currently provides advice. Prototype control software exists, but automatic instrument switching from forecast schedules is not yet operational. That connection needs local checks on battery condition, instrument status and operating limits, together with operator control. It also needs to confirm that a requested change has happened and that the instrument is collecting useful data. [5][6][7]

Archived forecasts can be checked against later observations to separate errors in generation, demand and battery charge. More detailed solar and load models are being evaluated, while recent work on grid charging still needs to be integrated into the operational forecast. We need independent checks of their performance before adopting them. [3][4][7]

## Sources for drafting

[1] [Measured instrument power use](https://docs.google.com/document/d/167FYmHPFyTaQTacYCw7SwVHv5CzlpKqM4_a5-Rg1KAE/edit)

[2] [Power forecasting and learned demand](https://github.com/GAMB2LE/aurora_cloud_dashboard/blob/7399c08a60e37fa2b7b48b21369b1bfb6ac54c1d/docs/analysis/power-load-forecast-model.md)

[3] [Experimental solar and load models](https://github.com/GAMB2LE/aurora_cloud_dashboard/blob/7399c08a60e37fa2b7b48b21369b1bfb6ac54c1d/docs/analysis/power-v12-hybrid-candidate.md)

[4] [Grid-supply integration](https://github.com/GAMB2LE/aurora_cloud_dashboard/blob/7399c08a60e37fa2b7b48b21369b1bfb6ac54c1d/docs/GRID_POWER.md)

[5] [Observing schedules and control](https://github.com/GAMB2LE/aurora_cloud_dashboard/blob/7399c08a60e37fa2b7b48b21369b1bfb6ac54c1d/docs/runtime/cl61-automation.md)

[6] [CL61 controller design](https://github.com/GAMB2LE/ac_instrument_control/blob/c94c91984c9eeeec00b6ca872f2aec7f50211946/README.md)

[7] [Local evidence and deployment review](integrating-autonomous-operation-review-20261028.md)

494 words, excluding title, citation markers and sources. Technical evidence is retained in the companion review.
