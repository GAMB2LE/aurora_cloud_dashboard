import SwiftUI

struct InteractiveDataView: View {
    let configuration: AppConfiguration

    private let instruments = [
        "Aurora Power Supply",
        "Ceilometer",
        "Cloud Radar",
        "Scanning Microwave Radiometer",
        "Meteorology",
        "Radiation",
        "WXcam"
    ]

    var body: some View {
        NavigationStack {
            List {
                Section(
                    header: Text("Dashboard instruments"),
                    footer: Text("These rows mirror the current Panel dashboard navigation. Native plots and product browsing will be wired to a future data API.")
                ) {
                    ForEach(instruments, id: \.self) { instrument in
                        Label(instrument, systemImage: iconName(for: instrument))
                    }
                }

                Section(header: Text("Starter state")) {
                    PlaceholderPanel(
                        title: "Native data browser placeholder",
                        subtitle: "Range controls, variable selection, and Plotly replacements are intentionally not connected yet.",
                        systemImage: "chart.xyaxis.line"
                    )
                    .listRowInsets(EdgeInsets(top: 8, leading: 16, bottom: 8, trailing: 16))
                    .listRowBackground(Color.clear)
                }
            }
            .navigationTitle("Interactive")
        }
    }

    private func iconName(for instrument: String) -> String {
        switch instrument {
        case "Aurora Power Supply":
            return "battery.100percent"
        case "Ceilometer":
            return "laser.burst"
        case "Cloud Radar":
            return "dot.radiowaves.left.and.right"
        case "Scanning Microwave Radiometer":
            return "antenna.radiowaves.left.and.right"
        case "Meteorology":
            return "cloud.sun"
        case "Radiation":
            return "sun.max"
        case "WXcam":
            return "video"
        default:
            return "waveform.path.ecg"
        }
    }
}
