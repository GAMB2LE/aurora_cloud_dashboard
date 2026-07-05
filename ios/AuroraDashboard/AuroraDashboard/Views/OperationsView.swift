import SwiftUI

struct OperationsView: View {
    let configuration: AppConfiguration

    var body: some View {
        NavigationStack {
            ScrollView {
                VStack(alignment: .leading, spacing: 16) {
                    Text("Traffic-light status for source sync, processing, archive transfer, storage, battery, and public endpoint checks.")
                        .font(.subheadline)
                        .foregroundStyle(.secondary)

                    PlaceholderPanel(
                        title: "System summary",
                        subtitle: "Overall health, service state, storage pressure, source lag, and battery conditions.",
                        systemImage: "gauge.with.dots.needle.bottom.50percent",
                        tint: .green
                    )

                    PlaceholderPanel(
                        title: "Root-cause groups",
                        subtitle: "Source, sync/network, local processing, GWS transfer, and dashboard render diagnostics.",
                        systemImage: "point.3.connected.trianglepath.dotted",
                        tint: .orange
                    )

                    PlaceholderPanel(
                        title: "Per-stream health",
                        subtitle: "Ceilometer, radar, radiometer, meteorology, radiation, power, WXcam, and operations streams.",
                        systemImage: "list.bullet.rectangle",
                        tint: .blue
                    )
                }
                .padding()
            }
            .background(Color(.systemGroupedBackground))
            .navigationTitle("Operations")
        }
    }
}
