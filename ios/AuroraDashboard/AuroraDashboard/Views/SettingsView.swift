import SwiftUI

struct SettingsView: View {
    let configuration: AppConfiguration

    var body: some View {
        NavigationStack {
            List {
                Section {
                    VStack(spacing: 10) {
                        Image("AppLogo")
                            .resizable()
                            .scaledToFit()
                            .frame(width: 92, height: 92)
                            .accessibilityLabel("GAMB2LE logo")

                        Text("Aurora Dashboard")
                            .font(.headline)
                    }
                    .frame(maxWidth: .infinity)
                    .padding(.vertical, 8)
                }
                .listRowBackground(Color.clear)

                Section(
                    header: Text("Dashboard endpoint"),
                    footer: Text("The starter app does not fetch live dashboard data yet. Add an endpoint here when the Panel app publishes a mobile API or static manifest.")
                ) {
                    LabeledContent("Status", value: endpointStatus)
                    LabeledContent("Refresh floor", value: "\(Int(configuration.minimumRefreshInterval)) seconds")
                }

                Section(header: Text("Links")) {
                    Link(destination: configuration.documentationURL) {
                        Label("Documentation", systemImage: "doc.text")
                    }

                    Link(destination: configuration.projectURL) {
                        Label("Project website", systemImage: "globe")
                    }
                }

                Section(header: Text("About")) {
                    LabeledContent("App", value: "Aurora Dashboard")
                    LabeledContent("Bundle", value: "uk.co.gamb2le.AuroraDashboard")
                    LabeledContent("Minimum iOS", value: "17.0")
                }
            }
            .navigationTitle("Settings")
        }
    }

    private var endpointStatus: String {
        configuration.dashboardBaseURL?.absoluteString ?? "Not configured"
    }
}
