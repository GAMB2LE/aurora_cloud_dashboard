import SwiftUI

struct SettingsView: View {
    @ObservedObject var store: DashboardStore

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

                        Text("Native live operations and media browser")
                            .font(.caption)
                            .foregroundStyle(.secondary)
                    }
                    .frame(maxWidth: .infinity)
                    .padding(.vertical, 8)
                }
                .listRowBackground(Color.clear)

                Section(
                    header: Text("Mobile API"),
                    footer: Text("The token is stored in Keychain. The API URL should include the /mobile/v1 prefix exposed by the deployment.")
                ) {
                    TextField("API base URL", text: $store.baseURLString)
                        .textInputAutocapitalization(.never)
                        .keyboardType(.URL)
                        .autocorrectionDisabled()

                    SecureField("Bearer token", text: $store.apiToken)
                        .textInputAutocapitalization(.never)
                        .autocorrectionDisabled()

                    Button {
                        store.saveSettings()
                    } label: {
                        Label("Save settings", systemImage: "checkmark.circle")
                    }

                    Button {
                        Task { await store.checkHealth() }
                    } label: {
                        Label("Check connection", systemImage: "network")
                    }

                    if let healthStatus = store.healthStatus {
                        LabeledContent("Health", value: healthStatus)
                    }
                }

                Section("Cache") {
                    Button(role: .destructive) {
                        store.clearCachedResponses()
                    } label: {
                        Label("Clear cached responses", systemImage: "trash")
                    }
                }

                Section(header: Text("Links")) {
                    Link(destination: store.configuration.documentationURL) {
                        Label("Documentation", systemImage: "doc.text")
                    }

                    Link(destination: store.configuration.projectURL) {
                        Label("Project website", systemImage: "globe")
                    }
                }

                Section(header: Text("About")) {
                    LabeledContent("App", value: "Aurora Dashboard")
                    LabeledContent("Bundle", value: "uk.co.gamb2le.AuroraDashboard")
                    LabeledContent("Minimum iOS", value: "17.0")
                    LabeledContent("API refresh floor", value: "\(Int(store.configuration.minimumRefreshInterval)) seconds")
                }
            }
            .navigationTitle("Settings")
        }
    }
}
