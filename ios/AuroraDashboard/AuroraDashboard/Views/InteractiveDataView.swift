import SwiftUI

struct InteractiveDataView: View {
    @ObservedObject var store: DashboardStore

    var body: some View {
        NavigationStack {
            List {
                if let error = store.lastError {
                    Section {
                        InlineErrorView(message: error)
                    }
                }

                Section("Dashboard instruments") {
                    ForEach(store.visibleInstruments.filter { $0.id != "wxcam" }) { instrument in
                        NavigationLink {
                            InstrumentDetailView(store: store, instrument: instrument)
                        } label: {
                            Label(instrument.title, systemImage: instrument.systemImage)
                        }
                    }
                }
            }
            .navigationTitle("Interactive")
            .refreshable {
                await store.refreshAll()
            }
        }
    }
}

private struct InstrumentDetailView: View {
    @ObservedObject var store: DashboardStore
    let instrument: InstrumentDescriptor
    @State private var window = "24h"

    private var summary: InstrumentSummaryPayload? {
        store.instrumentSummaries[instrument.id]
    }

    var body: some View {
        List {
            Picker("Window", selection: $window) {
                Text("24 h").tag("24h")
                Text("7 d").tag("7d")
            }
            .pickerStyle(.segmented)
            .listRowSeparator(.hidden)

            if let summary {
                Section {
                    StatusCard(
                        title: summary.instrument.title,
                        subtitle: summary.updatedAt.map { "Updated \($0)" } ?? "Latest generated products",
                        systemImage: summary.instrument.systemImage,
                        level: summary.panels.first?.level ?? "unknown"
                    ) {
                        Text(summary.window == "7d" ? "Seven-day view" : "Latest 24-hour view")
                            .font(.caption)
                            .foregroundStyle(.secondary)
                    }
                }
                .listRowInsets(EdgeInsets(top: 8, leading: 16, bottom: 8, trailing: 16))
                .listRowBackground(Color.clear)

                ForEach(summary.panels) { panel in
                    Section(panel.title) {
                        if panel.kind == "image", panel.imageURL != nil {
                            AuthenticatedRemoteImage(store: store, urlString: panel.imageURL)
                                .clipShape(RoundedRectangle(cornerRadius: 8, style: .continuous))
                                .listRowInsets(EdgeInsets(top: 8, leading: 16, bottom: 8, trailing: 16))
                                .listRowBackground(Color.clear)
                        } else {
                            ContentUnavailableView(panel.detail, systemImage: "chart.xyaxis.line")
                        }
                    }
                }

                if !summary.recentQuicklooks.isEmpty {
                    Section("Recent quicklooks") {
                        ForEach(summary.recentQuicklooks) { entry in
                            HStack {
                                VStack(alignment: .leading, spacing: 3) {
                                    Text(entry.title)
                                    if let modifiedAt = entry.modifiedAt {
                                        Text(modifiedAt)
                                            .font(.caption)
                                            .foregroundStyle(.secondary)
                                    }
                                }
                                Spacer()
                                Image(systemName: "photo")
                                    .foregroundStyle(.secondary)
                            }
                        }
                    }
                }
            } else {
                Section {
                    LoadingContentView(title: "Loading \(instrument.title)")
                }
            }
        }
        .navigationTitle(instrument.title)
        .navigationBarTitleDisplayMode(.inline)
        .task(id: "\(instrument.id)-\(window)") {
            await store.refreshInstrument(id: instrument.id, window: window)
        }
        .refreshable {
            await store.refreshInstrument(id: instrument.id, window: window)
        }
    }
}
