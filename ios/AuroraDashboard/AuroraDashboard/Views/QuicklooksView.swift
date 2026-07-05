import SwiftUI

struct QuicklooksView: View {
    @ObservedObject var store: DashboardStore
    @State private var kind = "science"
    @State private var instrumentID = "power"
    @State private var selectedToken = "latest"

    private var instruments: [InstrumentDescriptor] {
        store.visibleInstruments.filter { instrument in
            kind == "science" ? instrument.supportsScienceQuicklooks : instrument.supportsHousekeepingQuicklooks
        }
    }

    private var payload: QuicklooksPayload? {
        store.quicklooks(kind: kind, instrumentID: instrumentID)
    }

    private var selectedEntry: QuicklookEntry? {
        payload?.entries.first { $0.token == selectedToken } ?? payload?.latest ?? payload?.entries.first
    }

    var body: some View {
        NavigationStack {
            List {
                Picker("Quicklook type", selection: $kind) {
                    Text("Science").tag("science")
                    Text("Housekeeping").tag("housekeeping")
                }
                .pickerStyle(.segmented)
                .listRowSeparator(.hidden)

                Section("Product") {
                    Picker("Instrument", selection: $instrumentID) {
                        ForEach(instruments) { instrument in
                            Label(instrument.title, systemImage: instrument.systemImage)
                                .tag(instrument.id)
                        }
                    }

                    if let payload, !payload.entries.isEmpty {
                        Picker("Date", selection: $selectedToken) {
                            ForEach(payload.entries) { entry in
                                Text(entry.title).tag(entry.token)
                            }
                        }
                    }
                }

                if let entry = selectedEntry {
                    Section(entry.title) {
                        AuthenticatedRemoteImage(store: store, urlString: entry.imageURL)
                            .clipShape(RoundedRectangle(cornerRadius: 8, style: .continuous))
                            .listRowInsets(EdgeInsets(top: 8, leading: 16, bottom: 8, trailing: 16))
                            .listRowBackground(Color.clear)

                        if let modifiedAt = entry.modifiedAt {
                            Label(modifiedAt, systemImage: "clock")
                                .font(.caption)
                                .foregroundStyle(.secondary)
                        }
                    }
                } else if payload != nil {
                    Section {
                        ContentUnavailableView("No quicklooks", systemImage: "photo.on.rectangle.angled", description: Text("No generated image exists for this instrument and mode."))
                    }
                } else {
                    Section {
                        LoadingContentView(title: "Loading quicklooks")
                    }
                }
            }
            .navigationTitle("Quicklooks")
            .toolbar {
                ToolbarItem(placement: .topBarTrailing) {
                    Button {
                        Task { await reload() }
                    } label: {
                        Image(systemName: "arrow.clockwise")
                    }
                }
            }
            .task(id: "\(kind)-\(instrumentID)") {
                normalizeInstrumentSelection()
                await reload()
            }
            .onChange(of: kind) { _, _ in
                normalizeInstrumentSelection()
                selectedToken = "latest"
            }
            .onChange(of: instrumentID) { _, _ in
                selectedToken = "latest"
            }
            .refreshable {
                await reload()
            }
        }
    }

    private func normalizeInstrumentSelection() {
        if !instruments.contains(where: { $0.id == instrumentID }), let first = instruments.first {
            instrumentID = first.id
        }
    }

    private func reload() async {
        await store.refreshQuicklooks(kind: kind, instrumentID: instrumentID)
        if selectedToken == "latest", let latest = payload?.latest {
            selectedToken = latest.token
        }
    }
}
