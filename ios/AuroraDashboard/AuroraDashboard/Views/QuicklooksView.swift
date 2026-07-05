import SwiftUI

struct QuicklooksView: View {
    let configuration: AppConfiguration
    @State private var selection = QuicklookMode.science

    var body: some View {
        NavigationStack {
            List {
                Picker("Quicklook type", selection: $selection) {
                    ForEach(QuicklookMode.allCases) { mode in
                        Text(mode.title).tag(mode)
                    }
                }
                .pickerStyle(.segmented)
                .listRowSeparator(.hidden)

                Section(header: Text(selection.title), footer: Text(selection.footer)) {
                    ForEach(selection.items, id: \.self) { item in
                        Label(item, systemImage: selection.systemImage)
                    }
                }

                Section(header: Text("Starter state")) {
                    PlaceholderPanel(
                        title: "Image viewer placeholder",
                        subtitle: "Daily PNG browsing and latest quicklook loading will be added after the dashboard exposes mobile-friendly endpoints.",
                        systemImage: "photo.on.rectangle.angled"
                    )
                    .listRowInsets(EdgeInsets(top: 8, leading: 16, bottom: 8, trailing: 16))
                    .listRowBackground(Color.clear)
                }
            }
            .navigationTitle("Quicklooks")
        }
    }
}

private enum QuicklookMode: String, CaseIterable, Identifiable {
    case science
    case housekeeping

    var id: String { rawValue }

    var title: String {
        switch self {
        case .science:
            return "Science"
        case .housekeeping:
            return "Housekeeping"
        }
    }

    var systemImage: String {
        switch self {
        case .science:
            return "chart.line.uptrend.xyaxis"
        case .housekeeping:
            return "wrench.and.screwdriver"
        }
    }

    var items: [String] {
        switch self {
        case .science:
            return ["Power", "Ceilometer", "Cloud Radar", "Radiometer", "Meteorology", "Radiation", "WXcam"]
        case .housekeeping:
            return ["Ceilometer", "Cloud Radar", "ASFS Logger", "WXcam", "Operations"]
        }
    }

    var footer: String {
        switch self {
        case .science:
            return "Matches the Science Quicklooks tab in the Panel dashboard."
        case .housekeeping:
            return "Matches the House Keeping Quicklooks tab in the Panel dashboard."
        }
    }
}
