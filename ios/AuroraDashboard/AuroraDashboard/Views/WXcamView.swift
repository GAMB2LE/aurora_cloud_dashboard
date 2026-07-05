import SwiftUI

struct WXcamView: View {
    let configuration: AppConfiguration
    @State private var stream = WXcamStream.fish

    var body: some View {
        NavigationStack {
            ScrollView {
                VStack(alignment: .leading, spacing: 16) {
                    Picker("Stream", selection: $stream) {
                        ForEach(WXcamStream.allCases) { stream in
                            Text(stream.title).tag(stream)
                        }
                    }
                    .pickerStyle(.segmented)

                    PlaceholderPanel(
                        title: "\(stream.title) latest media",
                        subtitle: "A native still/video browser will eventually use the dashboard WXcam catalog and static media routes.",
                        systemImage: "video",
                        tint: .cyan
                    )

                    PlaceholderPanel(
                        title: "Hourly thumbnails",
                        subtitle: "The dashboard currently publishes daily MP4s and hourly thumbnail products for FISH HDR and PANO HDR streams.",
                        systemImage: "square.grid.3x3"
                    )
                }
                .padding()
            }
            .background(Color(.systemGroupedBackground))
            .navigationTitle("WXcam")
        }
    }
}

private enum WXcamStream: String, CaseIterable, Identifiable {
    case fish
    case pano

    var id: String { rawValue }

    var title: String {
        switch self {
        case .fish:
            return "FISH HDR"
        case .pano:
            return "PANO HDR"
        }
    }
}
