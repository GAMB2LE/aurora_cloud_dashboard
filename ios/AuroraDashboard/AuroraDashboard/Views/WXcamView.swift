import AVFoundation
import AVKit
import SwiftUI

struct WXcamView: View {
    @ObservedObject var store: DashboardStore
    @State private var stream = "fish_hdr"
    @State private var day = "latest"

    private var payload: WXcamPayload? {
        store.wxcam(stream: stream, day: day)
    }

    private var dayOptions: [String] {
        var values = ["latest"]
        for availableDay in payload?.availableDays ?? [] where !values.contains(availableDay) {
            values.append(availableDay)
        }
        return values
    }

    var body: some View {
        NavigationStack {
            List {
                Section("Stream") {
                    Picker("Stream", selection: $stream) {
                        ForEach(store.manifest.wxcamStreams) { stream in
                            Text(stream.title).tag(stream.id)
                        }
                    }
                    .pickerStyle(.segmented)

                    Picker("Day", selection: $day) {
                        ForEach(dayOptions, id: \.self) { day in
                            Text(day == "latest" ? "Latest" : day).tag(day)
                        }
                    }
                }

                if let payload {
                    Section(payload.stream.title) {
                        AuthenticatedVideoPlayer(store: store, urlString: payload.video.url)
                            .frame(height: stream == "fish_hdr" ? 360 : 220)
                            .clipShape(RoundedRectangle(cornerRadius: 8, style: .continuous))
                            .listRowInsets(EdgeInsets(top: 8, leading: 16, bottom: 8, trailing: 16))
                            .listRowBackground(Color.clear)

                        Label("Selected day: \(payload.selectedDay)", systemImage: "calendar")
                            .font(.caption)
                            .foregroundStyle(.secondary)
                    }

                    Section("Hourly thumbnails") {
                        if payload.thumbnails.isEmpty {
                            ContentUnavailableView("No thumbnails", systemImage: "square.grid.3x3", description: Text("No hourly thumbnails are available for this stream and day."))
                        } else {
                            LazyVGrid(columns: [GridItem(.adaptive(minimum: 96), spacing: 10)], spacing: 10) {
                                ForEach(payload.thumbnails) { thumb in
                                    VStack(alignment: .leading, spacing: 5) {
                                        AuthenticatedRemoteImage(store: store, urlString: thumb.imageURL, aspectRatio: 1, contentMode: .fill)
                                            .frame(height: 86)
                                            .clipped()
                                            .clipShape(RoundedRectangle(cornerRadius: 6, style: .continuous))
                                        Text(thumbLabel(thumb))
                                            .font(.caption2)
                                            .foregroundStyle(.secondary)
                                            .lineLimit(1)
                                    }
                                }
                            }
                            .listRowInsets(EdgeInsets(top: 8, leading: 16, bottom: 8, trailing: 16))
                            .listRowBackground(Color.clear)
                        }
                    }
                } else {
                    Section {
                        LoadingContentView(title: "Loading WXcam media")
                    }
                }
            }
            .navigationTitle("WXcam")
            .toolbar {
                ToolbarItem(placement: .topBarTrailing) {
                    Button {
                        Task { await reload() }
                    } label: {
                        Image(systemName: "arrow.clockwise")
                    }
                }
            }
            .task(id: "\(stream)-\(day)") {
                await reload()
            }
            .onChange(of: stream) { _, _ in
                day = "latest"
            }
            .refreshable {
                await reload()
            }
        }
    }

    private func reload() async {
        await store.refreshWXcam(stream: stream, day: day)
    }

    private func thumbLabel(_ thumb: WXcamThumbnail) -> String {
        if let hour = thumb.hourUTC {
            return String(format: "%02d:00 UTC", hour)
        }
        return thumb.title
    }
}

private struct AuthenticatedVideoPlayer: View {
    @ObservedObject var store: DashboardStore
    let urlString: String?
    @State private var player: AVPlayer?
    @State private var error: String?

    var body: some View {
        Group {
            if let player {
                VideoPlayer(player: player)
            } else if let error {
                InlineErrorView(message: error)
                    .frame(maxWidth: .infinity, minHeight: 180)
            } else {
                ContentUnavailableView("No video", systemImage: "video", description: Text("No MP4 is available for this selection."))
                    .frame(minHeight: 180)
            }
        }
        .task(id: store.absoluteURL(urlString)?.absoluteString) {
            configurePlayer()
        }
        .onDisappear {
            player?.pause()
        }
    }

    private func configurePlayer() {
        guard let url = store.absoluteURL(urlString) else {
            player = nil
            error = nil
            return
        }

        let headers = store.authenticatedHeaders()
        let options: [String: Any] = headers.isEmpty ? [:] : ["AVURLAssetHTTPHeaderFieldsKey": headers]
        let asset = AVURLAsset(url: url, options: options)
        player = AVPlayer(playerItem: AVPlayerItem(asset: asset))
        error = nil
    }
}
