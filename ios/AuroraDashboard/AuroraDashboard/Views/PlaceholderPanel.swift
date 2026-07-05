import SwiftUI
import UIKit

struct StatusBadge: View {
    let level: String

    var body: some View {
        Text(StatusStyle.label(for: level))
            .font(.caption.weight(.semibold))
            .padding(.horizontal, 9)
            .padding(.vertical, 5)
            .foregroundStyle(StatusStyle.color(for: level))
            .background(StatusStyle.color(for: level).opacity(0.14))
            .clipShape(Capsule())
    }
}

struct StatusCard<Content: View>: View {
    let title: String
    let subtitle: String
    let systemImage: String
    let level: String
    @ViewBuilder var content: Content

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            HStack(alignment: .top, spacing: 12) {
                Image(systemName: systemImage)
                    .font(.title3)
                    .foregroundStyle(StatusStyle.color(for: level))
                    .frame(width: 28)

                VStack(alignment: .leading, spacing: 3) {
                    Text(title)
                        .font(.headline)
                    Text(subtitle)
                        .font(.subheadline)
                        .foregroundStyle(.secondary)
                        .fixedSize(horizontal: false, vertical: true)
                }

                Spacer(minLength: 8)
                StatusBadge(level: level)
            }

            content
        }
        .padding()
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(Color(.secondarySystemGroupedBackground))
        .clipShape(RoundedRectangle(cornerRadius: 8, style: .continuous))
    }
}

struct MetricTile: View {
    let title: String
    let value: String
    let level: String

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack {
                Text(title)
                    .font(.caption)
                    .foregroundStyle(.secondary)
                Spacer()
                Circle()
                    .fill(StatusStyle.color(for: level))
                    .frame(width: 9, height: 9)
            }

            Text(value)
                .font(.title3.weight(.semibold))
                .monospacedDigit()
                .lineLimit(1)
                .minimumScaleFactor(0.75)
        }
        .padding(12)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(Color(.secondarySystemGroupedBackground))
        .clipShape(RoundedRectangle(cornerRadius: 8, style: .continuous))
    }
}

struct InlineErrorView: View {
    let message: String

    var body: some View {
        Label(message, systemImage: "exclamationmark.triangle")
            .font(.subheadline)
            .foregroundStyle(.orange)
            .padding(.vertical, 6)
    }
}

struct LoadingContentView: View {
    let title: String

    var body: some View {
        HStack(spacing: 12) {
            ProgressView()
            Text(title)
                .font(.subheadline)
                .foregroundStyle(.secondary)
        }
        .frame(maxWidth: .infinity, minHeight: 96)
    }
}

struct AuthenticatedRemoteImage: View {
    @ObservedObject var store: DashboardStore
    let urlString: String?
    var aspectRatio: CGFloat? = nil
    var contentMode: ContentMode = .fit

    @State private var image: UIImage?
    @State private var error: String?
    @State private var isLoading = false

    var body: some View {
        Group {
            if let image {
                Image(uiImage: image)
                    .resizable()
                    .aspectRatio(aspectRatio, contentMode: contentMode)
                    .frame(maxWidth: .infinity)
            } else if isLoading {
                LoadingContentView(title: "Loading media")
            } else if let error {
                InlineErrorView(message: error)
                    .frame(maxWidth: .infinity, minHeight: 96)
            } else {
                ContentUnavailableView("No media", systemImage: "photo", description: Text("No image is available for this selection."))
                    .frame(minHeight: 160)
            }
        }
        .task(id: store.absoluteURL(urlString)?.absoluteString) {
            await load()
        }
    }

    private func load() async {
        guard let url = store.absoluteURL(urlString) else {
            image = nil
            error = nil
            return
        }

        isLoading = true
        error = nil
        defer { isLoading = false }

        do {
            let data = try await store.imageData(from: url)
            guard let loaded = UIImage(data: data) else {
                error = "The downloaded image could not be decoded."
                image = nil
                return
            }
            image = loaded
        } catch {
            self.error = error.localizedDescription
            image = nil
        }
    }
}

enum StatusStyle {
    static func color(for level: String) -> Color {
        switch level.lowercased() {
        case "green", "ok", "healthy":
            return .green
        case "amber", "yellow", "warning":
            return .orange
        case "red", "critical", "failed", "error":
            return .red
        default:
            return .secondary
        }
    }

    static func label(for level: String) -> String {
        switch level.lowercased() {
        case "green":
            return "Green"
        case "amber":
            return "Amber"
        case "red":
            return "Red"
        default:
            return "Unknown"
        }
    }
}
