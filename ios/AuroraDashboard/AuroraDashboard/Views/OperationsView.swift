import SwiftUI

struct OperationsView: View {
    @ObservedObject var store: DashboardStore

    var body: some View {
        NavigationStack {
            List {
                if let error = store.lastError {
                    Section {
                        InlineErrorView(message: error)
                    }
                }

                if let operations = store.operations {
                    Section {
                        StatusCard(
                            title: "Operations Dashboard",
                            subtitle: operations.summary,
                            systemImage: "gauge.with.dots.needle.bottom.50percent",
                            level: operations.overallLevel
                        ) {
                            VStack(alignment: .leading, spacing: 6) {
                                if let updatedAt = operations.updatedAt {
                                    Label(updatedAt, systemImage: "clock")
                                        .font(.caption)
                                        .foregroundStyle(.secondary)
                                }
                                HStack(spacing: 8) {
                                    ForEach(["green", "amber", "red", "unknown"], id: \.self) { key in
                                        if let value = operations.checkCounts[key] {
                                            Text("\(key.capitalized): \(value)")
                                                .font(.caption)
                                                .foregroundStyle(.secondary)
                                        }
                                    }
                                }
                            }
                        }
                    }
                    .listRowInsets(EdgeInsets(top: 8, leading: 16, bottom: 8, trailing: 16))
                    .listRowBackground(Color.clear)

                    if !operations.trendCards.isEmpty {
                        Section("Seven-day context") {
                            LazyVGrid(columns: [GridItem(.flexible()), GridItem(.flexible())], spacing: 10) {
                                ForEach(operations.trendCards) { card in
                                    MetricTile(title: card.title, value: trendValue(card), level: card.level)
                                }
                            }
                            .listRowInsets(EdgeInsets(top: 8, leading: 16, bottom: 8, trailing: 16))
                            .listRowBackground(Color.clear)
                        }
                    }

                    if !operations.alerts.isEmpty {
                        Section("Active alerts") {
                            ForEach(operations.alerts) { alert in
                                HStack(alignment: .top, spacing: 12) {
                                    Image(systemName: "bell.badge")
                                        .foregroundStyle(StatusStyle.color(for: alert.level))
                                    VStack(alignment: .leading, spacing: 3) {
                                        Text(alert.title)
                                            .font(.headline)
                                        if !alert.detail.isEmpty {
                                            Text(alert.detail)
                                                .font(.subheadline)
                                                .foregroundStyle(.secondary)
                                        }
                                    }
                                    Spacer()
                                    StatusBadge(level: alert.level)
                                }
                            }
                        }
                    }

                    Section("Root-cause groups") {
                        ForEach(operations.rootCauseGroups) { group in
                            HStack(alignment: .top, spacing: 12) {
                                Circle()
                                    .fill(StatusStyle.color(for: group.level))
                                    .frame(width: 10, height: 10)
                                    .padding(.top, 5)
                                VStack(alignment: .leading, spacing: 3) {
                                    Text(group.title)
                                        .font(.headline)
                                    Text(group.detail)
                                        .font(.subheadline)
                                        .foregroundStyle(.secondary)
                                }
                            }
                        }
                    }

                    Section("Stream health") {
                        ForEach(operations.streamStates) { stream in
                            VStack(alignment: .leading, spacing: 8) {
                                HStack {
                                    Text(stream.title)
                                        .font(.headline)
                                    Spacer()
                                    StatusBadge(level: stream.level)
                                }
                                Text(stream.detail)
                                    .font(.subheadline)
                                    .foregroundStyle(.secondary)
                                if let healthy = stream.serviceHealthyCount, let total = stream.serviceCount {
                                    ProgressView(value: Double(healthy), total: Double(max(total, 1)))
                                }
                            }
                            .padding(.vertical, 4)
                        }
                    }
                } else {
                    Section {
                        LoadingContentView(title: "Loading operations")
                    }
                }
            }
            .navigationTitle("Operations")
            .toolbar {
                ToolbarItem(placement: .topBarTrailing) {
                    Button {
                        Task { await store.refreshOperations() }
                    } label: {
                        Image(systemName: "arrow.clockwise")
                    }
                    .disabled(store.isRefreshing)
                }
            }
            .refreshable {
                await store.refreshOperations()
            }
            .task {
                await store.refreshOperations()
            }
        }
    }

    private func trendValue(_ card: TrendCard) -> String {
        guard let value = card.value else {
            return "n/a"
        }
        let formatted = value.formatted(.number.precision(.fractionLength(0...1)))
        if let unit = card.unit, !unit.isEmpty {
            return "\(formatted) \(unit)"
        }
        return formatted
    }
}
