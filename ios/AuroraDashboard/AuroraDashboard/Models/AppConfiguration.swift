import Foundation
import Security
import SwiftUI

struct AppConfiguration: Equatable {
    var apiBaseURL: URL
    var documentationURL: URL
    var projectURL: URL
    var minimumRefreshInterval: TimeInterval

    static let `default` = AppConfiguration(
        apiBaseURL: URL(string: "https://data-ocean.gamb2le.co.uk/mobile/v1")!,
        documentationURL: URL(string: "https://gamb2le.pages.dev/documentation-docs/")!,
        projectURL: URL(string: "https://www.gamb2le.co.uk/")!,
        minimumRefreshInterval: 60
    )
}

struct MobileManifest: Decodable, Equatable {
    var serverTime: String?
    var minimumRefreshIntervalSeconds: Int?
    var sections: [SectionDescriptor]
    var instruments: [InstrumentDescriptor]
    var wxcamStreams: [WXcamStreamDescriptor]

    static let fallback = MobileManifest(
        serverTime: nil,
        minimumRefreshIntervalSeconds: 60,
        sections: DashboardSection.allCases.map {
            SectionDescriptor(id: $0.rawValue, title: $0.title, systemImage: $0.systemImage)
        },
        instruments: [
            InstrumentDescriptor(id: "power", title: "Aurora Power Supply", systemImage: "battery.100percent", visible: true, supportsSummary: true, supportsScienceQuicklooks: true, supportsHousekeepingQuicklooks: true),
            InstrumentDescriptor(id: "ceilometer", title: "Ceilometer", systemImage: "laser.burst", visible: true, supportsSummary: true, supportsScienceQuicklooks: true, supportsHousekeepingQuicklooks: true),
            InstrumentDescriptor(id: "cloud-radar", title: "Cloud Radar", systemImage: "dot.radiowaves.left.and.right", visible: true, supportsSummary: true, supportsScienceQuicklooks: true, supportsHousekeepingQuicklooks: true),
            InstrumentDescriptor(id: "hatpro", title: "Scanning Microwave Radiometer", systemImage: "antenna.radiowaves.left.and.right", visible: true, supportsSummary: true, supportsScienceQuicklooks: true, supportsHousekeepingQuicklooks: false),
            InstrumentDescriptor(id: "vaisalamet", title: "Meteorology", systemImage: "cloud.sun", visible: true, supportsSummary: true, supportsScienceQuicklooks: true, supportsHousekeepingQuicklooks: true),
            InstrumentDescriptor(id: "asfs-logger", title: "Radiation", systemImage: "sun.max", visible: true, supportsSummary: true, supportsScienceQuicklooks: true, supportsHousekeepingQuicklooks: true),
            InstrumentDescriptor(id: "ops-monitor", title: "Operations", systemImage: "gauge.with.dots.needle.bottom.50percent", visible: true, supportsSummary: true, supportsScienceQuicklooks: true, supportsHousekeepingQuicklooks: true),
            InstrumentDescriptor(id: "wxcam", title: "WXcam", systemImage: "video", visible: true, supportsSummary: false, supportsScienceQuicklooks: true, supportsHousekeepingQuicklooks: true)
        ],
        wxcamStreams: [
            WXcamStreamDescriptor(id: "fish_hdr", title: "FISH HDR", systemImage: "camera.aperture"),
            WXcamStreamDescriptor(id: "pano_hdr", title: "PANO HDR", systemImage: "photo")
        ]
    )
}

struct SectionDescriptor: Decodable, Equatable, Identifiable {
    var id: String
    var title: String
    var systemImage: String
}

struct InstrumentDescriptor: Decodable, Equatable, Identifiable, Hashable {
    var id: String
    var title: String
    var systemImage: String
    var visible: Bool
    var supportsSummary: Bool
    var supportsScienceQuicklooks: Bool
    var supportsHousekeepingQuicklooks: Bool

    init(
        id: String,
        title: String,
        systemImage: String,
        visible: Bool = true,
        supportsSummary: Bool = true,
        supportsScienceQuicklooks: Bool = true,
        supportsHousekeepingQuicklooks: Bool = false
    ) {
        self.id = id
        self.title = title
        self.systemImage = systemImage
        self.visible = visible
        self.supportsSummary = supportsSummary
        self.supportsScienceQuicklooks = supportsScienceQuicklooks
        self.supportsHousekeepingQuicklooks = supportsHousekeepingQuicklooks
    }

    enum CodingKeys: String, CodingKey {
        case id
        case title
        case systemImage
        case visible
        case supportsSummary
        case supportsScienceQuicklooks
        case supportsHousekeepingQuicklooks
    }

    init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        id = try container.decode(String.self, forKey: .id)
        title = try container.decode(String.self, forKey: .title)
        systemImage = try container.decodeIfPresent(String.self, forKey: .systemImage) ?? "waveform.path.ecg"
        visible = try container.decodeIfPresent(Bool.self, forKey: .visible) ?? true
        supportsSummary = try container.decodeIfPresent(Bool.self, forKey: .supportsSummary) ?? true
        supportsScienceQuicklooks = try container.decodeIfPresent(Bool.self, forKey: .supportsScienceQuicklooks) ?? true
        supportsHousekeepingQuicklooks = try container.decodeIfPresent(Bool.self, forKey: .supportsHousekeepingQuicklooks) ?? false
    }
}

struct WXcamStreamDescriptor: Decodable, Equatable, Identifiable, Hashable {
    var id: String
    var title: String
    var systemImage: String
}

struct OperationsPayload: Decodable, Equatable {
    var serverTime: String?
    var updatedAt: String?
    var overallLevel: String
    var summary: String
    var checkCounts: [String: Int]
    var streamStates: [StreamState]
    var rootCauseGroups: [RootCauseGroup]
    var alerts: [OperationAlert]
    var trendCards: [TrendCard]
}

struct StreamState: Decodable, Equatable, Identifiable {
    var id: String
    var title: String
    var level: String
    var detail: String
    var serviceHealthyCount: Int?
    var serviceCount: Int?
}

struct RootCauseGroup: Decodable, Equatable, Identifiable {
    var id: String
    var title: String
    var level: String
    var detail: String
}

struct OperationAlert: Decodable, Equatable, Identifiable {
    var id: String
    var title: String
    var level: String
    var detail: String
}

struct TrendCard: Decodable, Equatable, Identifiable {
    var id: String
    var title: String
    var value: Double?
    var unit: String?
    var level: String
}

struct InstrumentSummaryPayload: Decodable, Equatable {
    var serverTime: String?
    var instrument: InstrumentDescriptor
    var window: String
    var updatedAt: String?
    var panels: [SummaryPanel]
    var recentQuicklooks: [QuicklookEntry]
}

struct SummaryPanel: Decodable, Equatable, Identifiable {
    var id: String
    var title: String
    var kind: String
    var imageURL: String?
    var level: String
    var detail: String
}

struct QuicklooksPayload: Decodable, Equatable {
    var serverTime: String?
    var kind: String
    var instrument: InstrumentDescriptor
    var latest: QuicklookEntry?
    var entries: [QuicklookEntry]
}

struct QuicklookEntry: Decodable, Equatable, Identifiable, Hashable {
    var id: String
    var token: String
    var title: String
    var imageURL: String?
    var exists: Bool?
    var sizeBytes: Int?
    var modifiedAt: String?
}

struct WXcamPayload: Decodable, Equatable {
    var serverTime: String?
    var stream: WXcamStreamDescriptor
    var selectedDay: String
    var availableDays: [String]
    var video: MediaResource
    var posterURL: String?
    var thumbnails: [WXcamThumbnail]
}

struct WXcamThumbnail: Decodable, Equatable, Identifiable, Hashable {
    var id: String
    var title: String
    var hourUTC: Int?
    var imageURL: String?
    var exists: Bool?
    var sizeBytes: Int?
    var modifiedAt: String?
}

struct MediaResource: Decodable, Equatable {
    var url: String?
    var exists: Bool?
    var sizeBytes: Int?
    var modifiedAt: String?
}

struct HealthPayload: Decodable, Equatable {
    var status: String
    var serverTime: String?
    var authRequired: Bool?
    var tokenConfigured: Bool?
}

enum MobileAPIError: LocalizedError, Equatable {
    case invalidBaseURL
    case badStatus(Int)
    case unauthorized
    case emptyResponse

    var errorDescription: String? {
        switch self {
        case .invalidBaseURL:
            return "The API base URL is invalid."
        case .badStatus(let status):
            return "The API returned HTTP \(status)."
        case .unauthorized:
            return "The API token was rejected."
        case .emptyResponse:
            return "The API returned an empty response."
        }
    }
}

struct MobileAPIClient: Equatable {
    var baseURL: URL
    var token: String?

    func get<T: Decodable>(_ path: String, queryItems: [URLQueryItem] = []) async throws -> T {
        let data = try await data(for: makeURL(path: path, queryItems: queryItems))
        do {
            return try JSONDecoder().decode(T.self, from: data)
        } catch {
            throw error
        }
    }

    func data(from url: URL) async throws -> Data {
        try await data(for: url)
    }

    func authenticatedHeaders() -> [String: String] {
        guard let token, !token.isEmpty else {
            return [:]
        }
        return ["Authorization": "Bearer \(token)"]
    }

    func absoluteURL(for value: String?) -> URL? {
        guard let value, !value.isEmpty else {
            return nil
        }
        if let url = URL(string: value), url.scheme != nil {
            return url
        }
        var url = baseURL
        for component in value.split(separator: "/") {
            url.append(path: String(component))
        }
        return url
    }

    private func makeURL(path: String, queryItems: [URLQueryItem]) throws -> URL {
        var url = baseURL
        for component in path.split(separator: "/") {
            url.append(path: String(component))
        }
        guard var components = URLComponents(url: url, resolvingAgainstBaseURL: false) else {
            throw MobileAPIError.invalidBaseURL
        }
        components.queryItems = queryItems.isEmpty ? nil : queryItems
        guard let finalURL = components.url else {
            throw MobileAPIError.invalidBaseURL
        }
        return finalURL
    }

    private func data(for url: URL) async throws -> Data {
        var request = URLRequest(url: url)
        request.timeoutInterval = 20
        request.cachePolicy = .reloadRevalidatingCacheData
        for (field, value) in authenticatedHeaders() {
            request.setValue(value, forHTTPHeaderField: field)
        }

        let (data, response) = try await URLSession.shared.data(for: request)
        guard let http = response as? HTTPURLResponse else {
            throw MobileAPIError.emptyResponse
        }
        if http.statusCode == 401 {
            throw MobileAPIError.unauthorized
        }
        guard (200..<300).contains(http.statusCode) else {
            throw MobileAPIError.badStatus(http.statusCode)
        }
        return data
    }
}

@MainActor
final class DashboardStore: ObservableObject {
    private let baseURLDefaultsKey = "AuroraDashboard.apiBaseURL"
    private var hasLoadedInitialData = false

    @Published var configuration: AppConfiguration
    @Published var baseURLString: String
    @Published var apiToken: String
    @Published private(set) var manifest: MobileManifest = .fallback
    @Published private(set) var operations: OperationsPayload?
    @Published private(set) var instrumentSummaries: [String: InstrumentSummaryPayload] = [:]
    @Published private(set) var quicklooksByKey: [String: QuicklooksPayload] = [:]
    @Published private(set) var wxcamByKey: [String: WXcamPayload] = [:]
    @Published private(set) var isRefreshing = false
    @Published private(set) var lastError: String?
    @Published private(set) var healthStatus: String?
    @Published var selectedInstrumentID = "power"

    init(configuration: AppConfiguration) {
        let savedURL = UserDefaults.standard.string(forKey: baseURLDefaultsKey)
        let resolvedURL = savedURL.flatMap(URL.init(string:)) ?? configuration.apiBaseURL
        self.configuration = AppConfiguration(
            apiBaseURL: resolvedURL,
            documentationURL: configuration.documentationURL,
            projectURL: configuration.projectURL,
            minimumRefreshInterval: configuration.minimumRefreshInterval
        )
        self.baseURLString = resolvedURL.absoluteString
        self.apiToken = KeychainTokenStore.readToken() ?? ""
    }

    var client: MobileAPIClient {
        MobileAPIClient(baseURL: configuration.apiBaseURL, token: apiToken.nilIfBlank)
    }

    var visibleInstruments: [InstrumentDescriptor] {
        manifest.instruments.filter(\.visible)
    }

    func refreshAllIfNeeded() async {
        guard !hasLoadedInitialData else {
            return
        }
        hasLoadedInitialData = true
        await refreshAll()
    }

    func refreshAll() async {
        isRefreshing = true
        lastError = nil
        defer { isRefreshing = false }

        do {
            manifest = try await client.get("manifest")
            operations = try await client.get("operations")
            await refreshInstrument(id: selectedInstrumentID)
            await refreshQuicklooks(kind: "science", instrumentID: selectedInstrumentID)
            await refreshWXcam(stream: "fish_hdr", day: "latest")
        } catch {
            lastError = error.localizedDescription
        }
    }

    func refreshOperations() async {
        do {
            operations = try await client.get("operations")
            lastError = nil
        } catch {
            lastError = error.localizedDescription
        }
    }

    func refreshInstrument(id: String, window: String = "24h") async {
        selectedInstrumentID = id
        do {
            let response: InstrumentSummaryPayload = try await client.get(
                "instruments/\(id)/summary",
                queryItems: [URLQueryItem(name: "window", value: window)]
            )
            instrumentSummaries[id] = response
            lastError = nil
        } catch {
            lastError = error.localizedDescription
        }
    }

    func refreshQuicklooks(kind: String, instrumentID: String) async {
        do {
            let response: QuicklooksPayload = try await client.get(
                "quicklooks",
                queryItems: [
                    URLQueryItem(name: "kind", value: kind),
                    URLQueryItem(name: "instrument", value: instrumentID)
                ]
            )
            quicklooksByKey[quicklookKey(kind: kind, instrumentID: instrumentID)] = response
            lastError = nil
        } catch {
            lastError = error.localizedDescription
        }
    }

    func quicklooks(kind: String, instrumentID: String) -> QuicklooksPayload? {
        quicklooksByKey[quicklookKey(kind: kind, instrumentID: instrumentID)]
    }

    func refreshWXcam(stream: String, day: String) async {
        do {
            let response: WXcamPayload = try await client.get(
                "wxcam",
                queryItems: [
                    URLQueryItem(name: "stream", value: stream),
                    URLQueryItem(name: "day", value: day)
                ]
            )
            wxcamByKey[wxcamKey(stream: stream, day: day)] = response
            lastError = nil
        } catch {
            lastError = error.localizedDescription
        }
    }

    func wxcam(stream: String, day: String) -> WXcamPayload? {
        wxcamByKey[wxcamKey(stream: stream, day: day)]
    }

    func saveSettings() {
        guard let url = URL(string: baseURLString.trimmed), url.scheme != nil else {
            healthStatus = "Invalid API URL"
            return
        }
        configuration.apiBaseURL = url
        baseURLString = url.absoluteString
        UserDefaults.standard.set(url.absoluteString, forKey: baseURLDefaultsKey)
        KeychainTokenStore.saveToken(apiToken)
        healthStatus = "Saved"
    }

    func checkHealth() async {
        saveSettings()
        do {
            let health: HealthPayload = try await client.get("health")
            healthStatus = "API \(health.status)"
            lastError = nil
        } catch {
            healthStatus = error.localizedDescription
        }
    }

    func clearCachedResponses() {
        operations = nil
        instrumentSummaries = [:]
        quicklooksByKey = [:]
        wxcamByKey = [:]
        hasLoadedInitialData = false
        healthStatus = "Cleared"
    }

    func absoluteURL(_ value: String?) -> URL? {
        client.absoluteURL(for: value)
    }

    func imageData(from url: URL) async throws -> Data {
        try await client.data(from: url)
    }

    func authenticatedHeaders() -> [String: String] {
        client.authenticatedHeaders()
    }

    private func quicklookKey(kind: String, instrumentID: String) -> String {
        "\(kind):\(instrumentID)"
    }

    private func wxcamKey(stream: String, day: String) -> String {
        "\(stream):\(day)"
    }
}

enum KeychainTokenStore {
    private static let service = "uk.co.gamb2le.AuroraDashboard.mobileAPI"
    private static let account = "bearerToken"

    static func readToken() -> String? {
        let query: [String: Any] = [
            kSecClass as String: kSecClassGenericPassword,
            kSecAttrService as String: service,
            kSecAttrAccount as String: account,
            kSecReturnData as String: true,
            kSecMatchLimit as String: kSecMatchLimitOne
        ]
        var item: CFTypeRef?
        let status = SecItemCopyMatching(query as CFDictionary, &item)
        guard status == errSecSuccess, let data = item as? Data else {
            return nil
        }
        return String(data: data, encoding: .utf8)
    }

    static func saveToken(_ token: String) {
        let baseQuery: [String: Any] = [
            kSecClass as String: kSecClassGenericPassword,
            kSecAttrService as String: service,
            kSecAttrAccount as String: account
        ]

        if token.trimmed.isEmpty {
            SecItemDelete(baseQuery as CFDictionary)
            return
        }

        let data = Data(token.utf8)
        let update = [kSecValueData as String: data]
        let status = SecItemUpdate(baseQuery as CFDictionary, update as CFDictionary)
        if status == errSecItemNotFound {
            var addQuery = baseQuery
            addQuery[kSecValueData as String] = data
            SecItemAdd(addQuery as CFDictionary, nil)
        }
    }
}

extension String {
    var trimmed: String {
        trimmingCharacters(in: .whitespacesAndNewlines)
    }

    var nilIfBlank: String? {
        let value = trimmed
        return value.isEmpty ? nil : value
    }
}
