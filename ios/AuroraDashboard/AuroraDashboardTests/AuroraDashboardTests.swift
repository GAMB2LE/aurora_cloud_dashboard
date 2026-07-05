import XCTest
@testable import AuroraDashboard

final class AuroraDashboardTests: XCTestCase {
    func testDashboardSectionsMatchNativeTabs() {
        XCTAssertEqual(
            DashboardSection.allCases.map(\.title),
            ["Operations", "Interactive", "Quicklooks", "WXcam", "Settings"]
        )
    }

    func testDefaultConfigurationPointsAtMobileAPI() {
        let configuration = AppConfiguration.default

        XCTAssertEqual(configuration.apiBaseURL.absoluteString, "https://data-ocean.gamb2le.co.uk/mobile/v1")
        XCTAssertEqual(configuration.documentationURL.host(), "gamb2le.pages.dev")
        XCTAssertEqual(configuration.projectURL.host(), "www.gamb2le.co.uk")
        XCTAssertEqual(configuration.minimumRefreshInterval, 60)
    }

    func testFallbackManifestHasWorkingAppSections() {
        let manifest = MobileManifest.fallback

        XCTAssertEqual(manifest.sections.map(\.id), ["operations", "interactive", "quicklooks", "wxcam", "settings"])
        XCTAssertTrue(manifest.instruments.contains { $0.id == "power" && $0.supportsSummary })
        XCTAssertTrue(manifest.wxcamStreams.contains { $0.id == "fish_hdr" })
    }

    func testMobileAPIClientResolvesRelativeMediaURLsAgainstBaseURL() {
        let client = MobileAPIClient(baseURL: URL(string: "https://example.test/mobile/v1")!, token: "token")

        XCTAssertEqual(
            client.absoluteURL(for: "/media/quicklook/science/power/latest")?.absoluteString,
            "https://example.test/mobile/v1/media/quicklook/science/power/latest"
        )
        XCTAssertEqual(
            client.authenticatedHeaders()["Authorization"],
            "Bearer token"
        )
    }

    func testOperationsPayloadDecodesCompactResponse() throws {
        let data = """
        {
          "serverTime": "2026-07-05T07:30:00Z",
          "updatedAt": "2026-07-05T07:29:00Z",
          "overallLevel": "green",
          "summary": "All visible stream groups are healthy",
          "checkCounts": {"green": 7, "red": 0},
          "streamStates": [
            {"id": "power", "title": "Aurora Power Supply", "level": "green", "detail": "Source and processing services healthy"}
          ],
          "rootCauseGroups": [
            {"id": "source", "title": "Source freshness", "level": "green", "detail": "No source freshness issues"}
          ],
          "alerts": [],
          "trendCards": [
            {"id": "storage", "title": "Worst storage use", "value": 55.2, "unit": "%", "level": "green"}
          ]
        }
        """.data(using: .utf8)!

        let payload = try JSONDecoder().decode(OperationsPayload.self, from: data)

        XCTAssertEqual(payload.overallLevel, "green")
        XCTAssertEqual(payload.streamStates.first?.id, "power")
        XCTAssertEqual(payload.trendCards.first?.value, 55.2)
    }
}
