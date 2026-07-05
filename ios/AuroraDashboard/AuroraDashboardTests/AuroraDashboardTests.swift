import XCTest
@testable import AuroraDashboard

final class AuroraDashboardTests: XCTestCase {
    func testDashboardSectionsMatchStarterTabs() {
        XCTAssertEqual(
            DashboardSection.allCases.map(\.title),
            ["Operations", "Interactive", "Quicklooks", "WXcam", "Settings"]
        )
    }

    func testDefaultConfigurationIsOfflineFirst() {
        let configuration = AppConfiguration.default

        XCTAssertNil(configuration.dashboardBaseURL)
        XCTAssertEqual(configuration.documentationURL.host(), "gamb2le.pages.dev")
        XCTAssertEqual(configuration.projectURL.host(), "www.gamb2le.co.uk")
        XCTAssertEqual(configuration.minimumRefreshInterval, 60)
    }

    func testDashboardURLKeepsUnconfiguredEndpointNil() {
        XCTAssertNil(AppConfiguration.default.dashboardURL(for: .operations))
        XCTAssertNil(AppConfiguration.default.dashboardURL(for: .settings))
    }

    func testDashboardURLAddsTabQueryForConfiguredSections() {
        let configuration = AppConfiguration(
            dashboardBaseURL: URL(string: "https://dashboard.example.test/")!,
            documentationURL: AppConfiguration.default.documentationURL,
            projectURL: AppConfiguration.default.projectURL,
            minimumRefreshInterval: 60
        )

        XCTAssertEqual(
            configuration.dashboardURL(for: .operations)?.absoluteString,
            "https://dashboard.example.test/?tab=operations"
        )
        XCTAssertNil(configuration.dashboardURL(for: .settings))
    }
}
