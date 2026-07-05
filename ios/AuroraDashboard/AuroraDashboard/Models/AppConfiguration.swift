import Foundation

struct AppConfiguration: Equatable {
    var dashboardBaseURL: URL?
    var documentationURL: URL
    var projectURL: URL
    var minimumRefreshInterval: TimeInterval

    static let `default` = AppConfiguration(
        dashboardBaseURL: nil,
        documentationURL: URL(string: "https://gamb2le.pages.dev/documentation-docs/")!,
        projectURL: URL(string: "https://www.gamb2le.co.uk/")!,
        minimumRefreshInterval: 60
    )

    func dashboardURL(for section: DashboardSection) -> URL? {
        guard let dashboardBaseURL, let queryValue = section.dashboardQueryValue else {
            return nil
        }

        return dashboardBaseURL.appending(queryItems: [
            URLQueryItem(name: "tab", value: queryValue)
        ])
    }
}
