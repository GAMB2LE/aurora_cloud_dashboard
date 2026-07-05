import Foundation

enum DashboardSection: String, CaseIterable, Identifiable {
    case operations
    case interactive
    case quicklooks
    case wxcam
    case settings

    var id: String { rawValue }

    var title: String {
        switch self {
        case .operations:
            return "Operations"
        case .interactive:
            return "Interactive"
        case .quicklooks:
            return "Quicklooks"
        case .wxcam:
            return "WXcam"
        case .settings:
            return "Settings"
        }
    }

    var systemImage: String {
        switch self {
        case .operations:
            return "gauge.with.dots.needle.bottom.50percent"
        case .interactive:
            return "chart.xyaxis.line"
        case .quicklooks:
            return "photo.on.rectangle.angled"
        case .wxcam:
            return "video"
        case .settings:
            return "gearshape"
        }
    }

    var dashboardQueryValue: String? {
        switch self {
        case .operations:
            return "operations"
        case .interactive:
            return "interactive"
        case .quicklooks:
            return "science"
        case .wxcam:
            return "interactive"
        case .settings:
            return nil
        }
    }
}
