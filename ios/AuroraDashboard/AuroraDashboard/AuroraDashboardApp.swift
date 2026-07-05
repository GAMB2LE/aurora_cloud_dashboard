import SwiftUI

@main
struct AuroraDashboardApp: App {
    @StateObject private var store = DashboardStore(configuration: .default)

    var body: some Scene {
        WindowGroup {
            ContentView(store: store)
        }
    }
}
