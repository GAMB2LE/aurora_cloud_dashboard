import SwiftUI

struct ContentView: View {
    @ObservedObject var store: DashboardStore

    var body: some View {
        TabView {
            OperationsView(store: store)
                .tabItem {
                    Label(DashboardSection.operations.title, systemImage: DashboardSection.operations.systemImage)
                }

            InteractiveDataView(store: store)
                .tabItem {
                    Label(DashboardSection.interactive.title, systemImage: DashboardSection.interactive.systemImage)
                }

            QuicklooksView(store: store)
                .tabItem {
                    Label(DashboardSection.quicklooks.title, systemImage: DashboardSection.quicklooks.systemImage)
                }

            WXcamView(store: store)
                .tabItem {
                    Label(DashboardSection.wxcam.title, systemImage: DashboardSection.wxcam.systemImage)
                }

            SettingsView(store: store)
                .tabItem {
                    Label(DashboardSection.settings.title, systemImage: DashboardSection.settings.systemImage)
                }
        }
        .task {
            await store.refreshAllIfNeeded()
        }
    }
}
