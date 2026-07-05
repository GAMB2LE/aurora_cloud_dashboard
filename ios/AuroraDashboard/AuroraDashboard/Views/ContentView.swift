import SwiftUI

struct ContentView: View {
    let configuration: AppConfiguration

    var body: some View {
        TabView {
            OperationsView(configuration: configuration)
                .tabItem {
                    Label(DashboardSection.operations.title, systemImage: DashboardSection.operations.systemImage)
                }

            InteractiveDataView(configuration: configuration)
                .tabItem {
                    Label(DashboardSection.interactive.title, systemImage: DashboardSection.interactive.systemImage)
                }

            QuicklooksView(configuration: configuration)
                .tabItem {
                    Label(DashboardSection.quicklooks.title, systemImage: DashboardSection.quicklooks.systemImage)
                }

            WXcamView(configuration: configuration)
                .tabItem {
                    Label(DashboardSection.wxcam.title, systemImage: DashboardSection.wxcam.systemImage)
                }

            SettingsView(configuration: configuration)
                .tabItem {
                    Label(DashboardSection.settings.title, systemImage: DashboardSection.settings.systemImage)
                }
        }
    }
}
