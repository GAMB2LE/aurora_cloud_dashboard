# Aurora Dashboard iOS

Native SwiftUI starter for the Aurora Cloud Dashboard.

This project is a scaffold. It mirrors the current Panel dashboard navigation in native iOS tabs, but it does not embed the existing dashboard in a WebView and it does not fetch live dashboard data yet.

## Open

```bash
open ios/AuroraDashboard/AuroraDashboard.xcodeproj
```

## Build From The Command Line

```bash
xcodebuild \
  -project ios/AuroraDashboard/AuroraDashboard.xcodeproj \
  -scheme AuroraDashboard \
  -destination 'generic/platform=iOS Simulator' \
  -derivedDataPath /private/tmp/AuroraDashboardDerivedData \
  CODE_SIGNING_ALLOWED=NO \
  build
```

## Install On A Physical iPhone

Use Xcode automatic signing with the `AuroraDashboard` target:

- Bundle identifier: `uk.co.gamb2le.AuroraDashboard`
- Team: Ryan Neely Personal Team (`D863HTPFQC`)
- Provisioning profile: Xcode managed iOS development profile

Once Xcode has created the provisioning profile for the connected device, the app can be built and installed with:

```bash
xcodebuild \
  -project ios/AuroraDashboard/AuroraDashboard.xcodeproj \
  -scheme AuroraDashboard \
  -destination 'id=<device-identifier>' \
  -derivedDataPath /private/tmp/AuroraDashboardDeviceDerivedData \
  -allowProvisioningUpdates \
  -allowProvisioningDeviceRegistration \
  PRODUCT_BUNDLE_IDENTIFIER=uk.co.gamb2le.AuroraDashboard \
  DEVELOPMENT_TEAM=D863HTPFQC \
  CODE_SIGN_STYLE=Automatic \
  build

xcrun devicectl device install app \
  --device <device-identifier> \
  /private/tmp/AuroraDashboardDeviceDerivedData/Build/Products/Debug-iphoneos/AuroraDashboard.app
```

The first verified install was on `Overman` as `Aurora Dashboard` version `0.1`, side by side with the existing `UK WSR` app.

## Current Native Surface

- Operations: placeholders for traffic-light health, root-cause groups, and stream health.
- Interactive: instrument list matching the dashboard's interactive data browser.
- Quicklooks: science and housekeeping quicklook navigation placeholders.
- WXcam: FISH HDR and PANO HDR media browser placeholders.
- Settings: endpoint configuration placeholder plus documentation and project links.

## Future Integration Notes

The existing Python Panel app remains unchanged. A live native app should be wired through a small mobile API or static manifest rather than scraping Panel UI state. The starter configuration keeps `dashboardBaseURL` unset until that endpoint exists.
