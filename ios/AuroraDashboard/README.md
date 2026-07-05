# Aurora Dashboard iOS

Native SwiftUI app for the Aurora Cloud Dashboard.

The app uses the read-only Aurora mobile API. It does not embed the existing
Panel dashboard in a WebView and it does not read Zarr, SQLite, or filesystem
products directly from iOS.

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

## API Configuration

Default API base URL:

```text
https://data-ocean.gamb2le.co.uk/mobile/v1
```

The base URL can be edited in Settings. The bearer token is stored in Keychain
and sent as:

```text
Authorization: Bearer <token>
```

The matching backend lives in `mobile_api.py`; runtime notes are in
`docs/runtime/mobile-api.md`.

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

- Operations: live overall health, stream status, root-cause groups, active
  alerts, and compact trend cards.
- Interactive: manifest-driven instrument list and mobile summaries backed by
  generated quicklooks.
- Quicklooks: science/housekeeping browser with instrument/date selectors and
  authenticated image loading.
- WXcam: FISH HDR and PANO HDR day selection, stitched MP4 playback, and hourly
  thumbnails.
- Settings: API base URL, Keychain token, health check, response-cache reset,
  and documentation/project links.

## Integration Notes

The existing Python Panel app remains unchanged. The native app talks to the
mobile API contract rather than scraping Panel UI state.
