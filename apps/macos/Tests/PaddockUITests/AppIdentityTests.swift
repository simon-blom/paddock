import Foundation
import Testing

@Suite("Paddock product identity")
struct AppIdentityTests {
  private var repository: URL {
    URL(fileURLWithPath: #filePath).deletingLastPathComponent()
      .deletingLastPathComponent().deletingLastPathComponent()
      .deletingLastPathComponent().deletingLastPathComponent()
  }

  @Test func packagingUsesOneStableProductIdentity() throws {
    try verifyIdentity(repository.appending(path: "packaging/macos/Info.plist"))
    let script = try String(
      contentsOf: repository.appending(path: "apps/macos/scripts/build-app.sh"), encoding: .utf8)
    #expect(script.contains("--install) bundle=\"/Applications/Paddock.app\""))
    #expect(script.contains("\"$bundle/Contents/MacOS/Paddock\""))
    #expect(!script.contains("-test)"))
    #expect(!script.contains("plutil -replace CFBundleIdentifier"))
    #expect(script.contains("Helpers/paddock-runner"), "Refuse to replace a running bundled runner")
  }

  @Test(.enabled(if: ProcessInfo.processInfo.environment["PADDOCK_APP_BUNDLE"] != nil))
  func installedAppMatchesItsPackagingContract() throws {
    let path = try #require(ProcessInfo.processInfo.environment["PADDOCK_APP_BUNDLE"])
    let bundle = URL(fileURLWithPath: path, isDirectory: true)
    #expect(bundle.lastPathComponent == "Paddock.app")
    try verifyIdentity(bundle.appending(path: "Contents/Info.plist"))
    for executable in ["Contents/MacOS/Paddock", "Contents/Helpers/paddock-runner"] {
      #expect(FileManager.default.isExecutableFile(atPath: bundle.appending(path: executable).path))
    }
    #expect(
      !FileManager.default.fileExists(
        atPath: bundle.appending(path: "Contents/MacOS/PaddockMac").path))
    #expect(
      FileManager.default.fileExists(
        atPath: bundle.appending(path: "Contents/Frameworks/libpaddock_desktop.dylib").path))
  }

  private func verifyIdentity(_ url: URL) throws {
    let plist = try #require(
      PropertyListSerialization.propertyList(from: Data(contentsOf: url), format: nil)
        as? [String: Any])
    #expect(plist["CFBundleIdentifier"] as? String == "io.truespar.paddock")
    #expect(plist["CFBundleExecutable"] as? String == "Paddock")
    #expect(plist["CFBundleName"] as? String == "Paddock")
    #expect(plist["CFBundleDisplayName"] as? String == "Paddock")
    #expect((plist["NSMicrophoneUsageDescription"] as? String)?.isEmpty == false)
  }
}
