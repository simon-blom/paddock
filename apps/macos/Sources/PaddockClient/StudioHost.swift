import Foundation

/// Native-only bootstrap. This is an ephemeral UI session, not a runner key.
/// Native transport supplies the session only to this exact origin. Legacy
/// viewer hosts install it as HttpOnly; never expose it to JavaScript or logs.
public struct StudioHost: Decodable, Sendable {
  public let origin: URL
  public let cookieName: String
  public let session: String
  public var isValidPrivateHost: Bool {
    origin.scheme == "http" && origin.host == "127.0.0.1"
      && origin.port.map { $0 > 0 && $0 <= 65535 } == true
      && origin.user == nil && origin.password == nil
      && ["", "/"].contains(origin.path) && origin.query == nil && origin.fragment == nil
      && cookieName == "paddock_desktop_session" && session.count == 64
      && session.utf8.allSatisfy {
        (48...57).contains($0) || (65...70).contains($0) || (97...102).contains($0)
      }
  }
}
