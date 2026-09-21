import Foundation
import PaddockWebAssets

/// Lab keeps its isolated origin; both hosts share the same audited resource
/// resolver and off-main-thread scheme loader. No diagnostic bridge ships in
/// the production transcript target.
public struct LabAssets: Sendable {
  public static let origin = "paddock-lab://bundle"
  let bundled: BundledAssets
  public var root: URL { bundled.root }
  public init(root: URL) { bundled = BundledAssets(root: root, surface: .lab) }
  public func resource(for url: URL) -> URL? { bundled.resource(for: url) }
  public static let contentSecurityPolicy = BundledAssets.policy(surface: .lab)
}
typealias LabSchemeHandler = BundledSchemeHandler
