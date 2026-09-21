import Foundation
import WebKit

/// No general filesystem bridge: only immutable, bundled renderer resources are addressable.
public struct BundledAssets: Sendable {
  public enum Surface: String, Sendable {
    case lab = "paddock-lab"
    case transcript = "paddock-render"
  }
  public let surface: Surface
  public var origin: String { "\(surface.rawValue)://bundle" }
  public let root: URL
  public init(root: URL, surface: Surface) {
    self.root = root.resolvingSymlinksInPath().standardizedFileURL
    self.surface = surface
  }

  public func resource(for url: URL) -> URL? {
    guard url.scheme == surface.rawValue, url.host == "bundle", url.port == nil,
      url.user == nil, url.password == nil,
      let decoded = URLComponents(url: url, resolvingAgainstBaseURL: false)?.percentEncodedPath
        .removingPercentEncoding,
      !decoded.contains("\0"), !decoded.contains("\\"),
      !decoded.split(separator: "/").contains("..")
    else { return nil }
    let path = decoded == "/" ? "index.html" : String(decoded.drop(while: { $0 == "/" }))
    // Resolve existing parent links even when the leaf is absent. Resolving the
    // whole URL in one step can leave those links unresolved for a missing leaf.
    var file = root
    for component in path.split(separator: "/") {
      file =
        file.appendingPathComponent(String(component)).resolvingSymlinksInPath().standardizedFileURL
    }
    guard file.path.hasPrefix(root.path + "/"), Self.mime[file.pathExtension] != nil else {
      return nil
    }
    return file
  }

  public static let mime = [
    "html": "text/html", "js": "text/javascript", "css": "text/css",
    "wasm": "application/wasm", "json": "application/json", "svg": "image/svg+xml",
    "png": "image/png", "woff2": "font/woff2", "woff": "font/woff", "ttf": "font/ttf",
    "pdf": "application/pdf",
    "docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
  ]

  public var contentSecurityPolicy: String { Self.policy(surface: surface) }
  public static func policy(surface: Surface) -> String {
    let origin = surface.rawValue + ":"
    let frames = surface == .lab ? "blob:" : "'none'"
    return """
      default-src 'none'; script-src \(origin) blob: 'wasm-unsafe-eval'; \
      style-src \(origin) 'unsafe-inline'; img-src \(origin) data: blob:; \
      font-src \(origin) data: blob:; connect-src \(origin) blob:; \
      worker-src \(origin) blob:; frame-src \(frames); base-uri 'none'; form-action 'none';
      """
  }
}

@MainActor
public final class BundledSchemeHandler: NSObject, WKURLSchemeHandler {
  let assets: BundledAssets
  private var jobs: [ObjectIdentifier: Task<Void, Never>] = [:]
  public init(assets: BundledAssets) { self.assets = assets }

  public func webView(_ webView: WKWebView, start task: any WKURLSchemeTask) {
    let key = ObjectIdentifier(task)
    guard task.request.httpMethod == "GET", let url = task.request.url,
      let file = assets.resource(for: url)
    else {
      task.didFailWithError(URLError(.noPermissionsToReadFile))
      return
    }
    jobs[key] = Task {
      // Large WASM reads must not block the app's main thread. Cancellation means no
      // subsequent scheme callbacks (WebKit throws if a stopped task receives data).
      let result = await Task.detached {
        Result { try Data(contentsOf: file, options: .mappedIfSafe) }
      }.value
      guard !Task.isCancelled else { return }
      defer { jobs[key] = nil }
      switch result {
      case .success(let data):
        let headers = [
          "Content-Type": BundledAssets.mime[file.pathExtension] ?? "application/octet-stream",
          "Content-Length": String(data.count),
          "Content-Security-Policy": assets.contentSecurityPolicy,
          "X-Content-Type-Options": "nosniff",
          "Cache-Control": "no-store",
          // A custom-scheme worker may have an opaque origin. Resources here are public
          // bundle assets only; there is no authenticated or user-content endpoint.
          "Access-Control-Allow-Origin": "*",
        ]
        task.didReceive(
          HTTPURLResponse(
            url: url, statusCode: 200, httpVersion: "HTTP/1.1", headerFields: headers)!)
        task.didReceive(data)
        task.didFinish()
      case .failure(let error): task.didFailWithError(error)
      }
    }
  }
  public func webView(_ webView: WKWebView, stop task: any WKURLSchemeTask) {
    jobs.removeValue(forKey: ObjectIdentifier(task))?.cancel()
  }
}
