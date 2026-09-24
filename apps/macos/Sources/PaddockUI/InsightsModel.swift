import Foundation
import Observation
import PaddockClient

enum InsightPage: String, CaseIterable, Identifiable {
  case usage = "Usage"
  case activity = "Activity"
  case cache = "KV offloading"
  var id: Self { self }
}

@MainActor @Observable
final class InsightsModel {
  var page: InsightPage = .usage
  var port: UInt16?
  var days = 1
  var live = true
  private(set) var usage: UsageHistorySnapshot?
  private(set) var activity: ActivitySnapshot?
  private(set) var cache: CacheSnapshot?
  private(set) var error: String?
  private(set) var sampledAt: Date?
  private(set) var loading = false
  @ObservationIgnored private let client: any ManagerLoading
  @ObservationIgnored private var revision = 0
  init(client: any ManagerLoading) { self.client = client }
  var queryID: String { "\(page):\(port ?? 0):\(days):\(live)" }

  func observe() async {
    revision += 1
    let ownRevision = revision
    usage = nil
    activity = nil
    cache = nil
    sampledAt = nil
    repeat {
      loading = true
      do {
        let now = Int64(Date().timeIntervalSince1970 * 1000)
        switch page {
        case .usage:
          let value = try await client.inspect(
            .usage(from: now - Int64(days) * 86_400_000, to: now, port: port),
            as: UsageHistorySnapshot.self)
          guard !Task.isCancelled, ownRevision == revision else { return }
          usage = value
        case .activity:
          let value = try await client.inspect(.activity(port: port), as: ActivitySnapshot.self)
          guard !Task.isCancelled, ownRevision == revision else { return }
          activity = value
        case .cache:
          let value = try await client.inspect(.cache, as: CacheSnapshot.self)
          guard !Task.isCancelled, ownRevision == revision else { return }
          cache = value
        }
        error = nil
        sampledAt = Date()
      } catch {
        guard !Task.isCancelled, ownRevision == revision else { return }
        self.error = error.localizedDescription
      }
      loading = false
      guard live else { return }
      do { try await Task.sleep(for: .seconds(page == .usage ? 10 : 3)) } catch { return }
    } while !Task.isCancelled && ownRevision == revision
  }
}
