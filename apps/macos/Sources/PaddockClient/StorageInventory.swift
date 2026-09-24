import Foundation

public struct StorageInventory: Decodable, Sendable {
  public let artifacts: [Artifact]
  public struct Artifact: Decodable, Sendable, Identifiable {
    public let id, model, artifact: String
    public let bytes: UInt64?
    public let presentFiles, totalFiles: Int
    public let path: String?
    public let configuredPorts: [UInt16]
  }
}
