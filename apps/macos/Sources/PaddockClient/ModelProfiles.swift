import Foundation

public struct ModelProfiles: Decodable, Sendable {
  public let profiles: [ModelProfile]
}
public struct ModelProfile: Decodable, Sendable, Identifiable {
  public let id, name, model, artifact: String
  public let settings: Values
  public struct Values: Decodable, Sendable {
    public let maxCtx, maxBatch: Int?
    public let spec, kvCacheDtype: String?
    public let noSpec: Bool?
    public let vramBudget: Int?
    public let kvOffload: EndpointKVOffload?
    public let runtimeOptions: [EndpointRuntimeOption]?
  }
}
