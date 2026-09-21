import Foundation

/// AppHeader.vue's shared presentation. Names, target identity and capability
/// badges come from the same store projection, not native model heuristics.
public struct StudioModelHeader: Decodable, Sendable, Equatable {
  public struct Option: Decodable, Sendable, Identifiable, Equatable {
    public let value: String
    public let label: String
    public let hint: String
    public let vendor: String
    public let title: String
    public let available: Bool
    public var id: String { value }
  }
  public struct Lane: Decodable, Sendable, Identifiable, Equatable {
    public let id: String
    public let label: String
    public let vendor: String
    public let spec: String
  }
  public let currentModel: String
  public let pickerOptions: [Option]
  public let compareLanes: [Lane]
  public let comparing: Bool
  public let specLabel: String
  public let isVision: Bool
  public let soleEncoder: String?

  public var current: Option? { pickerOptions.first { $0.value == currentModel } }
}
