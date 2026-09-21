import Foundation

enum NativeComparePresentation {
  typealias O = [String: ConversationValue]
  /// Mirrors compare-presentation.ts. A chat answer's brevity is not a race.
  /// Never award a badge while another lane is unfinished, failed or sharing GPU.
  static func fastest(_ messages: [O]) -> String? {
    guard messages.count >= 2,
      !messages.contains(where: {
        $0["streaming"]?.bool == true || $0["stopped"]?.bool == true
          || $0["error"]?.string?.isEmpty == false || $0["run"]?["contended"]?.bool == true
      })
    else { return nil }
    let raced = messages.filter { $0["transcript"]?.object != nil }
    guard raced.count >= 2,
      raced.allSatisfy({
        ($0["usage"]?["ms"]?.double ?? 0) > 0 && ($0["usage"]?["ms"]?.double?.isFinite == true)
      }),
      let best = raced.min(by: { $0["usage"]!["ms"]!.double! < $1["usage"]!["ms"]!.double! }),
      let id = best["id"]?.string
    else { return nil }
    let time = best["usage"]!["ms"]!.double!
    return raced.filter { $0["id"]?.string != id }.allSatisfy {
      $0["usage"]!["ms"]!.double! > time * 1.1
    } ? id : nil
  }
}
