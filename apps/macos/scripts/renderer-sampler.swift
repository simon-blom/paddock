// Standalone diagnostic helper. Public libproc API; no private WKWebView PID selector.
import Darwin
import Foundation

let host = Int32(CommandLine.arguments[1])!
let excluded = Set(CommandLine.arguments[2].split(separator: ",").compactMap { Int32($0) })
var timebase = mach_timebase_info_data_t()
mach_timebase_info(&timebase)
let nanosPerTick = Double(timebase.numer) / Double(timebase.denom)
var missingHost = 0
let start = Date()
while Date().timeIntervalSince(start) < 900 {
  let count = proc_listallpids(nil, 0)
  var pids = [Int32](repeating: 0, count: Int(count) + 256)
  let actual = pids.withUnsafeMutableBytes { proc_listallpids($0.baseAddress, Int32($0.count)) }
  var rows: [[String: Any]] = []
  var hostFound = false
  for pid in pids.prefix(Int(max(0, actual))) where pid > 0 {
    var path = [CChar](repeating: 0, count: 4096)
    guard proc_pidpath(pid, &path, UInt32(path.count)) > 0 else { continue }
    let executable = String(cString: path)
    let isHost = pid == host
    let candidate =
      !excluded.contains(pid) && executable.hasPrefix("/System/")
      && executable.contains("/com.apple.WebKit.")
    guard isHost || candidate else { continue }
    var usage = rusage_info_v4()
    let result = withUnsafeMutablePointer(to: &usage) {
      $0.withMemoryRebound(to: rusage_info_t?.self, capacity: 1) {
        proc_pid_rusage(pid, RUSAGE_INFO_V4, $0)
      }
    }
    guard result == 0 else { continue }
    hostFound = hostFound || isHost
    rows.append([
      "pid": pid, "role": isHost ? "host" : "launch-delta-candidate",
      "name": URL(fileURLWithPath: executable).lastPathComponent,
      "startAbstime": usage.ri_proc_start_abstime,
      "footprint": usage.ri_phys_footprint, "peakFootprint": usage.ri_lifetime_max_phys_footprint,
      "resident": usage.ri_resident_size,
      "cpuUserNs": Double(usage.ri_user_time) * nanosPerTick,
      "cpuSystemNs": Double(usage.ri_system_time) * nanosPerTick,
    ])
  }
  let record: [String: Any] = [
    "at": Date().timeIntervalSince1970 * 1000, "nanosPerMachTick": nanosPerTick, "processes": rows,
  ]
  let data = try JSONSerialization.data(withJSONObject: record, options: [.sortedKeys])
  FileHandle.standardOutput.write(data + Data([10]))
  // A guardrail, not a workload pass. The orchestrator stops only its own app.
  if rows.contains(where: { ($0["footprint"] as? UInt64 ?? 0) > 8 * 1024 * 1024 * 1024 }) {
    exit(2)
  }
  missingHost = hostFound ? 0 : missingHost + 1
  if missingHost >= 10 { break }
  Thread.sleep(forTimeInterval: 0.5)
}
