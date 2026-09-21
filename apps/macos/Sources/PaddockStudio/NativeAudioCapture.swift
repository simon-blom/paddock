@preconcurrency import AVFoundation
import CoreMedia
import Foundation
import PaddockClient

/// AVFoundation delivers on a private serial queue, never the main thread.
/// That queue owns the WAV original; a bounded stream feeds native sockets.
/// An overrun is an explicit error, never silently missing PCM in one lane.
final class NativeAudioCapture: NSObject, AVCaptureAudioDataOutputSampleBufferDelegate,
  @unchecked Sendable
{
  struct Frame: Sendable {
    let pcm: Data
    let elapsed: Double
  }
  private let meter: StudioMicrophoneMeter
  private var spectrum = NativeMicrophoneSpectrum()
  private let control = DispatchQueue(label: "io.truespar.paddock.capture.control")
  private let samples = DispatchQueue(
    label: "io.truespar.paddock.capture.samples", qos: .userInitiated)
  private let session = AVCaptureSession()
  private let output = AVCaptureAudioDataOutput()
  private var sink: AsyncThrowingStream<Frame, Error>.Continuation?
  private var file: FileHandle?
  private var bytes = 0
  private var limitBytes = maximumBytes
  private var failed: (any Error)?
  let url = FileManager.default.temporaryDirectory.appending(
    path: "Paddock-recording-\(UUID().uuidString).wav")
  static let maximumBytes = 100 * 1024 * 1024 - 44

  init(meter: StudioMicrophoneMeter) {
    self.meter = meter
    super.init()
  }

  static func devices() -> [[String: String]] {
    AVCaptureDevice.DiscoverySession(
      deviceTypes: [.microphone], mediaType: .audio, position: .unspecified
    ).devices.map { ["id": $0.uniqueID, "label": $0.localizedName] }
  }
  func start(deviceID: String, maximumSeconds: Double = Double(maximumBytes) / 32000) async throws
    -> AsyncThrowingStream<Frame, Error>
  {
    guard maximumSeconds.isFinite, maximumSeconds > 0 else {
      throw ManagerError.core("Invalid recording limit")
    }
    let requestedBytes = Int(min(Double(Self.maximumBytes), maximumSeconds * 32000)) / 2 * 2
    let (stream, continuation) = AsyncThrowingStream<Frame, Error>.makeStream(
      bufferingPolicy: .bufferingOldest(32))
    try await withCheckedThrowingContinuation { (started: CheckedContinuation<Void, Error>) in
      control.async { [self] in
        do {
          guard AVCaptureDevice.authorizationStatus(for: .audio) == .authorized else {
            throw ManagerError.core("Microphone access is not granted")
          }
          let available = AVCaptureDevice.DiscoverySession(
            deviceTypes: [.microphone], mediaType: .audio, position: .unspecified
          ).devices
          let device =
            deviceID.isEmpty
            ? AVCaptureDevice.default(for: .audio) : available.first { $0.uniqueID == deviceID }
          guard let device else {
            throw ManagerError.core(
              "The selected microphone is disconnected. Choose an available input.")
          }
          let input = try AVCaptureDeviceInput(device: device)
          session.beginConfiguration()
          var configuring = true
          defer { if configuring { session.commitConfiguration() } }
          guard session.canAddInput(input), session.canAddOutput(output) else {
            throw ManagerError.core("The microphone cannot be configured")
          }
          session.addInput(input)
          session.addOutput(output)
          output.audioSettings = [
            AVFormatIDKey: kAudioFormatLinearPCM, AVSampleRateKey: 16000,
            AVNumberOfChannelsKey: 1, AVLinearPCMBitDepthKey: 16,
            AVLinearPCMIsFloatKey: false, AVLinearPCMIsBigEndianKey: false,
            AVLinearPCMIsNonInterleaved: false,
          ]
          guard
            FileManager.default.createFile(
              atPath: url.path, contents: Self.wavHeader(bytes: 0),
              attributes: [.posixPermissions: 0o600])
          else { throw ManagerError.core("The recording file could not be created") }
          let handle = try FileHandle(forWritingTo: url)
          try handle.seekToEnd()
          samples.sync {
            file = handle
            sink = continuation
            bytes = 0
            failed = nil
            limitBytes = requestedBytes
            spectrum = NativeMicrophoneSpectrum()
            meter.reset()
          }
          output.setSampleBufferDelegate(self, queue: samples)
          session.commitConfiguration()
          configuring = false
          session.startRunning()
          guard session.isRunning else { throw ManagerError.core("The microphone did not start") }
          started.resume()
        } catch {
          continuation.finish(throwing: error)
          started.resume(throwing: error)
        }
      }
    }
    return stream
  }
  func captureOutput(
    _ output: AVCaptureOutput, didOutput buffer: CMSampleBuffer,
    from connection: AVCaptureConnection
  ) {
    guard failed == nil, bytes < limitBytes, let file,
      let format = CMSampleBufferGetFormatDescription(buffer),
      let asbd = CMAudioFormatDescriptionGetStreamBasicDescription(format)
    else { return }
    do {
      guard asbd.pointee.mSampleRate == 16000, asbd.pointee.mChannelsPerFrame == 1,
        asbd.pointee.mBitsPerChannel == 16, asbd.pointee.mFormatID == kAudioFormatLinearPCM
      else {
        throw ManagerError.core("The microphone did not provide the requested 16 kHz PCM format")
      }
      guard let block = CMSampleBufferGetDataBuffer(buffer) else {
        throw ManagerError.core("The microphone delivered an empty audio buffer")
      }
      let delivered = CMBlockBufferGetDataLength(block)
      guard delivered > 0, delivered % 2 == 0 else {
        throw ManagerError.core("The microphone delivered invalid PCM")
      }
      // Stop at the exact model/attachment limit, including a partial final
      // callback, so the saved file cannot exceed the advertised ceiling.
      let count = min(delivered, limitBytes - bytes)
      var data = Data(count: count)
      let status = data.withUnsafeMutableBytes { ptr in
        CMBlockBufferCopyDataBytes(
          block, atOffset: 0, dataLength: count, destination: ptr.baseAddress!)
      }
      guard status == kCMBlockBufferNoErr else {
        throw ManagerError.core("The microphone samples could not be read")
      }
      try file.write(contentsOf: data)
      bytes += count
      if spectrum.consume(data) { meter.publish(spectrum.levels) }
      if case .dropped = sink?.yield(.init(pcm: data, elapsed: Double(bytes) / 32000)) {
        throw ManagerError.core(
          "The speech connection fell behind capture. Recording stopped; the captured original is retained."
        )
      }
      if bytes == limitBytes { sink?.finish() }
    } catch {
      failed = error
      meter.reset()
      sink?.finish(throwing: error)
    }
  }
  /// stopRunning returns before some queued callbacks. Drain the sample queue
  /// before writing RIFF lengths/closing; this preserves the final recording tail.
  func stop() async throws -> URL {
    try await withCheckedThrowingContinuation { (done: CheckedContinuation<URL, Error>) in
      control.async { [self] in
        session.stopRunning()
        output.setSampleBufferDelegate(nil, queue: nil)
        samples.async { [self] in
          meter.reset()
          do {
            if let file {
              try file.seek(toOffset: 0)
              try file.write(contentsOf: Self.wavHeader(bytes: bytes))
              try file.synchronize()
              try file.close()
              self.file = nil
            }
            sink?.finish()
            sink = nil
            done.resume(returning: url)
          } catch {
            sink?.finish(throwing: error)
            sink = nil
            done.resume(throwing: error)
          }
        }
      }
    }
  }
  static func wavHeader(bytes: Int) -> Data {
    var result = Data()
    func word<T: FixedWidthInteger>(_ value: T) {
      var little = value.littleEndian
      withUnsafeBytes(of: &little) { result.append(contentsOf: $0) }
    }
    result.append(Data("RIFF".utf8))
    word(UInt32(bytes + 36))
    result.append(Data("WAVEfmt ".utf8))
    word(UInt32(16))
    word(UInt16(1))
    word(UInt16(1))
    word(UInt32(16000))
    word(UInt32(32000))
    word(UInt16(2))
    word(UInt16(16))
    result.append(Data("data".utf8))
    word(UInt32(bytes))
    return result
  }
}
