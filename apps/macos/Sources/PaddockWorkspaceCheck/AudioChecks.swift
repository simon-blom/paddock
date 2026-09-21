import AVFoundation
import Foundation
import PaddockStudio
import WebKit

extension Checks {
  /// Real WebKit AudioWorklet + typed bridge; synthetic oscillator and socket.
  /// Never requests microphone permission or sends audio to a real endpoint.
  @MainActor func audioChecks(_ session: StudioWorkspace) async throws {
    try await session.command("renderer", ["mode": .string("native")])
    try await speechMenuChecks(session)
    _ = try await session.webView.callAsyncJavaScript(
      #"""
      const pinia=document.querySelector('#app').__vue_app__.config.globalProperties.$pinia;
      const models=pinia._s.get('models'), settings=pinia._s.get('settings');
      models.models=[{id:'audio-text',kind:'chat',status:'ok',port:16557},{id:'audio-speech',kind:'transcriber',status:'ok',port:16558}];
      models.refresh=async()=>{}; models.integrateRunnerRows=()=>{};
      for(const m of models.models)models.caps[m.id]={vision:false,webSearch:false,mcpServers:[],timestampGranularities:[],include:[],reasoning:'none',maxCtx:4096};
      settings.dictateWith='audio-speech'; window.audioFrames=[]; window.audioOpened=0; window.audioStopped=0;window.speechSockets=[];
      const Recorder=window.MediaRecorder;
      window.recordingEvents=[];
      window.MediaRecorder=class extends Recorder {
        constructor(...args){super(...args);const events={parts:[],mime:this.mimeType};window.recordingEvents.push(events);
          this.addEventListener('dataavailable',e=>{if(e.data.size)events.parts.push(e.data)});
        }
      };
      // Native meters must work even when WebKit never schedules a paint.
      window.requestAnimationFrame=()=>0;
      const synthetic=async()=>{
        window.audioOpened++;
        const ctx=new AudioContext({sampleRate:48000}),src=ctx.createOscillator(),dest=ctx.createMediaStreamDestination();
        src.frequency.value=440;src.connect(dest);src.start();await ctx.resume();
        window.audioTracks=dest.stream.getTracks();window.syntheticAudioContext=ctx;
        for(const track of window.audioTracks){const stop=track.stop.bind(track);track.stop=()=>{window.audioStopped++;stop();src.stop();void ctx.close()}}
        return dest.stream;
      };
      Object.defineProperty(navigator,'mediaDevices',{configurable:true,value:{getUserMedia:synthetic,enumerateDevices:async()=>[],addEventListener(){}}});
      class Socket {
        static CONNECTING=0;static OPEN=1;static CLOSED=3;readyState=0;heard=false;
        constructor(url){this.telemetry=url.includes('/api/gpu/stream');this.port=Number(url.match(/runners\/(\d+)/)?.[1]);if(!this.telemetry&&![16558,16559].includes(this.port))throw Error('Unexpected speech destination: '+new URL(url).pathname);if(!this.telemetry)window.speechSockets.push(this);setTimeout(()=>{this.readyState=1;this.onopen?.()},10)}
        send(raw){if(this.telemetry)return;const m=JSON.parse(raw);window.audioFrames.push({...m,port:this.port});
          if(m.type==='input_audio_buffer.append'&&!this.heard){this.heard=true;this.emit({type:'input_audio_buffer.speech_started'});this.emit({type:'conversation.item.input_audio_transcription.delta',delta:'Hej '})}
          if(m.type==='input_audio_buffer.commit')this.emit({type:'conversation.item.input_audio_transcription.completed',transcript:'Hej världen.',paddock_audio_start_ms:0});
        }
        emit(m){setTimeout(()=>this.onmessage?.({data:JSON.stringify(m)}),0)}
        close(){this.readyState=3}
      }
      window.WebSocket=Socket;
      """#, arguments: [:], in: nil, contentWorld: .page)
    try await session.command("models", ["ids": .array([.string("audio-text")])])
    try await session.command("microphoneSettings", ["language": .string("sv")])
    guard session.state?.audio?.jobs == ["dictate"] else {
      throw Failure(message: "Text arming did not offer dictation")
    }
    try await session.command("microphoneStart")
    try await until("synthetic microphone delivers provisional text") {
      session.state?.audio?.provisional == "Hej "
    }
    try await until("native meter samples actual audio without animation frames") {
      session.state?.audio?.levels.contains { $0 > 0.05 } == true
    }
    guard session.state?.audio?.dictation.isEmpty == true else {
      throw Failure(message: "Provisional text entered the draft")
    }
    try await check(
      "shared 16 kHz speech session and AudioWorklet",
      "window.audioFrames[0]?.session?.audio.input.format.rate===16000&&window.audioFrames[0]?.session?.audio.input.transcription.language==='sv'&&window.audioFrames.some(f=>f.type==='input_audio_buffer.append'&&atob(f.audio).length>0)"
    )
    try await session.command("microphoneStop", ["text": .string("Existing draft")])
    guard let audio = session.state?.audio, audio.phase == "idle",
      audio.dictation.first?.text == "Hej världen."
    else {
      throw Failure(message: "Final dictation was not retained for native insertion")
    }
    try await session.command(
      "dictationAck", ["session": .string(audio.session), "index": .number(0)])
    guard session.state?.audio?.dictation.isEmpty == true else {
      throw Failure(message: "Dictation acknowledgement failed")
    }
    let captureState = try await session.webView.evaluateJavaScript(
      "JSON.stringify({opened:window.audioOpened,stopped:window.audioStopped,tracks:window.audioTracks.map(t=>t.readyState),messages:document.querySelector('#app').__vue_app__.config.globalProperties.$pinia._s.get('chat').active.messages.length})"
    )
    print("Synthetic capture lifecycle: \(String(describing: captureState))")
    try await check(
      "microphone released and dictation never submits a turn",
      "window.audioOpened===1&&window.audioTracks.every(t=>t.readyState==='ended')&&document.querySelector('#app').__vue_app__.config.globalProperties.$pinia._s.get('chat').active.messages.length===0"
    )
    _ = try await session.webView.evaluateJavaScript("void window.syntheticAudioContext.close()")
    _ = try await session.webView.callAsyncJavaScript(
      #"""
      const pinia=document.querySelector('#app').__vue_app__.config.globalProperties.$pinia,models=pinia._s.get('models');
      models.models.push({id:'cloud:audio-fixture:whisper',kind:'transcriber',status:'ok',cloud:{endpoint:'audio-fixture',endpointName:'Fixture'}});
      window.recordedRequests=[];
      const original=window.fetch.bind(window);
      window.fetch=async(url,options={})=>{
        const path=String(url);
        if(path.split('?')[0]==='/api/cloud/audio-fixture/v1/audio/transcriptions'){
          const file=options.body.get('file');window.recordedRequests.push({size:file.size,type:file.type,language:options.body.get('language')});
          return new Response(JSON.stringify({text:'Recorded fixture.',language:'sv',duration:1}),{headers:{'Content-Type':'application/json'}});
        }
        if(/\/v1\/(responses|audio|chat)/.test(path))throw Error('Unexpected real inference request');
        return original(url,options);
      };
      """#, arguments: [:], in: nil, contentWorld: .page)
    try await session.command("models", ["ids": .array([.string("cloud:audio-fixture:whisper")])])
    guard session.state?.audio?.jobs == ["record"], session.state?.audio?.liveBlocked == true else {
      throw Failure(message: "Cloud speech was not offered recording with a live explanation")
    }
    try await session.command("microphoneStart")
    try await until("WebKit recorder has actual synthetic audio") {
      session.state?.audio?.arming == false && (session.state?.audio?.elapsed ?? 0) > 2.3
    }
    guard await session.stopMicrophone(text: "Hidden draft must not become transcription input")
    else {
      throw Failure(message: session.state?.audio?.error ?? "Recording was not accepted")
    }
    try await until("recorded transcription completes") { !session.busy }
    try await check(
      "one real recording uploaded once and transcribed through the cloud adapter",
      "window.recordedRequests.length===1&&window.recordedRequests[0].size>0&&window.recordedRequests[0].language==='sv'&&window.audioTracks.every(t=>t.readyState==='ended')"
    )
    try await check(
      "recorded turn has no hidden text and uses the shared transcript renderer",
      "const c=document.querySelector('#app').__vue_app__.config.globalProperties.$pinia._s.get('chat').active;return c.messages.find(m=>m.role==='user').content.every(p=>p.type==='audio')&&c.messages.some(m=>m.role==='assistant'&&m.content.some(p=>p.text==='Recorded fixture.'))"
    )
    guard session.attachments.isEmpty else {
      throw Failure(message: "Accepted recording stayed in the native tray")
    }
    try await validateStoredRecording(session, name: "record-and-send")
    _ = try await session.webView.evaluateJavaScript("void window.syntheticAudioContext.close()")
    let attachment =
      try await session.webView.evaluateJavaScript(
        "const c=document.querySelector('#app').__vue_app__.config.globalProperties.$pinia._s.get('chat').active,p=c.messages.find(m=>m.role==='user').content[0];JSON.stringify({id:p.attachmentId,name:p.name,mime:p.mime,size:p.size})"
      ) as? String
    guard let attachment, let bytes = attachment.data(using: .utf8) else {
      throw Failure(message: "Missing stored test recording")
    }
    let metadata = try JSONDecoder().decode([String: StudioValue].self, from: bytes)
    try await session.command("stage", metadata)
    try await session.command("preview", ["id": metadata["id"]!])
    guard session.state?.nativeAudioPreview?.id == metadata["id"]?.text,
      session.state?.nativeTranscript?.available == true
    else {
      throw Failure(message: "Audio preview switched away from native rendering")
    }
    try await check(
      "staged audio has no HTML player", "!document.querySelector('.native-audio-preview audio')")
    try await session.command("removeAttachment", ["id": metadata["id"]!])
    try await session.command("closePreview")
    _ = try await session.webView.callAsyncJavaScript(
      #"""
      const models=document.querySelector('#app').__vue_app__.config.globalProperties.$pinia._s.get('models');
      models.models.push({id:'audio-generative',kind:'chat',status:'ok',port:16559});
      models.caps['audio-generative']={...models.caps['audio-speech'],audio:true};
      models.caps['audio-speech'].timestampGranularities=['segment','word'];
      window.audioFrames=[];
      """#, arguments: [:], in: nil, contentWorld: .page)
    try await session.command(
      "models", ["ids": .array([.string("audio-speech"), .string("audio-generative")])])
    try await session.command("microphoneSettings", ["mode": .string("live")])
    try await session.command("microphoneStart")
    try await check(
      "both live lanes hear the same capture",
      "[16558,16559].every(port=>window.audioFrames.some(f=>f.port===port&&f.type==='input_audio_buffer.append'))"
    )
    try await until("multi-chunk live recording") { (session.state?.audio?.elapsed ?? 0) > 2.3 }
    guard session.state?.nativeTranscript?.available == true,
      session.state?.nativeTranscript?.messages.suffix(2).allSatisfy({
        $0.streaming && $0.speech != nil
      }) == true
    else {
      throw Failure(message: "Live transcription switched away from native rendering before Stop")
    }
    // Mimic a VAD pause: both lanes have already delivered their final text.
    // Stop therefore settles synchronously, before MediaRecorder's final event.
    _ = try await session.webView.evaluateJavaScript(
      "window.speechSockets.slice(-2).forEach(s=>{s.emit({type:'input_audio_buffer.speech_stopped'});s.emit({type:'conversation.item.input_audio_transcription.completed',transcript:'Hej världen.',paddock_audio_start_ms:0})})"
    )
    try await Task.sleep(for: .milliseconds(50))
    try await session.command("microphoneStop", ["text": .string("")])
    try await check(
      "identical PCM, per-lane capabilities and no second transcription",
      "const a=window.audioFrames.filter(f=>f.port===16558),b=window.audioFrames.filter(f=>f.port===16559);return JSON.stringify(a.filter(f=>f.audio).map(f=>f.audio))===JSON.stringify(b.filter(f=>f.audio).map(f=>f.audio))&&a[0].session.audio.input.transcription.paddock_verbose===true&&!b[0].session.audio.input.transcription.paddock_verbose&&window.recordedRequests.length===1&&window.audioTracks.every(t=>t.readyState==='ended')"
    )
    try await check(
      "live compare keeps both transcripts and its recording",
      "const c=document.querySelector('#app').__vue_app__.config.globalProperties.$pinia._s.get('chat').active,lanes=c.messages.slice(-2),u=c.messages.at(-3);return lanes.every(m=>m.transcript&&!m.streaming&&m.content[0].text==='Hej världen.')&&lanes[0].group===lanes[1].group&&!!u.content[0].attachmentId"
    )
    try await validateStoredRecording(session, name: "live-compare stop during pause")
    guard let projected = session.state?.nativeTranscript, projected.available,
      projected.messages.suffix(2).allSatisfy({ $0.speech != nil && $0.group != nil })
    else {
      throw Failure(message: "Transcription compare did not stay native")
    }
    _ = try await session.webView.evaluateJavaScript("void window.syntheticAudioContext.close()")
    print(
      "PASS: native audio projection, real WebKit capture/worklet, provisional/final dictation, acknowledgement and microphone teardown"
    )
  }

  @MainActor private func validateStoredRecording(_ session: StudioWorkspace, name: String)
    async throws
  {
    guard let transcript = session.state?.nativeTranscript, transcript.available,
      let clip = transcript.messages.last(where: { $0.role == "user" })?.audioClips?.first,
      await session.prepareAudio(clip), session.audioPlayback.duration > 2
    else {
      throw Failure(
        message:
          "\(name): native audio could not load a complete recording: \(session.audioPlayback.error ?? "missing native projection")"
      )
    }
    // Synthetic fixture bytes only. Compare the durable attachment byte-for-byte
    // with every real MediaRecorder event, including the header and final tail.
    let result = try await session.webView.callAsyncJavaScript(
      #"""
      const response=await fetch('/api/attachments/'+id),bytes=new Uint8Array(await response.arrayBuffer());
      const events=window.recordingEvents.at(-1),original=new Uint8Array(await new Blob(events.parts).arrayBuffer());
      if(events.parts.length<2||bytes.length!==original.length||bytes.some((b,i)=>b!==original[i]))throw Error('Saved recording lost or changed chunks');
      if(String.fromCharCode(...bytes.slice(4,8))!=='ftyp')throw Error('Missing MP4 container header');
      let binary='';for(let i=0;i<bytes.length;i+=8192)binary+=String.fromCharCode(...bytes.subarray(i,i+8192));
      return btoa(binary);
      """#, arguments: ["id": clip.id], in: nil, contentWorld: .page)
    guard let encoded = result as? String, let data = Data(base64Encoded: encoded) else {
      throw Failure(message: "Missing synthetic recording bytes")
    }
    let url = FileManager.default.temporaryDirectory.appendingPathComponent(
      "Paddock-audio-check-\(UUID()).m4a")
    try data.write(to: url, options: .withoutOverwriting)
    defer {
      try? FileManager.default.removeItem(at: url)
      session.audioPlayback.reset()
    }
    let audio = try AVAudioFile(forReading: url)
    guard let buffer = AVAudioPCMBuffer(pcmFormat: audio.processingFormat, frameCapacity: 4096)
    else {
      throw Failure(message: "Could not allocate synthetic audio decode buffer")
    }
    var frames: Int64 = 0
    var firstPeak: Float = 0
    var lastPeak: Float = 0
    while audio.framePosition < audio.length {
      try audio.read(into: buffer)
      guard buffer.frameLength > 0, let samples = buffer.floatChannelData?[0] else { break }
      let peak = (0..<Int(buffer.frameLength)).reduce(Float(0)) { max($0, abs(samples[$1])) }
      if frames == 0 { firstPeak = peak }
      lastPeak = peak
      frames += Int64(buffer.frameLength)
    }
    guard Double(frames) / audio.processingFormat.sampleRate > 2, firstPeak > 0.01, lastPeak > 0.01
    else {
      throw Failure(message: "\(name): recording did not decode audible beginning and tail")
    }
    print(
      "PASS: \(name), saved bytes match all recorder chunks; native AAC decode \(frames) frames, \(session.audioPlayback.duration)s, non-silent first and final blocks"
    )
  }
}
