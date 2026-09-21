import Foundation
import PaddockStudio
import WebKit

extension Checks {
  /// Inventory fixtures only. No configured endpoint files or real runners are
  /// created, started or stopped by this check.
  @MainActor func speechMenuChecks(_ session: StudioWorkspace) async throws {
    try await session.command("microphoneRefresh")
    // Discovery can legitimately see a speech endpoint already running here, even in
    // an isolated data root. Freeze only this test page's inventory; never
    // change, start or stop those real endpoints to manufacture an empty case.
    _ = try await session.webView.evaluateJavaScript(
      #"""
      { const p=document.querySelector('#app').__vue_app__.config.globalProperties.$pinia;
      const f=p._s.get('fleet'),m=p._s.get('models');
      f.refresh=async()=>{};f.rows=[];f.configured=[];f.deploying=[];
      m.refresh=async()=>{};m.integrateRunnerRows=()=>{};m.models=[]; } null;
      """#)
    try await session.command("microphoneRefresh")
    guard session.state?.audio?.menu?.needsSetup == true,
      session.state?.audio?.speechModels?.isEmpty == true
    else { throw Failure(message: "Empty speech inventory did not show the web setup flow") }
    _ = try await session.webView.evaluateJavaScript(
      #"""
      { const p=document.querySelector('#app').__vue_app__.config.globalProperties.$pinia;
      const f=p._s.get('fleet');
      f.configured=[{port:16558,model:'synthetic-speech',display:'Synthetic speech',vendor:'OpenAI',capability:['transcription'],running:false},
        {port:16557,model:'synthetic-chat',capability:['chat'],running:false}];
      f.rows=[]; f.deploying=[]; } null;
      """#)
    try await until("configured stopped speech row") {
      let rows = session.state?.audio?.speechModels ?? []
      return rows.count == 1 && rows[0].port == 16558 && rows[0].canStart && !rows[0].canStop
    }
    _ = try await session.webView.evaluateJavaScript(
      #"""
      document.querySelector('#app').__vue_app__.config.globalProperties.$pinia._s.get('fleet').deploying=[{port:16558,model:'synthetic-speech',phase:'starting',startedAt:Date.now(),log:[]}]; null;
      """#)
    try await until("speech start is working, with both actions disabled") {
      guard let row = session.state?.audio?.speechModels?.first else { return false }
      return row.busy && row.status == "working..." && !row.canStart && !row.canStop
    }
    _ = try await session.webView.evaluateJavaScript(
      #"""
      { const p=document.querySelector('#app').__vue_app__.config.globalProperties.$pinia,f=p._s.get('fleet'),m=p._s.get('models');
      f.deploying=[];f.rows=[{port:16558,status:'running',asr:'synthetic-speech'}];
      m.models=[{id:'synthetic-chat',kind:'chat',status:'ok',port:16557},{id:'synthetic-speech',kind:'transcriber',status:'ok',port:16558}];
      m.currentId='synthetic-chat';p._s.get('chat').startDraft('synthetic-chat'); } null;
      """#)
    try await until("running speech keeps Stop in the microphone menu") {
      guard let audio = session.state?.audio, let row = audio.speechModels?.first else {
        return false
      }
      return audio.menu?.offered == true && audio.menu?.needsSetup == false
        && audio.menu?.menu == true
        && row.canStop && !row.canStart && row.status == "running · port 16558"
    }
    _ = try await session.webView.evaluateJavaScript(
      #"""
      { const f=document.querySelector('#app').__vue_app__.config.globalProperties.$pinia._s.get('fleet');f.rows=[];f.configured=[];f.deploying=[]; } null;
      """#)
    print(
      "PASS: shared microphone empty/configured/starting/running states; no model lifecycle mutation"
    )
  }
}
