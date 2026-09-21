import AppKit
import Foundation
import PaddockClient
import PaddockStudio
import WebKit

/// Actual AppKit/WebKit + Rust-store functional checks. Only synthetic data in
/// a fresh directory. The product bundle never links this executable.
@main enum WorkspaceCheckApp {
  @MainActor static func main() {
    let app = NSApplication.shared
    let lifecycle = Checks()
    app.delegate = lifecycle
    app.setActivationPolicy(
      ProcessInfo.processInfo.environment["PADDOCK_TOOLS_CHECK_ONLY"] == "1"
        || ProcessInfo.processInfo.environment["PADDOCK_NATIVE_ONLY_CHECK"] == "1"
        || ProcessInfo.processInfo.environment["PADDOCK_AUDIO_CHECK_ONLY"] == "1"
        || ProcessInfo.processInfo.environment["PADDOCK_COMPARE_CHECK_ONLY"] == "1"
        || ProcessInfo.processInfo.environment["PADDOCK_PROMPTS_CHECK_ONLY"] == "1"
        || ProcessInfo.processInfo.environment["PADDOCK_HISTORY_CHECK_ONLY"] == "1"
        ? .accessory : .regular
    )
    withExtendedLifetime(lifecycle) { app.run() }
  }
}
@MainActor final class Checks: NSObject, NSApplicationDelegate {
  var window: NSWindow?
  var workspace: StudioWorkspace?
  var core: NativeManager?
  struct Failure: Error { let message: String }
  func applicationDidFinishLaunching(_ notification: Notification) {
    Task {
      do {
        try await run()
        await workspace?.shutdown()
        await core?.close()
        print("PASS: native content workspace functional checks")
        exit(0)
      } catch {
        print(
          "Native state: ready=\(workspace?.ready == true), busy=\(workspace?.busy == true), error=\(workspace?.error ?? "none")"
        )
        if let workspace,
          let detail = try? await workspace.webView.evaluateJavaScript(
            "JSON.stringify({composer:document.querySelectorAll('.composer').length,actions:document.querySelectorAll('.msg__actions').length,body:document.body.innerText.slice(-2000),errors:window.nativeCheckErrors})"
          )
        {
          print("Diagnostic: \(detail)")
        }
        await workspace?.shutdown()
        await core?.close()
        FileHandle.standardError.write(Data("FAIL: \(error)\n".utf8))
        exit(1)
      }
    }
  }
  func run() async throws {
    let env = ProcessInfo.processInfo.environment
    guard env["PADDOCK_DATA"]?.contains("paddock-workspace-check.") == true,
      CommandLine.arguments.count == 3
    else {
      throw Failure(
        message: "Supply content assets and synthetic fixtures; use an isolated data root")
    }
    let core = NativeManager()
    self.core = core
    let session = StudioWorkspace(client: core)
    workspace = session
    let window = NSWindow(
      contentRect: NSRect(x: 0, y: 0, width: 1200, height: 820),
      styleMask: [.titled, .closable, .resizable], backing: .buffered, defer: false)
    self.window = window
    window.isReleasedWhenClosed = false
    window.title = "Paddock content checks - synthetic files"
    window.contentView = session.webView
    window.center()
    if env["PADDOCK_TOOLS_CHECK_ONLY"] != "1" && env["PADDOCK_PROMPTS_CHECK_ONLY"] != "1"
      && env["PADDOCK_NATIVE_ONLY_CHECK"] != "1"
      && env["PADDOCK_AUDIO_CHECK_ONLY"] != "1"
      && env["PADDOCK_HISTORY_CHECK_ONLY"] != "1"
      && env["PADDOCK_COMPARE_CHECK_ONLY"] != "1"
    {
      window.makeKeyAndOrderFront(nil)
      NSApp.activate(ignoringOtherApps: true)
    }
    await session.start(assets: URL(fileURLWithPath: CommandLine.arguments[1]))
    try await until("workspace ready") {
      if let error = session.error { throw Failure(message: error) }
      return session.ready && session.state != nil
    }
    guard session.state?.composer != nil else {
      throw Failure(message: "Missing native composer policy")
    }
    if env["PADDOCK_NATIVE_ONLY_CHECK"] == "1" {
      try await session.command("autoTitle", ["enabled": .bool(false)])
      try await nativeOnlyChecks(session)
      if env["PADDOCK_SKIP_VISUAL_CHECKS"] == "1" {
        print("SKIP: PDF painted-page checks explicitly deferred; this is not a visual pass")
      } else {
        try await attachmentSelectionChecks(session)
      }
      try await audioChecks(session)
      return
    }
    if env["PADDOCK_AUDIO_CHECK_ONLY"] == "1" {
      try await audioChecks(session)
      return
    }
    if env["PADDOCK_COMPARE_CHECK_ONLY"] == "1" {
      try await compareChecks(session)
      return
    }
    if env["PADDOCK_PROMPTS_CHECK_ONLY"] == "1" {
      try await promptChecks(session)
      return
    }
    if env["PADDOCK_TOOLS_CHECK_ONLY"] == "1" {
      try await toolsChecks(session)
      return
    }
    if env["PADDOCK_HISTORY_CHECK_ONLY"] == "1" {
      try await historyChecks(session)
      return
    }
    if env["PADDOCK_MESSAGE_CHECK_ONLY"] == "1" {
      try await messageChecks(session)
      return
    }
    try await session.command("renderer", ["mode": .string("web")])
    try await session.command("autoTitle", ["enabled": .bool(false)])
    try await session.command("draft", ["text": .string("A draft estimate")])
    guard (session.state?.composer?.contextUsed ?? 0) > 0 else {
      throw Failure(message: "Missing context estimate")
    }
    try await session.command("settings", ["systemPrompt": .string("Keep instructions")])
    do {
      try await session.command(
        "settings", ["systemPrompt": .string("Must not persist"), "maxTokens": .number(-1)])
      throw Failure(message: "Invalid settings accepted")
    } catch is Failure { throw Failure(message: "Invalid settings accepted") } catch {
      // Expected validation failure; verify transactional settings below.
    }
    try await session.command("draft", ["text": .string("")])
    guard session.state?.settings["systemPrompt"]?.text == "Keep instructions" else {
      throw Failure(message: "Invalid patch partially mutated settings")
    }
    try await session.command("settings", ["systemPrompt": .string("")])
    try await session.command("samplerDefaults")
    guard session.state?.composer?.samplerSet == false else {
      throw Failure(message: "Default sampler is explicit")
    }
    try await check(
      "content only; no web composer or app chrome",
      "!document.querySelector('.composer') && !document.querySelector('.app-header') && !document.querySelector('.sidebar')"
    )
    _ = try await session.webView.evaluateJavaScript(
      "window.nativeCheckErrors=[];window.addEventListener('error',e=>window.nativeCheckErrors.push(e.message));window.addEventListener('unhandledrejection',e=>window.nativeCheckErrors.push(String(e.reason)))"
    )
    try await check(
      "private session is HttpOnly", "!document.cookie.includes('paddock_desktop_session')")
    try await check(
      "manager commands excluded",
      "(await fetch('/api/keys')).status === 403 && (await fetch('/api/servers/12481/file')).status === 403 && (await fetch('/api/runners', {method:'POST'})).status === 403"
    )
    try await check(
      "bundle asset misses are 404", "(await fetch('/assets/not-real.js')).status === 404")
    try await check(
      "artifact shell keeps opaque-origin sandbox policy",
      "const r=await fetch('/artifact-frame');const c=r.headers.get('Content-Security-Policy')||'';return r.ok&&c.includes('sandbox allow-scripts')&&c.includes(\"connect-src 'none'\")&&!c.includes('allow-same-origin')"
    )
    _ = try await session.webView.callAsyncJavaScript(
      #"""
      const frame=document.createElement('iframe');frame.id='native-artifact-check';frame.sandbox='allow-scripts';frame.src='/artifact-frame';
      window.addEventListener('message',event=>{if(event.source===frame.contentWindow&&event.data?.nativeArtifactCheck)window.nativeArtifactResult={...event.data.nativeArtifactCheck,origin:event.origin}});
      frame.onload=()=>frame.contentWindow.postMessage({type:'paddock:artifact',html:'<h1>Artifact check</h1><script>let isolated=false;try{void parent.document.body}catch{isolated=true}parent.postMessage({nativeArtifactCheck:{isolated}},"*")</script>'},'*');
      document.body.append(frame);
      """#, arguments: [:], in: nil, contentWorld: .page)
    try await check(
      "artifact script runs without parent access",
      "window.nativeArtifactResult?.isolated===true&&window.nativeArtifactResult?.origin==='null'")
    _ = try await session.webView.evaluateJavaScript(
      "document.querySelector('#native-artifact-check').remove()")
    try await check(
      "WebAssembly assets compile",
      "const manifest=await (await fetch('/manifest.json')).json(); for(const path of manifest.wasm){const r=await fetch('/'+path);if(!r.ok)return false;await WebAssembly.compile(await r.arrayBuffer())}return manifest.wasm.length===3"
    )
    for (name, selector) in [
      ("sample.pdf", ".lector-workspace"), ("sample.docx", ".scriptor-sheet"),
    ] {
      session.addFiles([URL(fileURLWithPath: CommandLine.arguments[2]).appending(path: name)])
      try await until("upload \(name)") {
        if let error = session.attachments.last?.error { throw Failure(message: error) }
        return session.attachments.last?.ready == true
      }
      guard let attachment = session.attachments.last else {
        throw Failure(message: "Missing attachment")
      }
      try await session.command("preview", ["id": .string(attachment.id)])
      try await check("shared viewer for \(name)", "!!document.querySelector('\(selector)')")
      for dark in [true, false, true] {
        await session.setDark(dark)
        let surface = dark ? "rgb(36, 36, 36)" : "rgb(245, 245, 245)"
        if name.hasSuffix("pdf") {
          try await check(
            "Lector nested chrome and page stage follow \(dark ? "dark" : "light")",
            """
            const roots=[...document.querySelectorAll('.docpane__lector :is(.lector-workspace,.lector-viewer)')];
            if(roots.length<2)return false;
            for(const el of roots){const s=getComputedStyle(el),v=k=>s.getPropertyValue(k).trim();
              if(v('--lector-bg')!==v('--pk-bg-surface')||v('--lector-accent-active')!==v('--pk-accent-active')||v('--lector-accent-light')!==v('--pk-accent-subtle')||v('--lector-radius')!=='6px'||v('--lector-radius-lg')!=='8px')return false;
            }
            const canvas=document.querySelector('.lector-canvas'), page=document.querySelector('.lector-page');
            if(!canvas||!page)return false;
            if(window.nativeThemePage&&window.nativeThemePage!==page)return false;
            window.nativeThemePage=page;
            return getComputedStyle(canvas).backgroundColor==='\(surface)'&&getComputedStyle(page).backgroundColor==='rgb(255, 255, 255)';
            """)
        } else {
          try await check(
            "Scriptor stage follows \(dark ? "dark" : "light") without recoloring paper",
            """
            const stage=document.querySelector('.docpane__body--docx'), sheet=document.querySelector('.scriptor-sheet');
            return stage&&sheet&&getComputedStyle(stage).backgroundColor==='\(surface)'&&getComputedStyle(sheet).filter==='none';
            """)
        }
      }
      if name.hasSuffix("pdf") { try await checkViewerOverlays(session) }
      try await checkContentColumn("composer column beside \(name)")
      guard session.conversation == nil else {
        throw Failure(message: "Preview created a sent conversation")
      }
      session.removeAttachment(attachment.id)
      try await check(
        "viewer closes without a duplicate instance", "!document.querySelector('\(selector)')")
    }
    // Seed durable rich content using the real store, then open with the
    // product's typed navigation. No test-only renderer or CSS fixtures.
    let fixture =
      #"{"id":"workspace-fixture","title":"Native workspace fixture","model":"fixture","systemPrompt":"","params":{"thinking":true,"reasoningEffort":"","stop":[]},"createdAt":1,"updatedAt":1,"messages":[{"id":"u","role":"user","content":[{"type":"text","text":"Keep this selection"}],"createdAt":1},{"id":"a","role":"assistant","content":[{"type":"text","text":"**Shared renderer**\n\n$$E=mc^2$$\n\n```swift\nlet value = 42\n```\n\n```mermaid\nflowchart LR\nA[Swift] --> B[Shared content]\n```"}],"reasoning":"Shared reasoning fold","createdAt":2}]}"#
    _ = try await session.webView.callAsyncJavaScript(
      """
      const doc=JSON.parse(fixture),m=doc.messages[1];
      m.model='cloud:removed-endpoint:anthropic/claude-fixture';
      m.usage={promptTokens:80,completionTokens:120,tps:60,ms:5600,ttftMs:350,reasoningTokens:200,reasoningMs:3000,reasoningTps:66.6,costUsd:0.012};
      m.run={model:m.model,spec:'off',params:{...doc.params},tools:[],systemPrompt:'Fixture instructions',at:2};
      const r=await fetch('/api/conversations/workspace-fixture',{method:'PUT',headers:{'Content-Type':'application/json'},body:JSON.stringify(doc)});
      if(!r.ok)throw new Error('Fixture save failed');
      const chat=document.querySelector('#app').__vue_app__.config.globalProperties.$pinia._s.get('chat');chat.loaded=false;await chat.hydrate();
      """,
      arguments: ["fixture": fixture], in: nil, contentWorld: .page)
    try await session.command("open", ["id": .string("workspace-fixture")])
    try await session.command("renderer", ["mode": .string("native")])
    try await until("native text projection") {
      session.state?.nativeTranscript?.messages.last?.text.contains("Shared renderer") == true
    }
    try await until("native identity and run metadata projection") {
      guard let chrome = session.state?.nativeTranscript?.messages.last?.chrome else {
        return false
      }
      return chrome.vendor == "Anthropic" && chrome.spec.isEmpty
        && chrome.footer == "120 tokens · 60 tok/s · 5.6s · $0.012"
        && chrome.thinkingMeta == "200 tokens · 67 tok/s"
        && chrome.promptText == "Fixture instructions"
        && chrome.sections.contains(where: { $0.id == "metrics" })
    }
    try await check(
      "native mode unmounts web Markdown",
      "!document.querySelector('.thread') && !document.querySelector('.msg--assistant')")
    try await session.command("renderer", ["mode": .string("web")])
    try await check(
      "shared Markdown, code, math and Mermaid",
      "document.querySelector('.msg--assistant strong')?.textContent==='Shared renderer' && !!document.querySelector('.katex') && document.body.textContent.includes('let value = 42') && !!document.querySelector('.msg--assistant svg')"
    )
    try await check(
      "message actions and native-only composer",
      "!!document.querySelector('.msg__actions button') && !document.querySelector('.composer')")
    for dark in [false, true] {
      await session.setDark(dark)
      let surface = dark ? "rgb(36, 36, 36)" : "rgb(245, 245, 245)"
      try await check(
        "shared Markdown chrome follows \(dark ? "dark" : "light")",
        """
        const root=document.querySelector('.markstream-vue'), block=document.querySelector('.pk-code');
        if(!root||!block)return false;
        const s=getComputedStyle(root);
        if(!s.getPropertyValue('--ms-ring').trim().startsWith('0.000 0.000%'))return false;
        const b=getComputedStyle(block);
        return b.backgroundColor==='\(surface)'&&b.borderRadius==='8px'&&block.textContent.includes('let value = 42');
        """)
    }
    try await check(
      "web and native footer use identical recorded statistics",
      "document.querySelector('.msg__usage')?.textContent==='120 tokens · 60 tok/s · 5.6s · $0.012' && !!document.querySelector('.msg__hd svg') && !document.querySelector('.msg__hd-spec')"
    )
    _ = try await session.webView.evaluateJavaScript(
      "document.querySelector('button[aria-label=\"Run details\"]').click()")
    try await check(
      "shared run details show recorded metrics",
      "document.querySelector('.run')?.textContent.includes('80 in · 120 out · 200 reasoning') && document.querySelector('.run')?.textContent.includes('TTFT 350ms')"
    )
    for width in [1200, 700, 360] {
      window.setContentSize(NSSize(width: width, height: 820))
      try await checkContentColumn("Markdown/composer column at \(width) points")
    }
    window.setContentSize(NSSize(width: 1400, height: 820))
    _ = try await session.webView.evaluateJavaScript(
      "document.querySelector('.thread__inner').classList.add('thread__inner--wide')")
    try await checkContentColumn("wide compare composer column", maximum: 1240)
    _ = try await session.webView.evaluateJavaScript(
      "document.querySelector('.thread__inner').classList.remove('thread__inner--wide')")
    window.setContentSize(NSSize(width: 1200, height: 820))
    try await checkContentColumn("composer column after compare closes")
    await session.setDark(true)
    try await check(
      "native charcoal palette",
      "getComputedStyle(document.documentElement).getPropertyValue('--pk-bg-base').trim()==='#1b1b1b'"
    )
    await session.setDark(false)
    try await check(
      "native light palette",
      "['#fff','#ffffff'].includes(getComputedStyle(document.documentElement).getPropertyValue('--pk-bg-base').trim())"
    )
    let retained = session.webView
    window.orderOut(nil)
    try await Task.sleep(for: .milliseconds(100))
    window.makeKeyAndOrderFront(nil)
    guard retained === session.webView else {
      throw Failure(message: "Reopen replaced the workspace")
    }
    try await check(
      "reopen preserves content", "document.body.textContent.includes('Shared renderer')")
    try await sendChecks(session)
    try await attachmentSelectionChecks(session)
    try await historyChecks(session)
    try await desktopChecks(session)
    try await messageChecks(session)
    let origin = session.webView.url!
    let (_, response) = try await URLSession(configuration: .ephemeral).data(
      from: origin.deletingLastPathComponent().appending(path: "api/server"))
    guard (response as? HTTPURLResponse)?.statusCode == 401 else {
      throw Failure(message: "Unauthenticated content access")
    }
    print("PASS: unauthenticated loopback access rejected")
    if let model = env["PADDOCK_WORKSPACE_LIVE_MODEL"] {
      let fleet = try await core.snapshot()
      guard
        fleet.runners.contains(where: {
          $0.model == model && $0.status == "ok" && ($0.inFlight ?? 0) == 0
        })
      else {
        throw Failure(
          message:
            "The requested live model must already be reachable and idle; this check never starts or stops it"
        )
      }
      await session.newChat()
      try await session.command("models", ["ids": .array([.string(model)])])
      try await session.command(
        "settings",
        [
          "maxTokens": .number(32), "webSearchEnabled": .bool(false),
          "params": .object(["thinking": .bool(false)]),
          "toolSelection": .object(["mode": .string("custom"), "picks": .array([])]),
        ])
      guard await session.send("Reply with exactly: Native composer ready.") else {
        throw Failure(message: session.error ?? "Live send refused")
      }
      try await until("real model response") {
        if let error = session.error { throw Failure(message: error) }
        return !session.busy
      }
      try await check(
        "real model reply persisted through shared relay",
        "const chat=document.querySelector('#app').__vue_app__.config.globalProperties.$pinia._s.get('chat');const c=await(await fetch('/api/conversations/'+chat.active.id)).json();const a=c.messages.at(-1);if(a.error)throw new Error(a.error);return a.role==='assistant'&&a.content.some(p=>p.type==='text'&&p.text.trim().length>0)&&!a.streaming"
      )
    }
  }
  func sendChecks(_ session: StudioWorkspace) async throws {
    try await session.command("renderer", ["mode": .string("native")])
    // Only the runner transport is synthetic. The command bridge, shared
    // orchestrator, streaming renderer and SQLite persistence are real.
    _ = try await session.webView.callAsyncJavaScript(
      #"""
      const pinia=document.querySelector('#app').__vue_app__.config.globalProperties.$pinia;
      const models=pinia._s.get('models');
      await models.refresh();
      models.models=[{id:'fixture',ownedBy:'test',kind:'chat',status:'ok',port:16555}];
      models.refresh=async()=>{};
      models.integrateRunnerRows=()=>{};
      models.caps.fixture={webSearch:false,mcpServers:[],taskTags:[],timestampGranularities:[],include:[],reasoning:'none',maxCtx:4096};
      models.currentId='fixture';pinia._s.get('chat').active.model='fixture';
      window.nativeRunnerCalls=0;window.nativeAborted=false;
      const original=window.fetch.bind(window);
      window.fetch=async (url,opts={})=>{
        if(window.failNextSave&&String(url).startsWith('/api/conversations/')&&opts.method==='PUT'){
          window.failNextSave=false;return new Response('Synthetic storage failure',{status:503});
        }
        if(String(url)==='/api/runners/16555/v1/responses'){
          window.nativeRunnerCalls++;window.nativeRequest=JSON.parse(opts.body);
          if(opts.signal.aborted)throw new DOMException('Aborted','AbortError');
          const encoder=new TextEncoder();
          const body=new ReadableStream({start(controller){
            const emit=value=>controller.enqueue(encoder.encode('data: '+JSON.stringify(value)+'\n\n'));
            emit({type:'response.output_text.delta',delta:'Native reply'});
            window.finishNativeResponse=()=>{
              emit({type:'response.completed',response:{id:'fixture-response',status:'completed',output:[{type:'message',role:'assistant',content:[{type:'output_text',text:'Native reply'}]}],usage:{input_tokens:5,output_tokens:2}}});
              controller.close();
            };
            opts.signal.addEventListener('abort',()=>{window.nativeAborted=true;controller.error(new DOMException('Aborted','AbortError'))},{once:true});
          }});
          return new Response(body,{headers:{'Content-Type':'text/event-stream'}});
        }
        if(/\/v1\/(responses|chat\/completions)/.test(String(url)))throw new Error('Unexpected live inference route in text fixture: '+url);
        return original(url,opts);
      };
      """#, arguments: [:], in: nil, contentWorld: .page)
    try await session.command(
      "settings",
      [
        "systemPrompt": .string("Native instructions"), "maxTokens": .number(32),
        "toolSelection": .object(["mode": .string("custom"), "picks": .array([])]),
      ])
    let payload: [String: StudioValue] = [
      "text": .string("Native send: `code` and \"quotes\""), "attachments": .array([]),
    ]
    let receipt = try await session.command("send", payload, id: "native-send-once")
    guard receipt["accepted"]?.boolean == true else {
      throw Failure(message: "No durable send receipt")
    }
    try await check(
      "native send persisted before receipt",
      "const c=await(await fetch('/api/conversations/workspace-fixture')).json();return c.messages.filter(m=>m.role==='user'&&m.content.some(p=>p.text?.startsWith('Native send:'))).length===1"
    )
    try await check(
      "native send uses shared Responses orchestration",
      "window.nativeRunnerCalls===1 && JSON.stringify(window.nativeRequest).includes('Native instructions')"
    )
    try await until("first streamed reply reaches native projection") {
      session.state?.nativeTranscript?.messages.last?.text == "Native reply"
    }
    try await session.command("send", payload, id: "native-send-once")
    try await check("duplicate command does not send twice", "window.nativeRunnerCalls===1")
    _ = try await session.webView.evaluateJavaScript("window.finishNativeResponse()")
    try await until("first response finishes") {
      if let error = session.error { throw Failure(message: error) }
      return !session.busy
    }
    try await until("settled stream publishes footer and run details") {
      guard let chrome = session.state?.nativeTranscript?.messages.last?.chrome else {
        return false
      }
      return chrome.footer.hasPrefix("2 tokens") && chrome.promptText == "Native instructions"
        && chrome.sections.contains(where: { $0.id == "provenance" })
    }
    guard await session.send("Stop this response") else {
      throw Failure(message: session.error ?? "Second send refused")
    }
    try await check("second response admitted", "window.nativeRunnerCalls===2")
    try await until("streaming text reaches native renderer") {
      session.state?.nativeTranscript?.messages.last?.text == "Native reply"
        && session.state?.nativeTranscript?.messages.last?.streaming == true
    }
    await session.cancel()
    try await until("stop drains the shared stream") { !session.busy }
    try await check(
      "stop aborts transport and retains partial text",
      "const chat=document.querySelector('#app').__vue_app__.config.globalProperties.$pinia._s.get('chat');return window.nativeAborted&&chat.active.messages.at(-1).stopped===true&&chat.active.messages.at(-1).content.some(p=>p.text==='Native reply')"
    )
    // Allow the final shared save to settle, then fail exactly the next PUT.
    try await check(
      "stopped turn persisted",
      "const c=await(await fetch('/api/conversations/workspace-fixture')).json();return c.messages.at(-1).stopped===true"
    )
    _ = try await session.webView.evaluateJavaScript("window.failNextSave=true")
    guard !(await session.send("Keep this unsent draft")) else {
      throw Failure(message: "Failed save falsely accepted the draft")
    }
    try await check(
      "failed receipt rolls back unaccepted turn",
      "const chat=document.querySelector('#app').__vue_app__.config.globalProperties.$pinia._s.get('chat');return !chat.active.messages.some(m=>m.content.some(p=>p.text==='Keep this unsent draft'))&&window.nativeRunnerCalls===2"
    )
    session.reload()
    try await until("reload restores the selected conversation") {
      session.ready && session.error == nil && session.conversation?.id == "workspace-fixture"
    }
    try await until("reload preserves native mode and stopped text") {
      session.state?.nativeTranscript?.messages.last?.text == "Native reply"
        && session.state?.nativeTranscript?.messages.last?.stopped == true
    }
    try await session.command("renderer", ["mode": .string("web")])
    try await check(
      "reload restores durable streamed content",
      "document.body.textContent.includes('Native reply') && document.body.textContent.includes('Stop this response')"
    )
  }
  func attachmentSelectionChecks(_ session: StudioWorkspace) async throws {
    await session.newChat()
    _ = try await session.webView.callAsyncJavaScript(
      #"""
      const pinia=document.querySelector('#app').__vue_app__.config.globalProperties.$pinia;
      const models=pinia._s.get('models');
      await models.refresh();
      models.models=[{id:'fixture',ownedBy:'test',kind:'chat',status:'ok',port:16555}];
      models.refresh=async()=>{};
      // Health push must not replace the synthetic catalog mid-test either.
      models.integrateRunnerRows=()=>{};
      models.caps.fixture={webSearch:false,mcpServers:[],taskTags:[],timestampGranularities:[],include:[],reasoning:'none',maxCtx:4096};
      models.currentId='fixture';pinia._s.get('chat').active.model='fixture';
      window.pdfRunnerCalls=0;
      const original=window.fetch.bind(window);
      window.fetch=async (url,opts={})=>{
        if(window.failPDFSave&&String(url).startsWith('/api/conversations/')&&opts.method==='PUT'){
          return new Response('Synthetic save failure',{status:503});
        }
        if(String(url)==='/api/runners')return Response.json([{model:'fixture',status:'ok',port:16555}]);
        if(String(url)==='/api/cloud')return Response.json([]);
        if(String(url)==='/api/runners/16555/v1/responses'){
          window.pdfRunnerCalls++;window.pdfRequest=JSON.parse(opts.body);
          const event={type:'response.completed',response:{id:'pdf-response',status:'completed',output:[{type:'message',role:'assistant',content:[{type:'output_text',text:'PDF selection received'}]}],usage:{input_tokens:5,output_tokens:3}}};
          return new Response('data: '+JSON.stringify(event)+'\n\n',{headers:{'Content-Type':'text/event-stream'}});
        }
        if(/\/v1\/(responses|chat\/completions)/.test(String(url)))throw new Error('Unexpected live inference route in PDF fixture: '+url);
        return original(url,opts);
      };
      """#, arguments: [:], in: nil, contentWorld: .page)
    try await session.command(
      "settings", ["toolSelection": .object(["mode": .string("custom"), "picks": .array([])])])
    let pdf = URL(fileURLWithPath: CommandLine.arguments[2]).appending(path: "page-selection.pdf")
    session.addFiles([pdf, pdf])
    try await until("two PDFs staged with worker page counts") {
      if let error = session.attachments.first(where: { $0.error != nil })?.error {
        throw Failure(message: error)
      }
      return !session.uploading && session.attachments.count == 2
        && session.attachments.allSatisfy { $0.ready && $0.pages == 6 }
    }
    session.attachments[0].firstPage = "invalid"
    guard !(await session.send("Must not send invalid pages")) else {
      throw Failure(message: "Invalid text admitted a PDF turn")
    }
    do {
      try await session.command(
        "send",
        [
          "text": .string("Invalid bridge range"),
          "attachments": .array([
            .object(["id": .string(session.attachments[0].id), "to": .number(7)])
          ]),
        ])
      throw Failure(message: "Bridge accepted pages beyond the known count")
    } catch is Failure {
      throw Failure(message: "Bridge accepted pages beyond the known count")
    } catch {}
    try await check("invalid PDF range never reaches runner", "window.pdfRunnerCalls===0")
    session.attachments[0].from = 2
    session.attachments[0].to = 4
    session.attachments[0].textOnly = true
    let ids = session.attachments.map(\.id)
    _ = try await session.webView.evaluateJavaScript("window.failPDFSave=true")
    let failedSend = await session.send("Keep PDF selections on save failure")
    _ = try await session.webView.evaluateJavaScript("window.failPDFSave=false")
    guard !failedSend,
      session.attachments.count == 2,
      session.attachments[0].from == 2, session.attachments[0].to == 4,
      session.attachments[0].textOnly
    else { throw Failure(message: "Failed admission discarded PDF selections") }
    try await check("failed PDF admission never reaches runner", "window.pdfRunnerCalls===0")
    guard await session.send("Compare the selected pages with the complete document") else {
      throw Failure(message: session.error ?? "PDF send refused")
    }
    try await until("PDF response finishes") { !session.busy }
    guard session.attachments.isEmpty else {
      throw Failure(message: "Accepted PDF attachments stayed staged")
    }
    try await check(
      "PDF range and reading mode reach the shared wire per file",
      "const files=window.pdfRequest?.input.flatMap(m=>m.content??[]).filter(p=>p.type==='input_file');return window.pdfRunnerCalls===1&&files?.length===2&&files[0].pages==='2-4'&&files[0].pdf_mode==='text'&&!files[1].pages&&!files[1].pdf_mode"
    )
    try await check(
      "PDF selection persists without embedding original bytes",
      "const chat=document.querySelector('#app').__vue_app__.config.globalProperties.$pinia._s.get('chat');const c=await(await fetch('/api/conversations/'+chat.active.id)).json();const files=c.messages.flatMap(m=>m.content).filter(p=>p.type==='file');return files.length===2&&files[0].pageRange==='2-4'&&files[0].pdfMode==='text'&&files[0].pages===6&&!files[1].pageRange&&!files.some(p=>p.file_data||p.path)"
    )
    let originals =
      try await session.webView.callAsyncJavaScript(
        "for(const id of ids){const r=await fetch('/api/attachments/'+id);if(!r.ok||!(await r.text()).startsWith('%PDF-'))return false}return true",
        arguments: ["ids": ids], in: nil, contentWorld: .page) as? Bool
    guard originals == true else { throw Failure(message: "PDF originals not retained for Lector") }
    print("PASS: original PDFs retained for Lector; only range metadata changes")
    try await session.command("renderer", ["mode": .string("native")])
    guard session.state?.nativeTranscript?.available == true,
      let message = session.state?.nativeTranscript?.messages.first(where: {
        $0.attachments?.count == 2
      })
    else { throw Failure(message: "Ordinary PDF turn lost its native transcript") }
    let savedConversation = session.conversation?.id
    for id in [ids[0], ids[1], ids[0]] {
      try await session.command(
        "openDocument", ["messageId": .string(message.id), "attachmentId": .string(id)])
      guard session.state?.nativeDocument?.id == id,
        session.state?.nativeTranscript?.available == true,
        session.conversation?.id == savedConversation
      else {
        throw Failure(message: "Document changed conversation or selected the wrong attachment")
      }
      try await check(
        "document-only Lector beside native chat",
        "document.querySelectorAll('.lector-workspace').length===1 && !!document.querySelector('.lector-page') && !document.querySelector('.thread') && !document.querySelector('.chatview__main') && !document.querySelector('.docpane__tabs') && !document.querySelector('[data-item-id=pk-fold]')"
      )
      let exact =
        try await session.webView.callAsyncJavaScript(
          "return window.paddockWorkspace.host.document.value.messages[0].content[0].attachmentId===id",
          arguments: ["id": id], in: nil, contentWorld: .page) as? Bool
      guard exact == true else {
        throw Failure(message: "Viewer did not receive selected original")
      }
      try await check(
        "native header replaces the viewer tab strip",
        "const tabs=document.querySelector('.lector-doctabs');return tabs&&getComputedStyle(tabs).display==='none'"
      )
      try await session.command("documentAction", ["action": .string("info")])
      try await check(
        "native details action opens shared metadata",
        "!!document.querySelector('.pv__content .pv__tabs')")
      if id == ids[0] { try await checkMetadataTheme(session) }
      _ = try await session.webView.evaluateJavaScript(
        "document.querySelector('.pv__content button[aria-label=Close]').click()")
      try await session.command("closePreview")
      try await check(
        "closing document unmounts viewer without mounting web chat",
        "!document.querySelector('.lector-workspace') && !document.querySelector('.lector-page') && !document.querySelector('.thread')"
      )
    }
    do {
      try await session.command(
        "openDocument", ["messageId": .string("missing-message"), "attachmentId": .string(ids[0])])
      throw Failure(message: "Unknown message opened a document")
    } catch is Failure { throw Failure(message: "Unknown message opened a document") } catch {}
    try await session.command("closePreview")
    try await checkLateDocumentOpen(session, messageId: message.id, ids: ids)
    try await check("document actions never invoke inference", "window.pdfRunnerCalls===1")
  }
  func checkLateDocumentOpen(_ session: StudioWorkspace, messageId: String, ids: [String])
    async throws
  {
    _ = try await session.webView.callAsyncJavaScript(
      """
      const original=window.fetch.bind(window);
      window.nativeDelayedStarted=false;window.nativeDelayedFinished=false;
      window.fetch=async(url,options)=>{
        if(String(url)==='/api/attachments/'+id&&!window.nativeDelayedStarted){
          window.nativeDelayedStarted=true;
          await new Promise(resolve=>window.nativeDelayedRelease=resolve);
          // Deliberately ignore cancellation to model a fetch that already
          // delivered its bytes. The pane's lifetime check must still win.
          const result=await original(url,{...options,signal:undefined});window.nativeDelayedFinished=true;return result;
        }
        return original(url,options);
      };
      """, arguments: ["id": ids[0]], in: nil, contentWorld: .page)
    try await session.command(
      "openDocument", ["messageId": .string(messageId), "attachmentId": .string(ids[0])])
    try await check("delayed PDF open starts", "window.nativeDelayedStarted")
    try await session.command("closePreview")
    try await session.command(
      "openDocument", ["messageId": .string(messageId), "attachmentId": .string(ids[1])])
    try await check(
      "new PDF opens while old fetch is pending", "!!document.querySelector('.lector-page')")
    _ = try await session.webView.evaluateJavaScript("window.nativeDelayedRelease()")
    try await check("old PDF fetch completes", "window.nativeDelayedFinished")
    try await Task.sleep(for: .milliseconds(500))
    try await check(
      "late open cannot retain an extra document or clear the new pane",
      "document.querySelectorAll('.lector-doctab').length===1 && !!document.querySelector('.lector-page') && !document.querySelector('.thread')"
    )
    guard session.state?.nativeDocument?.id == ids[1] else {
      throw Failure(message: "Late open replaced native document selection")
    }
    try await session.command("closePreview")
  }
  func check(_ name: String, _ expression: String) async throws {
    try await until(name) {
      guard let workspace else { return false }
      let source = expression.contains("return ") ? expression : "return (\(expression))"
      return try await workspace.webView.callAsyncJavaScript(
        source, arguments: [:], in: nil, contentWorld: .page) as? Bool == true
    }
    print("PASS: \(name)")
  }
  func checkContentColumn(_ name: String, maximum: Double = 760) async throws {
    try await until(name) {
      guard let workspace, let viewport = workspace.state?.viewport else { return false }
      let value =
        try await workspace.webView.callAsyncJavaScript(
          """
          const inner=document.querySelector('.thread__inner');
          const column=inner||document.querySelector('.native-content-column');
          const parent=document.querySelector('.thread')||document.querySelector('.chatview__main');
          if(!column||!parent)return null;
          const rect=column.getBoundingClientRect(),style=getComputedStyle(column);
          return {left:rect.left,width:rect.width,expected:Math.min(maximum,parent.clientWidth-56),padding:parseFloat(style.paddingLeft)+parseFloat(style.paddingRight)};
          """, arguments: ["maximum": maximum], in: nil, contentWorld: .page) as? [String: Double]
      guard let value, let left = value["left"], let width = value["width"],
        let expected = value["expected"], value["padding"] == 0
      else { return false }
      return abs(width - expected) < 0.75 && abs(viewport.left - left) < 0.75
        && abs(viewport.width - width) < 0.75
    }
    print("PASS: \(name)")
  }
  func until(_ name: String, _ predicate: () async throws -> Bool) async throws {
    let deadline = ContinuousClock.now.advanced(by: .seconds(30))
    while ContinuousClock.now < deadline {
      if try await predicate() { return }
      try await Task.sleep(for: .milliseconds(50))
    }
    throw Failure(message: "Timed out: \(name)")
  }
}
