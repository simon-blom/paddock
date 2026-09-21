import AppKit
import PaddockStudio
import ScreenCaptureKit
import SwiftUI
import WebKit

@testable import PaddockUI

extension Checks {
  @MainActor func messageChecks(_ session: StudioWorkspace) async throws {
    // All persistence uses the disposable functional-check database. Inference
    // is intercepted and unknown inference routes fail closed.
    _ = try await session.webView.callAsyncJavaScript(
      #"""
      const pinia=document.querySelector('#app').__vue_app__.config.globalProperties.$pinia;
      const chat=pinia._s.get('chat'),models=pinia._s.get('models');
      window.messageOriginalFetch=window.fetch.bind(window);window.messageRequests=[];window.messageMode='complete';
      models.models=[{id:'message-model',ownedBy:'test',kind:'chat',status:'ok',port:16556},{id:'original-model',ownedBy:'test',kind:'chat',status:'ok',port:16556}];
      models.refresh=async()=>{};models.integrateRunnerRows=()=>{};
      for(const id of ['message-model','original-model'])models.caps[id]={vision:true,webSearch:false,mcpServers:[],taskTags:[],timestampGranularities:[],include:[],reasoning:'none',maxCtx:4096};
      window.fetch=async(url,options={})=>{
        const path=String(url);
        if(path==='/api/conversations/message-fixture'&&options.method==='PUT'&&window.messageFailSave){window.messageFailSave=false;return new Response('Synthetic storage failure',{status:503})}
        if(path==='/api/runners/16556/v1/responses'){
          const body=JSON.parse(options.body);window.messageRequests.push(body);
          if(window.messageMode==='hold')return new Promise((resolve,reject)=>{options.signal.addEventListener('abort',()=>reject(new DOMException('Aborted','AbortError')),{once:true})});
          const incomplete=window.messageMode==='length',text=incomplete?'A partial answer.':' A completed answer.';
          const response={id:'synthetic-response',status:incomplete?'incomplete':'completed',output:[{type:'message',role:'assistant',content:[{type:'output_text',text}]}],usage:{input_tokens:8,output_tokens:4},...(incomplete?{incomplete_details:{reason:'max_output_tokens'}}:{})};
          return new Response([{type:'response.output_text.delta',delta:text},{type:incomplete?'response.incomplete':'response.completed',response}].map(e=>'data: '+JSON.stringify(e)+'\n\n').join(''),{headers:{'Content-Type':'text/event-stream'}});
        }
        if(/\/v1\/(responses|chat\/completions)/.test(path))throw new Error('Unexpected inference in message check');
        return window.messageOriginalFetch(url,options);
      };
      const c={id:'message-fixture',title:'Message actions fixture',model:'message-model',params:{thinking:false,reasoningEffort:'',stop:[]},systemPrompt:'',createdAt:1,updatedAt:2,leafId:'original-answer',messages:[{id:'original-question',parentId:null,role:'user',createdAt:1,content:[{type:'text',text:'Original question'}]},{id:'original-answer',parentId:'original-question',role:'assistant',model:'original-model',createdAt:2,content:[{type:'text',text:'Original answer'}]}]};
      const r=await fetch('/api/conversations/'+c.id,{method:'PUT',headers:{'Content-Type':'application/json'},body:JSON.stringify(c)});if(!r.ok)throw new Error('Fixture save failed');
      chat.conversations.push({...c,messages:[]});
      window.messageChat=chat;
      """#, arguments: [:], in: nil, contentWorld: .page)
    try await session.command("autoTitle", ["enabled": .bool(false)])
    try await session.command("renderer", ["mode": .string("native")])
    await session.open("message-fixture")
    func target(_ id: String) throws -> StudioMessageTarget {
      guard let transcript = session.state?.nativeTranscript,
        let target = StudioMessageTarget(transcript: transcript, messageId: id)
      else { throw Failure(message: "Missing native message target: \(id)") }
      return target
    }
    let original = try target("original-question")
    session.beginMessageEdit(original)
    session.messageEdit?.text = "Edited question"
    try await messageSurfaceChecks(session)
    await session.newChat()
    guard session.messageEdit?.text == "Edited question",
      session.conversation?.id == "message-fixture"
    else {
      throw Failure(message: "Navigation discarded the edit")
    }
    _ = try await session.webView.evaluateJavaScript("window.messageFailSave=true")
    guard await !session.submitMessageEdit(), session.messageEdit?.text == "Edited question" else {
      throw Failure(message: "Failed persistence discarded edit")
    }
    try await check(
      "failed native edit preserves draft/tree and sends no inference",
      "return window.messageChat.active.messages.length===2&&window.messageChat.active.leafId==='original-answer'&&window.messageRequests.length===0"
    )
    guard await session.submitMessageEdit(), !session.hasMessageEdit else {
      throw Failure(message: session.error ?? "Edit failed")
    }
    try await until("edited reply saved") { !session.busy }
    try await check(
      "edit creates a durable sibling and preserves original descendants",
      "const c=await(await fetch('/api/conversations/message-fixture')).json();return c.messages.length===4&&c.messages[0].content[0].text==='Original question'&&c.messages[1].content[0].text==='Original answer'&&c.messages[2].parentId===null&&c.messages[3].parentId===c.messages[2].id&&window.messageRequests.length===1"
    )
    let edited = try requireMessage(session.state?.nativeTranscript?.messages.first?.id)
    guard
      await session.messageAction("branch", target: try target(edited), branch: "original-question")
    else { throw Failure(message: session.error ?? "Branch failed") }
    guard session.state?.nativeTranscript?.leafId == "original-answer" else {
      throw Failure(message: "Original path not restored")
    }
    guard
      await session.messageAction("branch", target: try target("original-question"), branch: edited)
    else { throw Failure(message: "Edited path not restored") }
    let answer = try requireMessage(session.state?.nativeTranscript?.messages.last?.id)
    let stale = try target(answer)
    _ = try await session.webView.evaluateJavaScript("window.messageFailSave=true")
    guard await !session.messageAction("retry", target: stale) else {
      throw Failure(message: "Failed retry save accepted")
    }
    try await check(
      "failed retry admission retains the previous answer without inference",
      "return window.messageChat.active.messages.length===4&&window.messageRequests.length===1")
    _ = try await session.webView.evaluateJavaScript("window.messageMode='length'")
    guard await session.messageAction("retry", target: stale) else {
      throw Failure(message: session.error ?? "Retry failed")
    }
    try await until("retried answer ends") { !session.busy }
    let partial = try requireMessage(session.state?.nativeTranscript?.messages.last?.id)
    guard partial != answer,
      session.state?.nativeTranscript?.messages.last?.actions?.continueReply == true
    else { throw Failure(message: "Retry did not expose cutoff continuation") }
    guard await !session.messageAction("retry", target: stale) else {
      throw Failure(message: "Stale retry was admitted")
    }
    _ = try await session.webView.evaluateJavaScript("window.messageMode='complete'")
    _ = try await session.webView.evaluateJavaScript("window.messageFailSave=true")
    guard await !session.messageAction("continue", target: try target(partial)) else {
      throw Failure(message: "Failed continuation save accepted")
    }
    try await check(
      "failed continue admission preserves partial text and cutoff",
      "const m=window.messageChat.active.messages[4];return m.incomplete==='length'&&m.content[0].text==='A partial answer.'&&window.messageRequests.length===2"
    )
    try await session.command("models", ["ids": .array([.string("original-model")])])
    guard await session.messageAction("continue", target: try target(partial)) else {
      throw Failure(message: session.error ?? "Continue failed")
    }
    try await until("continued answer saved") { !session.busy }
    try await check(
      "continue appends to the same turn and retry keeps both answers",
      "const c=await(await fetch('/api/conversations/message-fixture')).json();const last=c.messages[c.messages.length-1];return c.messages.length===5&&last.content[0].text==='A partial answer. A completed answer.'&&!last.incomplete&&last.parentId===c.messages[3].parentId&&window.messageRequests.length===3&&window.messageRequests[2].model==='message-model'&&c.model==='original-model'"
    )
    _ = try await session.webView.evaluateJavaScript("window.messageFailSave=true")
    guard await !session.messageAction("branch", target: try target(partial), branch: answer) else {
      throw Failure(message: "Failed branch save accepted")
    }
    try await check(
      "failed branch persistence restores cursor and branch memory",
      "return window.messageChat.active.leafId===window.messageChat.active.messages[4].id")
    guard await session.messageAction("branch", target: try target(partial), branch: answer) else {
      throw Failure(message: "Answer branch failed")
    }
    guard
      await session.messageAction("branch", target: try target(edited), branch: "original-question")
    else { throw Failure(message: "Root branch failed") }
    guard
      await session.messageAction(
        "branch", target: try target("original-question"), branch: edited),
      session.state?.nativeTranscript?.leafId == answer
    else { throw Failure(message: "Nested branch choice not remembered") }
    // Retransmission uses one command receipt; concurrent callbacks cannot
    // create duplicate alternatives. Stop also interrupts this admitted action.
    _ = try await session.webView.evaluateJavaScript("window.messageMode='hold'")
    let retryPayload = try target(answer).payload(action: "retry")
    async let first = session.command("messageAction", retryPayload, id: "duplicate-retry")
    async let second = session.command("messageAction", retryPayload, id: "duplicate-retry")
    _ = try await (first, second)
    try await check(
      "duplicate native command admits one alternative",
      "return window.messageRequests.length===4&&window.messageChat.active.messages.length===6")
    await session.cancel()
    try await until("retry stops") { !session.busy }
    try await check(
      "stopped retry retains its branch and releases busy state",
      "const c=await(await fetch('/api/conversations/message-fixture')).json();return c.messages.length===6&&c.messages[5].stopped&&!c.messages[5].streaming"
    )
    session.reload()
    try await until("branches survive WebContent reload") {
      session.ready && !session.restoring && session.conversation?.id == "message-fixture"
        && session.state?.nativeTranscript?.messages.last?.stopped == true
    }
    print(
      "PASS: native edit/retry/continue/branch receipts, rollback, duplicate admission, cancellation and reload"
    )
  }

  @MainActor private func messageSurfaceChecks(_ session: StudioWorkspace) async throws {
    guard let window else { throw Failure(message: "No fixture window") }
    var draft = StudioDraft()
    draft.message = "Keep the separate composer draft"
    let host = NSHostingView(
      rootView: StudioConversationView(
        chat: session,
        draft: Binding(get: { draft }, set: { draft = $0 })))
    window.contentView = host
    window.makeKeyAndOrderFront(nil)
    defer { window.contentView = session.webView }
    func editors(_ view: NSView) -> [DraftTextView] {
      if let editor = view as? DraftTextView { return [editor] }
      return view.subviews.flatMap { editors($0) }
    }
    try await until("inline edit appears in the native transcript") {
      host.layoutSubtreeIfNeeded()
      return editors(host).contains {
        $0.accessibilityIdentifier() == "native-edit-text" && window.firstResponder === $0
          && $0.selectedRange().location == ("Edited question" as NSString).length
      }
    }
    let inline = editors(host).first { $0.accessibilityIdentifier() == "native-edit-text" }!
    let caret = NSRange(location: 3, length: 4)
    inline.setSelectedRange(caret)
    await session.perform("tools")  // An unrelated state projection cannot reset edit/caret.
    try await Task.sleep(for: .milliseconds(150))
    guard editors(host).contains(where: { $0 === inline }), inline.selectedRange() == caret,
      draft.message == "Keep the separate composer draft",
      session.messageEdit?.text == "Edited question"
    else {
      throw Failure(
        message:
          "Projection reset native editor: retained=\(editors(host).contains { $0 === inline }), selection=\(inline.selectedRange()), composerKept=\(draft.message == "Keep the separate composer draft"), editKept=\(session.messageEdit?.text == "Edited question")"
      )
    }
    if let directory = ProcessInfo.processInfo.environment["PADDOCK_UI_SNAPSHOT_DIR"] {
      let root = URL(fileURLWithPath: directory, isDirectory: true)
      try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
      for dark in [true, false] {
        window.appearance = NSAppearance(named: dark ? .darkAqua : .aqua)
        host.rootView = StudioConversationView(
          chat: session,
          draft: Binding(get: { draft }, set: { draft = $0 }))
        try await Task.sleep(for: .milliseconds(200))
        let content = try await SCShareableContent.excludingDesktopWindows(
          false, onScreenWindowsOnly: true)
        guard
          let source = content.windows.first(where: {
            $0.windowID == CGWindowID(window.windowNumber)
          })
        else { throw Failure(message: "Cannot capture native message fixture") }
        let config = SCStreamConfiguration()
        config.width = Int(window.frame.width * 2)
        config.height = Int(window.frame.height * 2)
        config.showsCursor = false
        let image = try await SCScreenshotManager.captureImage(
          contentFilter: SCContentFilter(desktopIndependentWindow: source), configuration: config)
        guard
          let png = NSBitmapImageRep(cgImage: image).representation(using: .png, properties: [:])
        else { throw Failure(message: "No native message screenshot") }
        try png.write(
          to: root.appending(path: "native-message-edit-\(dark ? "dark" : "light").png"))
      }
      window.appearance = nil
    }
    print("PASS: actual native editor retains identity, selection and independent composer draft")
  }
}

@MainActor private func requireMessage(_ value: String?) throws -> String {
  guard let value else { throw Checks.Failure(message: "Missing message ID") }
  return value
}
