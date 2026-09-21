import Foundation
import PaddockStudio
import SQLite3
import WebKit

extension Checks {
  @MainActor func nativeOnlyChecks(_ session: StudioWorkspace) async throws {
    guard session.state?.nativeTranscript?.available == true else {
      throw Failure(message: "Default app did not use native rendering")
    }
    do {
      try await session.command("renderer", ["mode": .string("web")])
      throw Failure(message: "Web renderer switch was accepted")
    } catch is Failure { throw Failure(message: "Web renderer switch was accepted") } catch {}
    let id =
      try await session.webView.callAsyncJavaScript(
        #"""
        const pinia=document.querySelector('#app').__vue_app__.config.globalProperties.$pinia;
        const models=pinia._s.get('models'),chat=pinia._s.get('chat');
        models.models=[{id:'fixture',kind:'chat',status:'ok',port:16555}];
        models.refresh=async()=>{};models.integrateRunnerRows=()=>{};models.currentId='fixture';
        models.caps.fixture={webSearch:false,mcpServers:[],reasoning:'none',maxCtx:4096};
        const c=chat.commitDraft('fixture');
        chat.addMessage(c,{id:'u',role:'user',content:[{type:'text',text:'Native rich messages'}],createdAt:1});
        chat.addMessage(c,{id:'a',role:'assistant',model:'fixture',group:'compare',content:[{type:'text',text:'Visible answer'}],createdAt:2,
          toolCalls:[{id:'call',name:'read_file',serverLabel:'files',arguments:'{"path":"synthetic.txt"}',status:'pending',approvalId:'approval'}],
          webSearches:[{id:'search',query:'Synthetic search',status:'completed',sources:[{url:'https://example.com',title:'Synthetic source'}]}]},'u');
        chat.addMessage(c,{id:'b',role:'assistant',model:'fixture',group:'compare',content:[{type:'text',text:'OCR answer'}],createdAt:2,
          docRun:{pages:[{state:'done',text:'Native document result',regions:[{label:'text',boxes:[[1,2,3,4]]}]}]}},'u');
        await chat.persistNow(c,true);
        window.nativeApprovalCalls=[];
        const original=window.fetch.bind(window);
        window.fetch=async(url,options={})=>{
          if(String(url)==='/api/runners/16555/mcp-approvals/approval'){
            window.nativeApprovalCalls.push(JSON.parse(options.body));return Response.json({ok:true,approved:JSON.parse(options.body).approve});
          }
          if(/\/v1\/(responses|chat\/completions)/.test(String(url)))throw Error('Unexpected inference in rendering boundary test');
          return original(url,options);
        };
        return c.id;
        """#, arguments: [:], in: nil, contentWorld: .page) as? String
    guard let id else { throw Failure(message: "Missing fixture conversation") }
    try await session.command("open", ["id": .string(id)])
    try await until("native tools, sources, OCR and whole Compare group") {
      guard let transcript = session.state?.nativeTranscript else { return false }
      return transcript.available && transcript.hasComparisons && transcript.messages.count == 3
        && transcript.messages[1].toolCalls?.first?.approvalId == "approval"
        && transcript.messages[1].searches?.first?.sources.count == 1
        && transcript.messages[2].documentResult?.pages.first?.text == "Native document result"
    }
    guard let transcript = session.state?.nativeTranscript,
      let target = StudioMessageTarget(transcript: transcript, messageId: "a")
    else {
      throw Failure(message: "Missing approval target")
    }
    var decision = target.payload(action: "approve")
    decision["callId"] = .string("call")
    decision["approvalId"] = .string("approval")
    decision["approve"] = .bool(true)
    try await session.command("toolApproval", decision)
    try await check(
      "native approval reaches the owning runner exactly once",
      "window.nativeApprovalCalls.length===1&&window.nativeApprovalCalls[0].approve===true")
    _ = try await session.webView.evaluateJavaScript(
      "document.querySelector('#app').__vue_app__.config.globalProperties.$pinia._s.get('chat').active.messages.find(m=>m.id==='a').toolCalls[0].status='completed'"
    )
    do {
      try await session.command("toolApproval", decision)
      throw Failure(message: "Stale approval accepted")
    } catch is Failure { throw Failure(message: "Stale approval accepted") } catch {}
    try await check(
      "stale native approval does not call the runner", "window.nativeApprovalCalls.length===1")
    _ = try await session.webView.evaluateJavaScript(
      "document.querySelector('#app').__vue_app__.config.globalProperties.$pinia._s.get('chat').active.messages.find(m=>m.id==='a').content=[{type:'text',text:'🦊'.repeat(160000)+'END'}]"
    )
    try await until("framed 640 KiB Unicode transcript arrives intact") {
      let text = session.state?.nativeTranscript?.messages.first { $0.id == "a" }?.text
      return text?.count == 160003 && text?.hasSuffix("END") == true
    }
    try await session.command("draft", ["text": .string("Command acknowledgment stays bounded")])
    try await check(
      "no web chat controls or hidden message bodies exist",
      "!!document.querySelector('[data-native-surface=embedded-viewers]')&&!document.querySelector('.thread,.msg,.composer,.native-audio-preview,.tc,.ws,.transcript')&&!document.body.textContent.includes('Visible answer')"
    )
    // Seed only our fresh synthetic database. The runner-only MCP callback
    // deliberately rejects browser cookies; never weaken that security gate
    // or expose its capability just to create a test artifact.
    let artifactId = try seedArtifact(conversation: id)
    try await session.command("refresh")
    try await until("native artifact listing") {
      session.state?.nativeArtifacts?.contains { $0.id == artifactId } == true
    }
    let original = try await session.artifactContent(artifactId)
    guard original.body == "<div>Native source</div>" else {
      throw Failure(message: "Native artifact body changed")
    }
    session.artifactDrafts[artifactId] = .init(saved: original.body)
    session.artifactDrafts[artifactId]?.text = "<div>Edited natively</div>"
    guard session.hasArtifactEdits else {
      throw Failure(message: "Native edits missing from quit protection")
    }
    try await session.saveArtifact(artifactId)
    let saved = try await session.artifactContent(artifactId)
    guard saved.body == "<div>Edited natively</div>", saved.versions.count == 2,
      session.artifactDrafts[artifactId] == nil
    else { throw Failure(message: "Native artifact version was not saved") }
    session.artifactDrafts[artifactId] = .init(saved: saved.body)
    session.artifactDrafts[artifactId]?.text = "Retain my pending edit"
    _ = try await session.webView.callAsyncJavaScript(
      "return (await fetch('/api/artifacts/'+id+'/content',{method:'PUT',headers:{'content-type':'text/plain'},body:'Concurrent synthetic update'})).ok",
      arguments: ["id": artifactId], in: nil, contentWorld: .page)
    do {
      try await session.saveArtifact(artifactId)
      throw Failure(message: "Stale artifact edit overwrote newer content")
    } catch is Failure {
      throw Failure(message: "Stale artifact edit overwrote newer content")
    } catch {}
    guard session.artifactDrafts[artifactId]?.text == "Retain my pending edit" else {
      throw Failure(message: "Conflict lost native draft")
    }
    session.artifactDrafts[artifactId] = nil  // Explicit cleanup of this synthetic edit only.
    try await check(
      "HTML artifact source never creates a browser frame", "!document.querySelector('iframe')")
    print("PASS: native artifact original/version reads, edit save and conflict retention")
    print("PASS: native-only rich message, approval and large-state checks")
  }
  private func seedArtifact(conversation: String) throws -> String {
    guard let root = ProcessInfo.processInfo.environment["PADDOCK_DATA"],
      root.hasPrefix("/tmp/paddock-workspace-check."),
      conversation.range(of: "^[a-zA-Z0-9_-]+$", options: .regularExpression) != nil
    else { throw Failure(message: "Refusing artifact fixture outside synthetic storage") }
    var database: OpaquePointer?
    guard sqlite3_open_v2(root + "/paddock.db", &database, SQLITE_OPEN_READWRITE, nil) == SQLITE_OK
    else {
      sqlite3_close(database)
      throw Failure(message: "Cannot open synthetic artifact database")
    }
    defer { sqlite3_close(database) }
    let id =
      "art_" + UUID().uuidString.replacingOccurrences(of: "-", with: "").lowercased().prefix(12)
    let sql = """
      BEGIN IMMEDIATE;
      INSERT INTO artifacts(id,conversation_id,kind,title,language,model,created_at,updated_at)
        VALUES('\(id)','\(conversation)','html','Native artifact fixture','html','fixture',1,1);
      INSERT INTO artifact_versions(artifact_id,seq,op,content,created_at)
        VALUES('\(id)',1,'create','<div>Native source</div>',1);
      COMMIT;
      """
    guard sqlite3_exec(database, sql, nil, nil, nil) == SQLITE_OK else {
      sqlite3_exec(database, "ROLLBACK", nil, nil, nil)
      throw Failure(message: "Cannot seed synthetic artifact")
    }
    return id
  }
}
