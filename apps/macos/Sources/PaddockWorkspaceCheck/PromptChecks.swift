import Foundation
import PaddockStudio
import WebKit

extension Checks {
  /// No visible fixtures or model requests. This exercises the real private
  /// relay, shared stores and SQLite in the harness's disposable data root.
  @MainActor func promptChecks(_ session: StudioWorkspace) async throws {
    _ = try await session.webView.callAsyncJavaScript(
      #"""
      window.promptOriginalFetch=window.fetch.bind(window);
      window.fetch=async(url,options={})=>{
        const path=String(url);
        if(/\/v1\/(responses|chat\/completions)/.test(path))throw new Error('Inference is forbidden in prompt checks');
        if(window.promptFailSettings&&path==='/api/settings'&&options.method==='PUT'){window.promptFailSettings=false;return new Response('',{status:503})}
        if(window.promptFailChat&&path==='/api/conversations/prompt-chat'&&options.method==='PUT'){window.promptFailChat=false;return new Response('',{status:503})}
        return window.promptOriginalFetch(url,options);
      };
      const pinia=document.querySelector('#app').__vue_app__.config.globalProperties.$pinia;
      window.promptChat=pinia._s.get('chat');window.promptSettings=pinia._s.get('settings');
      const c={id:'prompt-chat',title:'Prompt check',model:'fixture',params:{thinking:false,reasoningEffort:'',stop:[]},systemPrompt:'Previous instructions',createdAt:1,updatedAt:2,messages:[]};
      const r=await fetch('/api/conversations/'+c.id,{method:'PUT',headers:{'Content-Type':'application/json'},body:JSON.stringify(c)});
      if(!r.ok)throw new Error('Fixture write failed');window.promptChat.conversations.push(c);
      """#, arguments: [:], in: nil, contentWorld: .page)
    await session.open("prompt-chat")
    try await headerChecks(session)
    let first = try await session.command(
      "promptSave",
      [
        "id": .string("fixture-preset"), "name": .string("Evidence"),
        "body": .string("Cite evidence."), "revision": .string(""),
      ])
    guard let oldRevision = first["prompt"]?.object?["revision"]?.text else {
      throw Failure(message: "Missing acknowledged prompt revision")
    }
    try await check(
      "native-created preset is visible through the shared web API",
      "const p=await(await fetch('/api/prompts')).json();return p.length===1&&p[0].body==='Cite evidence.'&&p[0].revision.length===64"
    )
    let list = try await session.command(
      "promptList", ["search": .string("evidence"), "page": .number(0)])
    guard list["library"]?.object?["rows"]?.array?.count == 1 else {
      throw Failure(message: "Preset search failed")
    }
    let second = try await session.command(
      "promptSave",
      [
        "id": .string("fixture-preset"), "name": .string("Evidence revised"),
        "body": .string("Cite stronger evidence."), "revision": .string(oldRevision),
      ])
    guard let revision = second["prompt"]?.object?["revision"]?.text, revision != oldRevision else {
      throw Failure(message: "Prompt revision did not advance")
    }
    do {
      try await session.command(
        "promptDelete", ["id": .string("fixture-preset"), "revision": .string(oldRevision)])
      throw Failure(message: "Stale preset deletion succeeded")
    } catch is Failure { throw Failure(message: "Stale preset deletion succeeded") } catch {}
    try await check(
      "stale native deletion keeps the newer shared preset",
      "return (await(await fetch('/api/prompts')).json())[0].name==='Evidence revised'")
    try await session.command(
      "instructionsApply",
      [
        "conversationId": .string("prompt-chat"), "expected": .string("Previous instructions"),
        "body": .string("Cite stronger evidence."),
      ])
    try await check(
      "native instructions are acknowledged in the conversation document",
      "return (await(await fetch('/api/conversations/prompt-chat')).json()).systemPrompt==='Cite stronger evidence.'"
    )
    _ = try await session.webView.evaluateJavaScript("window.promptFailChat=true")
    do {
      try await session.command(
        "instructionsApply",
        [
          "conversationId": .string("prompt-chat"), "expected": .string("Cite stronger evidence."),
          "body": .string("Must roll back"),
        ])
      throw Failure(message: "Failed instruction save succeeded")
    } catch is Failure { throw Failure(message: "Failed instruction save succeeded") } catch {}
    try await check(
      "failed instruction persistence restores runtime text",
      "return window.promptChat.active.systemPrompt==='Cite stronger evidence.'")
    try await session.command(
      "promptDelete", ["id": .string("fixture-preset"), "revision": .string(revision)])
    try await check(
      "deleting a library preset does not change chats using its copy",
      "return (await(await fetch('/api/prompts')).json()).length===0&&(await(await fetch('/api/conversations/prompt-chat')).json()).systemPrompt==='Cite stronger evidence.'"
    )
    let before = try await session.command("preferencesGet")
    let expected = before["preferences"]?.object ?? [:]
    let layout = expected["layout"]?.object
    guard
      layout?["sections"]?.array?.compactMap({ $0.object?["id"]?.text })
        == ["maxTokens", "maxToolCalls", "summarize", "microphone", "mapTiles"],
      layout?["toolStops"]?.array?.compactMap({ $0.object?["value"]?.number })
        == [0, 5, 10, 25, 50, 100],
      layout?["replyLimit"]?.object?["maximum"] == .number(1_048_576)
    else {
      throw Failure(message: "Native settings presentation differs from the shared web layout")
    }
    print("PASS: native settings receives web section order and choices over the real bridge")
    let changed: [String: StudioValue] = [
      "maxTokens": .number(8192), "maxToolCalls": .number(25),
      "mapTiles": .string("https://example.invalid/{z}/{x}/{y}.png"),
    ]
    try await session.command(
      "preferencesSave", ["changes": .object(changed), "expected": .object(expected)])
    try await check(
      "native preferences share runtime settings and durable SQLite",
      "const p=await(await fetch('/api/settings')).json();return p['studio.pk_max_tokens']==='8192'&&p['studio.pk_max_tokens_v2']==='1'&&p['studio.pk_max_tool_calls']==='25'&&window.promptSettings.maxToolCalls===25"
    )
    _ = try await session.webView.evaluateJavaScript("window.promptFailSettings=true")
    do {
      try await session.command(
        "preferencesSave",
        [
          "changes": .object(["maxToolCalls": .number(50)]),
          "expected": .object(["maxToolCalls": .number(25)]),
        ])
      throw Failure(message: "Failed preference save succeeded")
    } catch is Failure { throw Failure(message: "Failed preference save succeeded") } catch {}
    try await check(
      "failed preference save rolls runtime back",
      "return window.promptSettings.maxToolCalls===25"
    )
    try await session.command("preferencesGet")
    // Clear the displayed command failure before recovery's ready check.
    await session.perform("preferencesGet")
    session.reload()
    try await until("preferences workspace reload") { session.ready && !session.restoring }
    let restored = try await session.command("preferencesGet")
    guard restored["preferences"]?.object?["maxTokens"] == .number(8192),
      restored["preferences"]?.object?["maxToolCalls"] == .number(25)
    else {
      throw Failure(message: "Explicit preferences lost across content restart")
    }
    print("PASS: explicit 8192 reply limit and tool budget survive a fresh WebKit page")
  }

  @MainActor private func headerChecks(_ session: StudioWorkspace) async throws {
    // Synthetic fleet only. Keep the actual command/store/SQLite path while
    // making model capability reads deterministic and forbidding inference.
    _ = try await session.webView.callAsyncJavaScript(
      #"""
      const pinia=document.querySelector('#app').__vue_app__.config.globalProperties.$pinia;
      const m=pinia._s.get('models');
      m.models=[
        {id:'fixture',display:'Fixture vision',ownedBy:'test',vendor:'Alibaba',kind:'chat',status:'ok',port:16555,vision:true,spec:'MTP'},
        {id:'other',display:'Other model',ownedBy:'test',vendor:'IBM',kind:'chat',status:'ok',port:16556,vision:false,spec:'off'}
      ];
      m.refresh=async()=>{};m.integrateRunnerRows=()=>{};m.fetchLimits=async()=>{};
      for(const x of m.models)m.caps[x.id]={vision:x.vision,reasoning:'none',maxCtx:4096,mcpServers:[],taskTags:[]};
      m.capsFor=async id=>m.caps[id];m.currentId='other';
      """#, arguments: [:], in: nil, contentWorld: .page)
    try await until("header follows conversation, not fleet seat") {
      session.state?.modelHeader?.currentModel == "fixture"
        && session.state?.modelHeader?.current?.label == "Fixture vision"
        && session.state?.modelHeader?.isVision == true
        && session.state?.modelHeader?.specLabel == "MTP"
    }
    print("PASS: shared header target and badges reach Swift independently of the fleet seat")
    try await session.command("models", ["ids": .array([.string("fixture"), .string("other")])])
    try await until("compare header names") {
      session.state?.modelHeader?.comparing == true
        && session.state?.modelHeader?.compareLanes.map(\.id) == ["fixture", "other"]
    }
    try await session.command("models", ["ids": .array([.string("other")])])
    try await until("header selection retargets chat and clears compare") {
      session.state?.modelHeader?.currentModel == "other"
        && session.state?.modelHeader?.comparing == false
        && session.state?.selectedModels == ["other"]
    }
    try await check(
      "header model selection persists the exact next-send target and clears compare",
      "const c=await(await fetch('/api/conversations/prompt-chat')).json();return c.model==='other'&&!c.compareModels"
    )
    try await session.command("models", ["ids": .array([.string("fixture")])])
  }
}
