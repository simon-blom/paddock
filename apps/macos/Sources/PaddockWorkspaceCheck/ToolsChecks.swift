import Foundation
import PaddockClient
import PaddockStudio
import WebKit

extension Checks {
  /// Offscreen, disposable SQLite + real Swift/WebKit command path. Listings
  /// are synthetic; no inference or external MCP request is allowed.
  @MainActor func toolsChecks(_ session: StudioWorkspace) async throws {
    guard let core else { throw Failure(message: "Missing fixture core") }
    var draft = ConnectorDraft()
    draft.label = "fixture-connector"
    draft.url = "https://example.invalid/mcp"
    let saved = try await core.integration(.save(draft))
    guard let connectorID = saved.savedId else {
      throw Failure(message: "Missing fixture connector")
    }
    _ = try await session.webView.callAsyncJavaScript(
      #"""
      const pinia=document.querySelector('#app').__vue_app__.config.globalProperties.$pinia;
      const chat=pinia._s.get('chat'),models=pinia._s.get('models');
      window.toolsOriginalFetch=window.fetch.bind(window);
      models.models=[{id:'tools-fixture-model',ownedBy:'test',kind:'chat',status:'ok'}];
      models.refresh=async()=>{};models.integrateRunnerRows=()=>{};
      models.caps['tools-fixture-model']={vision:false,webSearch:false,mcpServers:[],taskTags:[],timestampGranularities:[],include:[],reasoning:'none',maxCtx:4096};
      models.capsFor=async()=>models.caps['tools-fixture-model'];
      window.fetch=async(url,options={})=>{
        const path=String(url);
        if(/\/v1\/(responses|chat\/completions)/.test(path))throw new Error('Inference is not allowed in tools checks');
        if(path==='/api/mcp/tools')return new Response(JSON.stringify({ok:true,tools:[{name:'create_issue',description:'Create a repository issue'},{name:'get_issue',description:'Read a repository issue'}]}),{headers:{'Content-Type':'application/json'}});
        if(path==='/api/conversations/tools-parity-fixture'&&options.method==='PUT'&&window.toolsFailSave){window.toolsFailSave=false;return new Response('Synthetic save failure',{status:503})}
        return window.toolsOriginalFetch(url,options);
      };
      pinia._s.get('mcpTools').invalidate();
      const c={id:'tools-parity-fixture',title:'Tools parity fixture',model:'tools-fixture-model',params:{thinking:false,reasoningEffort:'',stop:[]},systemPrompt:'',createdAt:1,updatedAt:2,messages:[],toolSelection:{mode:'all'},connectorIds:[connectorID]};
      const r=await fetch('/api/conversations/'+c.id,{method:'PUT',headers:{'Content-Type':'application/json'},body:JSON.stringify(c)});
      if(!r.ok)throw new Error('Fixture save failed');chat.conversations.push(c);window.toolsChat=chat;
      """#, arguments: ["connectorID": connectorID], in: nil, contentWorld: .page)
    await session.open("tools-parity-fixture")
    try await session.command("tools")
    try await until("tool listings ready") {
      session.state?.tools.count == 2
        && session.state?.tools.allSatisfy { $0.status == "ok" } == true
    }
    guard session.state?.tools.first(where: { $0.id == "artifacts" })?.checked == "none",
      session.state?.tools.first(where: { $0.id == "fixture-connector" })?.checked == "all"
    else { throw Failure(message: "Native All-mode presentation diverged from web") }
    print("PASS: native All row owns server coverage; connectors remain opt-in")
    try await session.command("toolQuery", ["query": .string("crisu repository")])
    guard session.state?.tools.count == 2,
      session.state?.tools.allSatisfy({ $0.tools.count == 1 && $0.tools[0].name == "create_issue" })
        == true
    else { throw Failure(message: "Native multi-word fuzzy tool search diverged") }
    print("PASS: native tool search uses shared fuzzy multi-word matching")
    try await session.command("toolQuery", ["query": .string("")])
    try await session.command(
      "toolPicker",
      ["action": .string("tool"), "label": .string("artifacts"), "tool": .string("create_issue")])
    try await check(
      "immediate native tool selection is durable and preserves opt-in connectors",
      "const c=await(await fetch('/api/conversations/tools-parity-fixture')).json();return c.toolSelection.mode==='custom'&&c.toolSelection.picks.length===2&&c.toolSelection.picks.some(p=>p.label==='artifacts'&&p.tool==='create_issue')&&c.toolSelection.picks.some(p=>p.label==='fixture-connector'&&!p.tool)"
    )
    _ = try await session.webView.evaluateJavaScript("window.toolsFailSave=true")
    do {
      try await session.command(
        "toolPicker", ["action": .string("group"), "label": .string("fixture-connector")])
      throw Failure(message: "Failed selection save was accepted")
    } catch is Failure { throw Failure(message: "Failed selection save was accepted") } catch {}
    try await check(
      "failed native tool save restores selection without executing tools",
      "return window.toolsChat.active.toolSelection.picks.some(p=>p.label==='fixture-connector'&&!p.tool)"
    )
    try await session.command("toolPicker", ["action": .string("all")])
    try await check(
      "All mode is restored durably without clearing opt-in identities",
      "const c=await(await fetch('/api/conversations/tools-parity-fixture')).json();return c.toolSelection.mode==='all'&&c.connectorIds.length===1"
    )
    do {
      try await session.command(
        "toolPicker",
        ["action": .string("tool"), "label": .string("artifacts"), "tool": .string("missing")])
      throw Failure(message: "Unknown tool accepted")
    } catch is Failure { throw Failure(message: "Unknown tool accepted") } catch {}
    print("PASS: stale native tool choices rejected")
  }
}
