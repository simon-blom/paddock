import AppKit
import PaddockStudio
import WebKit

extension Checks {
  @MainActor func historyChecks(_ session: StudioWorkspace) async throws {
    // Isolated fixture DB and intercepted transport; never inspect a user's
    // conversations, credentials or model output in this executable.
    _ = try await session.webView.callAsyncJavaScript(
      #"""
      const pinia=document.querySelector('#app').__vue_app__.config.globalProperties.$pinia;
      const chat=pinia._s.get('chat'),models=pinia._s.get('models');
      await models.refresh();
      models.models=[{id:'fixture',ownedBy:'test',kind:'chat',status:'ok',port:16555}];
      models.refresh=async()=>{};
      models.integrateRunnerRows=()=>{};
      models.caps.fixture={vision:true,webSearch:false,mcpServers:[],taskTags:[],timestampGranularities:[],include:[],reasoning:'none',maxCtx:4096};
      models.currentId='fixture';
      const original=window.fetch.bind(window);
      window.historyTitleCalls=0;
      window.historyFailedReads=[];
      window.fetch=async(url,options={})=>{
        const path=String(url);
        if(window.historyFailDelete&&path==='/api/conversations/history-two'&&options.method==='DELETE'){
          window.historyFailDelete=false;return new Response('Storage failure',{status:503});
        }
        if(window.historyFailRename&&path==='/api/conversations/history-one'&&options.method==='PUT'){
          window.historyFailRename=false;return new Response('Storage failure',{status:503});
        }
        if(path==='/api/runners/16555/v1/responses'){
          const body=JSON.parse(options.body);
          if(body.stream===false){
            window.historyTitleCalls++;window.historyTitleRequest=body;
            return Response.json({status:'completed',output:[{type:'message',content:[{type:'output_text',text:'Designing a native workspace'}]}]});
          }
          window.historyChatRequest=body;
          const events=[{type:'response.output_text.delta',delta:'A synthetic design reply.'},{type:'response.completed',response:{id:'history-response',status:'completed',output:[{type:'message',role:'assistant',content:[{type:'output_text',text:'A synthetic design reply.'}]}],usage:{input_tokens:8,output_tokens:6}}}];
          return new Response(events.map(value=>'data: '+JSON.stringify(value)+'\n\n').join(''),{headers:{'Content-Type':'text/event-stream'}});
        }
        if(/\/v1\/(responses|chat\/completions)/.test(path))throw new Error('Unexpected inference in history check: '+path);
        const result=await original(url,options);
        if(path.startsWith('/api/conversations/')&&!options.method&&!result.ok)window.historyFailedReads.push({path,status:result.status,active:chat.active?.id,draft:chat.active?chat.isDraft(chat.active):null,stack:new Error().stack});
        return result;
      };
      for(const id of ['history-one','history-two']){
        const c={id,title:'Saved '+id,model:'fixture',params:{thinking:false,reasoningEffort:'',stop:[]},systemPrompt:'',createdAt:1,updatedAt:2,leafId:'answer',messages:[{id:'question',parentId:null,role:'user',createdAt:1,content:[{type:'text',text:'Preserve this question'}]},{id:'answer',parentId:'question',role:'assistant',createdAt:2,content:[{type:'text',text:'Preserve this answer'}]}]};
        const r=await fetch('/api/conversations/'+id,{method:'PUT',headers:{'Content-Type':'application/json'},body:JSON.stringify(c)});
        if(!r.ok)throw new Error('Fixture save failed');
        chat.conversations.push({...c,messages:[]});
      }
      """#, arguments: [:], in: nil, contentWorld: .page)
    guard
      await session.changeHistory(
        "renameChat", ids: ["history-one"], title: "Native renamed conversation")
    else {
      throw Failure(message: session.error ?? "Rename failed")
    }
    try await check(
      "native rename hydrates the full document and saves metadata only",
      "const c=await(await fetch('/api/conversations/history-one')).json();return c.title==='Native renamed conversation'&&c.titleSource==='manual'&&c.messages.length===2&&c.updatedAt===2"
    )
    let longTitle = String(repeating: "Read every word 👋 ", count: 20)
      .trimmingCharacters(in: .whitespacesAndNewlines)
    guard await session.changeHistory("renameChat", ids: ["history-one"], title: longTitle) else {
      throw Failure(message: "Long history title could not be saved")
    }
    try await session.command(
      "historyFilter",
      ["search": .string("Read every word"), "sort": .string("newest"), "page": .number(0)])
    guard session.state?.library?.rows.first?.title == longTitle else {
      throw Failure(message: "The history projection truncated a valid full title")
    }
    print(
      "PASS: full Unicode conversation title is preserved in the native history projection")
    guard
      await session.changeHistory(
        "renameChat", ids: ["history-one"], title: "Native renamed conversation")
    else {
      throw Failure(message: "Fixture rename restoration failed")
    }
    try await session.command(
      "historyFilter", ["search": .string(""), "sort": .string("newest"), "page": .number(0)])
    _ = try await session.webView.evaluateJavaScript("window.historyFailRename=true")
    guard
      !(await session.changeHistory("renameChat", ids: ["history-one"], title: "Must roll back"))
    else {
      throw Failure(message: "Failed rename reported success")
    }
    try await check(
      "failed rename preserves old native label",
      "return window.paddockWorkspace.state().history.some(c=>c.id==='history-one'&&c.title==='Native renamed conversation')"
    )
    guard await session.changeHistory("pinChat", ids: ["history-one"]) else {
      throw Failure(message: "Pin failed")
    }
    try await check(
      "pin is durable and does not change chat activity",
      "const c=await(await fetch('/api/conversations/history-one')).json();return c.pinned===true&&c.updatedAt===2"
    )
    _ = try await session.webView.evaluateJavaScript("window.historyFailDelete=true")
    guard !(await session.changeHistory("deleteChats", ids: ["history-two"])) else {
      throw Failure(message: "Failed delete reported success")
    }
    try await check(
      "failed deletion keeps the row and saved conversation",
      "return window.paddockWorkspace.state().history.some(c=>c.id==='history-two')&&(await fetch('/api/conversations/history-two')).ok"
    )
    guard await session.changeHistory("deleteChats", ids: ["history-two"]) else {
      throw Failure(message: "Delete retry failed")
    }
    try await check(
      "acknowledged delete removes exactly the target",
      "return (await fetch('/api/conversations/history-two')).status===404&&(await fetch('/api/conversations/history-one')).ok"
    )
    await session.newChat()
    try await session.command("models", ["ids": .array([.string("fixture")])])
    try await session.command(
      "settings",
      [
        "webSearchEnabled": .bool(false),
        "toolSelection": .object(["mode": .string("custom"), "picks": .array([])]),
      ])
    try await session.command("autoTitle", ["enabled": .bool(true)])
    guard await session.send("Help design a native workspace") else {
      throw Failure(message: session.error ?? "Title fixture send failed")
    }
    try await until("generated label reaches the native projection") {
      session.conversation?.title == "Designing a native workspace" && !session.busy
    }
    let titledID = session.conversation!.id
    try await check(
      "desktop activity reports a real completed turn without private content",
      "const a=window.paddockWorkspace.state().activity;return a.completedTurn?.conversationId===window.paddockWorkspace.state().conversation.id&&a.completedTurn.state==='completed'&&a.replies.every(r=>Object.keys(r).sort().join(',')==='id,state')&&!JSON.stringify(a).includes('native workspace')"
    )
    try await check(
      "one automatic title request is text-only, stateless and separate from chat",
      "const r=window.historyTitleRequest;return window.historyTitleCalls===1&&r.stream===false&&r.store===false&&r.max_output_tokens===128&&!r.tools&&!r.previous_response_id&&!r.input.includes('data:')"
    )
    guard await session.changeHistory("renameChat", ids: [titledID], title: "My permanent name")
    else { throw Failure(message: "Manual rename failed") }
    guard await session.send("Continue this synthetic conversation") else {
      let detail = try await session.webView.evaluateJavaScript(
        "JSON.stringify((()=>{const c=document.querySelector('#app').__vue_app__.config.globalProperties.$pinia._s.get('chat');return {activeId:c.activeId,active:c.active?.id,draft:c.active?c.isDraft(c.active):null,loading:c.activeLoading,failed:c.activeLoadFailed,ids:c.conversations.map(x=>x.id),reads:window.historyFailedReads,path:location.pathname}})())"
      )
      throw Failure(
        message: "Follow-up failed: \(session.error ?? "none"); \(String(describing: detail))")
    }
    try await until("follow-up ends") { !session.busy }
    try await check(
      "manual name survives another reply without a title request",
      "return window.historyTitleCalls===1&&window.paddockWorkspace.state().conversation.title==='My permanent name'"
    )
    // Paste a standards-based PNG into the real native upload pipeline. It
    // must be retained as an original, thumbnailed, previewable and sendable.
    let bitmap = NSBitmapImageRep(
      bitmapDataPlanes: nil, pixelsWide: 32, pixelsHigh: 24, bitsPerSample: 8, samplesPerPixel: 4,
      hasAlpha: true, isPlanar: false, colorSpaceName: .deviceRGB, bytesPerRow: 0, bitsPerPixel: 0)!
    bitmap.bitmapData!.initialize(repeating: 160, count: bitmap.bytesPerRow * bitmap.pixelsHigh)
    for y in 0..<bitmap.pixelsHigh {
      for x in 0..<bitmap.pixelsWide {
        bitmap.bitmapData![y * bitmap.bytesPerRow + x * 4 + 3] = 255
      }
    }
    let png = bitmap.representation(using: .png, properties: [:])!
    session.addPastedImage(png, png: true)
    try await until("pasted image upload and thumbnail") {
      session.attachments.last?.ready == true && !session.uploading
    }
    guard let image = session.attachments.last, image.width == 32, image.height == 24,
      image.thumbnail != nil
    else { throw Failure(message: "Image metadata did not reach native attachment chip") }
    try await session.command("preview", ["id": .string(image.id)])
    guard session.state?.nativeDocument?.kind == "image" else {
      throw Failure(message: "Image preview not exposed")
    }
    try await session.command("closePreview")
    let dropped = NSItemProvider(item: png as NSData, typeIdentifier: "public.png")
    guard session.addDroppedItems([dropped]), session.uploading else {
      throw Failure(message: "Image drop did not reserve attachment admission")
    }
    try await until("image provider drop becomes a second ready attachment") {
      session.attachments.count == 2 && session.attachments.allSatisfy(\.ready)
        && !session.uploading
    }
    guard await session.send("Describe this fixture image") else {
      throw Failure(message: session.error ?? "Image send failed")
    }
    try await until("image response ends") { !session.busy }
    try await check(
      "image bytes reach the shared request without entering the stored chat",
      "const chat=document.querySelector('#app').__vue_app__.config.globalProperties.$pinia._s.get('chat');const c=await(await fetch('/api/conversations/'+chat.active.id)).json();return JSON.stringify(window.historyChatRequest.input).includes('input_image')&&c.messages.some(m=>m.content.some(p=>p.type==='image'&&p.attachmentId&&!p.modelUrl&&!p.dataUrl))"
    )
    try await session.command("autoTitle", ["enabled": .bool(false)])
    session.reload()
    try await until("history metadata and preference survive reload") {
      session.ready && session.conversation?.title == "My permanent name"
        && session.state?.autoTitle == false
        && session.history.contains(where: { $0.id == "history-one" && $0.pinned == true })
    }
    print("PASS: history metadata, automatic titles and image attachment recovery")
  }
}
