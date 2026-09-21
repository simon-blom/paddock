import Foundation
import PaddockStudio
import WebKit

extension Checks {
  /// Offscreen, isolated store, synthetic local and cloud Responses streams.
  /// Unknown inference fails closed. No real provider credential or request.
  @MainActor func compareChecks(_ session: StudioWorkspace) async throws {
    _ = try await session.webView.callAsyncJavaScript(
      #"""
      const pinia=document.querySelector('#app').__vue_app__.config.globalProperties.$pinia;
      const chat=pinia._s.get('chat'),models=pinia._s.get('models');
      window.compareChat=chat;window.compareRequests=[];window.compareHold=false;
      models.models=[{id:'compare-local',display:'Local fixture',ownedBy:'test',kind:'chat',status:'ok',port:16557},
        {id:'cloud:fixture:meta/muse@meta',display:'Cloud fixture',ownedBy:'test',kind:'chat',status:'ok',cloud:{endpoint:'fixture',endpointName:'OpenRouter'}}];
      models.refresh=async()=>{};models.integrateRunnerRows=()=>{};
      for(const model of models.models)models.caps[model.id]={vision:true,webSearch:false,mcpServers:[],taskTags:[],timestampGranularities:[],include:[],reasoning:'none',maxCtx:4096};
      const original=window.fetch.bind(window);
      window.fetch=async(url,options={})=>{
        const path=String(url);
        if(path==='/api/conversations/compare-fixture'&&options.method==='PUT'&&window.compareFailSave){window.compareFailSave=false;return new Response('Synthetic storage failure',{status:503})}
        if(/\/v1\/(responses|chat\/completions)/.test(path)){
          if(!path.startsWith('/api/runners/16557/')&&!path.startsWith('/api/cloud/fixture/'))throw new Error('Unexpected inference destination');
          const body=JSON.parse(options.body);window.compareRequests.push({path,body});
          if(window.compareHold)return new Promise((resolve,reject)=>options.signal.addEventListener('abort',()=>reject(new DOMException('Aborted','AbortError')),{once:true}));
          const text=path.includes('/cloud/')?'## Cloud fixture\n\nCloud reply.':'## Local fixture\n\nLocal reply.';
          const response={id:'fixture',status:'completed',output:[{type:'message',role:'assistant',content:[{type:'output_text',text}]}],usage:{input_tokens:8,output_tokens:4}};
          return new Response([{type:'response.output_text.delta',delta:text},{type:'response.completed',response}].map(e=>'data: '+JSON.stringify(e)+'\n\n').join(''),{headers:{'Content-Type':'text/event-stream'}});
        }
        return original(url,options);
      };
      const q={id:'question',parentId:null,role:'user',createdAt:1,content:[{type:'text',text:'Compare question'}]};
      const lane=(id,group,model)=>({id,parentId:q.id,role:'assistant',model,group,createdAt:2,content:[{type:'text',text:'Stored '+id}]});
      const c={id:'compare-fixture',title:'Compare fixture',model:'compare-local',compareModels:models.models.map(m=>m.id),params:{thinking:false,reasoningEffort:'',stop:[]},systemPrompt:'',createdAt:1,updatedAt:2,leafId:'local-new',messages:[q,lane('local-old','old','compare-local'),lane('cloud-old','old',models.models[1].id),lane('local-new','new','compare-local'),lane('cloud-new','new',models.models[1].id)]};
      const r=await fetch('/api/conversations/'+c.id,{method:'PUT',headers:{'Content-Type':'application/json'},body:JSON.stringify(c)});if(!r.ok)throw new Error('Fixture save failed');
      chat.conversations.push({...c,messages:[]});
      """#, arguments: [:], in: nil, contentWorld: .page)
    try await session.command("autoTitle", ["enabled": .bool(false)])
    try await session.command("renderer", ["mode": .string("native")])
    await session.open("compare-fixture")
    try await until("native compare projection ready") {
      session.state?.nativeTranscript?.hasComparisons == true
    }
    guard session.state?.nativeTranscript?.available == true,
      session.state?.nativeTranscript?.blocks.last?.messages.count == 2
    else { throw Failure(message: "Compare fell back or lost a lane") }
    try await check(
      "native Compare does not mount a web transcript", "!document.querySelector('.thread')")
    func target(_ id: String) throws -> StudioMessageTarget {
      guard let transcript = session.state?.nativeTranscript,
        let target = StudioMessageTarget(transcript: transcript, messageId: id)
      else { throw Failure(message: "Missing compare target") }
      return target
    }
    guard
      await session.messageAction("branch", target: try target("local-new"), branch: "local-old")
    else { throw Failure(message: session.error ?? "Compare branch failed") }
    guard
      session.state?.nativeTranscript?.messages.map(\.id) == ["question", "local-old", "cloud-old"]
    else { throw Failure(message: "Only part of the group changed") }
    try await check(
      "whole compare branch switches without inference", "window.compareRequests.length===0")
    session.beginMessageEdit(try target("question"))
    session.messageEdit?.text = "Edited compare question"
    _ = try await session.webView.evaluateJavaScript("window.compareFailSave=true")
    guard await !session.submitMessageEdit(), session.hasMessageEdit else {
      throw Failure(message: "Failed compare save lost its draft")
    }
    try await check(
      "failed compare edit is not submitted",
      "window.compareRequests.length===0&&window.compareChat.active.messages.length===5")
    try await until("failed compare admission settles") { !session.busy }
    try await check("shared compare admission settles", "!window.paddockWorkspace.state().busy")
    guard await session.submitMessageEdit() else {
      throw Failure(message: session.error ?? "Compare edit failed")
    }
    try await until("both edited lanes complete") { !session.busy }
    guard session.state?.nativeTranscript?.available == true,
      session.state?.nativeTranscript?.blocks.last?.messages.count == 2,
      session.state?.nativeTranscript?.messages.last?.error.isEmpty == true
    else { throw Failure(message: "Edited comparison failed") }
    try await check(
      "edited compare persists both lanes and preserves provider pin",
      #"return window.compareRequests.length===2&&window.compareRequests.some(r=>r.path.includes('/cloud/')&&r.body.model==='cloud:fixture:meta/muse@meta')&&window.compareChat.active.messages.length===8"#
    )
    guard
      await session.messageAction(
        "branch", target: try target(session.state!.nativeTranscript!.messages[0].id),
        branch: "question")
    else { throw Failure(message: "Compare question branch restore failed") }
    guard
      session.state?.nativeTranscript?.messages.map(\.id) == ["question", "local-old", "cloud-old"]
    else { throw Failure(message: "Compare branch memory lost") }
    _ = try await session.webView.evaluateJavaScript("window.compareHold=true")
    session.beginMessageEdit(try target("question"))
    session.messageEdit?.text = "Stop both comparison lanes"
    guard await session.submitMessageEdit() else {
      throw Failure(message: session.error ?? "Streaming compare edit failed")
    }
    try await check(
      "both native comparison lanes begin streaming", "window.compareRequests.length===4")
    try await until("both streaming lanes project natively") {
      session.state?.nativeTranscript?.available == true
        && session.state?.nativeTranscript?.blocks.last?.messages.filter(\.streaming).count == 2
    }
    await session.perform("stop")
    try await until("comparison cancellation settles") { !session.busy }
    guard let stopped = session.state?.nativeTranscript?.blocks.last?.messages,
      stopped.count == 2, stopped.allSatisfy({ $0.stopped && !$0.streaming })
    else { throw Failure(message: "Compare stop did not settle both native lanes") }
    print(
      "PASS: native compare projection, whole-group navigation, durable edit fan-out and branch memory"
    )
    print("PASS: both comparison streams stay native and Stop cancels both")
  }
}
