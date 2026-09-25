import AppKit
import PaddockStudio
import WebKit

@testable import PaddockUI

extension Checks {
  func desktopChecks(_ session: StudioWorkspace) async throws {
    guard let core else { throw Failure(message: "Missing synthetic core") }
    let model = WorkspaceModel(client: core, preparedStudio: session)
    // The fixture runner is an intercepted transport, not a real listener.
    // Keep discovery isolated just as attachmentSelectionChecks does. A real
    // refresh correctly removes this synthetic model from the live inventory.
    _ = try await session.webView.callAsyncJavaScript(
      #"""
      const models=document.querySelector('#app').__vue_app__.config.globalProperties.$pinia._s.get('models');
      models.models=[{id:'fixture',ownedBy:'test',kind:'chat',status:'ok',port:16555}];
      models.caps.fixture={vision:true,webSearch:false,mcpServers:[],taskTags:[],timestampGranularities:[],include:[],reasoning:'none',maxCtx:4096};
      models.currentId='fixture';models.refresh=async()=>{};models.integrateRunnerRows=()=>{};
      const original=window.fetch.bind(window);
      window.fetch=async(url,options={})=>{
        const path=String(url);
        if(path==='/api/runners/16555/v1/responses'){
          const request=JSON.parse(options.body);
          if(request.stream!==true)throw new Error('Unexpected background inference in OS fixture');
          const event={type:'response.completed',response:{id:'desktop-response',status:'completed',output:[{type:'message',role:'assistant',content:[{type:'output_text',text:'Synthetic quick answer'}]}],usage:{input_tokens:5,output_tokens:3}}};
          return new Response('event: response.completed\ndata: '+JSON.stringify(event)+'\n\n',{headers:{'Content-Type':'text/event-stream'}});
        }
        if(/\/v1\/(responses|chat\/completions)/.test(path))throw new Error('Unexpected live inference route in OS fixture');
        return original(url,options);
      };
      """#, arguments: [:], in: nil, contentWorld: .page)
    try await session.command("autoTitle", ["enabled": .bool(false)])
    let originalID = session.conversation?.id
    let quick = QuickQuestionModel()
    quick.text = "Synthetic quick question through the shared orchestrator"
    quick.modelID = "fixture"
    model.draft.message = "Keep the original native draft"
    guard !(await quick.handoff(to: model) {}), model.draft.hasContent,
      session.conversation?.id == originalID, quick.hasContent
    else {
      throw Failure(message: "Quick question replaced existing draft")
    }
    print("PASS: quick question leaves the active conversation and both drafts intact on conflict")
    model.draft.message = ""
    var tracker = DesktopEventTracker()
    var events: [DesktopEvent] = []
    if let state = session.state { _ = tracker.studio(state) }
    model.onStudioState = { events.append(contentsOf: tracker.studio($0)) }
    var opens = 0
    guard await quick.handoff(to: model, openStudio: { opens += 1 }), !quick.hasContent else {
      throw Failure(message: quick.error ?? "Quick handoff failed")
    }
    try await until("quick question finishes through the existing stream") { !session.busy }
    guard opens == 1, !model.draft.hasContent, model.chat === session,
      session.conversation?.id != originalID
    else { throw Failure(message: "Quick handoff ownership failed") }
    try await until("one completion notification is emitted for quick question") {
      events.filter { $0.kind == .replyReady }.count == 1
    }
    print(
      "PASS: quick question uses one shared workspace, saves its conversation and emits one completion"
    )
    let quickID = session.conversation!.id
    await model.handleDesktopRequest(DesktopRequest(.conversation(quickID)))
    guard session.conversation?.id == quickID else {
      throw Failure(message: "Notification route did not reopen the exact conversation")
    }
    print("PASS: notification routing selects the exact existing conversation")
    quick.text = "Review synthetic image before sending"
    let image = NSBitmapImageRep(
      bitmapDataPlanes: nil, pixelsWide: 2, pixelsHigh: 2,
      bitsPerSample: 8, samplesPerPixel: 4, hasAlpha: true, isPlanar: false,
      colorSpaceName: .deviceRGB, bytesPerRow: 8, bitsPerPixel: 32)!
    image.bitmapData!.initialize(repeating: 180, count: 16)
    quick.addImage(image.representation(using: .png, properties: [:])!, png: true)
    guard await quick.handoff(to: model, openStudio: { opens += 1 }), !quick.hasContent else {
      throw Failure(message: quick.error ?? "Quick image handoff failed")
    }
    try await until("quick image is ready for review in the full composer") {
      !session.uploading && session.attachments.first?.ready == true
    }
    guard model.draft.message == "Review synthetic image before sending",
      session.conversation == nil,
      session.attachments.count == 1
    else { throw Failure(message: "Quick attachment sent without review") }
    print(
      "PASS: quick image transfers to the full composer without generating or discarding the draft")
    session.removeAttachment(session.attachments[0].id)
    model.draft.message = ""
    try await until("quick attachment cleanup") { !session.hasAttachments }
    model.onStudioState = nil
  }
}
