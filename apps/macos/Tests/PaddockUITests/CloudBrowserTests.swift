import Foundation
import PaddockClient
import Testing

@testable import PaddockUI

struct CloudFixture: OpenRouterLoading {
  func catalog() async throws -> CloudCatalog {
    try ManagerWire.decode(
      CloudCatalog.self,
      from: Data(
        #"""
        {"ranked":true,"models":[
        {"id":"anthropic/claude-test","display":"Anthropic: Claude Test","created":1786000000,"ctx":200000,"maxOut":32000,"promptPrice":0.000003,"completionPrice":0.000015,"vision":true,"reasoning":true,"tools":true,"blurb":"A model for coding and agents."},
        {"id":"alibaba/test","display":"Alibaba: Test","created":1786100000,"ctx":1000000,"maxOut":8192,"promptPrice":0.0000001,"completionPrice":0.0000002,"tools":true},
        {"id":"google/chirp-test","display":"Google: Chirp","created":1786200000,"asr":true,"promptPrice":0.006,"completionPrice":0},
        {"id":"openrouter/free","display":"OpenRouter: Free","created":1786300000,"promptPrice":0,"completionPrice":0,"free":true},
        {"id":"openrouter/unknown","created":0,"promptPrice":null,"completionPrice":null}
        ]}
        """#.utf8))
  }
  func providers(for model: String) async throws -> CloudProviders {
    try ManagerWire.decode(
      CloudProviders.self,
      from: Data(
        #"""
        {"providers":[
        {"name":"Provider with a long serving name","tag":"provider/eu-west-1","ctx":200000,"maxOut":8192,"promptPrice":0.000003,"completionPrice":0.000015,"quant":"bf16","tps":85},
        {"name":"Provider with a long serving name","tag":"provider/us-east-1","ctx":128000,"promptPrice":0.000002,"completionPrice":0.000012,"quant":"fp8","tps":110}
        ]}
        """#.utf8))
  }
}

struct LargeCloudFixture: OpenRouterLoading {
  func catalog() async throws -> CloudCatalog {
    let rows = (0..<1000).map { index in
      // Alternate missing and populated capabilities/context to exercise the
      // old variable-height-row bug, with a catalog larger than the live API.
      """
      {"id":"qwen/test-\(index)","display":"Model \(index)","created":\(1_786_000_000 + index),
       "promptPrice":0.0000002,"completionPrice":0.0000003\(index.isMultiple(of: 2) ? ",\"vision\":true,\"ctx\":100000" : "")}
      """
    }.joined(separator: ",")
    return try ManagerWire.decode(
      CloudCatalog.self, from: Data("{\"ranked\":true,\"models\":[\(rows)]}".utf8))
  }
  func providers(for model: String) async throws -> CloudProviders {
    try await CloudFixture().providers(for: model)
  }
}

@Suite("Cloud catalog presentation")
struct CloudCatalogTests {
  @Test func searchUsesAliasesMultipleTokensDescriptionsAndSpeechFlags() async throws {
    let models = try await CloudFixture().catalog().models
    func find(_ query: String) -> [String] {
      CloudCatalogPresentation.entries(models, query: query, filters: [], order: .newest).map(\.id)
    }
    #expect(find("qwen") == ["alibaba/test"])
    #expect(find("claude coding") == ["anthropic/claude-test"])
    #expect(find("CLAUDE nowhere").isEmpty)
    #expect(find("speech google") == ["google/chirp-test"])
    #expect(CloudCatalogPresentation.name(models[0]) == "Claude Test")
    #expect(CloudCatalogPresentation.vendor(models[1]) == "Alibaba")
    let filtered = CloudCatalogPresentation.entries(
      models, query: "", filters: [.vision, .tools], order: .newest)
    #expect(filtered.map(\.id) == ["anthropic/claude-test"])
  }

  @Test func priceAndDateSortingNeverInventValues() async throws {
    let models = try await CloudFixture().catalog().models
    func ordered(_ order: CloudOrder) -> [String] {
      CloudCatalogPresentation.entries(models, query: "", filters: [], order: order).map(\.id)
    }
    #expect(ordered(.newest).first == "openrouter/free")
    #expect(ordered(.oldest).first == "anthropic/claude-test")
    #expect(ordered(.newest).last == "openrouter/unknown")
    #expect(ordered(.oldest).last == "openrouter/unknown")
    #expect(ordered(.trending) == models.map(\.id))
    #expect(
      ordered(.cheapest).prefix(3) == ["openrouter/free", "alibaba/test", "anthropic/claude-test"])
    #expect(
      ordered(.expensive).prefix(3) == ["anthropic/claude-test", "alibaba/test", "openrouter/free"])
    #expect(ordered(.largestContext).first == "alibaba/test")
    #expect(ordered(.smallestContext).first == "anthropic/claude-test")
    #expect(ordered(.largestContext).last == "openrouter/unknown")
  }

  @Test func pricesHaveExplicitUnitsAndDoNotRoundSmallValuesToFree() async throws {
    let models = try await CloudFixture().catalog().models
    #expect(CloudCatalogPresentation.perMillion(0.000003) == "$3")
    #expect(CloudCatalogPresentation.perMillion(nil) == "Not listed")
    #expect(CloudCatalogPresentation.perMillion(-1) == "Not listed")
    #expect(CloudCatalogPresentation.perMillion(.infinity) == "Not listed")
    #expect(CloudCatalogPresentation.perMillion(0) == "$0")
    #expect(CloudCatalogPresentation.dollars(0.000035) == "$0.000035")
    #expect(CloudCatalogPresentation.dollars(0.0000001) != "$0")
    #expect(CloudCatalogPresentation.priceSummary(models[2]) == "$0.006 · audio rate")
    #expect(CloudCatalogPresentation.publication(models[4]) == nil)
    #expect(CloudCatalogPresentation.priceSummary(models[4]) == "Pricing not listed")
  }

  @Test func modelLinksCannotEscapeThePublicModelPath() {
    #expect(CloudCatalogPresentation.modelURL("qwen/qwen3.8:free")?.host == "openrouter.ai")
    #expect(CloudCatalogPresentation.modelURL("~openai/gpt-astra-latest")?.host == "openrouter.ai")
    for id in [
      "a/..", "a/b/c", "a/b?key=secret", "a/%2e%2e", "a/b#test", "https://evil.invalid", "a/тест",
      "a/",
    ] {
      #expect(CloudCatalogPresentation.modelURL(id) == nil)
    }
  }

  @Test func latestAliasesRetainTheirMakerArtworkAndCleanNames() throws {
    let model = try ManagerWire.decode(
      CloudModel.self,
      from: Data(#"{"id":"~openai/gpt-astra-latest","display":"OpenAI: GPT Astra Latest"}"#.utf8))
    #expect(CloudCatalogPresentation.vendor(model) == "OpenAI")
    #expect(CloudCatalogPresentation.name(model) == "GPT Astra Latest")
  }
}

@Suite("Cloud catalog lifetime", .timeLimit(.minutes(1))) @MainActor
struct CloudBrowserTests {
  @Test func explicitProviderChoiceSurvivesNavigationAndNeverSilentlyFallsBack() async throws {
    let client = CountingCloud()
    let model = CloudBrowserModel(client: client)
    await model.refresh()
    let entry = try #require(model.models.first)
    await model.loadProviders(entry.id, force: true)
    #expect(model.studioPick(for: entry)?.provider == nil)
    #expect(model.selectProvider("provider/us-east-1", for: entry.id))
    #expect(model.studioPick(for: entry)?.provider == "provider/us-east-1")
    #expect(model.studioPick(for: entry)?.ctx == 128000)
    #expect(model.selectedProvider(for: entry.id)?.promptPrice == 0.000002)
    #expect(!model.selectProvider("not-in-catalog", for: entry.id))
    await model.loadProviders("another/model", force: true)
    await model.loadProviders(entry.id)
    #expect(model.studioPick(for: entry)?.provider == "provider/us-east-1")
    #expect(model.selectProvider("provider/eu-west-1", for: entry.id))
    #expect(model.studioPick(for: entry)?.ctx == 200000)
    #expect(model.studioPick(for: entry)?.maxOut == 8192)
    await client.removeProviders()
    await model.loadProviders(entry.id, force: true)
    #expect(model.selectedProviders[entry.id] == "provider/eu-west-1")
    #expect(model.studioPick(for: entry) == nil)
    #expect(model.selectProvider(nil, for: entry.id))
    #expect(model.studioPick(for: entry)?.pickKey == entry.id)
  }

  @Test func navigatingAwayDoesNotDiscardTheSharedCatalogFetch() async {
    let client = CountingCloud(delay: .milliseconds(100))
    let model = CloudBrowserModel(client: client)
    let request = Task { await model.loadIfNeeded() }
    while !model.loading { await Task.yield() }
    request.cancel()
    await model.loadIfNeeded()
    await request.value
    #expect(await client.catalogCalls == 1)
    #expect(!model.models.isEmpty)
    #expect(!model.loading)
    #expect(model.error == nil)
  }

  @Test func cachedCatalogAndProvidersSurviveRefreshErrors() async throws {
    let client = CountingCloud()
    let model = CloudBrowserModel(client: client)
    await model.loadIfNeeded()
    await model.loadIfNeeded()
    #expect(await client.catalogCalls == 1)
    let previous = model.models
    let date = model.refreshedAt
    await model.loadProviders("a/b", force: true)
    await model.loadProviders("a/b")
    #expect(await client.providerCalls == 1)
    #expect(model.providers.count == 2)
    await client.fail()
    await model.refresh()
    #expect(model.models == previous)
    #expect(model.refreshedAt == date)
    #expect(model.error != nil)
    await model.loadProviders("a/b", force: true)
    #expect(model.providers.count == 2)
    #expect(model.providersError != nil)
    #expect(!model.loading && !model.providersLoading)
  }

  @Test func olderProviderResultCannotReplaceNewSelectionOrReopenStoppedState() async throws {
    let client = ControlledCloud()
    let model = CloudBrowserModel(client: client)
    let first = Task { await model.loadProviders("a/first", force: true) }
    try await waitFor(client, count: 1)
    let second = Task { await model.loadProviders("a/second", force: true) }
    try await waitFor(client, count: 2)
    try await client.finish("a/second")
    await second.value
    try await client.finish("a/first")
    await first.value
    #expect(model.providerModel == "a/second")
    #expect(model.providers.first?.name == "a/second")
    let last = Task { await model.loadProviders("a/last", force: true) }
    try await waitFor(client, count: 3)
    model.stop()
    try await client.finish("a/last")
    await last.value
    #expect(model.providers.isEmpty)
    #expect(!model.providersLoading)
  }

  private func waitFor(_ client: ControlledCloud, count: Int) async throws {
    for _ in 0..<1000 {
      if await client.calls >= count { return }
      try await Task.sleep(for: .milliseconds(2))
    }
    Issue.record("Provider request did not reach the fixture")
    throw CancellationError()
  }
}

private actor CountingCloud: OpenRouterLoading {
  let delay: Duration
  init(delay: Duration = .zero) { self.delay = delay }
  var catalogCalls = 0
  var providerCalls = 0
  var failing = false
  var emptyProviders = false
  func fail() { failing = true }
  func removeProviders() { emptyProviders = true }
  func catalog() async throws -> CloudCatalog {
    catalogCalls += 1
    try await Task.sleep(for: delay)
    if failing { throw ManagerError.core("Test offline") }
    return try await CloudFixture().catalog()
  }
  func providers(for model: String) async throws -> CloudProviders {
    providerCalls += 1
    if failing { throw ManagerError.core("Test offline") }
    if emptyProviders {
      return try ManagerWire.decode(CloudProviders.self, from: Data(#"{"providers":[]}"#.utf8))
    }
    return try await CloudFixture().providers(for: model)
  }
}

private actor ControlledCloud: OpenRouterLoading {
  var calls = 0
  var pending: [String: CheckedContinuation<CloudProviders, any Error>] = [:]
  func catalog() async throws -> CloudCatalog { try await CloudFixture().catalog() }
  func providers(for model: String) async throws -> CloudProviders {
    calls += 1
    return try await withCheckedThrowingContinuation { pending[model] = $0 }
  }
  func finish(_ id: String) throws {
    let result = try ManagerWire.decode(
      CloudProviders.self, from: Data("{\"providers\":[{\"name\":\"\(id)\"}]}".utf8))
    pending.removeValue(forKey: id)?.resume(returning: result)
  }
}
