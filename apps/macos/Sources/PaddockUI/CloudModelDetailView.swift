import PaddockClient
import SwiftUI

/// Compact model/provider comparison. Price, limits and routing sit together;
/// long descriptions and billing caveats are disclosures, not a full-page form.
struct CloudModelDetailView: View {
  let entry: CloudModel
  @Bindable var browser: CloudBrowserModel
  var service: CloudService = .openrouter
  var enabled: Set<String> = []
  var canAdd = false
  var onAdd: (CloudModelPick) -> Void = { _ in }
  @State private var aboutExpanded = false
  private var current: Bool { browser.providerModel == entry.id }
  private var audio: Bool { CloudCatalogPresentation.audioRate(entry) }

  var body: some View {
    PaddockScrollView {
      VStack(alignment: .leading, spacing: 26) {
        HStack(alignment: .top, spacing: 14) {
          ModelAvatar(vendor: CloudCatalogPresentation.vendor(entry) ?? service.vendor, size: 44)
          VStack(alignment: .leading, spacing: 6) {
            Text(CloudCatalogPresentation.name(entry)).font(.system(size: 23, weight: .semibold))
              .tracking(-0.5).fixedSize(horizontal: false, vertical: true).textSelection(.enabled)
            Text(entry.id).font(.system(size: 11)).foregroundStyle(.secondary).textSelection(
              .enabled
            )
            .fixedSize(horizontal: false, vertical: true)
          }
        }
        let features = CloudFeature.allCases.filter { $0.matches(entry) }
        if !features.isEmpty {
          LazyVGrid(
            columns: [GridItem(.adaptive(minimum: 80), alignment: .leading)], alignment: .leading,
            spacing: 8
          ) {
            ForEach(features) { feature in
              Label(feature.rawValue, systemImage: feature.symbol).font(.system(size: 11))
            }
          }.foregroundStyle(.secondary)
        }
        HStack(spacing: 8) {
          if let ctx = entry.ctx { metric("Context", CloudCatalogPresentation.tokens(ctx)) }
          if let maxOut = entry.maxOut {
            metric("Max output", CloudCatalogPresentation.tokens(maxOut))
          }
          if let date = CloudCatalogPresentation.publication(entry) { metric("Published", date) }
        }
        VStack(alignment: .leading, spacing: 0) {
          Text(service == .openrouter ? "Serving options" : "Model access")
            .font(.system(size: 13, weight: .semibold)).foregroundStyle(PaddockStyle.primary)
            .padding(.bottom, 12).accessibilityAddTraits(.isHeader)
          HStack {
            Text(service == .openrouter ? "Provider" : "Model").frame(
              maxWidth: .infinity, alignment: .leading)
            if hasPricing {
              Text(audio ? "Audio rate" : "$ in · out / M").frame(width: 104, alignment: .trailing)
            }
            Color.clear.frame(width: 40, height: 1)
          }.font(.system(size: 10)).foregroundStyle(.secondary).padding(.bottom, 8)
          routingRow(
            name: service == .openrouter ? "Auto · OpenRouter" : service.rawValue,
            subtitle: service == .openrouter ? "Provider chosen per request" : nil,
            input: entry.promptPrice, output: entry.completionPrice,
            pick: CloudModelPick(model: entry, provider: nil))
          if service == .openrouter {
            if !current || browser.providersLoading {
              ProgressView("Loading providers…").controlSize(.small).padding(.vertical, 12)
            }
            if current {
              if let error = browser.providersError {
                Text(error).font(.system(size: 11)).foregroundStyle(PaddockStyle.caution).padding(
                  .vertical, 10)
                Button("Try again") { Task { await browser.loadProviders(entry.id, force: true) } }
                  .buttonStyle(FlatButtonStyle()).disabled(browser.providersLoading)
              }
              ForEach(Array(browser.providers.enumerated()), id: \.offset) { _, provider in
                routingRow(
                  name: provider.name, subtitle: provider.tag,
                  input: provider.promptPrice, output: provider.completionPrice,
                  pick: CloudModelPick(model: entry, provider: provider), provider: provider)
              }
              if browser.providers.isEmpty && !browser.providersLoading
                && browser.providersError == nil
              {
                Text("No individual providers listed. Auto routing is available.")
                  .font(.system(size: 11)).foregroundStyle(.secondary).padding(.vertical, 10)
              }
            }
          }
        }.padding(12).background(PaddockStyle.surface)
          .clipShape(RoundedRectangle(cornerRadius: PaddockStyle.Radius.card))
          .accessibilityIdentifier("cloud-provider-comparison")
        if service == .openrouter || entry.blurb?.isEmpty == false {
          DisclosureGroup(
            service == .openrouter ? "About & pricing details" : "About", isExpanded: $aboutExpanded
          ) {
            VStack(alignment: .leading, spacing: 10) {
              if let blurb = entry.blurb {
                Text(blurb).textSelection(.enabled).fixedSize(horizontal: false, vertical: true)
              }
              if service == .openrouter {
                Text(
                  "Provider-specific picks never fall back to another provider. Prices are listed rates, not a quote; caching, context tiers and image/audio charges may change the final cost."
                )
                Text(
                  "Throughput is OpenRouter's last-30-minute report, not a Paddock benchmark or guarantee."
                )
                if let date = browser.providersAt {
                  Text("Provider snapshot · \(date.formatted(date: .omitted, time: .shortened))")
                }
                if let url = CloudCatalogPresentation.modelURL(entry.id) {
                  Link("Full details on OpenRouter ↗", destination: url)
                }
              }
            }.font(.system(size: 11)).foregroundStyle(.secondary).padding(.top, 8)
          }.font(.system(size: 12)).padding(12).background(PaddockStyle.surface)
            .clipShape(RoundedRectangle(cornerRadius: PaddockStyle.Radius.card))
        }
        if audio {
          Text(
            "Audio rate: billing time unit is not supplied by the catalog. Check the model page before comparing costs."
          )
          .font(.system(size: 10)).foregroundStyle(.secondary)
        }
      }.padding(.horizontal, 22).padding(.top, 24).padding(.bottom, 32)
        .frame(maxWidth: 840, alignment: .leading).frame(maxWidth: .infinity, alignment: .top)

    }.scrollBounceBehavior(.basedOnSize).background(PaddockStyle.canvas)
      .accessibilityIdentifier("cloud-model-details")
  }

  private var hasPricing: Bool {
    service == .openrouter || entry.promptPrice != nil || entry.completionPrice != nil
  }

  private func metric(_ name: String, _ value: String) -> some View {
    VStack(alignment: .leading, spacing: 4) {
      Text(name).font(.system(size: 10)).foregroundStyle(.secondary)
      Text(value).font(.system(size: 11)).monospacedDigit()
    }.frame(maxWidth: .infinity, alignment: .leading)
  }

  private func routingRow(
    name: String, subtitle: String?, input: Double?, output: Double?,
    pick: CloudModelPick, provider: CloudProvider? = nil
  ) -> some View {
    VStack(alignment: .leading, spacing: 6) {
      HStack(spacing: 8) {
        VStack(alignment: .leading, spacing: 3) {
          Text(name).font(.system(size: 12, weight: .medium)).lineLimit(2)
          if let subtitle {
            Text(subtitle).font(.system(size: 10)).foregroundStyle(.secondary).lineLimit(1).help(
              subtitle)
          }
        }.frame(maxWidth: .infinity, alignment: .leading)
        if hasPricing {
          Text(
            audio
              ? CloudCatalogPresentation.dollars(input)
              : "\(CloudCatalogPresentation.perMillion(input)) · \(CloudCatalogPresentation.perMillion(output))"
          )
          .font(.system(size: 10)).monospacedDigit().multilineTextAlignment(.trailing)
          .frame(width: 104, alignment: .trailing).textSelection(.enabled)
        }
        CloudAddButton(
          added: enabled.contains(pick.pickKey),
          enabled: canAdd && (provider.map { browser.canSelect($0, for: entry.id) } ?? true),
          label: pick.pickKey
        ) {
          onAdd(pick)
        }
      }
      if let provider {
        Text(providerFacts(provider)).font(.system(size: 10)).foregroundStyle(.secondary)
          .fixedSize(horizontal: false, vertical: true)
      }
    }.padding(.vertical, 10).accessibilityElement(children: .contain)
  }

  private func providerFacts(_ provider: CloudProvider) -> String {
    var facts: [String] = []
    if let ctx = provider.ctx { facts.append("\(CloudCatalogPresentation.tokens(ctx)) ctx") }
    if let maxOut = provider.maxOut {
      facts.append("\(CloudCatalogPresentation.tokens(maxOut)) max out")
    }
    if let tps = provider.tps, tps.isFinite, tps >= 0 {
      facts.append("\(tps.formatted(.number.precision(.fractionLength(0...1)))) tok/s")
    }
    if let quant = provider.quant { facts.append(quant) }
    return facts.joined(separator: " · ")
  }
}
