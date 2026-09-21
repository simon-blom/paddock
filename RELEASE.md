# Paddock 0.1.8

A feature release: a new model that puts 27B on a 16 GB card, a fix for image
requests under load, and a smoother Studio. Windows x64, Linux x64 and the
NVIDIA DGX Spark. NVIDIA GPUs, driver 580 or newer.

## New

- **Bonsai 2 27B: a 27B model on a 16 GB card.** Prism ML's ternary build of
  Qwen 3.8 27B is 5.9 GB of weights and is served exactly as shipped. Image
  input is a separate 0.9 GB download, on by default. A 16 GB card holds about
  43,000 tokens of context with text only and about 27,000 with images; a
  24 GB card reaches the full 256K with the 8-bit KV cache. Ternary weights
  are close to the original model, not identical to it.

- **Qwen 3.8 Flash-Next speculates for sampled requests.** The drafter used to
  help only requests that did not sample. Requests with a temperature, which
  is the default, now speculate too.

- **Qwen 3.8 Flash-Next has an NVFP4 build for Blackwell cards.** The catalog
  offers the distilled NVFP4 checkpoint for RTX 50-series class GPUs, and the
  engine reads MX-quantized checkpoints as they ship.

- **An image segmentation endpoint.** `POST /v1/segmentations` takes image
  chips and returns rasters from a DINOv3-based segmentation checkpoint served
  by path. No catalog model uses it yet.

## Improved

- **Long chats stay smooth in the Studio.** Formatting a long conversation now
  runs in the background and is spread across frames, so scrolling and
  streaming no longer stall on large replies, tables and diagrams.

- **Cloud models get reply lengths sized from the provider's own token
  counts**, not from an estimate.

- **Conversation summaries hold up under pressure.** Four fixes contributed by
  @DivyamTalwar: a window with no room left is no longer sent the whole
  transcript, a model switch mid-summary no longer saves the old model's
  summary under the new one, a stalled summary request times out, and an
  injected summary counts against the reply budget.

## Fixed

- **Several image requests at once could all answer with garbage.** On the
  Qwen 3.5, 3.6 and 3.8 vision models, two or more image requests arriving
  together, each with a prompt of 128 tokens or more, could all return a run
  of "!" in place of an answer. A single request at a time was never affected.

- **Qwen 3.8 Flash-Next tool calls with the IQ3 build.** Tool calls came back
  as plain text, `tool_choice: "required"` was refused and thinking budgets
  were rejected. The engine now reads the tool-call format from the model's
  own template when it has no entry of its own.

## Known

- **Switching Vision off does not unload the image tower** when its file sits
  beside the model: the model still answers images, and the memory estimate
  does not count the tower (about 0.9 GB).

- The fp8 KV cache's paged attention on RTX 50-series cards shows a small
  numeric deviation in one split configuration (#5).
