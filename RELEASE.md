# Paddock 0.1.9

A feature release: a diffusion language model with a structured-read API,
image generation and editing, a small MiniCPM5 model, Claude Code working
again with its current versions, and long-context fixes for Qwen 3.8
Flash-Next. Windows x64, Linux x64 and the NVIDIA DGX Spark. NVIDIA GPUs,
driver 580 or newer. The macOS pre-release for Apple Silicon is built from
the same commit.

## New

- **DiffusionGemma 26B A4B.** Google's diffusion version of Gemma 4 26B
  writes a whole block of text per step instead of one token at a time.
  27 GB of weights at full quality, text only for now.

- **Structured reads.** `POST /v1/systemone` asks DiffusionGemma a fixed set
  of yes/no, multiple-choice and scored questions about a text and returns
  every answer with its confidence, in the shape of the Jev API. The
  Studio's new Reads page builds the questions, runs them over a pasted or
  attached text and saves question sets for reuse.

- **Qwen-Image 2.1: pictures from a prompt, and edits of your own
  pictures.** `POST /v1/images/generations` and `POST /v1/images/edits`, with
  previews streamed while the picture renders. In the Studio a picture is a
  conversation of its own: ask for one, then ask for changes. Full quality
  needs about 17 GB of VRAM and the compact build about 10 GB, plus room
  that grows with the picture. The Qwen Research License allows research and
  evaluation only, not commercial use.

- **MiniCPM5 2B**, a small model with tool calling.

- **Whisper endpoints can load their model on demand.** An endpoint set to
  load on first request stays up without holding GPU memory, loads the model
  when a transcription arrives and unloads it after the idle time you
  choose. Set it under Model loading in the endpoint's settings; the change
  applies without a restart. Other models stay loaded as before.

## Improved

- **Qwen 3.8 Flash-Next at long context.** Past about 2,000 tokens the model
  attends sparsely, and Paddock now serves it that way. Before, longer
  prompts got full attention, which is not the model's own behaviour. The
  full 262K context now loads, a long prompt no longer freezes the other
  conversations while it is read, and a continued conversation resumes from
  its cache at any depth.

- **Long prompts no longer stall other conversations** on any model: while
  one conversation reads a long prompt, the others keep getting tokens at a
  steady pace.

- **Agent loops reuse more of their cache.** The next turn of an agent
  resumes at the tool call instead of reading the whole previous reply
  again, and the model's earlier reasoning now reaches its template on every
  model, as the model expects.

- **Replies run to the context window when no limit is set.** A request
  without `max_tokens` used to stop at 1,024 tokens, which cut reasoning
  models off before they answered.

- **Streams stay alive during long work.** A long prompt or a long tool call
  no longer leaves a stream silent long enough for a client to hang up.

- **Gemma 4 serves 4-bit k-quant GGUF files** such as Q4_K_M at 4 bits.

## Fixed

- **Claude Code works again.** Current Claude Code versions send system
  messages in the middle of a conversation, and every request failed with
  "invalid message role". They are accepted now.

- **Claude Code at long context.** Token usage counts the cached prefix
  once, so Claude Code no longer compacts at half the window. A prompt past
  the window returns Anthropic's own "prompt is too long" error, so Claude
  Code compacts and carries on. Requests with tools keep speculating after
  the first turn.

## macOS (pre-release)

- Image generation and editing with Qwen-Image on Metal.
- DiffusionGemma on Metal, with the Reads page in the native app.
- MiniCPM5 on Metal, and faster Bonsai.
- Usage, activity and cache views, storage per model, saved settings
  profiles and on-demand Whisper loading in Settings.
- Fixes to documents, transcripts, viewers and scrolling.

## Known

- **Switching Vision off does not unload the image tower** when its file sits
  beside the model: the model still answers images, and the memory estimate
  does not count the tower (about 0.9 GB).

- On-demand loading covers Whisper only.

- The fp8 KV cache's paged attention on RTX 50-series cards shows a small
  numeric deviation in one split configuration (#5).
