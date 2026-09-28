# Paddock 0.1.11

A memory release: the Qwen 3.5-3.8 models, Nemotron and Qwen 3.8 Flash-Next
fit more context on the same card, and long agent conversations resume from
the cache on every turn. Nemotron gains speculative decoding for agent
traffic and decodes long contexts much faster. Windows x64, Linux x64 and the
NVIDIA DGX Spark. NVIDIA GPUs, driver 580 or newer. The macOS pre-release for
Apple Silicon is built from the same commit.

## New

- **More context on the same card.** The resume points a long conversation
  picks up from used to take a fixed reserve before the context got any
  memory - on a 24 GB card that alone held a 27B model to about 21K tokens
  at one request. They now live inside the context cache itself and hand
  their memory back as a conversation grows. Measured on the same card and
  budget: Qwen 3.5 9B went from 23.8K to 41.6K tokens. This covers the
  Qwen 3.5-3.8 hybrids, Nemotron and Qwen 3.8 Flash-Next; a roomy card still
  keeps just as many resume points as before.

- **Qwen 3.8 Flash-Next's cache is paged.** Its memory is now planned inside
  the endpoint's budget (it used to take fixed amounts whatever the budget
  said), and a cached prefix is shared instead of copied, so resumed turns
  reach their first token 10-12% sooner.

- **Nemotron speculates on agent traffic** with its DFlash and DSpark
  drafters, including sampled and tool-calling requests and any number of
  concurrent requests. On the DGX Spark, tool-carrying sampled requests went
  from 80 to 118 tokens/s. DSpark, NVIDIA's recommended Spark drafter, is in
  the catalog.

## Improved

- **Nemotron decodes long contexts about four times faster** - 12.5 to 49.8
  tokens/s at 259K.

- **Long Nemotron agent conversations resume from the cache on every turn.**
  Past about 140K tokens every turn used to start over from the system
  prompt.

- **Bonsai runs faster on the DGX Spark**, with its ternary weights on the
  tensor cores at every batch size and an fp8 KV cache by default.

## Fixed

- **A Qwen 3.8 Flash-Next reply generated while speculating keeps its resume
  points**, so the next turn picks up at the end of the reply instead of
  reading the whole reply again.

- **A vision or drafter file beside the weights is checked against the model
  before it loads.** A folder several models share no longer hands one model
  another's file, and a mismatched file named in the configuration is
  refused with the reason.

- **A `vram_budget` written in GiB instead of MiB is refused** with the value
  it should have been.

- **Prefill on the DGX Spark** no longer fails for the Qwen 3.5-3.8
  full-attention layers with an fp8 KV cache.

## macOS (pre-release)

- Laya runs natively on Metal, with the native Reads page.
- Qwen 3.8 Flash-Next runs faster on Metal and reuses cached prefixes across
  agent turns.
- Qwen agent-turn resume points are kept on Metal.

## Known

- **Laya reads text only.**

- **Laya's confidence on choices with more than ten options is not
  calibrated:** the English checkpoint ships an out-of-range temperature for
  that case, which Paddock clamps, as Laya's own server does.

- **Switching Vision off does not unload the image tower** when its file sits
  beside the model: the model still answers images, and the memory estimate
  does not count the tower (about 0.9 GB).

- On-demand loading covers Whisper only.

- The fp8 KV cache's paged attention on RTX 50-series cards shows a small
  numeric deviation in one split configuration (#5).
