# Paddock 0.1.7

A feature release, mostly about Qwen 3.8 Flash-Next. Windows x64, Linux x64 and
the NVIDIA DGX Spark. NVIDIA GPUs, driver 580 or newer.

## New

- **Qwen 3.8 Flash-Next drafts its own tokens.** The model's companion drafter
  is downloaded with it and proposes the next tokens, which the model then
  verifies, so answers arrive faster with the same text. Measured on a DGX
  Spark: 26-28 tokens a second without it, 35-40 with it. Speculation stays a
  per-endpoint setting, off unless you turn it on.

- **Qwen 3.8 Flash-Next has a compact 4-bit build.** The catalog now offers the
  UD-Q4_K_XL export beside the smaller default, for machines with the memory
  to spare. It is not the default: it is 85 GB of weights.

## Improved

- **Qwen 3.8 Flash-Next reads prompts about twice as fast.** A 1024-token
  prompt went from roughly 750 to 1450 tokens a second on a DGX Spark, from
  the expert, attention and hyper-connection work now running on the tensor
  cores and from fewer passes over the same data.

- **A model keeps its full context after a restart on unified-memory machines.**
  On a DGX Spark the memory a stopped model left behind was not counted as
  free, so each restart planned a smaller conversation cache than the one
  before. It is now measured and counted.

- **A stopped model no longer looks like it is still running.** Stopping a
  model left a file behind that made its port read as occupied: the Studio
  stopped showing the model and starting it again was refused. The file now
  goes with the model, a leftover one is ignored, and a refusal says what is
  holding the port.

## Known

- The fp8 KV cache's paged attention on RTX 50-series cards shows a small
  numeric deviation in one split configuration (#5).
