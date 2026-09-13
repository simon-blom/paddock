# Paddock 0.1.6

A feature release. Windows x64, Linux x64 and, new in this release, the NVIDIA
DGX Spark. NVIDIA GPUs, driver 580 or newer.

## New

- **The DGX Spark is supported.** The GB10 in NVIDIA's DGX Spark joins the
  supported GPUs, with its own Linux build for the Spark (DGX OS 7 / Ubuntu
  24.04 or newer). Its kernels were tuned on the Spark and the model catalog
  was tested on it.

- **Gemma 4 reads small print.** A new endpoint setting, `max_image_tokens`,
  sets how much detail an image carries. Gemma 4's default is too coarse for
  fine print on a full page; raising it reads the figures correctly. More
  detail costs prompt tokens and time per image, so the default is unchanged.

- **Conversations resume from the last reply.** Qwen 3.5, 3.6 and 3.8,
  Nemotron 3.5 Lightning and Qwen 3.8 Flash-Next pick the next turn up from the
  end of the previous reply instead of reading the conversation again, and
  many conversations running at once each keep their place.

- **Qwen 3.8 Flash-Next caches repeated prompts.**

## Improved

- **Gemma 4 keeps to its memory budget.** With a tight `vram_budget` it now
  makes room for the conversation instead of going past the budget.

- **Nemotron 3.5 Lightning and 4-bit NVFP4 models are faster on the DGX
  Spark**, both reading prompts and generating text.

- **The Studio opens quickly** even with a very large chat history, and a
  stopped model keeps its logo and name.

## Fixed

- Opening a Studio chat before it had finished loading could save an empty
  copy over its history.

## Known

- The fp8 KV cache's paged attention on RTX 50-series cards shows a small
  numeric deviation in one split configuration (#5).
