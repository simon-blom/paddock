# Paddock 0.1.10

A feature release: Laya, an open model built only to answer questions about
a text, a much richer Reads page, image input for DiffusionGemma's reads, and
faster Qwen 3.8 Flash-Next. Windows x64, Linux x64 and the NVIDIA DGX Spark.
NVIDIA GPUs, driver 580 or newer. The macOS pre-release for Apple Silicon is
built from the same commit.

## New

- **Laya, a decision model.** Laya (ConvAI Innovations, Apache 2.0) answers
  yes/no, multiple-choice and scored questions about a text with a calibrated
  probability each, in one pass per question and without writing any text -
  a few milliseconds a question. It serves the same `POST /v1/systemone` as
  DiffusionGemma, so the Reads page works with it unchanged. The download
  (2.4 GB) holds an English model, a multilingual one for 100+ languages and
  a typed-decisions one; Paddock picks between the first two from the text
  itself, as Laya's own server does. A text longer than one question's window
  is read in overlapping windows instead of being cut.

- **Images in reads.** DiffusionGemma reads a question set over pictures as
  well as text, with its own vision companion (1.2 GB) built from Google's
  checkpoint. Attach pictures on the Reads page by button, drop or paste.

- **Conditional questions.** A question can be asked only when an earlier
  answer is one of the values you pick, or read with an earlier question's
  answer already known. A question that is skipped answers `null` and says
  why.

- **More ways to read.** Several denoising steps with the answer template
  held in place, a thought written before the answers, and repeated reads
  that report how much the answer moves between them.

## Improved

- **The Reads page keeps its history** in a side panel beside the questions,
  stored by the manager like conversations, and stays in the sidebar with no
  model running so earlier reads open any time.

- **DiffusionGemma's compact build starts on NVIDIA GPUs.**

- **Qwen 3.8 Flash-Next decodes faster,** with and without speculative
  decoding, and requests that carry tools now speculate too.

- **Paddock's own app icon** in the Studio, the Windows executables and the
  menu bar.

## Fixed

- **Flash-Next greedy decoding with speculation** no longer drifts from the
  answer it gives without speculation on a small share of requests.

- **The Studio over the network** opens at the key prompt again instead of
  reporting that it could not open its saved data.

## macOS (pre-release)

- DiffusionGemma reads images on Metal.
- The native Reads page takes pictures by paste and import, keeps its history
  in the shared database the web Studio uses, and navigates it from the
  sidebar.
- Fixes to notification previews and benchmark forms.

## Known

- **Laya runs on NVIDIA GPUs only** for now, and reads text only.

- **Laya's confidence on choices with more than ten options is not
  calibrated:** the English checkpoint ships an out-of-range temperature for
  that case, which Paddock clamps, as Laya's own server does.

- **Switching Vision off does not unload the image tower** when its file sits
  beside the model: the model still answers images, and the memory estimate
  does not count the tower (about 0.9 GB).

- On-demand loading covers Whisper only.

- The fp8 KV cache's paged attention on RTX 50-series cards shows a small
  numeric deviation in one split configuration (#5).
